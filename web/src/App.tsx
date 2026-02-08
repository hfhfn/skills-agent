import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";

import { ChatTimeline } from "./components/ChatTimeline";
import { Composer } from "./components/Composer";
import { ConversationList, type ConversationInfo } from "./components/ConversationList";
import { SkillPanel } from "./components/SkillPanel";
import { openChatStream } from "./lib/sse";
import {
  chatReducer,
  createInitialState,
  getNextConversationNumber,
  type SkillSummary,
  type TimelineEntry,
} from "./state/chatReducer";
import type { AgentStreamEvent } from "./types/events";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";

function makeId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function skillsAsMarkdown(skills: SkillSummary[]): string {
  if (!skills.length) {
    return "未发现可用技能。";
  }

  return [
    "## 可用技能",
    ...skills.map(
      (skill) =>
        `- **${skill.name}**: ${skill.description || "暂无描述"}\n  - 路径: \`${skill.path}\``,
    ),
  ].join("\n");
}

function promptAsMarkdown(prompt: string): string {
  const escaped = prompt.replaceAll("```", "` ` `");
  return `## 系统提示词\n\n\`\`\`text\n${escaped}\n\`\`\``;
}

const ACTIVE_THREAD_KEY = "skills-agent-active-thread";

function loadActiveThreadId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_THREAD_KEY);
  } catch {
    return null;
  }
}

function saveActiveThreadId(threadId: string) {
  try {
    localStorage.setItem(ACTIVE_THREAD_KEY, threadId);
  } catch {
    // localStorage might be full or disabled
  }
}

export default function App() {
  const [state, dispatch] = useReducer(chatReducer, undefined, createInitialState);
  const streamCloserRef = useRef<(() => void) | null>(null);

  const activeThread = state.threads[state.activeThreadId];

  // Load conversations from backend on mount
  useEffect(() => {
    let cancelled = false;

    const loadConversations = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/conversations`);
        if (!response.ok) return;
        const payload = (await response.json()) as { conversations: ConversationInfo[] };
        if (!cancelled && payload.conversations && payload.conversations.length > 0) {
          dispatch({
            type: "conversations_loaded",
            conversations: payload.conversations,
          });
        }
      } catch {
        // silently ignore — will use default state
      }
    };

    loadConversations();
    return () => { cancelled = true; };
  }, []);

  // Load skills
  useEffect(() => {
    let cancelled = false;

    const loadSkills = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/skills`);
        if (!response.ok) {
          throw new Error(`加载技能失败 (${response.status})`);
        }
        const payload = (await response.json()) as { skills: SkillSummary[] };
        if (!cancelled) {
          dispatch({ type: "skills_loaded", skills: payload.skills || [] });
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!cancelled) {
          dispatch({ type: "skills_failed", message });
        }
      }
    };

    loadSkills();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    return () => {
      streamCloserRef.current?.();
    };
  }, []);

  // Persist active thread ID to localStorage
  useEffect(() => {
    saveActiveThreadId(state.activeThreadId);
  }, [state.activeThreadId]);

  // Restore active thread from localStorage after conversations loaded
  useEffect(() => {
    const saved = loadActiveThreadId();
    if (saved && state.threads[saved] && saved !== state.activeThreadId) {
      dispatch({ type: "switch_thread", threadId: saved });
    }
  }, [state.conversationsVersion]); // run after conversations_loaded

  // Load history from backend on thread switch or after conversations loaded
  const loadedThreadsRef = useRef<Set<string>>(new Set());
  const prevConvVersionRef = useRef(state.conversationsVersion);
  useEffect(() => {
    // Reset loaded cache when conversations are freshly loaded from backend
    if (state.conversationsVersion !== prevConvVersionRef.current) {
      loadedThreadsRef.current.clear();
      prevConvVersionRef.current = state.conversationsVersion;
    }

    const threadId = state.activeThreadId;
    const thread = state.threads[threadId];
    if (!thread) return;
    // Only load if timeline is empty and we haven't already tried
    if (thread.timeline.length > 0 || loadedThreadsRef.current.has(threadId)) return;
    loadedThreadsRef.current.add(threadId);

    const loadHistory = async () => {
      try {
        const resp = await fetch(
          `${API_BASE_URL}/api/chat/history?thread_id=${encodeURIComponent(threadId)}`,
        );
        if (!resp.ok) return;
        const payload = (await resp.json()) as { entries: TimelineEntry[] };
        if (payload.entries && payload.entries.length > 0) {
          dispatch({
            type: "restore_thread",
            threadId,
            entries: payload.entries,
          });
        }
      } catch {
        // silently ignore — thread will just be empty
      }
    };
    loadHistory();
  }, [state.activeThreadId, state.conversationsVersion]);

  // Build conversations list for sidebar
  const conversations: ConversationInfo[] = useMemo(
    () =>
      state.threadOrder.map((threadId) => {
        const thread = state.threads[threadId];
        return {
          id: threadId,
          label: thread?.label || threadId,
          lastActiveAt: "",
        };
      }),
    [state.threadOrder, state.threads],
  );

  const appendSystemMessage = (content: string, markdown = true) => {
    dispatch({
      type: "append_system_message",
      threadId: state.activeThreadId,
      entryId: makeId("system"),
      message: content,
      markdown,
      createdAt: Date.now(),
    });
  };

  const handleSend = async (text: string) => {
    if (state.isStreaming) {
      return;
    }

    if (text === "/skills") {
      appendSystemMessage(skillsAsMarkdown(state.skills));
      return;
    }

    if (text === "/prompt") {
      try {
        const response = await fetch(`${API_BASE_URL}/api/prompt`);
        if (!response.ok) {
          throw new Error(`加载系统提示词失败 (${response.status})`);
        }
        const payload = (await response.json()) as { prompt: string };
        dispatch({ type: "prompt_loaded", prompt: payload.prompt || "" });
        appendSystemMessage(promptAsMarkdown(payload.prompt || ""));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`错误: ${message}`, false);
      }
      return;
    }

    const threadId = state.activeThreadId;
    const userEntryId = makeId("user");
    const assistantEntryId = makeId("assistant");

    dispatch({
      type: "submit_user_message",
      threadId,
      message: text,
      userEntryId,
      assistantEntryId,
      createdAt: Date.now(),
    });

    streamCloserRef.current?.();
    const threadLabel = activeThread?.label || "";
    streamCloserRef.current = openChatStream({
      apiBaseUrl: API_BASE_URL,
      message: text,
      threadId,
      label: threadLabel,
      onEvent: (event: AgentStreamEvent) => {
        dispatch({
          type: "stream_event",
          threadId,
          assistantEntryId,
          event,
        });

        if (event.type === "done" || event.type === "error") {
          streamCloserRef.current = null;
        }
      },
      onError: (message) => {
        dispatch({
          type: "stream_failed",
          threadId,
          assistantEntryId,
          message,
        });
        streamCloserRef.current = null;
      },
    });
  };

  const handleToggleToolExpand = useCallback(
    (assistantId: string, toolId: string) => {
      dispatch({
        type: "toggle_tool_expand",
        threadId: state.activeThreadId,
        assistantEntryId: assistantId,
        toolId,
      });
    },
    [state.activeThreadId],
  );

  const handleToggleCollapse = useCallback(
    (assistantId: string) => {
      dispatch({
        type: "toggle_collapse",
        threadId: state.activeThreadId,
        assistantEntryId: assistantId,
      });
    },
    [state.activeThreadId],
  );

  const createThread = () => {
    if (state.isStreaming) {
      return;
    }
    const threadNumber = getNextConversationNumber(state.threads);
    const threadId = makeId("thread");
    dispatch({
      type: "create_thread",
      threadId,
      label: `会话 ${threadNumber}`,
    });
  };

  const handleDeleteThread = async (threadId: string) => {
    if (state.isStreaming) return;
    dispatch({ type: "delete_thread", threadId });
    try {
      await fetch(`${API_BASE_URL}/api/conversations/${encodeURIComponent(threadId)}`, {
        method: "DELETE",
      });
    } catch {
      // ignore — local state already updated
    }
  };

  const handleRenameThread = async (threadId: string, label: string) => {
    dispatch({ type: "rename_thread", threadId, label });
    try {
      await fetch(
        `${API_BASE_URL}/api/conversations/${encodeURIComponent(threadId)}/label?label=${encodeURIComponent(label)}`,
        { method: "PUT" },
      );
    } catch {
      // ignore — local state already updated
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <p className="eyebrow">Skills Agent</p>
          <h1>实时交互控制台</h1>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <ConversationList
            conversations={conversations}
            activeId={state.activeThreadId}
            onSelect={(threadId) => dispatch({ type: "switch_thread", threadId })}
            onCreate={createThread}
            onDelete={handleDeleteThread}
            onRename={handleRenameThread}
            disabled={state.isStreaming}
          />

          <SkillPanel
            skills={state.skills}
            activeSkillName={activeThread?.activeSkillName}
            loading={!state.skillsLoaded && !state.skillsError}
            error={state.skillsError}
          />
        </aside>

        <main className="chat-panel">
          <ChatTimeline
            entries={activeThread?.timeline || []}
            onToggleToolExpand={handleToggleToolExpand}
            onToggleCollapse={handleToggleCollapse}
          />

          {state.streamError && <p className="global-error">{state.streamError}</p>}

          <Composer disabled={state.isStreaming} onSubmit={handleSend} />
        </main>
      </div>
    </div>
  );
}
