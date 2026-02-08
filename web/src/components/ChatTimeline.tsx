import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { AssistantEntry, SystemEntry, TimelineEntry, UserEntry } from "../state/chatReducer";
import { ToolCallItem } from "./ToolCallItem";

type ChatTimelineProps = {
  entries: TimelineEntry[];
  onToggleToolExpand: (assistantId: string, toolId: string) => void;
  onToggleCollapse: (assistantId: string) => void;
};

function phaseLabel(phase: string): string {
  switch (phase) {
    case "waiting":
      return "AI 正在思考中...";
    case "thinking":
      return "深度思考";
    case "analyzing":
      return "AI 正在分析结果...";
    case "responding":
      return "正在生成回复";
    case "done":
      return "已完成";
    case "error":
      return "出错";
    default:
      return phase;
  }
}

function showSpinner(phase: string): boolean {
  return (
    phase === "waiting" ||
    phase === "thinking" ||
    phase === "analyzing" ||
    phase === "responding"
  );
}

const UserMessage = memo(function UserMessage({ entry }: { entry: UserEntry }) {
  return (
    <article className="message message--user">
      <header>用户</header>
      <p>{entry.text}</p>
    </article>
  );
});

const SystemMessage = memo(function SystemMessage({ entry }: { entry: SystemEntry }) {
  return (
    <article className="message message--system">
      <header>命令</header>
      {entry.markdown ? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {entry.text}
        </ReactMarkdown>
      ) : (
        <p>{entry.text}</p>
      )}
    </article>
  );
});

const AssistantMessage = memo(function AssistantMessage({
  entry,
  onToggleToolExpand,
  onToggleCollapse,
}: {
  entry: AssistantEntry;
  onToggleToolExpand: (assistantId: string, toolId: string) => void;
  onToggleCollapse: (assistantId: string) => void;
}) {
  const isCollapsed = Boolean(entry.collapsed);
  const toolCount = entry.tools.length;
  const hasThinking = Boolean(entry.thinking);
  const hasCollapsibleContent = hasThinking || toolCount > 0;

  return (
    <article className="message message--assistant">
      <header className="assistant-header">
        <span>助手</span>
        <span className="phase-pill">
          {showSpinner(entry.phase) && <span className="inline-spinner" aria-hidden />}
          {phaseLabel(entry.phase)}
        </span>
        {hasCollapsibleContent && (
          <button
            type="button"
            className="collapse-header-btn"
            onClick={() => onToggleCollapse(entry.id)}
            title={isCollapsed ? "展开详情" : "折叠详情"}
          >
            {isCollapsed ? "\u25b6" : "\u25bc"}
          </button>
        )}
      </header>

      {isCollapsed && hasCollapsibleContent && (
        <div className="collapse-summary">
          <button
            type="button"
            className="collapse-toggle"
            onClick={() => onToggleCollapse(entry.id)}
          >
            <span className="collapse-arrow">&#9654;</span>
            {hasThinking && <span className="collapse-tag">思考过程</span>}
            {toolCount > 0 && (
              <span className="collapse-tag">{toolCount} 个工具调用</span>
            )}
            <span className="collapse-hint">点击展开</span>
          </button>
        </div>
      )}

      {!isCollapsed && entry.thinking && (
        <section className="panel panel--thinking">
          <h4>
            思考过程
            <button
              type="button"
              className="collapse-inline-btn"
              onClick={() => onToggleCollapse(entry.id)}
            >
              收起
            </button>
          </h4>
          <pre>{entry.thinking}</pre>
        </section>
      )}

      {!isCollapsed && entry.tools.length > 0 && (
        <section className="panel panel--tools">
          <h4>
            工具调用
            {!hasThinking && (
              <button
                type="button"
                className="collapse-inline-btn"
                onClick={() => onToggleCollapse(entry.id)}
              >
                收起
              </button>
            )}
          </h4>
          <div className="tools-list">
            {entry.tools.map((tool) => (
              <ToolCallItem
                key={tool.id}
                assistantId={entry.id}
                tool={tool}
                onToggleExpand={onToggleToolExpand}
              />
            ))}
          </div>
        </section>
      )}

      {entry.response && (
        <section className="panel panel--response">
          <h4>回复</h4>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {entry.response}
          </ReactMarkdown>
        </section>
      )}

      {entry.error && <p className="error-text">{entry.error}</p>}
    </article>
  );
});

export const ChatTimeline = memo(function ChatTimeline({ entries, onToggleToolExpand, onToggleCollapse }: ChatTimelineProps) {
  return (
    <section className="chat-timeline" aria-live="polite">
      {entries.length === 0 && (
        <div className="empty-state">
          <h3>开始对话</h3>
          <p>输入任务、URL 或指令，完整的执行过程将在此实时展示。</p>
        </div>
      )}

      {entries.map((entry) => {
        if (entry.kind === "user") {
          return <UserMessage key={entry.id} entry={entry} />;
        }

        if (entry.kind === "system") {
          return <SystemMessage key={entry.id} entry={entry} />;
        }

        return (
          <AssistantMessage
            key={entry.id}
            entry={entry}
            onToggleToolExpand={onToggleToolExpand}
            onToggleCollapse={onToggleCollapse}
          />
        );
      })}
    </section>
  );
});
