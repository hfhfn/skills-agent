"""
LangChain Skills Agent 主体

使用 LangChain 1.0 的 create_agent API 实现 Skills Agent，演示三层加载机制：
- Level 1: 启动时将 Skills 元数据注入 system_prompt
- Level 2: load_skill tool 加载详细指令
- Level 3: bash tool 执行脚本

与 claude-agent-sdk 实现的对比：
- claude-agent-sdk: setting_sources=["user", "project"] 自动处理
- LangChain 实现: 显式调用 SkillLoader，过程透明可见

流式输出支持：
- 支持 Extended Thinking 显示模型思考过程
- 事件级流式输出 (thinking / text / tool_call / tool_result)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .memory import MemoryConfig, MemoryMiddleware, MemoryAgentMiddleware, MemoryRetriever, ConversationSummarizer
from .skill_loader import SkillLoader
from .tools import ALL_TOOLS, SkillAgentContext
from .stream import StreamEventEmitter, ToolCallTracker, is_success, DisplayLimits

logger = logging.getLogger(__name__)


# 加载环境变量（override=True 确保 .env 文件覆盖系统环境变量）
load_dotenv(override=True)


# 默认配置
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TEMPERATURE = 1.0  # Extended Thinking 要求温度为 1.0
DEFAULT_THINKING_BUDGET = 10000


def get_credentials() -> tuple[str | None, str | None]:
    """
    获取 API 认证信息

    支持通用和 Anthropic 专属环境变量（通用优先）：
    - API Key: API_KEY > ANTHROPIC_API_KEY > ANTHROPIC_AUTH_TOKEN
    - Base URL: API_BASE_URL > ANTHROPIC_BASE_URL

    Returns:
        (api_key, base_url) 元组
    """
    api_key = (
        os.getenv("API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
    )
    base_url = os.getenv("API_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    return api_key, base_url


def _is_anthropic_provider() -> bool:
    """
    判断当前是否使用 Anthropic provider

    判断逻辑：
    1. MODEL_PROVIDER 显式设为 "anthropic" → True
    2. MODEL_PROVIDER 未设置且模型名包含 "claude" → True
    3. 其他情况 → False
    """
    provider = os.getenv("MODEL_PROVIDER", "").lower()
    if provider:
        return provider == "anthropic"
    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
    return "claude" in model.lower()


def check_api_credentials() -> bool:
    """检查是否配置了 API 认证"""
    api_key, _ = get_credentials()
    return api_key is not None


class LangChainSkillsAgent:
    """
    基于 LangChain 1.0 的 Skills Agent

    演示目的：展示 Skills 三层加载机制的底层原理

    使用示例：
        agent = LangChainSkillsAgent()

        # 查看 system prompt（展示 Level 1）
        print(agent.get_system_prompt())

        # 运行 agent
        for chunk in agent.stream("提取这篇公众号文章"):
            response = agent.get_last_response(chunk)
            if response:
                print(response)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        skill_paths: Optional[list[Path]] = None,
        working_directory: Optional[Path] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ):
        """
        初始化 Agent

        Args:
            model: 模型名称，默认 claude-sonnet-4-5-20250929
            skill_paths: Skills 搜索路径
            working_directory: 工作目录
            max_tokens: 最大 tokens
            temperature: 温度参数 (Anthropic + thinking 启用时强制为 1.0)
            enable_thinking: 是否启用 Extended Thinking
            thinking_budget: thinking 的 token 预算
        """
        # thinking 配置
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.is_anthropic = _is_anthropic_provider()

        # 配置
        self.model_name = model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        if self.is_anthropic and enable_thinking:
            self.temperature = 1.0  # Anthropic 要求启用 thinking 时温度为 1.0
        else:
            self.temperature = temperature or float(os.getenv("MODEL_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.working_directory = working_directory or Path.cwd()

        # 初始化 SkillLoader
        self.skill_loader = SkillLoader(skill_paths)

        # Level 1: 构建 system prompt（将 Skills 元数据注入）
        self.system_prompt = self._build_system_prompt()

        # 创建上下文（供 tools 使用）
        self.context = SkillAgentContext(
            skill_loader=self.skill_loader,
            working_directory=self.working_directory,
        )

        # 初始化 checkpointer（PostgreSQL 或 InMemory）
        self.checkpointer = self._get_checkpointer()

        # 初始化三层记忆系统
        self.memory_config = MemoryConfig.from_env()
        self.summarizer = self._create_summarizer()
        self.retriever = self._create_retriever()
        self.memory_middleware = MemoryMiddleware(
            config=self.memory_config,
            summarizer=self.summarizer,
            retriever=self.retriever,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
        )

        # 将 retriever 注入到上下文中供 recall_memory 工具使用
        self.context.memory_retriever = self.retriever

        # 创建 LangChain Agent
        self.agent = self._create_agent()

    def _get_checkpointer(self):
        """
        获取 checkpointer 实例

        优先使用 PostgreSQL 持久化（需要 DATABASE_URL 环境变量），
        未配置时 fallback 到 InMemorySaver（保持零依赖可运行）。
        """
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            try:
                from psycopg import Connection
                from psycopg.rows import dict_row
                from langgraph.checkpoint.postgres import PostgresSaver

                conn = Connection.connect(
                    db_url,
                    autocommit=True,
                    prepare_threshold=0,
                    row_factory=dict_row,
                )
                checkpointer = PostgresSaver(conn)
                checkpointer.setup()
                self._pg_conn = conn  # keep reference for conversation management
                self._setup_thread_activity_table()
                return checkpointer
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to initialize PostgreSQL checkpointer: {e}. "
                    "Falling back to InMemorySaver."
                )
        return InMemorySaver()

    def _setup_thread_activity_table(self):
        """创建 thread_activity 和 conversation_summaries 表。"""
        conn = getattr(self, "_pg_conn", None)
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS thread_activity (
                    thread_id TEXT PRIMARY KEY,
                    label TEXT,
                    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            # Add label column if table already exists without it
            cur.execute("""
                ALTER TABLE thread_activity ADD COLUMN IF NOT EXISTS label TEXT
            """)

    def _prune_thread_messages(self, thread_id: str):
        """如果线程消息数超过限制，摘要旧消息并修剪。"""
        max_messages = self.memory_config.max_messages_per_thread
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.agent.get_state(config)
        except Exception:
            return
        if not state or not state.values:
            return
        messages = state.values.get("messages", [])
        if len(messages) <= max_messages:
            return

        prune_count = len(messages) - max_messages + 20
        to_prune = messages[:prune_count]

        # Extract topics from pruned messages
        topics = []
        if self.summarizer:
            topics = self.summarizer.extract_topics(to_prune)

        # Store pruned messages to vector DB before removing
        if self.retriever:
            self.retriever.store_messages(thread_id, to_prune, topics=topics)

        # Update summary
        existing_summary = self.summarizer.load_summary(thread_id) if self.summarizer else None
        if self.summarizer:
            self.summarizer.generate_and_store(thread_id, to_prune, existing_summary)

        # Remove old messages from state via RemoveMessage
        try:
            from langchain_core.messages import RemoveMessage
            removals = [RemoveMessage(id=m.id) for m in to_prune if m.id]
            if removals:
                self.agent.update_state(config, {"messages": removals})
                logger.info(
                    "Pruned %d messages from thread %s (had %d, limit %d, topics: %s)",
                    len(removals), thread_id, len(messages), max_messages, topics,
                )
        except Exception as e:
            logger.warning("Failed to prune messages for thread %s: %s", thread_id, e)

    def _touch_thread_activity(self, thread_id: str, label: str | None = None):
        """更新线程的最后活跃时间，首次创建时可设置 label。"""
        conn = getattr(self, "_pg_conn", None)
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                if label:
                    cur.execute(
                        "INSERT INTO thread_activity (thread_id, label, last_active_at) "
                        "VALUES (%s, %s, now()) "
                        "ON CONFLICT (thread_id) DO UPDATE SET last_active_at = now()",
                        (thread_id, label),
                    )
                else:
                    cur.execute(
                        "INSERT INTO thread_activity (thread_id, last_active_at) "
                        "VALUES (%s, now()) "
                        "ON CONFLICT (thread_id) DO UPDATE SET last_active_at = now()",
                        (thread_id,),
                    )
        except Exception:
            pass  # non-critical, don't break the chat flow

    def _create_summarizer(self) -> Optional[ConversationSummarizer]:
        """Create conversation summarizer if PostgreSQL is available."""
        pg_conn = getattr(self, "_pg_conn", None)
        if not pg_conn or not self.memory_config.enable_summary:
            return None

        # Create a lightweight LLM for summarization (use same credentials)
        try:
            api_key, base_url = get_credentials()
            init_kwargs = {"temperature": 0.3, "max_tokens": 2000}
            if api_key:
                init_kwargs["api_key"] = api_key
            if base_url:
                init_kwargs["base_url"] = base_url

            model_provider = os.getenv("MODEL_PROVIDER")
            provider_kwargs = {}
            if model_provider:
                provider_kwargs["model_provider"] = model_provider

            summary_llm = init_chat_model(
                self.model_name,
                **provider_kwargs,
                **init_kwargs,
            )
            return ConversationSummarizer(
                pg_conn=pg_conn,
                llm=summary_llm,
                max_summary_tokens=self.memory_config.max_summary_tokens,
            )
        except Exception as e:
            logger.warning("Failed to create summarizer LLM: %s", e)
            return ConversationSummarizer(
                pg_conn=pg_conn,
                llm=None,
                max_summary_tokens=self.memory_config.max_summary_tokens,
            )

    def _create_retriever(self) -> Optional[MemoryRetriever]:
        """Create memory retriever if PostgreSQL is available."""
        pg_conn = getattr(self, "_pg_conn", None)
        if not pg_conn:
            return None

        # Create embedding model
        embedding_model = None
        try:
            api_key, base_url = get_credentials()
            model_provider = os.getenv("MODEL_PROVIDER")

            # Embedding credentials: dedicated env vars > general credentials
            embed_base_url = self.memory_config.embedding_base_url or base_url
            embed_api_key = self.memory_config.embedding_api_key or api_key

            if model_provider == "openai" or (embed_base_url and not model_provider):
                from langchain_openai import OpenAIEmbeddings
                embed_kwargs = {}
                if embed_api_key:
                    embed_kwargs["api_key"] = embed_api_key
                if embed_base_url:
                    embed_kwargs["base_url"] = embed_base_url
                embedding_model = OpenAIEmbeddings(
                    model=self.memory_config.embedding_model,
                    **embed_kwargs,
                )
            else:
                # For Anthropic, use a lightweight OpenAI-compatible embedding
                # or skip embeddings (fallback to keyword search)
                logger.info(
                    "No embedding model configured for provider '%s', "
                    "recall_memory will use keyword search",
                    model_provider,
                )
        except Exception as e:
            logger.warning("Failed to create embedding model: %s", e)

        try:
            return MemoryRetriever(
                pg_conn=pg_conn,
                embedding_model=embedding_model,
            )
        except Exception as e:
            logger.warning("Failed to create memory retriever: %s", e)
            return None

    def list_conversations(self) -> list[dict]:
        """返回所有会话的元数据列表。"""
        conn = getattr(self, "_pg_conn", None)
        if not conn:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ta.thread_id, ta.label, ta.last_active_at
                    FROM thread_activity ta
                    ORDER BY ta.last_active_at DESC
                """)
                rows = cur.fetchall()
                result = []
                for row in rows:
                    if isinstance(row, dict):
                        tid = row["thread_id"]
                        label = row.get("label") or tid
                        last_active = row["last_active_at"]
                    else:
                        tid = row[0]
                        label = row[1] or tid
                        last_active = row[2]
                    result.append({
                        "id": tid,
                        "label": label,
                        "lastActiveAt": last_active.isoformat() if hasattr(last_active, "isoformat") else str(last_active),
                    })
                return result
        except Exception as e:
            logger.warning("Failed to list conversations: %s", e)
            return []

    def delete_conversation(self, thread_id: str):
        """删除指定会话的所有数据。"""
        conn = getattr(self, "_pg_conn", None)
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM conversation_summaries WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM thread_activity WHERE thread_id = %s", (thread_id,))
            # Clean up vector store
            if self.retriever:
                self.retriever.delete_thread(thread_id)
            logger.info("Deleted conversation: %s", thread_id)
        except Exception as e:
            logger.warning("Failed to delete conversation %s: %s", thread_id, e)

    def rename_conversation(self, thread_id: str, label: str):
        """重命名会话。"""
        conn = getattr(self, "_pg_conn", None)
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE thread_activity SET label = %s WHERE thread_id = %s",
                    (label, thread_id),
                )
        except Exception as e:
            logger.warning("Failed to rename conversation %s: %s", thread_id, e)

    def _build_system_prompt(self) -> str:
        """
        构建 system prompt

        这是 Level 1 的核心：将所有 Skills 的元数据注入到 system prompt。
        每个 skill 约 100 tokens，启动时一次性加载。
        """
        base_prompt = f"""You are a helpful coding assistant with access to specialized skills.

Your capabilities include:
- Loading and using specialized skills for specific tasks
- Executing bash commands and scripts
- Reading and writing files
- Following skill instructions to complete complex tasks

Working directory: {self.working_directory}
Output directory: {self.working_directory / "output"}

IMPORTANT: When creating output files, generated scripts, temporary files, or any other artifacts,
always place them in the output directory ({self.working_directory / "output"}).
Do NOT create files in the project root directory.

When a user request matches a skill's description, use the load_skill tool to get detailed instructions before proceeding."""

        return self.skill_loader.build_system_prompt(base_prompt)

    def _create_agent(self):
        """
        创建 LangChain Agent

        使用 LangChain 1.0 的 create_agent API:
        - model: 可以是字符串 ID 或 model 实例
        - tools: 工具列表
        - system_prompt: 系统提示（Level 1 注入 Skills 元数据）
        - context_schema: 上下文类型（供 ToolRuntime 使用）
        - checkpointer: 会话记忆

        Extended Thinking 支持 (仅 Anthropic):
        - 启用后可获取模型的思考过程
        - 温度必须为 1.0

        多 Provider 支持:
        - MODEL_PROVIDER 指定 provider（如 openai、anthropic）
        - 未设置时由 init_chat_model 自动推断
        """
        # 获取认证信息
        api_key, base_url = get_credentials()

        # 构建初始化参数
        init_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # 添加认证参数
        if api_key:
            init_kwargs["api_key"] = api_key
        if base_url:
            init_kwargs["base_url"] = base_url

        # Extended Thinking 配置（仅 Anthropic provider）
        if self.enable_thinking and self.is_anthropic:
            init_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        # model_provider 参数（未设置时不传，保持自动推断）
        model_provider = os.getenv("MODEL_PROVIDER")
        provider_kwargs = {}
        if model_provider:
            provider_kwargs["model_provider"] = model_provider

        # 初始化模型
        model = init_chat_model(
            self.model_name,
            **provider_kwargs,
            **init_kwargs,
        )

        # 创建 Agent（注入三层记忆中间件）
        agent_middleware = MemoryAgentMiddleware(self.memory_middleware)
        agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=self.system_prompt,
            context_schema=SkillAgentContext,
            checkpointer=self.checkpointer,
            middleware=[agent_middleware],
        )

        return agent

    @property
    def use_streaming(self) -> bool:
        """
        是否使用流式输出

        优先级：
        1. ENABLE_STREAMING 环境变量（显式覆盖）
        2. enable_thinking 构造参数（默认 True，即默认流式）

        思考模型（Claude/DeepSeek-R1/GLM 等）推荐流式，可实时显示推理过程。
        非思考模型也可以使用流式，不会有副作用。
        需要非流式时，设置 ENABLE_STREAMING=false 或使用 --no-thinking。
        """
        env_streaming = os.getenv("ENABLE_STREAMING", "").lower()
        if env_streaming:
            return env_streaming in ("1", "true", "yes")
        return self.enable_thinking

    def get_system_prompt(self) -> str:
        """
        获取当前 system prompt

        用于演示和调试，展示 Level 1 注入的内容。
        """
        return self.system_prompt

    def get_discovered_skills(self) -> list[dict]:
        """
        获取发现的 Skills 列表

        用于演示 Level 1 的 Skills 发现过程。
        """
        skills = self.skill_loader.scan_skills()
        return [
            {
                "name": s.name,
                "description": s.description,
                "path": str(s.skill_path),
            }
            for s in skills
        ]

    def get_thread_history(self, thread_id: str) -> list[dict]:
        """
        从 checkpointer 获取线程的消息历史，转换为前端事件格式。

        Args:
            thread_id: 会话 ID

        Returns:
            前端 timeline entries 列表
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.agent.get_state(config)
        except Exception:
            return []
        if not state or not state.values:
            return []
        messages = state.values.get("messages", [])
        return self._messages_to_timeline(messages)

    def _messages_to_timeline(self, messages: list) -> list[dict]:
        """
        将 LangChain messages 转换为前端 timeline entries 格式。

        - HumanMessage → {kind: "user", ...}
        - AIMessage → {kind: "assistant", ...}
        - ToolMessage → 附加到前一个 assistant entry 的 tools 中
        """
        entries: list[dict] = []
        entry_counter = 0

        for msg in messages:
            if isinstance(msg, HumanMessage):
                entry_counter += 1
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                entries.append({
                    "kind": "user",
                    "id": f"hist-user-{entry_counter}",
                    "text": text,
                    "createdAt": 0,
                })

            elif isinstance(msg, AIMessage):
                entry_counter += 1
                thinking = ""
                response = ""
                tools: list[dict] = []

                content = msg.content
                if isinstance(content, str):
                    response = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            response += block
                        elif isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype in ("thinking", "reasoning"):
                                thinking += block.get("thinking", "") or block.get("reasoning", "")
                            elif btype == "text":
                                response += block.get("text", "")

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_id = tc.get("id", f"hist-tool-{entry_counter}-{len(tools)}")
                        tools.append({
                            "id": tool_id,
                            "name": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                            "status": "success",
                            "expanded": False,
                        })

                entries.append({
                    "kind": "assistant",
                    "id": f"hist-assistant-{entry_counter}",
                    "createdAt": 0,
                    "phase": "done",
                    "thinking": thinking,
                    "response": response,
                    "tools": tools,
                    "collapsed": True,
                })

            elif isinstance(msg, ToolMessage):
                # 附加到最近一个 assistant entry 的 tools 中
                if entries and entries[-1].get("kind") == "assistant":
                    assistant_entry = entries[-1]
                    tool_name = getattr(msg, "name", "unknown")
                    raw_content = str(getattr(msg, "content", ""))
                    success = not raw_content.strip().startswith("[FAILED]")

                    matched = False
                    for tool in assistant_entry["tools"]:
                        if tool["name"] == tool_name and "result" not in tool:
                            tool["result"] = raw_content[:2000]
                            tool["success"] = success
                            tool["status"] = "success" if success else "failed"
                            matched = True
                            break

                    if not matched:
                        tool_id = getattr(msg, "tool_call_id", f"hist-toolresult-{entry_counter}")
                        assistant_entry["tools"].append({
                            "id": tool_id,
                            "name": tool_name,
                            "args": {},
                            "status": "success" if success else "failed",
                            "result": raw_content[:2000],
                            "success": success,
                            "expanded": False,
                        })

        return entries

    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """
        同步调用 Agent

        Args:
            message: 用户消息
            thread_id: 会话 ID（用于多轮对话）

        Returns:
            Agent 响应
        """
        config = {"configurable": {"thread_id": thread_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=self.context,
        )

        return result

    def stream(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        流式调用 Agent (state 级别)

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            流式响应块 (完整状态更新)
        """
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=self.context,
            stream_mode="values",
        ):
            yield chunk

    def stream_events(self, message: str, thread_id: str = "default", label: str = "") -> Iterator[dict]:
        """
        事件级流式输出，支持 thinking 和 token 级流式

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            事件字典，格式如下:
            - {"type": "thinking", "content": "..."} - 思考内容片段
            - {"type": "text", "content": "..."} - 响应文本片段
            - {"type": "tool_call", "name": "...", "args": {...}} - 工具调用
            - {"type": "tool_result", "name": "...", "content": "...", "success": bool} - 工具结果
            - {"type": "done", "response": "..."} - 完成标记，包含完整响应
        """
        config = {"configurable": {"thread_id": thread_id}}
        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()

        # 设置当前 thread_id 供 recall_memory 工具使用
        self.context.current_thread_id = thread_id

        self._touch_thread_activity(thread_id, label=label or None)
        self._prune_thread_messages(thread_id)

        full_response = ""
        debug = os.getenv("SKILLS_DEBUG", "").lower() in ("1", "true", "yes")

        # 使用 messages 模式获取 token 级流式
        try:
            for event in self.agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                context=self.context,
                stream_mode="messages",
            ):
                # event 可能是 tuple(message, metadata) 或直接 message
                if isinstance(event, tuple) and len(event) >= 2:
                    chunk = event[0]
                else:
                    chunk = event

                if debug:
                    chunk_type = type(chunk).__name__
                    print(f"[DEBUG] Event: {chunk_type}")

                # 处理 AIMessageChunk / AIMessage
                if isinstance(chunk, (AIMessageChunk, AIMessage)):
                    # 处理 content
                    for ev in self._process_chunk_content(chunk, emitter, tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

                    # 处理 tool_calls (有些情况下在 chunk.tool_calls 中)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for ev in self._process_tool_calls(chunk.tool_calls, emitter, tracker):
                            if debug:
                                print(f"[DEBUG] Yielding from tool_calls: {ev.type}")
                            yield ev.data

                # 处理 ToolMessage (工具执行结果)
                elif hasattr(chunk, "type") and chunk.type == "tool":
                    if debug:
                        tool_name = getattr(chunk, "name", "unknown")
                        print(f"[DEBUG] Processing tool result: {tool_name}")
                    for ev in self._process_tool_result(chunk, emitter, tracker):
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

            if debug:
                print("[DEBUG] Stream completed normally")

        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG] Stream error: {e}")
                traceback.print_exc()
            # 发送错误事件让用户知道发生了什么
            yield emitter.error(str(e)).data
            raise

        # 发送完成事件
        yield emitter.done(full_response).data

    def _process_chunk_content(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 chunk 的 content"""
        content = chunk.content

        if isinstance(content, str):
            if content:
                yield emitter.text(content)
                return

        blocks = None
        if hasattr(chunk, "content_blocks"):
            try:
                blocks = chunk.content_blocks
            except Exception:
                blocks = None

        if blocks is None:
            if isinstance(content, dict):
                blocks = [content]
            elif isinstance(content, list):
                blocks = content
            else:
                return

        for raw_block in blocks:
            block = raw_block
            if not isinstance(block, dict):
                if hasattr(block, "model_dump"):
                    block = block.model_dump()
                elif hasattr(block, "dict"):
                    block = block.dict()
                else:
                    continue

            block_type = block.get("type")

            if block_type in ("thinking", "reasoning"):
                thinking_text = block.get("thinking") or block.get("reasoning") or ""
                if thinking_text:
                    yield emitter.thinking(thinking_text)

            elif block_type == "text":
                text = block.get("text") or block.get("content") or ""
                if text:
                    yield emitter.text(text)

            elif block_type in ("tool_use", "tool_call"):
                tool_id = block.get("id", "")
                name = block.get("name", "")
                args = block.get("input") if block_type == "tool_use" else block.get("args")
                args_payload = args if isinstance(args, dict) else {}

                if tool_id:
                    tracker.update(tool_id, name=name, args=args_payload)
                    # 立即发送（显示"执行中"状态），参数可能尚不完整
                    if tracker.is_ready(tool_id):
                        tracker.mark_emitted(tool_id)
                        yield emitter.tool_call(name, args_payload, tool_id)

            elif block_type == "input_json_delta":
                # 累积 JSON 片段（args 分批到达）
                partial_json = block.get("partial_json", "")
                if partial_json:
                    tracker.append_json_delta(partial_json, block.get("index", 0))

            elif block_type == "tool_call_chunk":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                if tool_id:
                    tracker.update(tool_id, name=name)
                partial_args = block.get("args", "")
                if isinstance(partial_args, str) and partial_args:
                    tracker.append_json_delta(partial_args, block.get("index", 0))

    def _handle_tool_use_block(self, block: dict, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 tool_use 块 - 立即发送 tool_call 事件

        在收到 tool_use 时立即发送，让 CLI 可以显示"正在执行"状态。
        避免重复发送（同一 tool 可能通过多个路径到达）。
        """
        tool_id = block.get("id", "")
        if tool_id:
            name = block.get("name", "")
            args = block.get("input", {})
            args_payload = args if isinstance(args, dict) else {}

            tracker.update(tool_id, name=name, args=args_payload)
            if tracker.is_ready(tool_id):
                tracker.mark_emitted(tool_id)
                yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_calls(self, tool_calls: list, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 chunk.tool_calls - 立即发送 tool_call 事件

        避免重复发送（同一 tool 可能通过 tool_use block 已发送）。
        """
        for tc in tool_calls:
            tool_id = tc.get("id", "")
            if tool_id:
                name = tc.get("name", "")
                args = tc.get("args", {})
                args_payload = args if isinstance(args, dict) else {}

                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_result(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理工具结果"""
        # 最终化：解析累积的 JSON 片段为 args
        tracker.finalize_all()

        # 发送所有工具调用的更新（参数现在是完整的）
        # CLI 会用 tool_id 去重和更新
        for info in tracker.get_all():
            yield emitter.tool_call(info.name, info.args, info.id)

        # 发送结果
        name = getattr(chunk, "name", "unknown")
        raw_content = str(getattr(chunk, "content", ""))
        content = raw_content[:DisplayLimits.TOOL_RESULT_MAX]
        if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
            content += "\n... (truncated)"

        # 基于内容判断是否成功（统一使用 is_success）
        success = is_success(content)

        yield emitter.tool_result(name, content, success)

    def get_last_response(self, result: dict) -> str:
        """
        从结果中提取最后的 AI 响应文本

        Args:
            result: invoke 或 stream 的结果

        Returns:
            AI 响应文本
        """
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # 处理多部分内容
                    text_parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return "\n".join(text_parts)
        return ""


def create_skills_agent(
    model: Optional[str] = None,
    skill_paths: Optional[list[Path]] = None,
    working_directory: Optional[Path] = None,
    enable_thinking: bool = True,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> LangChainSkillsAgent:
    """
    便捷函数：创建 Skills Agent

    Args:
        model: 模型名称
        skill_paths: Skills 搜索路径
        working_directory: 工作目录
        enable_thinking: 是否启用 Extended Thinking
        thinking_budget: thinking 的 token 预算

    Returns:
        配置好的 LangChainSkillsAgent 实例
    """
    return LangChainSkillsAgent(
        model=model,
        skill_paths=skill_paths,
        working_directory=working_directory,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )
