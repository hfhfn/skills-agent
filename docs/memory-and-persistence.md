# 数据库设计与三层记忆管理

## 一、数据库表结构

项目使用 PostgreSQL + pgvector 扩展，共 7 张表，分两大类。

### 1.1 LangGraph Checkpoint 表（3 张，自动创建）

由 `PostgresSaver.setup()` 自动创建，存储完整对话状态。

| 表名 | 作用 |
|------|------|
| `checkpoints` | 状态快照——每次 agent 交互结束后保存完整 state |
| `checkpoint_blobs` | 大对象存储——序列化后的消息列表等二进制数据 |
| `checkpoint_writes` | 写入日志——支持并发安全和版本回溯 |

**核心机制**：checkpoint 存的不是单条消息，而是**整个会话的完整 state 快照**。每次交互结束后 LangGraph 自动保存，下次请求时自动恢复。这是历史记录持久化的基础。

```
用户发消息 → agent.stream({messages: [user_msg]}, config={thread_id: "xxx"})
                │
                ├─ 1. 用 thread_id 从 checkpoints 加载上一次的完整 state
                │     state["messages"] = [之前所有 HumanMessage, AIMessage, ToolMessage ...]
                │
                ├─ 2. 追加新的 HumanMessage
                │
                ├─ 3. 调用 LLM → AIMessage → 如有 tool_call → 执行 → ToolMessage → 再调 LLM
                │
                └─ 4. 交互结束，自动将新的完整 state 写入 checkpoints
```

### 1.2 自建表（4 张）

#### thread_activity — 会话元数据

```sql
CREATE TABLE thread_activity (
    thread_id      TEXT PRIMARY KEY,   -- UUID，和 checkpoints 的 thread_id 一致
    label          TEXT,               -- 显示名称，如"会话 1"
    last_active_at TIMESTAMPTZ         -- 最后活跃时间，用于列表排序
);
```

LangGraph 的 checkpoint 表没有"会话名称"或"活跃时间"等元数据，这张表补充这些信息。前端 `GET /api/conversations` 查的就是这张表。

#### conversation_summaries — 长期摘要（Tier 1）

```sql
CREATE TABLE conversation_summaries (
    thread_id     TEXT PRIMARY KEY,    -- 一个会话一条摘要
    summary       TEXT NOT NULL,       -- LLM 生成的压缩摘要
    message_count INT DEFAULT 0,       -- 已被摘要的消息数
    updated_at    TIMESTAMPTZ          -- 最后更新时间
);
```

当会话消息被物理修剪时，旧消息的语义信息以摘要形式保留在这里。

#### memory_vectors — 向量化对话片段（RAG 检索）

```sql
CREATE TABLE memory_vectors (
    id         TEXT PRIMARY KEY,       -- hash(thread_id + chunk) 去重
    thread_id  TEXT NOT NULL,
    chunk      TEXT NOT NULL,          -- 对话片段文本
    topics     TEXT[] DEFAULT '{}',    -- 主题标签数组
    embedding  vector(1536),           -- pgvector 向量嵌入
    created_at TIMESTAMPTZ
);

-- 索引
CREATE INDEX idx_memory_vectors_thread ON memory_vectors (thread_id);
CREATE INDEX idx_memory_vectors_topics ON memory_vectors USING GIN (topics);
```

被修剪的旧消息向量化后存入此表，供 `recall_memory` 工具做语义检索。

#### memory_topics — 主题索引

```sql
CREATE TABLE memory_topics (
    thread_id     TEXT NOT NULL,
    topic         TEXT NOT NULL,        -- 如"数据库设计"、"API认证"
    mention_count INT DEFAULT 1,
    last_seen_at  TIMESTAMPTZ,
    PRIMARY KEY (thread_id, topic)
);
```

轻量级主题元数据。Tier 1 注入时把主题列表告诉 LLM，LLM 如需详情可调用 `recall_memory` 从 `memory_vectors` 检索。

### 1.3 表关系

```
                    thread_id（逻辑关联，无外键）
                    ┌──────────────────────────────────────┐
                    │                                      │
  ┌─────────────────┼───────────────┐                     │
  │ LangGraph 自动管理               │    thread_activity  │
  │                                 │    ┌────────────┐   │
  │  checkpoints                    │    │ thread_id  │◄──┘
  │  checkpoint_blobs               │    │ label      │
  │  checkpoint_writes              │    │ last_active │
  └─────────────────────────────────┘    └──────┬─────┘
                                                │
                          ┌─────────────────────┼─────────────────┐
                          ▼                     ▼                 ▼
             conversation_summaries     memory_vectors     memory_topics
             (长期摘要)                 (向量片段)          (主题索引)
```

所有表通过 `thread_id` 逻辑关联。删除会话时，7 张表中该 `thread_id` 的数据全部清除。

---

## 二、Checkpoint 如何保证历史记录

### 2.1 读写流程

```
首次对话（thread_id = "thread-a1b2c3"）:
  1. LangGraph 在 checkpoints 中查找 thread_id → 不存在 → 从空 state 开始
  2. 用户消息追加到 state.messages
  3. LLM 回复 → state.messages 追加 AIMessage
  4. 交互结束 → 完整 state 写入 checkpoints

后续对话（同一 thread_id）:
  1. LangGraph 从 checkpoints 恢复完整 state（包含之前所有消息）
  2. 追加新消息 → LLM 回复 → 新 state 快照覆盖写入
```

### 2.2 持久化保证

| 场景 | 数据是否保留 |
|------|------------|
| 刷新页面 | 保留。前端从 `GET /api/conversations` 和 `GET /api/chat/history` 重新加载 |
| 重启后端服务 | 保留。数据在 PostgreSQL 中，与进程无关 |
| 重启 Docker 容器 | 保留。数据存在 Docker volume (`pgdata`) 中 |
| 删除 Docker volume | **丢失**。这是唯一会丢数据的操作 |
| 用户主动删除会话 | 丢失（预期行为）。7 张表全部清除 |

### 2.3 前端加载流程

```
页面加载
    │
    ├─ GET /api/conversations
    │   → 查 thread_activity 表 → 返回会话列表 [{id, label, lastActiveAt}]
    │   → dispatch("conversations_loaded") → 前端 state 填充
    │
    ├─ GET /api/chat/history?thread_id=xxx
    │   → agent.get_state({thread_id}) → 从 checkpoints 读取完整 messages
    │   → 转换为前端 timeline 格式 → dispatch("restore_thread")
    │
    └─ 用户看到历史消息
```

---

## 三、摘要触发条件

系统中有**两个独立的触发机制**，解决不同层面的问题。

### 3.1 物理修剪 — `_prune_thread_messages()`

**触发时机**：每次用户发消息时，在 `stream_events()` 中执行。

**触发条件**：checkpoint 中的消息数 > `MEMORY_MAX_MESSAGES_PER_THREAD`（默认 200）。

```
stream_events() 被调用
    │
    ├─ _touch_thread_activity()         ← 更新最后活跃时间
    ├─ _prune_thread_messages()         ← 检查是否需要修剪
    │     │
    │     ├─ agent.get_state(config)    ← 从 checkpoint 取出所有消息
    │     ├─ len(messages) > 200 ?
    │     │     │
    │     │     ├─ NO  → 跳过，不做任何处理
    │     │     │
    │     │     └─ YES → 取最早的 N 条消息（N = 总数 - 200 + 20 buffer）
    │     │           ├─ extract_topics()        → LLM 提取主题标签
    │     │           ├─ store_messages()         → 向量化存入 memory_vectors
    │     │           ├─ generate_and_store()     → LLM 生成摘要存入 conversation_summaries
    │     │           └─ RemoveMessage() × N      → 从 checkpoint state 中删除旧消息
    │     │
    │
    └─ agent.stream(...)                ← 正常执行对话（checkpoint 中现在只有 ~180 条）
```

**摘要递归压缩**：当现有摘要 + 新摘要超过 `MEMORY_MAX_SUMMARY_TOKENS`（默认 1000）时：

1. 用 LLM 将旧摘要压缩到约 1/3 长度
2. 合并旧摘要（压缩后）+ 新摘要
3. 如果仍然超限，对整体再压缩一次

### 3.2 虚拟裁剪 — `MemoryAgentMiddleware`

**触发时机**：LangGraph 每次调用 LLM 之前，通过 `AgentMiddleware.wrap_model_call()` 自动执行。

**触发条件**：所有对话消息的估算 token 总量 > 上下文预算。

```
LangGraph 准备调用 LLM
    │
    ├─ wrap_model_call(request, handler)
    │     │
    │     ├─ 取出 request.messages（不含 system message）
    │     ├─ 估算总 token 数
    │     ├─ 总 token ≤ 预算 ? → 不裁剪，原样传递
    │     ├─ 总 token > 预算 ? → 应用三层策略
    │     │     ├─ Tier 1: 注入摘要 + 主题索引
    │     │     ├─ Tier 2: 压缩中间消息
    │     │     └─ Tier 3: 完整保留最近消息
    │     │
    │     └─ request.override(messages=trimmed) → handler(request)
    │
    └─ LLM 只看到裁剪后的消息，checkpoint 中的完整历史不受影响
```

**关键区别**：物理修剪真正删除旧消息（但保留摘要和向量）；虚拟裁剪只影响 LLM 单次看到的内容，checkpoint 完整性不受影响。

### 3.3 两个机制的协作

```
消息数持续增长
    │
    ├─ < 200 条：物理修剪不触发，虚拟裁剪看 token 总量
    │              token 够用 → 全部消息原样发给 LLM
    │              token 超限 → 三层裁剪（仅当次生效）
    │
    ├─ > 200 条：物理修剪触发
    │     ├─ 旧消息 → 摘要 + 向量化 → 从 checkpoint 删除
    │     └─ checkpoint 回到 ~180 条
    │
    └─ 无论哪种情况，LLM 都不会收到超出上下文窗口的消息
```

---

## 四、三层记忆策略

### 4.1 预算分配

以 128K token 上下文窗口为例：

```
上下文窗口: 128,000 tokens
  - 安全裕量 10%  → 剩余 115,200
  - 输出预留 16K  → 剩余  99,200
  - system prompt → 剩余  ~95,000（可用预算）

Tier 1 (10%):  ~9,500 tokens   → 长期摘要 + 主题索引
Tier 2 (35%): ~33,000 tokens   → 压缩的中期消息
Tier 3 (55%): ~52,000 tokens   → 完整的近期消息
```

### 4.2 Tier 3 — 短期完整记忆

从最新消息往前取，最多 `MEMORY_FULL_RECENT_COUNT`（默认 10）条，不超过 55% 预算。

**特点**：消息**原封不动**，包含完整 thinking blocks、完整 tool output。这确保 LLM 对最近的对话有完整理解。

### 4.3 Tier 2 — 中期压缩记忆

处理 Tier 3 未覆盖的中间消息。从剩余消息末尾往前填充，不超过 35% 预算。

**压缩方式**（`condenser.py`）：

| 消息类型 | 处理方式 |
|---------|---------|
| `HumanMessage` | 原样保留 |
| `SystemMessage` | 原样保留 |
| `AIMessage` | **删除 thinking/reasoning blocks**，仅保留 text 和 tool_calls |
| `ToolMessage` | **截断到 200 字符**，超出部分替换为 `... (truncated)` |

**压缩效果示例**：

一条 AIMessage 原始内容：
```json
[
  {"type": "thinking", "thinking": "让我分析一下这个SQL查询...（2000字）"},
  {"type": "text", "text": "查询结果如下"},
  {"type": "tool_use", "name": "bash", "input": {...}}
]
```

压缩后：
```json
[
  {"type": "text", "text": "查询结果如下"},
  {"type": "tool_use", "name": "bash", "input": {...}}
]
```

thinking block 通常占单条消息 50% 以上的 token，删除后大幅节省预算。

### 4.4 Tier 1 — 长期摘要记忆

从 `conversation_summaries` 和 `memory_topics` 表加载，注入为一条消息：

```
[Conversation Summary]
用户在之前的对话中讨论了 PostgreSQL 表设计、三层记忆管理、前端中文化...

[Recallable Topics]: 数据库设计, 前端中文化, Docker部署, 三层记忆
Use the recall_memory tool to retrieve details about any topic above.
```

LLM 看到这条消息后知道之前讨论过什么。如果需要具体细节，可以主动调用 `recall_memory` 工具，从 `memory_vectors` 表做向量相似度搜索。

### 4.5 LLM 最终看到的消息结构

```
┌─────────────────────────────────────────────────────┐
│ SystemMessage（原始 system prompt，不动）              │
├─────────────────────────────────────────────────────┤
│ Tier 1: "[Conversation Summary] ..."                │
│         "[Recallable Topics]: ..."                  │
│         （~10% 预算，来自 conversation_summaries）     │
├─────────────────────────────────────────────────────┤
│ Tier 2: msg[70] ~ msg[170] 的压缩版                 │
│         （无 thinking，tool output ≤200 字）          │
│         （~35% 预算，从后往前填充直到预算用完）         │
├─────────────────────────────────────────────────────┤
│ Tier 3: msg[171] ~ msg[180] 完整原文                 │
│         （含 thinking，完整 tool output）              │
│         （~55% 预算，最近 10 条消息）                  │
└─────────────────────────────────────────────────────┘
```

---

## 五、环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | 无 | PostgreSQL 连接字符串，未设置时 fallback 到内存模式 |
| `MEMORY_CONTEXT_WINDOW` | `0` | 上下文窗口大小（0 = 根据模型自动推断） |
| `MEMORY_FULL_RECENT_COUNT` | `10` | Tier 3 完整保留的最近消息数 |
| `MEMORY_ENABLE_SUMMARY` | `true` | 是否启用 LLM 摘要 |
| `MEMORY_SUMMARY_THRESHOLD` | `30` | 触发摘要的最小消息数 |
| `MEMORY_MAX_SUMMARY_TOKENS` | `1000` | 摘要最大 token 数 |
| `MEMORY_CONDENSED_TOOL_MAX` | `200` | Tier 2 工具结果截断字符数 |
| `MEMORY_MAX_MESSAGES_PER_THREAD` | `200` | 单会话最大消息数（超出则物理修剪） |
| `MEMORY_EMBEDDING_MODEL` | `embedding-3` | 向量嵌入模型 |
| `MEMORY_EMBEDDING_BASE_URL` | 空 | 嵌入模型 API 地址（空则复用 `API_BASE_URL`） |
| `MEMORY_EMBEDDING_API_KEY` | 空 | 嵌入模型 API Key（空则复用 `API_KEY`） |
| `MEMORY_RECALL_TOP_K` | `5` | `recall_memory` 返回的最相关片段数 |

---

## 六、降级策略

| 场景 | 行为 |
|------|------|
| 无 PostgreSQL（`DATABASE_URL` 未设置） | 使用 InMemorySaver，重启后数据丢失；无摘要、无向量检索、无会话列表 API |
| LLM 摘要调用失败 | warning 日志，使用旧摘要或跳过 Tier 1 |
| 嵌入模型不可用 | 向量搜索降级为关键词搜索 |
| 摘要本身过长 | 递归压缩（LLM 将旧摘要压缩到 1/3 长度） |
| Token 计数不精确 | 10% 安全裕量兜底 |
