"""
Tests for the three-tier memory management system.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_skills.memory.config import MemoryConfig, MODEL_CONTEXT_WINDOWS
from langchain_skills.memory.token_budget import (
    estimate_tokens,
    estimate_message_tokens,
    calculate_budget,
)
from langchain_skills.memory.condenser import condense_message
from langchain_skills.memory.summarizer import ConversationSummarizer
from langchain_skills.memory.middleware import MemoryMiddleware
from langchain_skills.memory.retriever import MemoryRetriever, _chunk_by_turns, _split_long_text, _messages_to_text


# ── Config Tests ──


class TestMemoryConfig:
    def test_default_values(self):
        config = MemoryConfig()
        assert config.context_window == 0
        assert config.full_recent_count == 10
        assert config.enable_summary is True
        assert config.max_messages_per_thread == 200

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONTEXT_WINDOW", "50000")
        monkeypatch.setenv("MEMORY_FULL_RECENT_COUNT", "5")
        monkeypatch.setenv("MEMORY_ENABLE_SUMMARY", "false")
        monkeypatch.setenv("MEMORY_MAX_MESSAGES_PER_THREAD", "100")
        config = MemoryConfig.from_env()
        assert config.context_window == 50000
        assert config.full_recent_count == 5
        assert config.enable_summary is False
        assert config.max_messages_per_thread == 100

    def test_get_context_window_explicit(self):
        config = MemoryConfig(context_window=50000)
        assert config.get_context_window("any-model") == 50000

    def test_get_context_window_auto_detect(self):
        config = MemoryConfig(context_window=0)
        assert config.get_context_window("claude-sonnet-4-5-20250929") == 200_000
        assert config.get_context_window("gpt-4o") == 128_000

    def test_get_context_window_fallback(self):
        config = MemoryConfig(context_window=0)
        assert config.get_context_window("unknown-model") == 128_000


# ── Token Budget Tests ──


class TestTokenBudget:
    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_nonempty(self):
        tokens = estimate_tokens("Hello world, this is a test message.")
        assert tokens > 0

    def test_estimate_message_tokens_string(self):
        msg = HumanMessage(content="Hello")
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_estimate_message_tokens_list(self):
        msg = AIMessage(content=[
            {"type": "thinking", "thinking": "Let me think..."},
            {"type": "text", "text": "Here is my answer."},
        ])
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_calculate_budget(self):
        config = MemoryConfig(context_window=100_000)
        budget = calculate_budget(config, "test-model", system_prompt_tokens=500)
        assert budget.total > 0
        assert budget.tier1_max > 0
        assert budget.tier2_max > 0
        assert budget.tier3_max > 0
        # Tier budgets should sum approximately to total
        tier_sum = budget.tier1_max + budget.tier2_max + budget.tier3_max
        assert abs(tier_sum - budget.total) <= 2


# ── Condenser Tests ──


class TestCondenser:
    def test_condense_human_message_unchanged(self):
        msg = HumanMessage(content="Hello")
        result = condense_message(msg)
        assert result is msg

    def test_condense_system_message_unchanged(self):
        msg = SystemMessage(content="You are a helpful assistant.")
        result = condense_message(msg)
        assert result is msg

    def test_condense_ai_strips_thinking(self):
        msg = AIMessage(content=[
            {"type": "thinking", "thinking": "Deep reasoning here..."},
            {"type": "text", "text": "My answer."},
        ])
        result = condense_message(msg)
        assert isinstance(result, AIMessage)
        # Should only have the text block
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"

    def test_condense_ai_string_unchanged(self):
        msg = AIMessage(content="Simple string response")
        result = condense_message(msg)
        assert result is msg

    def test_condense_tool_message_truncate(self):
        long_content = "x" * 500
        msg = ToolMessage(content=long_content, tool_call_id="test-id")
        result = condense_message(msg, max_tool_chars=200)
        assert isinstance(result, ToolMessage)
        assert len(result.content) < len(long_content)
        assert "truncated" in result.content

    def test_condense_tool_message_short_unchanged(self):
        msg = ToolMessage(content="short", tool_call_id="test-id")
        result = condense_message(msg, max_tool_chars=200)
        assert result is msg


# ── Summarizer Tests ──


class TestSummarizer:
    def test_load_summary_no_pg(self):
        summarizer = ConversationSummarizer(pg_conn=None, llm=None)
        assert summarizer.load_summary("thread-1") is None

    def test_save_summary_no_pg(self):
        summarizer = ConversationSummarizer(pg_conn=None, llm=None)
        # Should not raise
        summarizer.save_summary("thread-1", "test summary", 10)

    def test_generate_summary_no_llm(self):
        summarizer = ConversationSummarizer(pg_conn=None, llm=None)
        messages = [HumanMessage(content="Hello")]
        assert summarizer.generate_summary(messages) is None

    def test_generate_summary_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary of conversation")
        summarizer = ConversationSummarizer(pg_conn=None, llm=mock_llm)
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]
        result = summarizer.generate_summary(messages)
        assert result == "Summary of conversation"
        mock_llm.invoke.assert_called_once()

    def test_generate_and_store_no_messages(self):
        summarizer = ConversationSummarizer(pg_conn=None, llm=None)
        result = summarizer.generate_and_store("thread-1", [])
        assert result is None

    def test_generate_and_store_merges_summaries(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="New summary")
        summarizer = ConversationSummarizer(
            pg_conn=None, llm=mock_llm, max_summary_tokens=10000
        )
        messages = [HumanMessage(content="Hello")]
        result = summarizer.generate_and_store(
            "thread-1", messages, existing_summary="Old context"
        )
        assert result is not None
        assert "Old context" in result
        assert "New summary" in result


# ── Middleware Tests ──


class TestMemoryMiddleware:
    def _make_messages(self, count: int) -> list:
        """Create a list of alternating human/AI messages."""
        messages = []
        for i in range(count):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User message {i}", id=f"msg-{i}"))
            else:
                messages.append(AIMessage(content=f"AI response {i}", id=f"msg-{i}"))
        return messages

    def test_small_conversation_no_trimming(self):
        config = MemoryConfig(context_window=200_000)
        middleware = MemoryMiddleware(config=config, model_name="test-model")
        messages = self._make_messages(6)
        result = middleware.apply(messages, "thread-1")
        # Small conversation should be returned as-is
        assert len(result) == 6

    def test_large_conversation_is_trimmed(self):
        # Use a very small context window to force trimming
        config = MemoryConfig(
            context_window=200,
            full_recent_count=2,
        )
        middleware = MemoryMiddleware(config=config, model_name="test-model")
        messages = self._make_messages(50)
        result = middleware.apply(messages, "thread-1")
        # Should be trimmed to fewer messages
        assert len(result) < 50

    def test_system_messages_preserved(self):
        config = MemoryConfig(context_window=500, full_recent_count=2)
        middleware = MemoryMiddleware(config=config, model_name="test-model")
        messages = [
            SystemMessage(content="You are helpful."),
            *self._make_messages(50),
        ]
        result = middleware.apply(messages, "thread-1")
        # System message should be at the front
        assert isinstance(result[0], SystemMessage)

    def test_tier3_recent_messages_at_end(self):
        config = MemoryConfig(
            context_window=2000,
            full_recent_count=3,
        )
        middleware = MemoryMiddleware(config=config, model_name="test-model")
        messages = self._make_messages(40)
        result = middleware.apply(messages, "thread-1")
        if len(result) < len(messages):
            # Last messages should be the most recent ones from original
            last_original = messages[-1]
            last_result = result[-1]
            assert last_result.content == last_original.content

    def test_with_summary(self):
        config = MemoryConfig(
            context_window=200,
            full_recent_count=2,
            enable_summary=True,
        )
        mock_summarizer = MagicMock()
        mock_summarizer.load_summary.return_value = "Previous conversation about Python."
        mock_summarizer.compress_summary.return_value = "Previous conversation about Python."

        middleware = MemoryMiddleware(
            config=config,
            summarizer=mock_summarizer,
            model_name="test-model",
        )
        messages = self._make_messages(40)
        result = middleware.apply(messages, "thread-1")
        # Should include a summary message
        summary_msgs = [
            m for m in result
            if isinstance(m, HumanMessage) and "[Conversation Summary]" in str(m.content)
        ]
        assert len(summary_msgs) == 1
        mock_summarizer.load_summary.assert_called_once_with("thread-1")

    def test_with_topic_index(self):
        config = MemoryConfig(
            context_window=200,
            full_recent_count=2,
            enable_summary=True,
        )
        mock_summarizer = MagicMock()
        mock_summarizer.load_summary.return_value = "Summary of past conversation."
        mock_summarizer.compress_summary.return_value = "Summary of past conversation."

        mock_retriever = MagicMock()
        mock_retriever.load_topic_index.return_value = ["Python基础", "数据库设计", "API认证"]

        middleware = MemoryMiddleware(
            config=config,
            summarizer=mock_summarizer,
            retriever=mock_retriever,
            model_name="test-model",
        )
        messages = self._make_messages(40)
        result = middleware.apply(messages, "thread-1")
        # Should include both summary and topic index
        summary_msgs = [
            m for m in result
            if isinstance(m, HumanMessage) and "[Recallable Topics]" in str(m.content)
        ]
        assert len(summary_msgs) == 1
        assert "recall_memory" in str(summary_msgs[0].content)
        mock_retriever.load_topic_index.assert_called_once_with("thread-1")


# ── Retriever Tests ──


class TestRetriever:
    def test_chunk_by_turns_basic(self):
        """Each Human+AI pair should become one chunk."""
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
            HumanMessage(content="How about Java?"),
            AIMessage(content="Java is also a popular language."),
        ]
        chunks = _chunk_by_turns(messages)
        assert len(chunks) == 2
        assert "What is Python?" in chunks[0]
        assert "Python is a programming language." in chunks[0]
        assert "How about Java?" in chunks[1]

    def test_chunk_by_turns_single_exchange(self):
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        chunks = _chunk_by_turns(messages)
        assert len(chunks) == 1
        assert "Hello" in chunks[0]
        assert "Hi there" in chunks[0]

    def test_chunk_by_turns_empty(self):
        chunks = _chunk_by_turns([])
        assert chunks == []

    def test_split_long_text(self):
        text = "x" * 3000
        chunks = _split_long_text(text, max_chars=1200, overlap=150)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 1200

    def test_split_long_text_short(self):
        chunks = _split_long_text("short text", max_chars=1200)
        assert len(chunks) == 1

    def test_messages_to_text(self):
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]
        text = _messages_to_text(messages)
        assert "Human: What is Python?" in text
        assert "AI: Python is a programming language." in text

    def test_messages_to_text_with_list_content(self):
        messages = [
            AIMessage(content=[
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Here is my answer."},
            ]),
        ]
        text = _messages_to_text(messages)
        assert "Let me think" in text
        assert "Here is my answer" in text

    def test_retriever_no_pg(self):
        retriever = MemoryRetriever(pg_conn=None)
        assert retriever.store_messages("t1", [HumanMessage(content="hi")]) == 0
        assert retriever.search("t1", "hello") == []
        assert retriever.load_topic_index("t1") == []

    def test_make_id_deterministic(self):
        retriever = MemoryRetriever(pg_conn=None)
        id1 = retriever._make_id("thread-1", "some chunk text")
        id2 = retriever._make_id("thread-1", "some chunk text")
        id3 = retriever._make_id("thread-1", "different chunk")
        assert id1 == id2
        assert id1 != id3


# ── Topic Extraction Tests ──


class TestTopicExtraction:
    def test_extract_topics_no_llm(self):
        summarizer = ConversationSummarizer(pg_conn=None, llm=None)
        result = summarizer.extract_topics([HumanMessage(content="hello")])
        assert result == []

    def test_extract_topics_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='["Python基础", "数据库设计", "API认证"]'
        )
        summarizer = ConversationSummarizer(pg_conn=None, llm=mock_llm)
        messages = [
            HumanMessage(content="Tell me about Python"),
            AIMessage(content="Python is great for web development"),
        ]
        topics = summarizer.extract_topics(messages)
        assert len(topics) == 3
        assert "Python基础" in topics

    def test_extract_topics_handles_markdown_code_block(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n["topic1", "topic2"]\n```'
        )
        summarizer = ConversationSummarizer(pg_conn=None, llm=mock_llm)
        topics = summarizer.extract_topics([HumanMessage(content="test")])
        assert len(topics) == 2

    def test_extract_topics_handles_invalid_json(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json at all")
        summarizer = ConversationSummarizer(pg_conn=None, llm=mock_llm)
        topics = summarizer.extract_topics([HumanMessage(content="test")])
        assert topics == []
