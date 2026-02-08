"""
Three-tier memory middleware.

Intercepts messages before they are sent to the LLM, applying the three-tier
memory strategy:

- Tier 1 (Summary): Long-term conversation summary (~10% budget)
- Tier 2 (Condensed): Recent messages with thinking stripped and tool results
  truncated (~35% budget)
- Tier 3 (Full): Most recent messages kept intact (~55% budget)

The checkpointer still stores the complete history; this middleware only
affects what the LLM sees.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .condenser import condense_message
from .config import MemoryConfig
from .retriever import MemoryRetriever
from .summarizer import ConversationSummarizer
from .token_budget import (
    TokenBudget,
    calculate_budget,
    estimate_message_tokens,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class MemoryMiddleware:
    """
    Three-tier memory middleware.

    Usage:
        middleware = MemoryMiddleware(config, summarizer, model_name)
        trimmed = middleware.apply(messages, thread_id)
        # Send trimmed messages to LLM instead of full history
    """

    def __init__(
        self,
        config: MemoryConfig,
        summarizer: Optional[ConversationSummarizer] = None,
        retriever: Optional[MemoryRetriever] = None,
        model_name: str = "",
        system_prompt: str = "",
    ):
        self.config = config
        self.summarizer = summarizer
        self.retriever = retriever
        self.model_name = model_name
        self.system_prompt_tokens = estimate_tokens(system_prompt)

    def apply(
        self,
        messages: list,
        thread_id: str = "default",
    ) -> list:
        """
        Apply three-tier memory strategy to messages.

        Returns a new list of messages trimmed to fit the context window budget.
        The original list is not modified.
        """
        if not messages:
            return messages

        # Separate system messages from conversation messages
        system_msgs = []
        conversation_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                conversation_msgs.append(msg)

        # If conversation is small enough, return as-is
        total_tokens = sum(estimate_message_tokens(m) for m in conversation_msgs)
        budget = calculate_budget(
            self.config, self.model_name, self.system_prompt_tokens
        )

        if total_tokens <= budget.total:
            logger.debug(
                "All %d messages fit in budget (%d/%d tokens), no trimming needed",
                len(conversation_msgs),
                total_tokens,
                budget.total,
            )
            return messages

        logger.info(
            "Trimming %d messages (%d tokens) to fit budget (%d tokens)",
            len(conversation_msgs),
            total_tokens,
            budget.total,
        )

        # Apply three-tier strategy
        result = list(system_msgs)

        # --- Tier 1: Summary ---
        tier1_messages = self._build_tier1(thread_id, budget)
        result.extend(tier1_messages)

        # --- Tier 3: Full recent (work backwards from end) ---
        tier3_msgs, tier3_tokens, tier3_count = self._build_tier3(
            conversation_msgs, budget
        )

        # --- Tier 2: Condensed middle (between summarized and full recent) ---
        remaining = conversation_msgs[: len(conversation_msgs) - tier3_count]
        tier2_msgs = self._build_tier2(remaining, budget)

        result.extend(tier2_msgs)
        result.extend(tier3_msgs)

        logger.info(
            "Memory tiers: T1=%d msgs, T2=%d msgs, T3=%d msgs (of %d total)",
            len(tier1_messages),
            len(tier2_msgs),
            len(tier3_msgs),
            len(conversation_msgs),
        )

        return result

    def _build_tier1(
        self, thread_id: str, budget: TokenBudget
    ) -> list:
        """Build Tier 1: conversation summary + topic index."""
        if not self.summarizer or not self.config.enable_summary:
            return []

        summary = self.summarizer.load_summary(thread_id)

        # Load topic index from retriever
        topics = []
        if self.retriever:
            topics = self.retriever.load_topic_index(thread_id)

        if not summary and not topics:
            return []

        # Build the Tier 1 content
        parts = []
        if summary:
            parts.append(f"[Conversation Summary]\n{summary}")
        if topics:
            parts.append(f"[Recallable Topics]: {', '.join(topics)}\n"
                         f"Use the recall_memory tool to retrieve details about any topic above.")

        content = "\n\n".join(parts)

        content_tokens = estimate_tokens(content)
        if content_tokens > budget.tier1_max:
            # Try to compress summary part
            if summary:
                compressed = self.summarizer.compress_summary(summary)
                if compressed and estimate_tokens(compressed) < estimate_tokens(summary):
                    parts[0] = f"[Conversation Summary]\n{compressed}"
                    content = "\n\n".join(parts)

        return [
            HumanMessage(
                content=content,
                id="memory-summary",
            )
        ]

    def _build_tier3(
        self, conversation_msgs: list, budget: TokenBudget
    ) -> tuple[list, int, int]:
        """
        Build Tier 3: full recent messages (last N messages).

        Returns (messages, total_tokens, count_from_end).
        """
        max_count = self.config.full_recent_count
        tier3_budget = budget.tier3_max

        result = []
        total_tokens = 0
        count = 0

        # Walk backwards from the most recent message
        for msg in reversed(conversation_msgs):
            if count >= max_count:
                break
            msg_tokens = estimate_message_tokens(msg)
            if total_tokens + msg_tokens > tier3_budget:
                break
            result.insert(0, msg)
            total_tokens += msg_tokens
            count += 1

        return result, total_tokens, count

    def _build_tier2(
        self, remaining_msgs: list, budget: TokenBudget
    ) -> list:
        """
        Build Tier 2: condensed recent messages.

        Condense messages (strip thinking, truncate tool results) and fill
        backwards from the remaining messages until budget is used.
        """
        tier2_budget = budget.tier2_max
        max_tool_chars = self.config.condensed_tool_max

        result = []
        total_tokens = 0

        # Walk backwards through remaining messages
        for msg in reversed(remaining_msgs):
            condensed = condense_message(msg, max_tool_chars)
            msg_tokens = estimate_message_tokens(condensed)
            if total_tokens + msg_tokens > tier2_budget:
                break
            result.insert(0, condensed)
            total_tokens += msg_tokens

        return result
