"""
Message condenser for Tier 2.

Strips thinking/reasoning blocks and truncates tool results to create
condensed versions of messages that use less token budget.
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def condense_message(msg, max_tool_chars: int = 200):
    """
    Create a condensed copy of a message for Tier 2.

    - HumanMessage / SystemMessage: kept as-is
    - AIMessage: strip thinking/reasoning blocks, keep text + tool_calls
    - ToolMessage: truncate content to max_tool_chars
    """
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg

    if isinstance(msg, AIMessage):
        return _condense_ai_message(msg)

    if isinstance(msg, ToolMessage):
        return _condense_tool_message(msg, max_tool_chars)

    # Unknown message type: return as-is
    return msg


def _condense_ai_message(msg: AIMessage) -> AIMessage:
    """Strip thinking/reasoning blocks from AIMessage content."""
    content = msg.content

    if isinstance(content, str):
        # String content has no thinking blocks
        return msg

    if isinstance(content, list):
        condensed_blocks = []
        for block in content:
            if isinstance(block, str):
                condensed_blocks.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype in ("thinking", "reasoning"):
                    continue  # strip thinking blocks
                condensed_blocks.append(block)
            else:
                condensed_blocks.append(block)

        # Reconstruct with same metadata
        return AIMessage(
            content=condensed_blocks,
            id=msg.id,
            tool_calls=msg.tool_calls if hasattr(msg, "tool_calls") else [],
            additional_kwargs=msg.additional_kwargs,
        )

    return msg


def _condense_tool_message(msg: ToolMessage, max_chars: int) -> ToolMessage:
    """Truncate tool message content."""
    content = str(msg.content) if msg.content else ""

    if len(content) <= max_chars:
        return msg

    truncated = content[:max_chars] + "\n... (truncated)"
    return ToolMessage(
        content=truncated,
        tool_call_id=msg.tool_call_id,
        name=getattr(msg, "name", None),
        id=msg.id,
    )
