"""
Token budget calculator for the three-tier memory system.

Estimates token counts for messages and allocates budget across tiers.
"""

from dataclasses import dataclass

from .config import MemoryConfig


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for mixed CJK/English."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def estimate_message_tokens(msg) -> int:
    """Estimate tokens for a LangChain message."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return estimate_tokens(content) + 4  # overhead for role etc.
    elif isinstance(content, list):
        total = 4
        for block in content:
            if isinstance(block, str):
                total += estimate_tokens(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype in ("thinking", "reasoning"):
                    total += estimate_tokens(
                        block.get("thinking", "") or block.get("reasoning", "")
                    )
                elif btype == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif btype in ("tool_use", "tool_call"):
                    import json

                    args = block.get("input") or block.get("args") or {}
                    total += estimate_tokens(json.dumps(args, ensure_ascii=False)) + 10
        return total
    return 4


@dataclass
class TokenBudget:
    """Calculated token budgets for each tier."""

    total: int
    tier1_max: int  # summary budget
    tier2_max: int  # condensed recent budget
    tier3_max: int  # full recent budget


def calculate_budget(
    config: MemoryConfig,
    model_name: str,
    system_prompt_tokens: int = 0,
) -> TokenBudget:
    """
    Calculate token budgets for each tier.

    Available = context_window * (1 - safety_margin) - system_prompt - output_reserve
    Then split across tiers by ratio.
    """
    context_window = config.get_context_window(model_name)

    # Reserve space for safety margin and output (20% for output, capped at 16k)
    usable = int(context_window * (1 - config.safety_margin))
    output_reserve = min(int(context_window * 0.2), 16000)
    available = max(usable - system_prompt_tokens - output_reserve, 0)

    return TokenBudget(
        total=available,
        tier1_max=int(available * config.tier1_ratio),
        tier2_max=int(available * config.tier2_ratio),
        tier3_max=int(available * config.tier3_ratio),
    )
