"""
Memory configuration and model context window mappings.
"""

import os
from dataclasses import dataclass, field

# Model → context window size (tokens)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-opus-4-5-20251101": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # OpenAI compatible
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    # DeepSeek
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    # GLM
    "glm-4": 128_000,
    "glm-4-flash": 128_000,
    "glm-4.7-flash": 128_000,
}

DEFAULT_CONTEXT_WINDOW = 128_000


@dataclass
class MemoryConfig:
    """Configuration for the three-tier memory system."""

    # Context window (0 = auto-detect from model name)
    context_window: int = 0

    # Tier 3: full recent messages count
    full_recent_count: int = 10

    # Tier 1: summary settings
    enable_summary: bool = True
    summary_threshold: int = 30  # min messages before generating summary
    max_summary_tokens: int = 1000

    # Tier 2: condensed tool result max chars
    condensed_tool_max: int = 200

    # Thread pruning
    max_messages_per_thread: int = 200

    # RAG recall
    embedding_model: str = "embedding-3"
    embedding_base_url: str = ""  # empty = reuse API_BASE_URL
    embedding_api_key: str = ""  # empty = reuse API_KEY
    recall_top_k: int = 5

    # Budget ratios (must sum to 1.0)
    tier1_ratio: float = 0.10  # summary
    tier2_ratio: float = 0.35  # condensed recent
    tier3_ratio: float = 0.55  # full recent

    # Safety margin (reserve this fraction of the budget)
    safety_margin: float = 0.10

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Load configuration from environment variables."""
        return cls(
            context_window=int(os.getenv("MEMORY_CONTEXT_WINDOW", "0")),
            full_recent_count=int(os.getenv("MEMORY_FULL_RECENT_COUNT", "10")),
            enable_summary=os.getenv("MEMORY_ENABLE_SUMMARY", "true").lower()
            in ("1", "true", "yes"),
            summary_threshold=int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "30")),
            max_summary_tokens=int(os.getenv("MEMORY_MAX_SUMMARY_TOKENS", "1000")),
            condensed_tool_max=int(os.getenv("MEMORY_CONDENSED_TOOL_MAX", "200")),
            max_messages_per_thread=int(
                os.getenv("MEMORY_MAX_MESSAGES_PER_THREAD", "200")
            ),
            embedding_model=os.getenv("MEMORY_EMBEDDING_MODEL", "embedding-3"),
            embedding_base_url=os.getenv("MEMORY_EMBEDDING_BASE_URL", ""),
            embedding_api_key=os.getenv("MEMORY_EMBEDDING_API_KEY", ""),
            recall_top_k=int(os.getenv("MEMORY_RECALL_TOP_K", "5")),
        )

    def get_context_window(self, model_name: str) -> int:
        """Resolve context window size from config or model name."""
        if self.context_window > 0:
            return self.context_window
        # Try exact match first, then prefix match
        if model_name in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model_name]
        for key, size in MODEL_CONTEXT_WINDOWS.items():
            if model_name.startswith(key) or key.startswith(model_name):
                return size
        return DEFAULT_CONTEXT_WINDOW
