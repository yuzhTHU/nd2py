# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ....utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_api import LLMAPI
    from .llm_result import LLMResult
    from .openai_api import OpenAIAPI
    from .manual_api import ManualAPI
    from .gemini_api import GeminiAPI
    from .deepseek_api import DeepSeekAPI
    from .openrouter_api import OpenRouterAPI
    from .siliconflow_api import SiliconFlowAPI

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "LLMAPI": (".llm_api", "all"),
    "LLMResult": (".llm_result", "all"),
    "OpenAIAPI": (".openai_api", "all"),
    "ManualAPI": (".manual_api", "all"),
    "GeminiAPI": (".gemini_api", "all"),
    "DeepSeekAPI": (".deepseek_api", "all"),
    "OpenRouterAPI": (".openrouter_api", "all"),
    "SiliconFlowAPI": (".siliconflow_api", "all"),
})
