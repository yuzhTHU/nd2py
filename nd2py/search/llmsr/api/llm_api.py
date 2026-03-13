import logging
from typing import List, Generator, Dict, Type
from .llm_result import LLMResult

_logger = logging.getLogger(__name__)


class LLMAPI:
    supported_models = []

    @classmethod
    def load(cls, args) -> 'LLMAPI':
        from . import (
            SiliconFlowAPI, OpenRouterAPI, OpenAIAPI,
            GeminiAPI, DeepSeekAPI, ManualAPI,
        )
        if args.llm_provider:
            if args.llm_provider.lower() == "siliconflow":
                return SiliconFlowAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_provider.lower() == "openrouter":
                return OpenRouterAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_provider.lower() == "openai":
                return OpenAIAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_provider.lower() == "gemini":
                return GeminiAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_provider.lower() == "deepseek":
                return DeepSeekAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_provider.lower() == "manual":
                return ManualAPI(model=args.llm_model, n=args.generate_per_prompt)
            else:
                raise ValueError(f"Unsupported provider: {args.llm_provider}.")
        elif args.llm_model:
            if args.llm_model in ManualAPI.supported_models:
                return ManualAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_model in DeepSeekAPI.supported_models:
                return DeepSeekAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_model in OpenAIAPI.supported_models:
                return OpenAIAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_model in GeminiAPI.supported_models:
                return GeminiAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_model in SiliconFlowAPI.supported_models:
                return SiliconFlowAPI(model=args.llm_model, n=args.generate_per_prompt)
            elif args.llm_model in OpenRouterAPI.supported_models:
                return OpenRouterAPI(model=args.llm_model, n=args.generate_per_prompt)
            else:
                raise ValueError(f"Unsupported model: {args.llm_model}.")
        else:
            _logger.warning("No llm_provider or llm_model specified, returning base LLMAPI.")
            return LLMAPI()

    def __init__(self):
        self.model = None

    def __call__(self, messages: List | str, **kwargs) -> LLMResult:
        """Request the LLM and return an LLMResult wrapper.

        This is the main entry point for requesting the LLM API.
        It returns an LLMResult object that:
        - Can be iterated to get generated content (streaming)
        - Provides property access to usage, messages, response, etc.

        Args:
            messages (List | str): The input messages or prompt string.

        Returns:
            LLMResult: A wrapper that yields content and provides property access to results.

        Example:
            >>> api = OpenAIAPI(model='gpt-4o-mini')
            >>> result = api("Hello")
            >>> for content in result:
            ...     print(content)
            >>> print(result.usage)      # Token usage
            >>> print(result.contents)   # List of generated contents
        """
        return LLMResult(self._request(messages, **kwargs))

    def _request(self, messages: List | str, **kwargs) -> Generator[str, None, Dict]:
        """Internal method that implements the actual LLM request.

        Subclasses should override this method instead of __call__.

        Args:
            messages (List | str): The input messages or prompt string.

        Yields:
            content (str): The generated content from the LLM.

        Returns:
            dict: The usage statistics and other relevant information.
        """
        raise NotImplementedError("Subclasses should implement this method.")
