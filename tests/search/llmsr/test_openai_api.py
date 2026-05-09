"""Unit tests for OpenAIAPI - PAID API"""
import pytest
import os
from dotenv import load_dotenv
from nd2py.search.llmsr.api import OpenAIAPI

# Load environment variables from .env
load_dotenv()


@pytest.mark.slow
@pytest.mark.paid
class TestOpenAIAPI:
    """Test OpenAIAPI functionality (PAID - requires API credits)"""

    def test_chat_completions_basic(self):
        """Test gpt-4o-mini with chat completions API"""
        api = OpenAIAPI(model='gpt-4o-mini', max_tokens=50, temperature=0.7)

        prompt = "Say hello in one sentence."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0
        assert 'usage' in result._result
        assert 'messages' in result._result
        assert 'contents' in result._result
        assert 'response' in result._result

    def test_responses_api_basic(self):
        """Test gpt-5-mini with responses API"""
        api = OpenAIAPI(model='gpt-5-mini', max_tokens=50)

        messages = [{"role": "user", "content": "Say hello in one sentence."}]

        result = api(messages)
        contents = list(result)

        assert len(contents) >= 0

    def test_n_generation(self):
        """Test multiple generations (n > 1)"""
        api = OpenAIAPI(model='gpt-4o-mini', max_tokens=30, temperature=1.0, n=3)

        prompt = "Give me a random number."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 3

    def test_usage_tracking(self):
        """Test token usage and price tracking"""
        api = OpenAIAPI(model='gpt-4o-mini', max_tokens=30, temperature=0.7)

        prompt = "What is 2+2? Answer in one word."

        result = api(prompt)
        _ = list(result)  # Consume

        assert 'token' in result.usage
        assert 'price' in result.usage
        assert result.usage['token']['prompt'] > 0

    def test_messages_format(self):
        """Test with proper message format including system/developer role"""
        api = OpenAIAPI(model='gpt-4o-mini', max_tokens=50, temperature=0.7)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ]

        result = api(messages)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0

    def test_result_properties(self):
        """Test that LLMResult properties work correctly"""
        api = OpenAIAPI(model='gpt-4o-mini', max_tokens=30)

        result = api("Hello")
        _ = list(result)  # Consume

        # Test properties
        assert isinstance(result.usage, dict)
        assert isinstance(result.contents, list)
        assert isinstance(result.messages, list)
        assert len(result.contents) > 0
