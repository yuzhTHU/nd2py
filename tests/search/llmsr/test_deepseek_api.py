"""Unit tests for DeepSeekAPI"""
import pytest
import os
from dotenv import load_dotenv
from nd2py.search.llmsr.api import DeepSeekAPI

# Load environment variables from .env
load_dotenv()


@pytest.mark.slow
@pytest.mark.paid
class TestDeepSeekAPI:
    """Test DeepSeekAPI functionality (PAID - requires API credits)"""

    def test_deepseek_chat_basic(self):
        """Test deepseek-chat model basic generation"""
        api = DeepSeekAPI(model='deepseek-chat', max_tokens=50)

        prompt = "Say hello in one sentence."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0
        assert 'usage' in result._result
        assert 'messages' in result._result
        assert 'contents' in result._result

    def test_deepseek_reasoner_basic(self):
        """Test deepseek-reasoner model basic generation"""
        api = DeepSeekAPI(model='deepseek-reasoner', max_tokens=1000)

        prompt = "What is 2+2?"

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0

    def test_n_generation(self):
        """Test multiple generations (n > 1)"""
        api = DeepSeekAPI(model='deepseek-chat', max_tokens=30, n=3)

        prompt = "Give me a random number."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 3
        assert all(len(c) > 0 for c in contents)

    def test_usage_tracking(self):
        """Test token usage and price tracking"""
        api = DeepSeekAPI(model='deepseek-chat', max_tokens=30)

        prompt = "What is 2+2? Answer in one word."

        result = api(prompt)
        _ = list(result)

        assert 'token' in result.usage
        assert 'price' in result.usage
        assert result.usage['token']['prompt'] > 0
        assert result.usage['token']['answer'] > 0

    def test_result_properties(self):
        """Test that LLMResult properties work correctly"""
        api = DeepSeekAPI(model='deepseek-chat', max_tokens=30)

        result = api("Hello")
        _ = list(result)

        assert isinstance(result.usage, dict)
        assert isinstance(result.contents, list)
        assert len(result.contents) > 0
