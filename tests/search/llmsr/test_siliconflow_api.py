"""Unit tests for SiliconFlowAPI"""
import pytest
import os
from dotenv import load_dotenv
from nd2py.search.llmsr.api import SiliconFlowAPI

# Load environment variables from .env
load_dotenv()


@pytest.mark.slow
class TestSiliconFlowAPI:
    """Test SiliconFlowAPI functionality"""

    def test_qwen3_8b_basic(self):
        """Test Qwen3-8B model basic generation (FREE)"""
        api = SiliconFlowAPI(model='Qwen3-8B', max_tokens=50, temperature=0.7)

        prompt = "Say hello in one sentence."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0
        assert 'usage' in result._result
        assert 'messages' in result._result
        assert 'contents' in result._result

    def test_n_generation_qwen(self):
        """Test multiple generations with Qwen3-8B (FREE)"""
        api = SiliconFlowAPI(model='Qwen3-8B', max_tokens=30, temperature=1.0, n=3)

        prompt = "Give me a random number."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 3
        assert all(len(c) > 0 for c in contents)

    def test_messages_format(self):
        """Test with proper message format (FREE)"""
        api = SiliconFlowAPI(model='Qwen3-8B', max_tokens=50, temperature=0.7)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ]

        result = api(messages)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0

    def test_result_properties(self):
        """Test that LLMResult properties work correctly (FREE)"""
        api = SiliconFlowAPI(model='Qwen3-8B', max_tokens=30)

        result = api("Hello")
        _ = list(result)

        # Test properties
        assert isinstance(result.usage, dict)
        assert isinstance(result.contents, list)
        assert len(result.contents) > 0


@pytest.mark.slow
@pytest.mark.paid
class TestSiliconFlowAPIPaid:
    """Test SiliconFlowAPI DeepSeek-V3 (PAID - requires API credits)"""

    def test_deepseek_v3_basic(self):
        """Test Deepseek-V3 model basic generation"""
        api = SiliconFlowAPI(model='Deepseek-V3', max_tokens=50, temperature=0.7)

        prompt = "Say hello in one sentence."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 1
        assert len(contents[0]) > 0
        assert 'usage' in result._result

    def test_n_generation_deepseek(self):
        """Test multiple generations with Deepseek-V3"""
        api = SiliconFlowAPI(model='Deepseek-V3', max_tokens=30, temperature=1.0, n=3)

        prompt = "Give me a random number."

        result = api(prompt)
        contents = list(result)

        assert len(contents) == 3
        assert all(len(c) > 0 for c in contents)
