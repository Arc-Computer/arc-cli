"""
LLM Client Interface

Interface for LLM integration using OpenRouter API.
"""

import logging
import json
import os
import aiohttp
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class OpenRouterLLMClient(LLMClient):
    """OpenRouter LLM client for OpenAI and Anthropic models."""
    
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3-haiku"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        logger.info(f"Initialized OpenRouter client with model: {model}")
    
    async def generate(self, prompt: str) -> str:
        """Generate response using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://arc-eval.com",
            "X-Title": "Arc Evaluation Platform"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI system analyzer that provides detailed, actionable recommendations based on simulation data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.4,
            "max_tokens": 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error {response.status}: {error_text}")
                    raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                
                result = await response.json()
                content = result['choices'][0]['message']['content']
                
                logger.debug(f"Generated response using {self.model}: {len(content)} characters")
                return content


class OpenAILLMClient(OpenRouterLLMClient):
    """OpenAI LLM client via OpenRouter."""
    
    def __init__(self, api_key: str = None, model: str = "openai/gpt-4o-mini"):
        super().__init__(api_key, model)


class AnthropicLLMClient(OpenRouterLLMClient):
    """Anthropic LLM client via OpenRouter."""
    
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3-haiku"):
        super().__init__(api_key, model)


def create_llm_client(provider: str = "openrouter", **kwargs) -> Optional[LLMClient]:
    """Factory function to create LLM clients."""
    try:
        api_key = kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        
        if not api_key:
            logger.error("API key required for LLM client. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
            return None
        
        if provider.lower() in ["openrouter", "anthropic"]:
            model = kwargs.get("model", "anthropic/claude-3-haiku")
            return OpenRouterLLMClient(api_key, model)
        elif provider.lower() == "openai":
            model = kwargs.get("model", "openai/gpt-4o-mini")
            return OpenAILLMClient(api_key, model)
        else:
            logger.warning(f"Unknown LLM provider: {provider}, falling back to OpenRouter")
            model = kwargs.get("model", "anthropic/claude-3-haiku")
            return OpenRouterLLMClient(api_key, model)
            
    except Exception as e:
        logger.error(f"Error creating LLM client: {e}")
        return None 