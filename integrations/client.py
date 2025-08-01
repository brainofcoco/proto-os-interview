"""
Unified LLM Client.

This module provides a unified interface for working with multiple LLM providers.
It automatically handles provider registration, configuration, and routing.
"""

import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Type, Union

from .base import LLMProvider, provider_registry
from .config import config
from .openrouter.provider import OpenRouterProvider
from .openai.provider import OpenAIProvider
from .openrouter.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    GenerationInfo,
    ChatMessage
)

# Configure logging
logger = logging.getLogger(__name__)


# Auto-register providers
provider_registry.register(OpenRouterProvider)
provider_registry.register(OpenAIProvider)


class UnifiedLLMClient:
    """
    Unified client that can work with multiple LLM providers.
    
    This client provides a consistent interface regardless of the underlying provider,
    making it easy to switch between OpenRouter, OpenAI, Claude, etc.
    """
    
    def __init__(self, provider_name: str = "openrouter", **config_kwargs):
        """
        Initialize the unified client with a specific provider.
        
        Args:
            provider_name: Name of the provider to use ("openrouter", "openai", etc.)
            **config_kwargs: Configuration parameters for the provider
        """
        self.provider_name = provider_name
        self.provider = self._create_provider(provider_name, **config_kwargs)
        logger.info(f"Unified LLM client initialized with provider: {provider_name}")
    
    def _create_provider(self, provider_name: str, **config_kwargs) -> LLMProvider:
        """Create and configure a provider instance."""
        # Get configuration from the centralized config system
        provider_config = config.get_provider_config(provider_name)
        
        # Override with any provided config kwargs
        if config_kwargs:
            provider_config = provider_config.model_copy(update=config_kwargs)
        
        return provider_registry.create_provider(provider_name, provider_config)
    
    @property
    def capabilities(self):
        """Get the capabilities of the current provider."""
        return self.provider.capabilities
    
    def switch_provider(self, provider_name: str, **config_kwargs):
        """Switch to a different provider."""
        self.provider_name = provider_name
        self.provider = self._create_provider(provider_name, **config_kwargs)
        logger.info(f"Switched to provider: {provider_name}")
    
    def list_available_providers(self) -> List[str]:
        """List all registered providers."""
        return provider_registry.list_providers()
    
    # Core LLM methods - these delegate to the current provider
    
    async def chat_completion(self, request: Union[ChatCompletionRequest, Dict[str, Any]]) -> ChatCompletionResponse:
        """Generate a chat completion using the current provider."""
        if isinstance(request, dict):
            request = ChatCompletionRequest(**request)
        
        return await self.provider.chat_completion(request)
    
    async def completion(self, request: Union[CompletionRequest, Dict[str, Any]]) -> CompletionResponse:
        """Generate a text completion using the current provider."""
        if isinstance(request, dict):
            request = CompletionRequest(**request)
        
        return await self.provider.completion(request)
    
    async def chat_completion_stream(self, request: Union[ChatCompletionRequest, Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming chat completion using the current provider."""
        if isinstance(request, dict):
            request = ChatCompletionRequest(**request)
        
        async for chunk in self.provider.chat_completion_stream(request):
            yield chunk
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from the current provider."""
        return await self.provider.list_models()
    
    async def get_generation(self, generation_id: str) -> GenerationInfo:
        """Get generation information from the current provider."""
        return await self.provider.get_generation(generation_id)
    
    async def get_credits(self) -> Optional[Dict[str, Any]]:
        """Get account credits (if supported by the provider)."""
        return await self.provider.get_credits()
    
    async def structured_completion(self, request: Union[ChatCompletionRequest, Dict[str, Any]], response_model: Type) -> Any:
        """Generate structured output using the current provider."""
        if isinstance(request, dict):
            request = ChatCompletionRequest(**request)
        
        return await self.provider.structured_completion(request, response_model)
    
    # Convenience methods
    
    def create_chat_messages(self, *messages) -> List[ChatMessage]:
        """Helper method to create chat messages."""
        chat_messages = []
        for msg in messages:
            if isinstance(msg, str):
                chat_messages.append(ChatMessage(role="user", content=msg))
            elif isinstance(msg, dict):
                chat_messages.append(ChatMessage(**msg))
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")
        return chat_messages
    
    async def simple_chat(self, message: str, model: str = "gpt-3.5-turbo") -> str:
        """Simple chat interface - send a message and get a response."""
        request = ChatCompletionRequest(
            model=model,
            messages=self.create_chat_messages(message),
            max_tokens=150
        )
        
        response = await self.chat_completion(request)
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        
        return "No response generated"
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "name": self.provider.provider_name,
            "capabilities": self.provider.capabilities.model_dump(),
            "config": {
                "base_url": self.provider.config.base_url,
                "timeout": self.provider.config.timeout,
                "max_retries": self.provider.config.max_retries,
            }
        }
