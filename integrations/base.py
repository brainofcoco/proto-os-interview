"""
Abstract base classes for LLM provider integration.

This module defines the interfaces that all LLM providers must implement
to ensure consistent behavior across different AI services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncIterator
from pydantic import BaseModel
import logging

# Import these at runtime to avoid circular imports


class ProviderConfig(BaseModel):
    """Base configuration for LLM providers."""
    api_key: str
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    site_url: Optional[str] = None
    site_name: Optional[str] = None


class ProviderCapabilities(BaseModel):
    """Defines what features a provider supports."""
    supports_streaming: bool
    supports_structured_output: bool
    supports_function_calling: bool
    supports_image_input: bool
    supports_audio_input: bool
    max_tokens: Optional[int] = None
    rate_limit_requests_per_minute: Optional[int] = None


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        pass
    
    @abstractmethod
    async def chat_completion(
        self, 
        request: Any  # ChatCompletionRequest
    ) -> Any:  # ChatCompletionResponse
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    async def completion(
        self, 
        request: Any  # CompletionRequest
    ) -> Any:  # CompletionResponse
        """Generate a text completion."""
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self, 
        request: Any  # ChatCompletionRequest
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming chat completion."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Any]:  # List[ModelInfo]
        """List available models."""
        pass
    
    @abstractmethod
    async def get_generation(self, generation_id: str) -> Any:  # GenerationInfo
        """Get information about a specific generation."""
        pass
    
    # Optional methods that providers can override
    async def get_credits(self) -> Optional[Dict[str, Any]]:
        """Get account credit information. Not all providers support this."""
        return None
    
    async def structured_completion(
        self, 
        request: Any,  # ChatCompletionRequest
        response_model: type
    ) -> Any:
        """Generate structured output. Default implementation uses regular completion."""
        response = await self.chat_completion(request)
        # This is a basic implementation - providers should override for better support
        try:
            import json
            content = response.choices[0].message.content
            return response_model.model_validate(json.loads(content))
        except Exception as e:
            self.logger.warning(f"Failed to parse structured output: {e}")
            raise ValueError(f"Could not parse response as {response_model.__name__}")


class ProviderRegistry:
    """Registry for managing multiple LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, type] = {}
        self._instances: Dict[str, LLMProvider] = {}
    
    def register(self, provider_class: type) -> None:
        """Register a provider class."""
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(f"{provider_class} must inherit from LLMProvider")
        
        # Use the provider_name from an instance to get the name
        temp_config = ProviderConfig(api_key="temp", base_url="temp")
        temp_instance = provider_class(temp_config)
        provider_name = temp_instance.provider_name
        
        self._providers[provider_name] = provider_class
    
    def create_provider(self, provider_name: str, config: ProviderConfig) -> LLMProvider:
        """Create a provider instance."""
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = self._providers[provider_name]
        instance = provider_class(config)
        self._instances[provider_name] = instance
        return instance
    
    def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Get an existing provider instance."""
        return self._instances.get(provider_name)
    
    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())


# Global registry instance
provider_registry = ProviderRegistry()
