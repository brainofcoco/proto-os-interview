"""
Configuration management for multi-provider LLM integration.

This module handles configuration loading, validation, and provider-specific settings.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from .base import ProviderConfig

# Load environment variables
load_dotenv()


class MultiProviderConfig(BaseModel):
    """Configuration for multiple LLM providers."""
    
    # OpenRouter configuration
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", description="OpenRouter base URL")
    openrouter_site_url: Optional[str] = Field(None, description="Site URL for OpenRouter rankings")
    openrouter_site_name: Optional[str] = Field(None, description="Site name for OpenRouter rankings")
    
    # OpenAI configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_base_url: str = Field("https://api.openai.com/v1", description="OpenAI base URL")
    openai_organization: Optional[str] = Field(None, description="OpenAI organization ID")
    
    # Anthropic/Claude configuration
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    anthropic_base_url: str = Field("https://api.anthropic.com", description="Anthropic base URL")
    
    # Global settings
    default_provider: str = Field("openrouter", description="Default provider to use")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, description="Base retry delay in seconds")
    
    @classmethod
    def from_env(cls) -> 'MultiProviderConfig':
        """Load configuration from environment variables."""
        return cls(
            # OpenRouter
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            openrouter_base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
            openrouter_site_url=os.getenv('OPENROUTER_SITE_URL'),
            openrouter_site_name=os.getenv('OPENROUTER_SITE_NAME'),
            
            # OpenAI
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            openai_organization=os.getenv('OPENAI_ORGANIZATION'),
            
            # Anthropic
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            anthropic_base_url=os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
            
            # Global
            default_provider=os.getenv('DEFAULT_LLM_PROVIDER', 'openrouter'),
            timeout=int(os.getenv('LLM_TIMEOUT', '30')),
            max_retries=int(os.getenv('LLM_MAX_RETRIES', '3')),
            retry_delay=float(os.getenv('LLM_RETRY_DELAY', '1.0')),
        )
    
    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        if provider_name == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
            
            return ProviderConfig(
                api_key=self.openrouter_api_key,
                base_url=self.openrouter_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                site_url=self.openrouter_site_url,
                site_name=self.openrouter_site_name
            )
        
        elif provider_name == "openai":
            # Use mock key for demo if real key not provided
            api_key = self.openai_api_key or "mock-openai-key-for-demo"
            
            return ProviderConfig(
                api_key=api_key,
                base_url=self.openai_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay
            )
        
        elif provider_name == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
            
            return ProviderConfig(
                api_key=self.anthropic_api_key,
                base_url=self.anthropic_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get list of providers and whether they're configured."""
        return {
            "openrouter": bool(self.openrouter_api_key),
            "openai": True,  # Always available (mock or real)
            "anthropic": bool(self.anthropic_api_key)
        }
    
    def validate_provider(self, provider_name: str) -> bool:
        """Check if a provider is properly configured."""
        try:
            self.get_provider_config(provider_name)
            return True
        except ValueError:
            return False


# Global configuration instance
config = MultiProviderConfig.from_env()
