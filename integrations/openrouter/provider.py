"""
OpenRouter Provider Implementation.

This module implements the LLMProvider interface for OpenRouter API integration.
"""

import asyncio
from typing import List, Dict, Any, AsyncIterator, Optional, Union
from ..base import LLMProvider, ProviderCapabilities, ProviderConfig
from .client import OpenRouterClient
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    GenerationInfo,
    ErrorResponse
)


class OpenRouterProvider(LLMProvider):
    """OpenRouter implementation of the LLMProvider interface."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = OpenRouterClient(
            api_key=config.api_key,
            base_url=config.base_url,
            site_url=config.site_url,
            site_name=config.site_name
        )
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return "openrouter"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_structured_output=True,
            supports_function_calling=True,
            supports_image_input=True,
            supports_audio_input=False,
            max_tokens=None,  # Varies by model
            rate_limit_requests_per_minute=None  # Varies by plan
        )
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a chat completion."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.client.chat_completion, request)
        
        if isinstance(response, ErrorResponse):
            raise ValueError(f"Chat completion failed: {response.message}")
        
        return response
    
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.client.completion, request)
        
        if isinstance(response, ErrorResponse):
            raise ValueError(f"Completion failed: {response.message}")
        
        return response
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming chat completion."""
        loop = asyncio.get_event_loop()
        
        def _stream():
            return self.client.chat_completion_stream(request)
        
        stream = await loop.run_in_executor(None, _stream)
        
        for chunk in stream:
            if isinstance(chunk, ErrorResponse):
                raise ValueError(f"Streaming failed: {chunk.message}")
            
            yield chunk.model_dump()
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.client.list_models)
        
        if isinstance(response, ErrorResponse):
            raise ValueError(f"Failed to list models: {response.message}")
        
        # Convert ModelsResponse to List[ModelInfo]
        return response.data if hasattr(response, 'data') else []
    
    async def get_generation(self, generation_id: str) -> GenerationInfo:
        """Get information about a specific generation."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.client.get_generation, generation_id)
        
        if isinstance(response, ErrorResponse):
            raise ValueError(f"Failed to get generation: {response.message}")
        
        return response
    
    async def get_credits(self) -> Optional[Dict[str, Any]]:
        """Get account credit information."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.client.get_credits)
        
        if isinstance(response, ErrorResponse):
            self.logger.warning(f"Failed to get credits: {response.message}")
            return None
        
        # Return the data part of the credit response
        return response.data.model_dump() if response and hasattr(response, 'data') else None
    
    async def structured_completion(self, request: ChatCompletionRequest, response_model: type) -> Any:
        """Generate structured output with validation."""
        loop = asyncio.get_event_loop()
        
        # Convert to StructuredOutputRequest if needed
        from .models import StructuredOutputRequest
        structured_request = StructuredOutputRequest(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            user=request.user,
            response_format={"type": "json_object"}  # Enable JSON mode
        )
        
        response = await loop.run_in_executor(
            None, 
            self.client.structured_completion, 
            structured_request, 
            response_model
        )
        
        if isinstance(response, ErrorResponse):
            raise ValueError(f"Structured completion failed: {response.message}")
        
        # Parse the validated JSON content
        if response.choices and response.choices[0].message.content:
            import json
            try:
                return response_model.model_validate(json.loads(response.choices[0].message.content))
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse structured output: {e}")
                raise ValueError(f"Could not parse response as {response_model.__name__}")
        
        raise ValueError("No content in structured completion response")
