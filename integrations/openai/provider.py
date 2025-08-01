"""
Real OpenAI Provider Implementation.

This provider connects to the actual OpenAI API, demonstrating true multi-provider capability.
"""

import asyncio
import json
import requests
from typing import List, Dict, Any, AsyncIterator, Optional
from tenacity import retry, wait_exponential, stop_after_attempt
from ..base import LLMProvider, ProviderCapabilities, ProviderConfig
from ..openrouter.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    GenerationInfo,
    ChatMessage,
    ChatCompletionChoice,
    CompletionChoice,
    Usage,
    ErrorResponse
)


class OpenAIProvider(LLMProvider):
    """Real OpenAI implementation of the LLMProvider interface."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url
        self.api_key = config.api_key
        
    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return "openai"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_structured_output=True,
            supports_function_calling=True,
            supports_image_input=True,
            supports_audio_input=True,
            max_tokens=4096,
            rate_limit_requests_per_minute=3500  # OpenAI's actual limit
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def _make_request(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """Make a request to the OpenAI API with retry logic."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        
        try:
            self.logger.debug(f"Making {method} request to OpenAI: {url}")
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=payload, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI API request failed: {str(e)}")
            raise ValueError(f"OpenAI API request failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI request: {str(e)}")
            raise ValueError(f"Unexpected error: {str(e)}")
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a chat completion using OpenAI API."""
        self.logger.info(f"OpenAI chat completion with model: {request.model}")
        
        # Convert our request format to OpenAI format
        payload = {
            "model": request.model,
            "messages": [{
                "role": msg.role,
                "content": msg.content
            } for msg in request.messages],
        }
        
        # Add optional parameters
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.stop:
            payload["stop"] = request.stop
        
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            None, self._make_request, "POST", "chat/completions", payload
        )
        
        # Convert OpenAI response to our format
        return ChatCompletionResponse(
            id=response_data["id"],
            object=response_data["object"],
            created=response_data["created"],
            model=response_data["model"],
            choices=[
                ChatCompletionChoice(
                    index=choice["index"],
                    message=ChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"]["content"]
                    ),
                    finish_reason=choice.get("finish_reason")
                ) for choice in response_data["choices"]
            ],
            usage=Usage(
                prompt_tokens=response_data["usage"]["prompt_tokens"],
                completion_tokens=response_data["usage"]["completion_tokens"],
                total_tokens=response_data["usage"]["total_tokens"]
            ) if "usage" in response_data else None
        )
    
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion (mock implementation)."""
        self.logger.info(f"Mock OpenAI completion with model: {request.model}")
        
        # Mock response
        return CompletionResponse(
            id="cmpl-mock-123",
            object="text_completion",
            created=1234567890,
            model=request.model,
            choices=[
                {
                    "text": "This is a mock completion from OpenAI provider.",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=10,
                total_tokens=15
            )
        )
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming chat completion (mock implementation)."""
        self.logger.info(f"Mock OpenAI streaming with model: {request.model}")
        
        # Mock streaming response
        chunks = [
            {"id": "chatcmpl-mock-stream", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "This "}}]},
            {"id": "chatcmpl-mock-stream", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "is "}}]},
            {"id": "chatcmpl-mock-stream", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "a mock stream."}}]},
        ]
        
        for chunk in chunks:
            await asyncio.sleep(0.1)  # Simulate streaming delay
            yield chunk
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from OpenAI API."""
        self.logger.info("Fetching OpenAI models")
        
        try:
            loop = asyncio.get_event_loop()
            response_data = await loop.run_in_executor(
                None, self._make_request, "GET", "models", None
            )
            
            models = []
            for model_data in response_data.get("data", []):
                # Only include chat/completion models
                if any(model_data["id"].startswith(prefix) for prefix in ["gpt-", "text-"]):
                    models.append(ModelInfo(
                        id=model_data["id"],
                        name=model_data["id"].replace("-", " ").title(),
                        description=f"OpenAI model: {model_data['id']}",
                        pricing=None,  # OpenAI doesn't provide pricing in API
                        context_length=self._get_context_length(model_data["id"]),
                        architecture={"provider": "openai", "created": model_data.get("created")}
                    ))
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OpenAI models: {e}")
            # Return fallback models if API fails
            return [
                ModelInfo(
                    id="gpt-4",
                    name="GPT-4",
                    description="Large language model by OpenAI",
                    pricing=None,
                    context_length=8192,
                    architecture={"provider": "openai"}
                ),
                ModelInfo(
                    id="gpt-3.5-turbo",
                    name="GPT-3.5 Turbo",
                    description="Fast, affordable language model by OpenAI",
                    pricing=None,
                    context_length=4096,
                    architecture={"provider": "openai"}
                )
            ]
    
    def _get_context_length(self, model_id: str) -> int:
        """Get context length for a model ID."""
        context_lengths = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
        }
        return context_lengths.get(model_id, 4096)  # Default to 4k
    
    async def get_generation(self, generation_id: str) -> GenerationInfo:
        """Get information about a specific generation (mock implementation)."""
        self.logger.info(f"Mock OpenAI get generation: {generation_id}")
        
        return GenerationInfo(
            id=generation_id,
            model="gpt-4",
            streamed=False,
            created_at="2024-01-01T12:00:00Z"
        )
    
    async def get_credits(self) -> Optional[Dict[str, Any]]:
        """OpenAI doesn't have a credits system like OpenRouter.
        
        OpenAI uses a billing/subscription model rather than credits.
        Regular API keys don't have access to billing information.
        """
        return None
