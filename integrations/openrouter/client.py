"""OpenRouter API Client.

This module provides a comprehensive client for interacting with the OpenRouter API.
It includes methods for completions, chat completions, model information, credits, and generation tracking.
"""

import os
import json
import logging
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
from dotenv import load_dotenv
from .models import (
    CompletionRequest, CompletionResponse, ChatCompletionRequest, 
    ChatCompletionResponse, StructuredOutputRequest, ErrorResponse,
    ModelsResponse, ModelEndpointsResponse, CreditResponse, GenerationInfo,
    StreamingResponse, ChatMessage, ModelInfo
)
# Provider interface imports removed to avoid circular imports

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter API client with comprehensive functionality.
    
    This client provides methods for all major OpenRouter API endpoints including
    completions, chat completions, streaming, model information, and account management.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 site_url: Optional[str] = None, site_name: Optional[str] = None):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, will use OPENROUTER_API_KEY env var.
            base_url: Base URL for OpenRouter API. If None, will use OPENROUTER_BASE_URL env var.
            site_url: Site URL for rankings. If None, will use OPENROUTER_SITE_URL env var.
            site_name: Site name for rankings. If None, will use OPENROUTER_SITE_NAME env var.
            
        Raises:
            ValueError: If required environment variables are not set.
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = base_url or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.site_url = site_url or os.getenv('OPENROUTER_SITE_URL')
        self.site_name = site_name or os.getenv('OPENROUTER_SITE_NAME')
        
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable")
        
        logger.info(f"OpenRouter client initialized with base URL: {self.base_url}")
    
    def _get_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get standard headers for API requests.
        
        Args:
            extra_headers: Additional headers to include.
            
        Returns:
            Dictionary of headers for the request.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add optional site information for rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        if extra_headers:
            headers.update(extra_headers)
            
        return headers
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def _make_request(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None, 
                     response_model = None, extra_headers: Optional[Dict[str, str]] = None) -> Any:
        """Make a request to the OpenRouter API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            payload: Request payload for POST requests
            response_model: Pydantic model to parse response
            extra_headers: Additional headers
            
        Returns:
            Parsed response object or ErrorResponse on failure.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers(extra_headers)
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=payload, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if response_model:
                return response_model.model_validate(response.json())
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            error_details = {"message": str(e), "type": "RequestException"}
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_details.update(error_data)
                except (ValueError, json.JSONDecodeError):
                    error_details["raw_response"] = e.response.text
                    
            return ErrorResponse(
                error=error_details,
                message=f"Request to {endpoint} failed",
                type="HTTPError",
                code=str(getattr(e.response, 'status_code', 'unknown')) if hasattr(e, 'response') else 'unknown'
            )
        except RetryError as e:
            logger.error(f"Retry limit reached: {str(e)}")
            return ErrorResponse(
                error={"message": "Retry limit reached", "detail": str(e)} ,
                message="Retry limit reached",
                type="RetryError",
                code="unknown"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return ErrorResponse(
                error={"message": str(e)},
                message="Unexpected error occurred",
                type="UnexpectedError",
                code="unknown"
            )
    
    def completion(self, request: Union[CompletionRequest, Dict[str, Any]]) -> Union[CompletionResponse, ErrorResponse]:
        """Generate a text completion.
        
        Args:
            request: Completion request object or dictionary.
            
        Returns:
            CompletionResponse object or ErrorResponse on failure.
        """
        if isinstance(request, dict):
            request = CompletionRequest(**request)
        
        payload = request.model_dump(exclude_none=True)
        logger.info(f"Making completion request with model: {request.model}")
        
        return self._make_request("POST", "completions", payload, CompletionResponse)
    
    def chat_completion(self, request: Union[ChatCompletionRequest, Dict[str, Any]]) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Generate a chat completion.
        
        Args:
            request: Chat completion request object or dictionary.
            
        Returns:
            ChatCompletionResponse object or ErrorResponse on failure.
        """
        if isinstance(request, dict):
            request = ChatCompletionRequest(**request)
        
        payload = request.model_dump(exclude_none=True)
        logger.info(f"Making chat completion request with model: {request.model}")
        
        return self._make_request("POST", "chat/completions", payload, ChatCompletionResponse)
    
    def get_generation(self, generation_id: str) -> Union[GenerationInfo, ErrorResponse]:
        """Get information about a specific generation.
        
        Args:
            generation_id: ID of the generation to retrieve.
            
        Returns:
            GenerationInfo object or ErrorResponse on failure.
        """
        logger.info(f"Retrieving generation info for ID: {generation_id}")
        return self._make_request("GET", f"generation/{generation_id}", response_model=GenerationInfo)
    
    def list_models(self) -> Union[ModelsResponse, ErrorResponse]:
        """List all available models.
        
        Returns:
            ModelsResponse object containing list of available models or ErrorResponse on failure.
        """
        logger.info("Retrieving list of available models")
        return self._make_request("GET", "models", response_model=ModelsResponse)
    
    def list_model_endpoints(self, model_id: str) -> Union[ModelEndpointsResponse, ErrorResponse]:
        """List endpoints for a specific model.
        
        Args:
            model_id: The model ID to get endpoints for.
            
        Returns:
            ModelEndpointsResponse object or ErrorResponse on failure.
        """
        logger.info(f"Retrieving endpoints for model: {model_id}")
        return self._make_request("GET", f"models/{model_id}/providers", response_model=ModelEndpointsResponse)
    
    def get_credits(self) -> Union[CreditResponse, ErrorResponse]:
        """Get current credit balance and usage information.
        
        Returns:
            CreditResponse object or ErrorResponse on failure.
        """
        logger.info("Retrieving credit information")
        return self._make_request("GET", "credits", response_model=CreditResponse)
    
    def chat_completion_stream(self, request: Union[ChatCompletionRequest, Dict[str, Any]]) -> Iterator[Union[StreamingResponse, ErrorResponse]]:
        """Generate a streaming chat completion.
        
        Args:
            request: Chat completion request object or dictionary.
            
        Yields:
            StreamingResponse objects for each chunk or ErrorResponse on failure.
        """
        if isinstance(request, dict):
            request = ChatCompletionRequest(**request)
        
        # Ensure streaming is enabled
        request.stream = True
        payload = request.model_dump(exclude_none=True)
        
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        
        logger.info(f"Making streaming chat completion request with model: {request.model}")
        
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                        
                        # Parse Server-Sent Events format
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            
                            # Check for end of stream
                            if data.strip() == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                yield StreamingResponse.model_validate(chunk_data)
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"Failed to parse streaming chunk: {e}")
                                continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming request failed: {str(e)}")
            error_details = {"message": str(e), "type": "StreamingRequestException"}
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_details.update(error_data)
                except (ValueError, json.JSONDecodeError):
                    error_details["raw_response"] = e.response.text
            
            yield ErrorResponse(
                error=error_details,
                message="Streaming chat completion failed",
                type="HTTPError",
                code=str(getattr(e.response, 'status_code', 'unknown')) if hasattr(e, 'response') else 'unknown'
            )
        except Exception as e:
            logger.error(f"Unexpected streaming error: {str(e)}")
            yield ErrorResponse(
                error={"message": str(e)},
                message="Unexpected streaming error occurred",
                type="UnexpectedError",
                code="unknown"
            )
    
    def structured_completion(self, request: Union[StructuredOutputRequest, Dict[str, Any]], 
                            validation_model: Optional[Any] = None) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Generate a chat completion with structured output validation.
        
        Args:
            request: Structured output request object or dictionary.
            validation_model: Optional Pydantic model to validate the structured output.
            
        Returns:
            ChatCompletionResponse with validated structured output or ErrorResponse on failure.
        """
        if isinstance(request, dict):
            request = StructuredOutputRequest(**request)
        
        payload = request.model_dump(exclude_none=True)
        logger.info(f"Making structured completion request with model: {request.model}")
        
        response = self._make_request("POST", "chat/completions", payload, ChatCompletionResponse)
        
        # If validation model is provided and response is successful, validate the output
        if validation_model and isinstance(response, ChatCompletionResponse):
            try:
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    
                    # Try to parse as JSON and validate with the provided model
                    try:
                        parsed_content = json.loads(content)
                        validated_output = validation_model.model_validate(parsed_content)
                        
                        # Replace the content with the validated and potentially transformed output
                        response.choices[0].message.content = validated_output.model_dump_json()
                        logger.info("Structured output validation successful")
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse structured output as JSON: {e}")
                        # Return original response if parsing fails
                        
            except Exception as e:
                logger.error(f"Structured output validation failed: {e}")
                # Return original response if validation fails
        
        return response
    
    def create_chat_messages(self, *messages: Union[str, Dict[str, str]]) -> List[ChatMessage]:
        """Helper method to create ChatMessage objects from various input formats.
        
        Args:
            *messages: Variable number of messages. Can be strings (treated as user messages)
                      or dictionaries with 'role' and 'content' keys.
        
        Returns:
            List of ChatMessage objects.
        """
        chat_messages = []
        
        for msg in messages:
            if isinstance(msg, str):
                chat_messages.append(ChatMessage(role="user", content=msg))
            elif isinstance(msg, dict):
                chat_messages.append(ChatMessage(**msg))
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")
        
        return chat_messages


# Global client instance for backward compatibility
client = OpenRouterClient()
