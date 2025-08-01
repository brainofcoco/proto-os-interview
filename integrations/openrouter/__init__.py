"""
OpenRouter API Integration.

This module provides a comprehensive client for interacting with the OpenRouter API,
including support for completions, chat completions, streaming, structured outputs,
and account management.
"""

from .client import OpenRouterClient, client
from .models import (
    CompletionRequest, CompletionResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    StructuredOutputRequest, ChatMessage, MessageRole,
    StreamingResponse, ErrorResponse,
    ModelsResponse, ModelEndpointsResponse,
    CreditInfo, GenerationInfo
)

__version__ = "1.0.0"
__all__ = [
    "OpenRouterClient", "client",
    "CompletionRequest", "CompletionResponse",
    "ChatCompletionRequest", "ChatCompletionResponse", 
    "StructuredOutputRequest", "ChatMessage", "MessageRole",
    "StreamingResponse", "ErrorResponse",
    "ModelsResponse", "ModelEndpointsResponse",
    "CreditInfo", "GenerationInfo"
]
