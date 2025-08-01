"""
Pydantic models for OpenRouter API request and response objects.

This module defines all the data structures used for interacting with the OpenRouter API,
including completion requests, chat completion requests, streaming responses, and structured outputs.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class MessageRole(str, Enum):
    """Enumeration for message roles in chat completion."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    
    model_config = ConfigDict(extra="allow")
    
    role: MessageRole = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the message sender")


class CompletionRequest(BaseModel):
    """Request model for the completion endpoint."""
    
    model_config = ConfigDict(extra="allow")
    
    model: str = Field(..., description="The model to use for completion")
    prompt: str = Field(..., description="The prompt to complete")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")


class ChatCompletionRequest(BaseModel):
    """Request model for the chat completion endpoint."""
    
    model_config = ConfigDict(extra="allow")
    
    model: str = Field(..., description="The model to use for chat completion")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Response format for structured outputs")


class StructuredOutputRequest(ChatCompletionRequest):
    """Request model for structured outputs using Pydantic validation."""
    
    response_format: Dict[str, Any] = Field(..., description="JSON schema for structured output")


class CompletionChoice(BaseModel):
    """Represents a single completion choice."""
    
    model_config = ConfigDict(extra="allow")
    
    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="The index of this choice")
    finish_reason: Optional[str] = Field(None, description="Reason why generation finished")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")


class ChatCompletionChoice(BaseModel):
    """Represents a single chat completion choice."""
    
    model_config = ConfigDict(extra="allow")
    
    index: int = Field(..., description="The index of this choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(None, description="Reason why generation finished")


class Usage(BaseModel):
    """Token usage information."""
    
    model_config = ConfigDict(extra="allow")
    
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class CompletionResponse(BaseModel):
    """Response model for the completion endpoint."""
    
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(..., description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="The model used for completion")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class ChatCompletionResponse(BaseModel):
    """Response model for the chat completion endpoint."""
    
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Unique identifier for the chat completion")
    object: str = Field(..., description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="The model used for chat completion")
    choices: List[ChatCompletionChoice] = Field(..., description="List of chat completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class StreamingDelta(BaseModel):
    """Represents a delta in streaming response."""
    
    model_config = ConfigDict(extra="allow")
    
    content: Optional[str] = Field(None, description="Incremental content")
    role: Optional[MessageRole] = Field(None, description="Message role (only in first chunk)")


class StreamingChoice(BaseModel):
    """Represents a streaming choice."""
    
    model_config = ConfigDict(extra="allow")
    
    index: int = Field(..., description="The index of this choice")
    delta: StreamingDelta = Field(..., description="The delta content")
    finish_reason: Optional[str] = Field(None, description="Reason why generation finished")


class StreamingResponse(BaseModel):
    """Response model for streaming completions."""
    
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Unique identifier for the streaming response")
    object: str = Field(..., description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="The model used")
    choices: List[StreamingChoice] = Field(..., description="List of streaming choices")


class ModelInfo(BaseModel):
    """Information about an available model."""
    
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    pricing: Optional[Dict[str, Any]] = Field(None, description="Pricing information")
    context_length: Optional[int] = Field(None, description="Maximum context length")
    architecture: Optional[Dict[str, Any]] = Field(None, description="Model architecture details")
    top_provider: Optional[Dict[str, Any]] = Field(None, description="Top provider information")


class ModelsResponse(BaseModel):
    """Response model for listing available models."""
    
    model_config = ConfigDict(extra="allow")
    
    object: str = Field(..., description="Object type")
    data: List[ModelInfo] = Field(..., description="List of available models")


class ModelEndpoint(BaseModel):
    """Information about a model endpoint."""
    
    model_config = ConfigDict(extra="allow")
    
    provider: str = Field(..., description="Provider name")
    endpoint: str = Field(..., description="Endpoint URL")
    pricing: Optional[Dict[str, Any]] = Field(None, description="Pricing information")
    status: Optional[str] = Field(None, description="Endpoint status")


class ModelEndpointsResponse(BaseModel):
    """Response model for listing model endpoints."""
    
    model_config = ConfigDict(extra="allow")
    
    object: str = Field(..., description="Object type")
    data: List[ModelEndpoint] = Field(..., description="List of model endpoints")


class CreditInfo(BaseModel):
    """Information about account credits."""
    
    model_config = ConfigDict(extra="allow")
    
    balance: float = Field(..., description="Current credit balance")
    usage: Optional[float] = Field(None, description="Total usage")
    limit: Optional[float] = Field(None, description="Credit limit")


class GenerationInfo(BaseModel):
    """Information about a specific generation."""
    
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Generation ID")
    model: str = Field(..., description="Model used")
    streamed: bool = Field(..., description="Whether generation was streamed")
    generation_time: Optional[float] = Field(None, description="Time taken for generation")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    tokens_prompt: Optional[int] = Field(None, description="Prompt tokens")
    tokens_completion: Optional[int] = Field(None, description="Completion tokens")
    native_tokens_prompt: Optional[int] = Field(None, description="Native prompt tokens")
    native_tokens_completion: Optional[int] = Field(None, description="Native completion tokens")
    num_media: Optional[int] = Field(None, description="Number of media items")
    origin: Optional[str] = Field(None, description="Request origin")
    total_cost: Optional[float] = Field(None, description="Total cost")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    model_config = ConfigDict(extra="allow")
    
    error: Dict[str, Any] = Field(..., description="Error details")
    message: Optional[str] = Field(None, description="Error message")
    type: Optional[str] = Field(None, description="Error type")
    code: Optional[str] = Field(None, description="Error code")
