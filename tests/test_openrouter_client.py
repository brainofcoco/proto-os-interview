"""
Unit tests for the OpenRouterClient.

This test suite uses pytest to verify the functionality of the OpenRouterClient.
The tests are designed to ensure that the client interfaces correctly with the OpenRouter API,
handles errors gracefully, and provides the expected outputs.
"""

import pytest
from unittest.mock import patch
from integrations.openrouter.client import OpenRouterClient
from integrations.openrouter.models import CompletionRequest, ChatCompletionRequest

@pytest.fixture
def client():
    """Fixture for OpenRouterClient instance."""
    return OpenRouterClient(api_key="dummy_api_key", base_url="https://dummyapi.test/v1")

@patch('requests.post')
def test_completion(mock_post, client):
    """Test the completion method."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "id": "test123",
        "object": "completion",
        "created": 1620479960,
        "model": "text-davinci-003",
        "choices": [{
            "text": "Hello, World!",
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8
        }
    }

    request = CompletionRequest(model="text-davinci-003", prompt="Say Hello")
    response = client.completion(request)

    assert response.id == "test123"
    assert response.choices[0].text == "Hello, World!"

@patch('requests.get')
def test_list_models(mock_get, client):
    """Test the list_models method."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "Fast and versatile AI model."
            }
        ]
    }

    response = client.list_models()

    assert len(response.data) == 1
    assert response.data[0].id == "gpt-3.5-turbo"

@patch('requests.post')
def test_chat_completion(mock_post, client):
    """Test the chat_completion method."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "id": "chat123",
        "object": "chat.completion",
        "created": 1620479960,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }

    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    response = client.chat_completion(request)

    assert response.id == "chat123"
    assert response.choices[0].message.content == "Hello! How can I help you today?"

@patch('requests.get')
def test_get_credits(mock_get, client):
    """Test the get_credits method."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "data": {
            "total_credits": 10.50,
            "total_usage": 5.25
        }
    }

    response = client.get_credits()

    assert response.data.total_credits == 10.50
    assert response.data.total_usage == 5.25

@patch('requests.get')
def test_get_generation(mock_get, client):
    """Test the get_generation method."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "id": "gen123",
        "model": "gpt-3.5-turbo",
        "streamed": False,
        "generation_time": 1.5,
        "tokens_prompt": 10,
        "tokens_completion": 15,
        "total_cost": 0.002
    }

    response = client.get_generation("gen123")

    assert response.id == "gen123"
    assert response.model == "gpt-3.5-turbo"
    assert response.total_cost == 0.002

@patch('requests.get')
def test_list_model_endpoints(mock_get, client):
    """Test the list_model_endpoints method."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "object": "list",
        "data": [
            {
                "provider": "openai",
                "endpoint": "https://api.openai.com/v1",
                "status": "active"
            }
        ]
    }

    response = client.list_model_endpoints("gpt-3.5-turbo")

    assert len(response.data) == 1
    assert response.data[0].provider == "openai"
    assert response.data[0].status == "active"

@patch('requests.post')
def test_error_handling(mock_post, client):
    """Test error handling for failed requests."""
    mock_post.side_effect = Exception("Connection error")

    request = CompletionRequest(model="test-model", prompt="Test")
    response = client.completion(request)

    # Should return ErrorResponse on failure
    assert hasattr(response, 'error')
    assert response.type == "UnexpectedError"

def test_create_chat_messages(client):
    """Test the create_chat_messages helper method."""
    messages = client.create_chat_messages(
        "Hello",
        {"role": "assistant", "content": "Hi there!"},
        "How are you?"
    )

    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there!"
    assert messages[2].role == "user"
    assert messages[2].content == "How are you?"

def test_client_initialization():
    """Test client initialization with different parameters."""
    # Test with all parameters
    client = OpenRouterClient(
        api_key="test_key",
        base_url="https://test.api/v1",
        site_url="https://test.site",
        site_name="Test Site"
    )
    
    assert client.api_key == "test_key"
    assert client.base_url == "https://test.api/v1"
    assert client.site_url == "https://test.site"
    assert client.site_name == "Test Site"

    # Test missing API key should raise ValueError
    # We need to clear the environment variable first
    import os
    original_key = os.environ.get('OPENROUTER_API_KEY')
    if 'OPENROUTER_API_KEY' in os.environ:
        del os.environ['OPENROUTER_API_KEY']
    
    try:
        with pytest.raises(ValueError, match="OpenRouter API key must be provided"):
            OpenRouterClient(api_key=None)
    finally:
        # Restore the original key
        if original_key:
            os.environ['OPENROUTER_API_KEY'] = original_key

@patch('requests.post')
def test_streaming_response_parsing(mock_post, client):
    """Test streaming response parsing (mocked)."""
    # This is a simplified test as actual streaming requires more complex mocking
    mock_response = mock_post.return_value.__enter__.return_value  # For context manager
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.iter_lines.return_value = [
        b'data: {"id":"stream123","object":"chat.completion.chunk","created":1620479960,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"}}]}',
        b'data: [DONE]'
    ]
    
    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Get first chunk from the stream
    stream = client.chat_completion_stream(request)
    first_chunk = next(stream)
    
    assert first_chunk.id == "stream123"
    assert first_chunk.object == "chat.completion.chunk"

@patch('requests.post')
def test_structured_completion(mock_post, client):
    """Test the structured_completion method."""
    from integrations.openrouter.models import StructuredOutputRequest
    from pydantic import BaseModel
    
    # Define a test validation model
    class TestResponse(BaseModel):
        name: str
        age: int
    
    # Mock the API response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "id": "struct123",
        "object": "chat.completion",
        "created": 1620479960,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"name": "John", "age": 30}'
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25
        }
    }
    
    request = StructuredOutputRequest(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Generate a person"}],
        response_format={"type": "json_object"}
    )
    
    response = client.structured_completion(request, validation_model=TestResponse)
    
    assert response.id == "struct123"
    assert response.choices[0].message.role == "assistant"
    # The content should be validated and reformatted
    import json
    content_data = json.loads(response.choices[0].message.content)
    assert content_data["name"] == "John"
    assert content_data["age"] == 30

@patch('requests.post')
def test_http_error_handling(mock_post, client):
    """Test HTTP error handling with status codes."""
    import requests
    
    # Mock HTTP 429 (Rate Limit) error
    mock_response = mock_post.return_value
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Rate Limit")
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
    
    request = CompletionRequest(model="test-model", prompt="Test")
    response = client.completion(request)
    
    assert hasattr(response, 'error')
    assert response.type == "HTTPError"
    assert "429 Rate Limit" in str(response.error)  # The actual error message from the exception

@patch('requests.post')
def test_json_decode_error_handling(mock_post, client):
    """Test handling of malformed JSON responses."""
    import requests
    
    # Mock response with invalid JSON
    mock_response = mock_post.return_value
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
    mock_response.status_code = 400
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.text = "Invalid JSON response"
    
    request = CompletionRequest(model="test-model", prompt="Test")
    response = client.completion(request)
    
    assert hasattr(response, 'error')
    assert response.type == "HTTPError"
    assert "400 Bad Request" in str(response.error)  # Check for the actual exception message

@patch('requests.post')
def test_retry_error_handling(mock_post, client):
    """Test retry logic when all attempts fail."""
    import requests
    
    # Mock consistent failures that trigger retry exhaustion
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
    
    request = CompletionRequest(model="test-model", prompt="Test")
    response = client.completion(request)
    
    assert hasattr(response, 'error')
    # After retries are exhausted, it becomes an HTTPError in our implementation
    assert response.type in ["HTTPError", "RetryError"]
    assert "Connection failed" in str(response.error)

def test_create_chat_messages_invalid_input(client):
    """Test create_chat_messages with invalid input types."""
    with pytest.raises(ValueError, match="Unsupported message type"):
        client.create_chat_messages(123)  # Invalid type

def test_structured_completion_invalid_json(client):
    """Test structured completion with invalid JSON response."""
    from integrations.openrouter.models import StructuredOutputRequest
    from pydantic import BaseModel
    
    class TestModel(BaseModel):
        name: str
    
    with patch.object(client, '_make_request') as mock_request:
        from integrations.openrouter.models import ChatCompletionResponse, ChatCompletionChoice, ChatMessage
        
        # Mock response with invalid JSON content
        mock_response = ChatCompletionResponse(
            id="test123",
            object="chat.completion",
            created=1620479960,
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="invalid json content"),
                    finish_reason="stop"
                )
            ]
        )
        mock_request.return_value = mock_response
        
        request = StructuredOutputRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"}
        )
        
        # Should handle invalid JSON gracefully
        response = client.structured_completion(request, validation_model=TestModel)
        assert response.choices[0].message.content == "invalid json content"  # Original content preserved

@patch('requests.post')
def test_streaming_connection_error(mock_post, client):
    """Test streaming error handling."""
    import requests
    
    # Mock connection error during streaming
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
    
    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Get error from stream
    stream = client.chat_completion_stream(request)
    error_response = next(stream)
    
    assert hasattr(error_response, 'error')
    assert error_response.type == "HTTPError"
    assert "Connection failed" in str(error_response.error)

@patch('requests.post')
def test_streaming_invalid_chunk(mock_post, client):
    """Test streaming with invalid JSON chunks."""
    # Mock streaming response with invalid JSON chunk
    mock_response = mock_post.return_value.__enter__.return_value
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.iter_lines.return_value = [
        b'data: invalid json chunk',  # This should be skipped
        b'data: {"id":"stream123","object":"chat.completion.chunk","created":1620479960,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"}}]}',
        b'data: [DONE]'
    ]
    
    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Should skip invalid chunk and return valid one
    stream = client.chat_completion_stream(request)
    first_chunk = next(stream)
    
    assert first_chunk.id == "stream123"
    assert first_chunk.object == "chat.completion.chunk"

@patch('requests.post')
def test_dict_input_conversion(mock_post, client):
    """Test that dict inputs are properly converted to Pydantic models."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "id": "test123",
        "object": "completion",
        "created": 1620479960,
        "model": "text-davinci-003",
        "choices": [{"text": "Hello", "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    }
    
    # Test with dict input instead of Pydantic model
    request_dict = {
        "model": "text-davinci-003",
        "prompt": "Say Hello",
        "max_tokens": 50
    }
    
    response = client.completion(request_dict)
    assert response.id == "test123"
    assert response.choices[0].text == "Hello"

def test_headers_with_site_info():
    """Test header generation with site information."""
    client = OpenRouterClient(
        api_key="test_key",
        site_url="https://myapp.com",
        site_name="My App"
    )
    
    headers = client._get_headers()
    assert headers["HTTP-Referer"] == "https://myapp.com"
    assert headers["X-Title"] == "My App"
    assert headers["Authorization"] == "Bearer test_key"

def test_headers_with_extra_headers():
    """Test header generation with additional headers."""
    client = OpenRouterClient(api_key="test_key")
    
    extra_headers = {"Custom-Header": "custom-value"}
    headers = client._get_headers(extra_headers)
    
    assert headers["Custom-Header"] == "custom-value"
    assert headers["Authorization"] == "Bearer test_key"
