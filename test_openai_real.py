#!/usr/bin/env python3
"""
Test script to verify OpenAI provider real API integration.
"""

import asyncio
import os
from integrations.client import UnifiedLLMClient

async def test_openai_real():
    """Test the OpenAI provider with real API calls."""
    
    print("🧪 Testing OpenAI Provider (Real API Integration)")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   This test requires a valid OpenAI API key")
        return
    
    try:
        # Initialize client with OpenAI provider
        client = UnifiedLLMClient(provider_name="openai")
        print("✅ OpenAI client initialized successfully")
        
        # Test 1: List available models
        print("\n📋 Testing list_models()...")
        models = await client.list_models()
        print(f"✅ Retrieved {len(models)} models from OpenAI API")
        
        # Show first few models
        for i, model in enumerate(models[:5]):
            print(f"   {i+1}. {model.id} - {model.name}")
            print(f"      Context: {model.context_length} tokens")
        
        if len(models) > 5:
            print(f"   ... and {len(models) - 5} more models")
        
        # Test 2: Chat completion with a real model
        print("\n💬 Testing chat completion...")
        
        # Use GPT-3.5-turbo as it's commonly available and cheaper
        test_model = "gpt-3.5-turbo"
        available_models = [m.id for m in models]
        
        if test_model not in available_models:
            # Fallback to the first available GPT model
            gpt_models = [m.id for m in models if m.id.startswith("gpt-")]
            if gpt_models:
                test_model = gpt_models[0]
                print(f"   Using model: {test_model}")
            else:
                print("❌ No GPT models available for testing")
                return
        
        from integrations.openrouter.models import ChatCompletionRequest, ChatMessage
        
        request = ChatCompletionRequest(
            model=test_model,
            messages=[
                ChatMessage(role="user", content="Hello! Can you respond with exactly: 'OpenAI API test successful'")
            ],
            max_tokens=50
        )
        
        response = await client.chat_completion(request)
        
        print("✅ Chat completion successful!")
        print(f"   Model used: {response.model}")
        print(f"   Response: {response.choices[0].message.content}")
        
        if response.usage:
            print(f"   Token usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total")
        
        # Test 3: Credits (should return None for OpenAI)
        print("\n💳 Testing credits retrieval...")
        credits = await client.get_credits()
        if credits is None:
            print("✅ Correctly returned None (OpenAI doesn't have credits system)")
        else:
            print(f"ℹ️  Unexpected credits response: {credits}")
        
        print("\n🎉 All OpenAI provider tests completed successfully!")
        print("   The provider is now using real OpenAI API calls.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide helpful debugging info
        if "api_key" in str(e).lower():
            print("   💡 This might be an API key issue")
        elif "rate limit" in str(e).lower():
            print("   💡 This might be a rate limiting issue")
        elif "quota" in str(e).lower():
            print("   💡 This might be a quota/billing issue")

if __name__ == "__main__":
    asyncio.run(test_openai_real())
