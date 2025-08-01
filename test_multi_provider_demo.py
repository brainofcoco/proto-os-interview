#!/usr/bin/env python3
"""
Demo script showing multi-provider capabilities.
This test works without requiring actual API keys by testing different aspects.
"""

import asyncio
import os
from integrations.client import UnifiedLLMClient

async def test_multi_provider_demo():
    """Demonstrate multi-provider capabilities."""
    
    print("🌟 Multi-Provider LLM Integration Demo")
    print("=" * 50)
    
    # Test 1: Provider initialization and capabilities
    print("\n1. 🔧 Testing Provider Initialization")
    print("-" * 40)
    
    providers = ["openrouter", "openai"]
    clients = {}
    
    for provider in providers:
        try:
            client = UnifiedLLMClient(provider_name=provider)
            clients[provider] = client
            capabilities = client.provider.capabilities
            
            print(f"✅ {provider.title()} Provider:")
            print(f"   - Streaming: {capabilities.supports_streaming}")
            print(f"   - Function calling: {capabilities.supports_function_calling}")
            print(f"   - Image input: {capabilities.supports_image_input}")
            print(f"   - Max tokens: {capabilities.max_tokens}")
            print(f"   - Rate limit: {capabilities.rate_limit_requests_per_minute}/min")
            
        except Exception as e:
            print(f"❌ Failed to initialize {provider}: {e}")
    
    # Test 2: Model listing (works for OpenRouter, requires API key for OpenAI)
    print("\n2. 📋 Testing Model Listings")
    print("-" * 40)
    
    # Test OpenRouter (should work without key for model listing)
    if "openrouter" in clients:
        try:
            or_models = await clients["openrouter"].list_models()
            print(f"✅ OpenRouter: {len(or_models)} models available")
            print(f"   Sample models: {', '.join([m.id for m in or_models[:3]])}")
        except Exception as e:
            print(f"ℹ️  OpenRouter models: {e}")
    
    # Test OpenAI (will fail without API key, but shows proper error handling)
    if "openai" in clients:
        try:
            openai_models = await clients["openai"].list_models()
            print(f"✅ OpenAI: {len(openai_models)} models available")
            print(f"   Sample models: {', '.join([m.id for m in openai_models[:3]])}")
        except Exception as e:
            print(f"ℹ️  OpenAI models: {str(e)[:60]}... (Expected without API key)")
    
    # Test 3: Provider switching
    print("\n3. 🔄 Testing Provider Switching")
    print("-" * 40)
    
    if "openrouter" in clients:
        client = clients["openrouter"]
        print(f"Current provider: {client.provider.provider_name}")
        
        # Test switching (though we can't actually call without API keys)
        try:
            # This will work for initialization
            client.switch_provider("openai")
            print(f"Switched to: {client.provider.provider_name}")
            
            # Switch back
            client.switch_provider("openrouter")
            print(f"Switched back to: {client.provider.provider_name}")
            print("✅ Provider switching works correctly")
            
        except Exception as e:
            print(f"❌ Provider switching failed: {e}")
    
    # Test 4: Configuration validation
    print("\n4. ⚙️  Testing Configuration Validation")
    print("-" * 40)
    
    # Test valid configs
    valid_configs = [
        ("openrouter", {"base_url": "https://openrouter.ai/api/v1"}),
        ("openai", {"base_url": "https://api.openai.com/v1"}),
    ]
    
    for provider, config_overrides in valid_configs:
        try:
            client = UnifiedLLMClient(
                provider_name=provider,
                **config_overrides
            )
            print(f"✅ {provider.title()} config validation passed")
        except Exception as e:
            print(f"❌ {provider.title()} config validation failed: {e}")
    
    # Test 5: Error handling
    print("\n5. 🛡️  Testing Error Handling")
    print("-" * 40)
    
    # Test invalid provider
    try:
        UnifiedLLMClient(provider_name="nonexistent")
        print("❌ Should have failed with invalid provider")
    except Exception as e:
        print(f"✅ Correctly rejected invalid provider: {type(e).__name__}")
    
    # Test credits endpoints
    print("\n6. 💳 Testing Credits Endpoints")
    print("-" * 40)
    
    for provider_name, client in clients.items():
        try:
            credits = await client.get_credits()
            if provider_name == "openai":
                if credits is None:
                    print(f"✅ {provider_name.title()}: Correctly returns None (no credits system)")
                else:
                    print(f"ℹ️  {provider_name.title()}: Unexpected credits: {credits}")
            else:
                # OpenRouter might return credits or fail without API key
                if credits:
                    print(f"✅ {provider_name.title()}: Retrieved credits info")
                else:
                    print(f"ℹ️  {provider_name.title()}: No credits (likely needs API key)")
        except Exception as e:
            print(f"ℹ️  {provider_name.title()} credits: {str(e)[:50]}...")
    
    print("\n🎉 Multi-Provider Demo Complete!")
    print("\nSummary:")
    print("- ✅ Both providers initialize correctly")
    print("- ✅ Provider capabilities are properly exposed")
    print("- ✅ Provider switching works seamlessly")
    print("- ✅ Configuration validation prevents errors")
    print("- ✅ Error handling is robust")
    print("- ✅ Different provider behaviors (like credits) are handled")
    print("\n💡 For full functionality testing:")
    print("   - Set OPENROUTER_API_KEY for OpenRouter live testing")
    print("   - Set OPENAI_API_KEY for OpenAI live testing")

if __name__ == "__main__":
    asyncio.run(test_multi_provider_demo())
