#!/usr/bin/env python3
"""Quick script to list all available Gemini models and their capabilities."""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storyforge.model_discovery import list_gemini_models

print("=" * 80)
print("Available Gemini Models")
print("=" * 80)

try:
    models = list_gemini_models()
    
    if not models:
        print("No models found. This could mean:")
        print("1. GEMINI_API_KEY is not set")
        print("2. API connection failed")
        print("3. No models are available")
        sys.exit(1)
    
    # Separate by type
    image_models = []
    text_models = []
    other_models = []
    
    for model in models:
        name = model.get("name", "")
        display_name = model.get("display_name", "")
        methods = model.get("supported_generation_methods", [])
        
        if "image" in name.lower() or "imagen" in name.lower():
            image_models.append(model)
        elif "generateContent" in methods:
            text_models.append(model)
        else:
            other_models.append(model)
    
    # Print image models
    if image_models:
        print("\n📸 IMAGE GENERATION MODELS:")
        print("-" * 80)
        for model in image_models:
            name = model.get("name", "").replace("models/", "")
            display_name = model.get("display_name", "")
            is_preview = "⚠️  PREVIEW" if "preview" in name.lower() else "✅ STABLE"
            print(f"  {is_preview:12} {name:50} {display_name}")
    else:
        print("\n⚠️  NO IMAGE GENERATION MODELS FOUND")
        print("This might explain why image generation is failing.")
    
    # Print text models
    if text_models:
        print("\n📝 TEXT GENERATION MODELS:")
        print("-" * 80)
        for model in text_models[:10]:  # Limit to first 10
            name = model.get("name", "").replace("models/", "")
            display_name = model.get("display_name", "")
            is_preview = "⚠️  PREVIEW" if "preview" in name.lower() else "✅ STABLE"
            print(f"  {is_preview:12} {name:50}")
        if len(text_models) > 10:
            print(f"  ... and {len(text_models) - 10} more text models")
    
    print("\n" + "=" * 80)
    print(f"Total models found: {len(models)}")
    print(f"  - Image models: {len(image_models)}")
    print(f"  - Text models: {len(text_models)}")
    print(f"  - Other models: {len(other_models)}")
    print("=" * 80)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    stable_image_models = [m for m in image_models if "preview" not in m.get("name", "").lower()]
    if not stable_image_models:
        print("  ⚠️  No stable image generation models available!")
        print("  → Use OpenAI (DALL-E 3) for image generation instead")
        print("  → Set: export OPENAI_API_KEY='your-key'")
        print("  → Run: storyforge --backend openai")
    else:
        print(f"  ✅ Use stable image model: {stable_image_models[0].get('name', '').replace('models/', '')}")

except Exception as e:
    print(f"Error listing models: {e}")
    print("\nMake sure GEMINI_API_KEY is set:")
    print("  export GEMINI_API_KEY='your-api-key'")
    sys.exit(1)
