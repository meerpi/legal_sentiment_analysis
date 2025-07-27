#!/usr/bin/env python3
"""
Example script demonstrating how to use Gemini AI as a fallback for sentiment analysis
when Watson API credits are exhausted or unavailable.
"""

import os
import pandas as pd
from legal_sentiment_evaluation_hf import GeminiSentimentEvaluator, GEMINI_AVAILABLE

def test_gemini_fallback():
    """Test the Gemini AI fallback functionality."""
    
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        print("❌ Gemini AI not available. Install with: pip install google-genai")
        return False
    
    # Set up API key (you can also set GEMINI_API_KEY environment variable)
    api_key = "AIzaSyBBT6CQs4PCXGRfKbRHx1nk84PUrcVGFus"  # Your provided API key
    
    # For security, it's better to use environment variable:
    # os.environ["GEMINI_API_KEY"] = api_key
    
    try:
        # Initialize Gemini evaluator
        print("🚀 Initializing Gemini AI sentiment evaluator...")
        gemini_evaluator = GeminiSentimentEvaluator(api_key=api_key)
        
        # Test with sample legal texts
        test_texts = [
            "The court ruled in favor of the plaintiff and granted the motion for summary judgment.",
            "The appellant's case was dismissed and the lower court's decision was affirmed.",
            "The matter is remanded to the trial court for further proceedings consistent with this opinion."
        ]
        
        print("📝 Testing with sample legal texts...")
        
        # Create a test DataFrame
        test_df = pd.DataFrame({
            'text_content': test_texts,
            'case_id': ['TEST_001', 'TEST_002', 'TEST_003']
        })
        
        # Generate predictions
        predictions = gemini_evaluator.generate_predictions(test_df)
        
        # Display results
        print("\n📊 Results:")
        print("-" * 60)
        for i, (text, prediction) in enumerate(zip(test_texts, predictions)):
            print(f"Case {i+1}: {prediction.upper()}")
            print(f"Text: {text[:100]}...")
            print("-" * 60)
        
        print("✅ Gemini AI fallback test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini AI: {e}")
        return False

def setup_environment():
    """Setup instructions for using Gemini AI."""
    print("🔧 Setup Instructions for Gemini AI Fallback:")
    print("=" * 50)
    print("1. Install Google GenAI: pip install google-genai")
    print("2. Set your API key as environment variable:")
    print("   export GEMINI_API_KEY='your_api_key_here'")
    print("3. Or pass it directly to the GeminiSentimentEvaluator constructor")
    print()
    print("💡 Usage in your code:")
    print("   from legal_sentiment_evaluation_hf import GeminiSentimentEvaluator")
    print("   evaluator = GeminiSentimentEvaluator()")
    print("   predictions = evaluator.generate_predictions(your_dataframe)")
    print()

if __name__ == "__main__":
    print("🧪 Gemini AI Fallback Test Script")
    print("=" * 40)
    
    setup_environment()
    
    # Test the fallback
    success = test_gemini_fallback()
    
    if success:
        print("\n🎉 Gemini AI is ready to use as a Watson API fallback!")
    else:
        print("\n⚠️  Please check the setup instructions above.")
