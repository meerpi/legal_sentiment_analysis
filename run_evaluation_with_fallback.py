#!/usr/bin/env python3
"""
Enhanced evaluation script with Gemini AI fallback for when Watson API is unavailable.
This script demonstrates how to gracefully fallback to Gemini when Watson API credits are exhausted.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation_with_fallback():
    """
    Run sentiment evaluation with Watson API and Gemini AI fallback.
    """
    
    print("🚀 Starting Legal Sentiment Analysis with Fallback Support")
    print("=" * 60)
    
    # --- Step 1: Data Preparation ---
    print("📋 Step 1: Data Preparation")
    try:
        from data_preparation_script import prepare_legal_sentiment_data
        prepare_legal_sentiment_data()
        print("✅ Data preparation completed successfully")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return False
    
    # --- Step 2: Load Dataset ---
    print("\n📊 Step 2: Loading Dataset")
    input_path = Path('legal_sentiment_data/all_legal_documents.csv')
    if not input_path.exists():
        logger.error(f"Dataset not found: {input_path}")
        return False
    
    df = pd.read_csv(input_path)
    # For testing, use a smaller subset
    df = df.head(5)
    print(f"✅ Loaded {len(df)} documents for analysis")
    
    # --- Step 3: Try Watson API First ---
    print("\n🔬 Step 3: Attempting Watson API Analysis")
    watson_success = False
    
    try:
        # Check if Watson credentials are available
        watson_api_key = os.environ.get("WATSON_API_KEY")
        watson_url = os.environ.get("WATSON_URL") 
        watson_project_id = os.environ.get("WATSON_PROJECT_ID")
        
        if all([watson_api_key, watson_url, watson_project_id]):
            from legal_sentiment_evaluation import LegalSentimentEvaluator
            
            print("🔑 Watson credentials found, attempting analysis...")
            evaluator = LegalSentimentEvaluator(watson_api_key, watson_url, watson_project_id)
            
            # Try to analyze a single document first
            test_predictions = evaluator.generate_predictions(df.head(1))
            
            if test_predictions and test_predictions['predicted_sentiment'].iloc[0] != 'error':
                # Watson API is working, proceed with full analysis
                print("✅ Watson API is working, proceeding with full analysis...")
                watson_predictions = evaluator.generate_predictions(df)
                df['watson_sentiment'] = watson_predictions['predicted_sentiment']
                watson_success = True
            else:
                print("❌ Watson API returned errors")
                
        else:
            print("⚠️  Watson credentials not found in environment")
            
    except Exception as e:
        logger.warning(f"Watson API failed: {e}")
        print("❌ Watson API unavailable or failed")
    
    # --- Step 4: Use Gemini AI Fallback ---
    if not watson_success:
        print("\n🤖 Step 4: Using Gemini AI Fallback")
        
        try:
            from legal_sentiment_evaluation_hf import GeminiSentimentEvaluator, GEMINI_AVAILABLE
            
            if not GEMINI_AVAILABLE:
                print("❌ Gemini AI not available. Install with: pip install google-genai")
                return False
            
            # Initialize Gemini with your API key
            gemini_api_key = "AIzaSyBBT6CQs4PCXGRfKbRHx1nk84PUrcVGFus"  # Your provided key
            
            print("🚀 Initializing Gemini AI as fallback...")
            gemini_evaluator = GeminiSentimentEvaluator(api_key=gemini_api_key)
            
            print("🔍 Generating predictions with Gemini AI...")
            gemini_predictions = gemini_evaluator.generate_predictions(df)
            df['gemini_sentiment'] = gemini_predictions
            
            print("✅ Gemini AI analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Gemini AI fallback failed: {e}")
            print("❌ Both Watson and Gemini AI failed")
            return False
    
    # --- Step 5: Add Local Hugging Face Models ---
    print("\n🤗 Step 5: Adding Local Hugging Face Models")
    
    try:
        from legal_sentiment_evaluation_hf import LegalSentimentEvaluatorHF, PromptBasedSentimentLLM
        
        # Quick local models (no API required)
        print("📥 Loading local BERT model...")
        bert_model = LegalSentimentEvaluatorHF('cardiffnlp/twitter-roberta-base-sentiment')
        df['local_sentiment'] = bert_model.generate_predictions(df)
        
        print("✅ Local model analysis completed")
        
    except Exception as e:
        logger.warning(f"Local models failed: {e}")
        df['local_sentiment'] = ['error'] * len(df)
    
    # --- Step 6: Save Results ---
    print("\n💾 Step 6: Saving Results")
    
    # Create output filename based on what worked
    if watson_success:
        output_file = 'legal_sentiment_data/sentiment_analysis_watson_primary.csv'
        print("📄 Using Watson API as primary method")
    else:
        output_file = 'legal_sentiment_data/sentiment_analysis_gemini_fallback.csv'
        print("📄 Using Gemini AI as fallback method")
    
    df.to_csv(output_file, index=False)
    
    # --- Step 7: Display Summary ---
    print("\n📈 Step 7: Analysis Summary")
    print("=" * 50)
    
    if watson_success:
        watson_counts = df['watson_sentiment'].value_counts()
        print("Watson API Results:")
        for sentiment, count in watson_counts.items():
            print(f"  {sentiment.capitalize()}: {count}")
    
    if 'gemini_sentiment' in df.columns:
        gemini_counts = df['gemini_sentiment'].value_counts()
        print("\nGemini AI Results:")
        for sentiment, count in gemini_counts.items():
            print(f"  {sentiment.capitalize()}: {count}")
    
    if 'local_sentiment' in df.columns:
        local_counts = df['local_sentiment'].value_counts()
        print("\nLocal Model Results:")
        for sentiment, count in local_counts.items():
            print(f"  {sentiment.capitalize()}: {count}")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("🎉 Analysis completed successfully!")
    
    return True

def setup_instructions():
    """Display setup instructions."""
    print("🔧 Setup Instructions")
    print("=" * 30)
    print()
    print("For Watson API (Primary):")
    print("  export WATSON_API_KEY='your_watson_key'")
    print("  export WATSON_URL='your_watson_url'") 
    print("  export WATSON_PROJECT_ID='your_project_id'")
    print()
    print("For Gemini AI (Fallback):")
    print("  pip install google-genai")
    print("  export GEMINI_API_KEY='your_gemini_key'")
    print("  # Or use the hardcoded key in this script")
    print()
    print("For Local Models:")
    print("  pip install transformers torch")
    print("  # No API keys required")
    print()

if __name__ == "__main__":
    print("🏛️  Legal Sentiment Analysis with AI Fallback")
    print("=" * 50)
    
    setup_instructions()
    
    success = run_evaluation_with_fallback()
    
    if success:
        print("\n🎯 All done! Check the output files for results.")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")
