import pandas as pd
from data_preparation_script import prepare_legal_sentiment_data
from legal_sentiment_evaluation_hf import LegalSentimentEvaluatorHF, PromptBasedSentimentLLM, GeminiSentimentEvaluator, GEMINI_AVAILABLE
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    The main function to run the complete local legal sentiment analysis pipeline
    with two standard classifiers and two instruction-tuned LLMs.
    """
    
    # --- Step 1: Data Preparation ---
    print("--- Starting Data Preparation ---")
    prepare_legal_sentiment_data()
    print("--- Data Preparation Complete ---\n")
    
    # --- Step 2: Load Dataset ---
    print("--- Loading the Dataset ---")
    input_path = Path('legal_sentiment_data/all_legal_documents.csv')
    if not input_path.exists():
        print(f"Error: The file {input_path} was not found.")
        return
    
    df = pd.read_csv(input_path)
    # For faster testing, uncomment the line below
    # df = df.head(5) 
    print("--- Dataset Loaded ---\n")

    # --- Step 3: Initialize Models ---
    print("--- Initializing Hugging Face Models ---")
    # Classifier 1: BERT-based multilingual sentiment model
    bert_model = LegalSentimentEvaluatorHF(model_name='nlptown/bert-base-multilingual-uncased-sentiment')
    
    # Classifier 2: RoBERTa-based model fine-tuned on Twitter data
    roberta_model = LegalSentimentEvaluatorHF(model_name='cardiffnlp/twitter-roberta-base-sentiment')
    
    # LLM 1: DistilBART for instruction-based sentiment analysis
    distilbart_model = PromptBasedSentimentLLM(model_name='sshleifer/distilbart-cnn-12-6')
    
    # Initialize Gemini AI as fallback if available
    gemini_model = None
    if GEMINI_AVAILABLE:
        try:
            gemini_model = GeminiSentimentEvaluator()
            logger.info("Gemini AI fallback model initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini AI: {e}")
    else:
        logger.info("Gemini AI not available - install with: pip install google-genai")
    
    print("--- All Models Initialized ---\n")

    # --- Step 4: Generate Predictions ---
    print("--- Generating Sentiment Predictions ---")
    df['bert_sentiment'] = bert_model.generate_predictions(df)
    df['roberta_sentiment'] = roberta_model.generate_predictions(df)
    df['distilbart_sentiment'] = distilbart_model.generate_predictions(df)
    
    # Add Gemini predictions if available
    if gemini_model:
        try:
            print("--- Generating Gemini AI Predictions ---")
            df['gemini_sentiment'] = gemini_model.generate_predictions(df)
            logger.info("Gemini AI predictions completed successfully.")
        except Exception as e:
            logger.error(f"Error during Gemini predictions: {e}")
            df['gemini_sentiment'] = ['error'] * len(df)
    else:
        df['gemini_sentiment'] = ['not_available'] * len(df)
    
    print("--- All Predictions Generated ---\n")

    # --- Step 5: Save Results ---
    output_path = Path('legal_sentiment_data/sentiment_analysis_results_advanced.csv')
    print(f"--- Saving Final Results to {output_path} ---")
    df.to_csv(output_path, index=False)
    
    print("======================================================")
    print("Pipeline Finished! Your results are ready.")
    print(f"The combined sentiment analysis has been saved to: {output_path}")
    print("======================================================")

if __name__ == "__main__":
    main()
