"""
Configuration file for Legal Sentiment Analysis project.
Centralizes all constants and configuration parameters.
"""

# Text processing constants
MAX_TEXT_LENGTH_STANDARD = 512  # For standard BERT/RoBERTa models
MAX_TEXT_LENGTH_LLM = 300       # For LLM models (leaving room for prompt)
MAX_TEXT_LENGTH_WATSON = 10000  # For IBM Watson API

# Batch sizes for processing
BATCH_SIZE_STANDARD = 32        # For standard models
BATCH_SIZE_LLM = 16            # For LLM models (more memory intensive)
BATCH_SIZE_WATSON = 10         # For IBM Watson API calls
BATCH_SIZE_GEMINI = 8          # For Gemini API calls (rate limiting)

# File paths and directories
DEFAULT_JSON_PATH = "14/json"
DATA_OUTPUT_DIR = "legal_sentiment_data"
OUTPUT_CSV_FILENAME = "all_legal_documents.csv"

# Model configurations
HUGGINGFACE_MODELS = {
    'bert': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'roberta': 'cardiffnlp/twitter-roberta-base-sentiment',
    'distilbart': 'sshleifer/distilbart-cnn-12-6'
}

# Sentiment label mappings
POSITIVE_LABELS = ['positive', 'pos', 'favor', 'won', 'grant', '5 stars', '4 stars']
NEGATIVE_LABELS = ['negative', 'neg', 'against', 'lost', 'deni', '1 star', '2 stars']
NEUTRAL_LABELS = ['neutral', 'neut', 'mixed', 'procedural', '3 stars']

# Watson API configuration
WATSON_MODEL_ID = "google/flan-t5-xxl"
WATSON_MAX_NEW_TOKENS = 5
WATSON_API_VERSION = "2024-03-19"

# Gemini AI configuration  
GEMINI_MODEL_ID = "gemma-3-27b-it"
GEMINI_MAX_OUTPUT_TOKENS = 10
GEMINI_TEMPERATURE = 0.1
