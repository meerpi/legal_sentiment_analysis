import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from abc import ABC, abstractmethod
import logging
import os

# Import configuration
try:
    from config import MAX_TEXT_LENGTH_STANDARD, MAX_TEXT_LENGTH_LLM, BATCH_SIZE_STANDARD, BATCH_SIZE_LLM
except ImportError:
    # Fallback values if config not available
    MAX_TEXT_LENGTH_STANDARD = 512
    MAX_TEXT_LENGTH_LLM = 300
    BATCH_SIZE_STANDARD = 32
    BATCH_SIZE_LLM = 16

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Gemini AI (optional dependency)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenAI not available. Install with: pip install google-genai")

def get_device():
    """
    Auto-detect and return the appropriate device for PyTorch operations.
    
    Returns:
        int: 0 for GPU, -1 for CPU
    """
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        return 0  # Use the first GPU
    else:
        logger.info("CUDA not available. Using CPU.")
        return -1  # Use CPU

# Set the global device
DEVICE = get_device()

class BaseSentimentEvaluator(ABC):
    """
    Base class for sentiment evaluators with common functionality.
    """
    
    def validate_text(self, text):
        """Validate and clean input text."""
        if not text or not isinstance(text, str) or not text.strip():
            return None
        return text.strip()
    
    @abstractmethod
    def _predict_sentiment(self, text):
        """Abstract method for sentiment prediction."""
        pass
    
    def generate_predictions(self, df, text_column='text_content', batch_size=None):
        """
        Generate predictions with batching for efficiency.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Column name containing text to analyze
            batch_size (int): Number of texts to process at once
        """
        if batch_size is None:
            batch_size = BATCH_SIZE_STANDARD
            
        logger.info(f"Generating predictions for {len(df)} documents...")
        df[text_column] = df[text_column].fillna('')
        
        predictions = []
        texts = df[text_column].tolist()
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}...")
            
            batch_texts = texts[i:i+batch_size]
            batch_predictions = []
            
            for text in batch_texts:
                try:
                    prediction = self._predict_sentiment(text)
                    batch_predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    batch_predictions.append("error")
            
            predictions.extend(batch_predictions)
        
        logger.info("Prediction generation complete.")
        return predictions

class LegalSentimentEvaluatorHF(BaseSentimentEvaluator):
    """
    A class to evaluate the sentiment of legal documents using standard Hugging Face sentiment models.
    """
    def __init__(self, model_name):
        logger.info(f"Loading standard sentiment model: {model_name}")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=DEVICE
        )
        logger.info(f"Model '{model_name}' loaded successfully.")

    def _predict_sentiment(self, text):
        """Predict sentiment for a single text."""
        validated_text = self.validate_text(text)
        if not validated_text:
            return "neutral"
            
        # Truncate using configuration
        truncated_text = validated_text[:MAX_TEXT_LENGTH_STANDARD]
        
        result = self.sentiment_pipeline(truncated_text)
        label = result[0]['label'].lower()
        
        # Use more flexible label mapping
        return self._map_sentiment_label(label)
    
    def _map_sentiment_label(self, label):
        """Map model-specific labels to standard sentiment categories."""
        # Positive indicators
        positive_indicators = ['pos', 'positive', '5', '4', 'good', 'favorable']
        # Negative indicators  
        negative_indicators = ['neg', 'negative', '1', '2', 'bad', 'unfavorable']
        
        label_lower = label.lower()
        
        if any(indicator in label_lower for indicator in positive_indicators):
            return 'positive'
        elif any(indicator in label_lower for indicator in negative_indicators):
            return 'negative'
        else:
            return 'neutral'

class PromptBasedSentimentLLM(BaseSentimentEvaluator):
    """
    A class to evaluate legal sentiment using an instruction-tuned LLM with prompt engineering.
    """
    def __init__(self, model_name, max_length=15, temperature=0.1):
        logger.info(f"Loading instruction-tuned LLM: {model_name}")
        
        # Create pipeline directly to avoid loading model twice
        self.llm_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=DEVICE
        )
        
        # Get tokenizer from pipeline to avoid duplicate loading
        self.tokenizer = self.llm_pipeline.tokenizer
        
        # Configuration parameters
        self.max_length = max_length
        self.temperature = temperature
        
        logger.info(f"LLM '{model_name}' loaded successfully.")

    def _create_prompt(self, text):
        """Create a concise, effective prompt for sentiment analysis."""
        return f"""Analyze this legal text sentiment. Reply with only: positive, negative, or neutral.

Legal outcome for plaintiff/appellant:
- Positive: won case, appeal granted, favorable ruling
- Negative: lost case, appeal denied, unfavorable ruling  
- Neutral: procedural matter, remand, mixed result

Text: {text}

Sentiment:"""

    def _predict_sentiment(self, text):
        """Predict sentiment using LLM with improved parsing."""
        validated_text = self.validate_text(text)
        if not validated_text:
            return "neutral"
            
        # Truncate text using configuration
        truncated_text = self.tokenizer.decode(
            self.tokenizer.encode(validated_text, max_length=MAX_TEXT_LENGTH_LLM, truncation=True),
            skip_special_tokens=True
        )
        
        prompt = self._create_prompt(truncated_text)
        
        result = self.llm_pipeline(
            prompt,
            max_length=self.max_length,
            num_return_sequences=1,
            do_sample=False,
            temperature=self.temperature
        )
        
        generated_text = result[0]['generated_text'].lower().strip()
        return self._parse_llm_response(generated_text)
    
    def _parse_llm_response(self, response):
        """Parse LLM response to extract sentiment."""
        # More robust parsing with priority order
        response_lower = response.lower()
        
        # Check for explicit sentiment words first
        if 'positive' in response_lower:
            return 'positive'
        elif 'negative' in response_lower:
            return 'negative'
        elif 'neutral' in response_lower:
            return 'neutral'
        
        # Check for outcome indicators
        positive_outcomes = ['won', 'grant', 'favor', 'success', 'affirm']
        negative_outcomes = ['lost', 'deni', 'against', 'reject', 'reverse']
        neutral_outcomes = ['remand', 'procedural', 'mixed']
        
        if any(word in response_lower for word in positive_outcomes):
            return 'positive'
        elif any(word in response_lower for word in negative_outcomes):
            return 'negative'
        elif any(word in response_lower for word in neutral_outcomes):
            return 'neutral'
        
        # Default to neutral if unclear
        return 'neutral'
    
    def generate_predictions(self, df, text_column='text_content', batch_size=None):
        """Override with configuration-based batch size for LLM."""
        if batch_size is None:
            batch_size = BATCH_SIZE_LLM
        return super().generate_predictions(df, text_column, batch_size)


class GeminiSentimentEvaluator(BaseSentimentEvaluator):
    """
    A fallback class for sentiment analysis using Google Gemini AI when Watson API is unavailable.
    """
    def __init__(self, api_key=None, model_name="gemma-3-27b-it"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI not available. Install with: pip install google-genai")
        
        # Use provided API key or environment variable
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"Gemini AI evaluator initialized with model: {model_name}")
    
    def _create_legal_prompt(self, text):
        """Create a prompt specifically designed for legal sentiment analysis."""
        return f"""You are an expert legal analyst. Analyze the sentiment of the following legal text and classify it as exactly one word: "positive", "negative", or "neutral".

Classification criteria:
- POSITIVE: Favorable outcome for plaintiff/appellant (case won, appeal granted, favorable ruling)
- NEGATIVE: Unfavorable outcome for plaintiff/appellant (case lost, appeal denied, unfavorable ruling)
- NEUTRAL: Procedural matters, remands, mixed results, or inconclusive outcomes

Legal Text: {text}

Respond with only one word: positive, negative, or neutral."""

    def _predict_sentiment(self, text):
        """Predict sentiment using Gemini AI."""
        validated_text = self.validate_text(text)
        if not validated_text:
            return "neutral"
        
        # Truncate text to reasonable length for API
        truncated_text = validated_text[:MAX_TEXT_LENGTH_LLM * 2]  # Gemini can handle longer texts
        
        prompt = self._create_legal_prompt(truncated_text)
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation
            generate_config = types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent results
                top_p=0.8,
                top_k=40,
                max_output_tokens=10,  # We only need one word
                stop_sequences=[".", "\n"]  # Stop at punctuation or newline
            )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_config,
            )
            
            # Extract and parse response
            if response and response.text:
                return self._parse_gemini_response(response.text.strip())
            else:
                logger.warning("Empty response from Gemini API")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error during Gemini sentiment prediction: {e}")
            return "error"
    
    def _parse_gemini_response(self, response):
        """Parse Gemini response to extract sentiment."""
        response_lower = response.lower().strip()
        
        # Direct sentiment matches
        if 'positive' in response_lower:
            return 'positive'
        elif 'negative' in response_lower:
            return 'negative'
        elif 'neutral' in response_lower:
            return 'neutral'
        
        # Check for outcome indicators if direct match fails
        positive_indicators = ['favor', 'won', 'grant', 'success', 'affirm', 'uphold']
        negative_indicators = ['against', 'lost', 'deni', 'reject', 'reverse', 'dismiss']
        neutral_indicators = ['remand', 'procedural', 'mixed', 'inconclusive']
        
        if any(indicator in response_lower for indicator in positive_indicators):
            return 'positive'
        elif any(indicator in response_lower for indicator in negative_indicators):
            return 'negative'
        elif any(indicator in response_lower for indicator in neutral_indicators):
            return 'neutral'
        
        # Default to neutral for unclear responses
        logger.warning(f"Unclear Gemini response: {response}")
        return 'neutral'
    
    def generate_predictions(self, df, text_column='text_content', batch_size=None):
        """Override with smaller batch size for API rate limiting."""
        if batch_size is None:
            batch_size = 8  # Smaller batch size for API calls
        return super().generate_predictions(df, text_column, batch_size)
