import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- Auto-detect and set the device for PyTorch ---
if torch.cuda.is_available():
    DEVICE = 0  # Use the first GPU
    print("CUDA is available. Using GPU.")
else:
    DEVICE = -1  # Use CPU
    print("CUDA not available. Using CPU.")

class LegalSentimentEvaluatorHF:
    """
    A class to evaluate the sentiment of legal documents using standard Hugging Face sentiment models.
    """
    def __init__(self, model_name):
        print(f"Loading standard sentiment model: {model_name}")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=DEVICE
        )
        print(f"Model '{model_name}' loaded successfully.")

    def _predict_sentiment(self, text):
        truncated_text = text[:512]
        if not truncated_text or truncated_text.isspace():
            return "neutral"
        try:
            result = self.sentiment_pipeline(truncated_text)
            return result[0]['label'].lower()
        except Exception as e:
            print(f"Error during standard sentiment prediction: {e}")
            return "error"

    def generate_predictions(self, df, text_column='text_content'):
        print(f"Generating predictions with {self.sentiment_pipeline.model.name_or_path}...")
        df[text_column] = df[text_column].fillna('')
        predictions = [self._predict_sentiment(text) for text in df[text_column]]
        print("Prediction generation complete.")
        return predictions

class PromptBasedSentimentLLM:
    """
    A class to evaluate legal sentiment using an instruction-tuned LLM with prompt engineering.
    """
    def __init__(self, model_name):
        print(f"Loading instruction-tuned LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=DEVICE
        )
        print(f"LLM '{model_name}' loaded successfully.")

    def _create_prompt(self, text):
        return f"""
        Instruction: You are an expert legal analyst. Read the following legal opinion and classify its sentiment as "positive", "negative", or "neutral" based on the outcome for the primary litigant.
        - A "positive" outcome is a judgment in favor of the plaintiff or appellant.
        - A "negative" outcome is a judgment against the plaintiff or appellant.
        - A "neutral" outcome involves procedural matters, mixed results, or remands without a final decision on the merits.

        Legal Text: "{text}"

        Sentiment Classification:
        """

    def _predict_sentiment(self, text):
        # T5 models have a 512 token limit, so we truncate the text.
        # We leave some space for the prompt itself.
        truncated_text = self.tokenizer.decode(self.tokenizer.encode(text, max_length=450, truncation=True))
        
        if not truncated_text or truncated_text.isspace():
            return "neutral"

        prompt = self._create_prompt(truncated_text)
        
        try:
            generated_text = self.llm_pipeline(
                prompt,
                max_length=10,  # Max length for the generated answer
                num_return_sequences=1
            )[0]['generated_text']
            
            answer = generated_text.lower()
            
            if 'positive' in answer:
                return 'positive'
            elif 'negative' in answer:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"Error during LLM sentiment prediction: {e}")
            return "error"

    def generate_predictions(self, df, text_column='text_content'):
        print(f"Generating LLM predictions with {self.model.name_or_path}...")
        df[text_column] = df[text_column].fillna('')
        predictions = [self._predict_sentiment(text) for text in df[text_column]]
        print("LLM prediction generation complete.")
        return predictions
