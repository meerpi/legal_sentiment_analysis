import pandas as pd
import requests
from pathlib import Path

class LegalSentimentEvaluator:
    """
    A class to evaluate the sentiment of legal documents using IBM's Watsonx.ai API.

    This evaluator handles API authentication, sends text for sentiment prediction,
    and saves the results.
    """
    def __init__(self, api_key, url, project_id):
        """
        Initializes the evaluator with the necessary API credentials.

        Args:
            api_key (str): Your IBM Cloud API key.
            url (str): The base URL for the Watsonx.ai API.
            project_id (str): Your Watsonx.ai project ID.
        """
        self.api_key = api_key
        self.url = url
        self.project_id = project_id
        self.token = self._get_iam_token()
        print("LegalSentimentEvaluator initialized successfully.")

    def _get_iam_token(self):
        """
        Retrieves an IAM token from IBM Cloud for API authentication.

        Returns:
            str: The access token for making authenticated API requests.
        """
        print("Authenticating with IBM Cloud and retrieving IAM token...")
        try:
            response = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key}
            )
            response.raise_for_status()
            print("Successfully retrieved IAM token.")
            return response.json()["access_token"]
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving IAM token: {e}")
            return None

    def _predict_sentiment(self, text):
        """
        Predicts the sentiment of a single piece of text using the Watsonx.ai API.

        Args:
            text (str): The text for which to predict sentiment.

        Returns:
            str: The predicted sentiment ('positive', 'negative', or 'neutral').
        """
        max_length = 10000
        truncated_text = text[:max_length]

        if not truncated_text or truncated_text.isspace():
            return "neutral"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "project_id": self.project_id,
            "input": f"Analyze the sentiment of the following legal text and return only one word: 'positive', 'negative', or 'neutral'. Text: {truncated_text}",
            "model_id": "google/flan-t5-xxl",
            "parameters": {
                "max_new_tokens": 5
            }
        }
        
        try:
            response = requests.post(
                f"{self.url}/ml/v1-beta/generation/text?version=2024-03-19",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["results"][0]["generated_text"].strip().lower()
        except requests.exceptions.RequestException as e:
            print(f"API Error during sentiment prediction: {e}")
            return None

    def generate_predictions(self, df, batch_size=10):
        """
        Generates sentiment predictions for an entire DataFrame of documents.
        Uses batching to handle large datasets efficiently.

        Args:
            df (pd.DataFrame): A DataFrame with a 'text_content' column.
            batch_size (int): Number of documents to process at once.

        Returns:
            pd.DataFrame: The DataFrame with an added 'predicted_sentiment' column.
        """
        print(f"Generating sentiment predictions for {len(df)} documents...")
        df['text_content'] = df['text_content'].fillna('')
        
        predictions = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_num = (i // batch_size) + 1
            print(f"Processing batch {batch_num}/{total_batches}...")
            
            batch = df['text_content'].iloc[i:i+batch_size]
            batch_predictions = [self._predict_sentiment(text) for text in batch]
            predictions.extend(batch_predictions)
        
        df['predicted_sentiment'] = predictions
        print("Prediction generation complete.")
        return df

    def save_predictions_to_cos(self, cos_utils, df, object_key):
        """
        Saves the DataFrame with predictions to a specified IBM COS bucket.

        Args:
            cos_utils (IBMCOSUtils): The initialized IBM COS utility.
            df (pd.DataFrame): The DataFrame to save.
            object_key (str): The desired key for the object in COS.
        """
        import tempfile
        import os
        
        # Use temporary file with automatic cleanup
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False)

        try:
            print(f"Uploading predictions to COS bucket '{cos_utils.bucket_name}'...")
            cos_utils.upload_file(temp_path, object_key)
            print("Predictions successfully uploaded to COS.")
        finally:
            # Ensure cleanup even if upload fails
            if os.path.exists(temp_path):
                os.unlink(temp_path)

