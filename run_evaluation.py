import os
import pandas as pd
from legal_sentiment_evaluation import LegalSentimentEvaluator
from data_preparation_script import prepare_legal_sentiment_data
from ibm_cos_utils import IBMCOSUtils

def main():
    """
    The main function to run the complete legal sentiment analysis pipeline with IBM COS.
    
    This script will:
    1. Initialize the IBM COS utility.
    2. Prepare and upload data to a COS bucket.
    3. Download the data from COS for processing.
    4. Initialize the LegalSentimentEvaluator with IBM Watson credentials.
    5. Generate sentiment predictions for all documents.
    6. Upload the final predictions back to the COS bucket.
    """
    
    # --- Step 1: Load Credentials from Environment Variables ---
    watson_api_key = os.environ.get("WATSON_API_KEY")
    watson_url = os.environ.get("WATSON_URL")
    watson_project_id = os.environ.get("WATSON_PROJECT_ID")
    cos_api_key = os.environ.get("COS_API_KEY")
    cos_service_instance_id = os.environ.get("COS_SERVICE_INSTANCE_ID")
    cos_endpoint_url = os.environ.get("COS_ENDPOINT_URL")
    cos_bucket_name = os.environ.get("COS_BUCKET_NAME")

    if not all([watson_api_key, watson_url, watson_project_id, cos_api_key, cos_service_instance_id, cos_endpoint_url, cos_bucket_name]):
        print("Error: Missing one or more required environment variables.")
        return

    # --- Step 2: Initialize IBM COS Utility ---
    print("--- Initializing IBM COS Utility ---")
    cos_utils = IBMCOSUtils(cos_api_key, cos_service_instance_id, cos_endpoint_url, cos_bucket_name)
    print("--- COS Utility Initialized ---\n")

    # --- Step 3: Data Preparation and Upload to COS ---
    print("--- Starting Data Preparation and Upload to COS ---")
    prepare_legal_sentiment_data(cos_utils, output_object_key="all_legal_documents.csv")
    print("--- Data Preparation and Upload Complete ---\n")

    # --- Step 4: Download Data from COS ---
    print("--- Downloading Dataset from COS ---")
    local_csv_path = "downloaded_documents.csv"
    cos_utils.download_file("all_legal_documents.csv", local_csv_path)
    all_docs_df = pd.read_csv(local_csv_path)
    print("--- Dataset Downloaded ---\n")

    # --- Step 5: Initialize the Evaluator ---
    print("--- Initializing Sentiment Evaluator ---")
    evaluator = LegalSentimentEvaluator(watson_api_key, watson_url, watson_project_id)
    print("--- Evaluator Initialized ---\n")

    # --- Step 6: Generate and Save Predictions to COS ---
    print("--- Generating and Uploading Sentiment Predictions ---")
    predictions_df = evaluator.generate_predictions(all_docs_df)
    evaluator.save_predictions_to_cos(cos_utils, predictions_df, "sentiment_predictions.csv")
    print("--- Prediction Generation and Upload Complete ---\n")

    print("======================================================")
    print("Pipeline Finished! Your results are ready in your COS bucket.")
    print(f"The sentiment predictions have been saved to '{cos_bucket_name}/sentiment_predictions.csv'.")
    print("======================================================")

if __name__ == "__main__":
    main()
