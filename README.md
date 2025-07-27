# Legal Sentiment Analysis

This project provides a framework for performing sentiment analysis on legal documents. It offers two main approaches:
1.  **IBM Watsonx.ai Integration**: Utilizes IBM's powerful AI capabilities for sentiment prediction, with data handling via IBM Cloud Object Storage (COS).
2.  **Hugging Face Models**: Leverages open-source sentiment analysis models and instruction-tuned Large Language Models (LLMs) from Hugging Face for local execution.

The goal is to extract and analyze sentiment from legal texts, which can be crucial for understanding case outcomes, judicial opinions, or legal document nuances.

## Features

*   **Data Preparation**: Extracts and preprocesses legal case data from JSON files into a structured CSV format.
*   **IBM Watsonx.ai Integration**:
    *   Authenticates with IBM Cloud IAM.
    *   Predicts sentiment using the `google/flan-t5-xxl` model on Watsonx.ai.
    *   Integrates with IBM Cloud Object Storage (COS) for seamless data upload and download.
*   **Hugging Face Model Support**:
    *   Supports standard sentiment analysis models (e.g., `nlptown/bert-base-multilingual-uncased-sentiment`, `cardiffnlp/twitter-roberta-base-sentiment`).
    *   Includes functionality for prompt-based sentiment analysis using instruction-tuned LLMs (e.g., `sshleifer/distilbart-cnn-12-6`).
    *   Automatically detects and utilizes GPU if available for faster processing.
*   **Modular Design**: Clear separation of concerns for data handling, IBM COS utilities, and sentiment evaluation.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/legal_sentiment_analysis.git
    cd legal_sentiment_analysis
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

The `data_preparation_script.py` is used to extract legal case data from JSON files and prepare it for sentiment analysis. The script expects JSON files in the `14/json` directory.

To run data preparation:

```bash
python data_preparation_script.py
```

This will generate `all_legal_documents.csv` in the `legal_sentiment_data/` directory.

### 2. Running Sentiment Analysis with IBM Watsonx.ai

This approach requires IBM Cloud credentials and a Watsonx.ai project.

1.  **Set up Environment Variables**:
    You need to set the following environment variables with your IBM Cloud credentials. These are crucial for authentication and access to IBM Watsonx.ai and IBM COS services. The `run_evaluation.py` script reads these values directly from your environment.

    ```bash
    export WATSON_API_KEY="YOUR_WATSON_API_KEY"
    export WATSON_URL="YOUR_WATSON_URL" # e.g., https://us-south.ml.cloud.ibm.com
    export WATSON_PROJECT_ID="YOUR_WATSON_PROJECT_ID"
    export COS_API_KEY="YOUR_COS_API_KEY"
    export COS_SERVICE_INSTANCE_ID="YOUR_COS_SERVICE_INSTANCE_ID"
    export COS_ENDPOINT_URL="YOUR_COS_ENDPOINT_URL" # e.g., https://s3.us-south.cloud-object-storage.appdomain.cloud
    export COS_BUCKET_NAME="YOUR_COS_BUCKET_NAME"
    ```

2.  **Service Availability**:
    Please note that the successful execution of `run_evaluation.py` is dependent on the availability and responsiveness of the IBM Watsonx.ai service and IBM Cloud Object Storage. If these services are unavailable or experience issues, the script will encounter errors and fail to generate predictions. Error handling is implemented to catch API-related exceptions, but successful operation requires active services.

3.  **Run the evaluation script**:
    ```bash
    python run_evaluation.py
    ```
    This script will:
    *   Prepare data (if not already done).
    *   Upload `all_legal_documents.csv` to your specified COS bucket.
    *   Download the data from COS.
    *   Generate sentiment predictions using Watsonx.ai.
    *   Upload the `sentiment_predictions.csv` back to your COS bucket.

### 3. Running Sentiment Analysis with Hugging Face Models

This approach runs locally and does not require IBM Cloud credentials.

1.  **Run the evaluation script**:
    ```bash
    python run_evaluation_hf.py
    ```
    This script will:
    *   Prepare data (if not already done).
    *   Load the `all_legal_documents.csv` dataset.
    *   Initialize and run sentiment prediction using:
        *   `nlptown/bert-base-multilingual-uncased-sentiment`
        *   `cardiffnlp/twitter-roberta-base-sentiment`
        *   `sshleifer/distilbart-cnn-12-6` (for prompt-based LLM sentiment)
    *   Save the combined results to `legal_sentiment_data/sentiment_analysis_results_advanced.csv`.

## Data

The project expects legal case data in JSON format, typically located in the `14/json` directory. The `data_preparation_script.py` processes these JSON files.

The `legal_sentiment_data/` directory contains:
*   `all_legal_documents.csv`: The consolidated dataset of legal documents after preprocessing.
*   `sample_dataset.csv`, `test_dataset.csv`, `train_dataset.csv`, `validation_dataset.csv`: These are likely sample or split datasets for training and testing models.
*   `sentiment_predictions.csv`: (Generated by `run_evaluation.py`) Contains sentiment predictions from IBM Watsonx.ai.
*   `test_predictions.csv`, `validation_predictions.csv`: Likely contain predictions for test and validation sets.
*   `sentiment_analysis_results_advanced.csv`: (Generated by `run_evaluation_hf.py`) Contains sentiment predictions from various Hugging Face models.

## Project Structure

```
.
├── .git/
├── .gitignore
├── 14/
│   ├── html/           # Contains HTML versions of legal documents
│   └── json/           # Contains JSON formatted legal case data (input for data_preparation_script.py)
├── data_preparation_script.py  # Script to extract and preprocess legal data
├── ibm_cos_utils.py            # Utility functions for IBM Cloud Object Storage (COS)
├── legal_sentiment_data/       # Directory for processed data and results
│   ├── all_legal_documents.csv
│   ├── sample_dataset.csv
│   ├── sentiment_predictions.csv
│   ├── test_dataset.csv
│   ├── test_predictions.csv
│   ├── train_dataset.csv
│   ├── validation_dataset.csv
│   └── validation_predictions.csv
├── legal_sentiment_evaluation_hf.py # Sentiment evaluation using Hugging Face models
├── legal_sentiment_evaluation.py    # Sentiment evaluation using IBM Watsonx.ai
├── requirements.txt            # Python dependencies
├── run_evaluation_hf.py        # Script to run evaluation with Hugging Face models
├── run_evaluation.py           # Script to run evaluation with IBM Watsonx.ai and COS
└── README.md                   # This file
```
