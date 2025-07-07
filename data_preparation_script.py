import json
import pandas as pd
from pathlib import Path
import re

def extract_cases(path="legal_sentiment_analysis/14/json"):
    """
    Extracts case data from individual JSON files located in the specified directory.

    This function iterates through all JSON files in the given path, parsing each
    to extract key information such as case ID, name, court, decision date,
    jurisdiction, and the full text of all legal opinions.

    Args:
        path (str): The directory containing the JSON files.

    Returns:
        list: A list of dictionaries, where each dictionary represents a case.
    """
    law_text = []
    for file in Path(path).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            json1 = json.load(f)
        
        case_id = json1.get('id')
        case_name = json1.get('name')
        case_court = json1.get('court')
        case_date = json1.get("decision_date")
        case_jurisdiction = json1.get("jurisdiction", {}).get("name")
        opinions = json1.get("casebody", {}).get("opinions", [])
        case_body = "\n".join([opinion.get('text', '') for opinion in opinions])
        
        law_text.append({
            'case_id': case_id,
            'case_name': case_name,
            'case_court': case_court,
            'case_date': case_date,
            'case_jurisdiction': case_jurisdiction,
            'case_body': case_body
        })
    return law_text

def clean_text(text):
    """
    Cleans the input text by removing numerical citations and excess newlines.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[\\d+]', '', text)
    text = re.sub(r'No\\.\\s\\d+-\\d+', '', text)
    text = re.sub(r'\\n+', '\\n', text).strip()
    return text

def preprocess_legal_text(law_data):
    """
    Applies the text cleaning function to the case body of each legal document.

    Args:
        law_data (list): A list of case dictionaries.

    Returns:
        list: The list of case dictionaries with cleaned text.
    """
    for entry in law_data:
        entry['case_body'] = clean_text(entry['case_body'])
    return law_data

def prepare_legal_sentiment_data(json_path="legal_sentiment_analysis/14/json"):
    """
    The main function to orchestrate the data preparation pipeline.

    This function extracts data from JSON files, preprocesses the text,
    and saves the consolidated data to a single CSV file.
    """
    print("Step 1: Extracting and preprocessing legal case data...")
    law_data = extract_cases(json_path)
    preprocessed_data = preprocess_legal_text(law_data)
    
    df = pd.DataFrame(preprocessed_data)
    # Rename columns for clarity and consistency
    df = df.rename(columns={'case_id': 'document_id', 'case_body': 'text_content'})
    df['document_type'] = 'legal_case'

    # Ensure the output directory exists
    output_dir = Path('legal_sentiment_data')
    output_dir.mkdir(exist_ok=True)

    print("Step 2: Saving all documents to 'all_legal_documents.csv'...")
    df.to_csv(output_dir / 'all_legal_documents.csv', index=False)
    
    print("\nData preparation is complete. The dataset is ready for use.")
    return df
