import json
import pandas as pd
from pathlib import Path
import re

def extract_cases(path="14/json"):
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
    json_path = Path(path)
    
    if not json_path.exists():
        print(f"Warning: Directory {path} does not exist.")
        return law_text
    
    json_files = list(json_path.glob("*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {path}")
        return law_text
    
    print(f"Processing {len(json_files)} JSON files from {path}")
    
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                json1 = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
            print(f"Error processing file {file}: {e}")
            continue
        
        case_id = json1.get('id', 'unknown')
        case_name = json1.get('name', 'unknown')
        case_court = json1.get('court', 'unknown')
        case_date = json1.get("decision_date", 'unknown')
        case_jurisdiction = json1.get("jurisdiction", {}).get("name", 'unknown')
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
    if not isinstance(text, str):
        return ""
    
    # Remove standalone numbers and numerical citations
    text = re.sub(r'\b\d+\b', '', text)
    # Remove case citation patterns like "No. 123-456"
    text = re.sub(r'No\.\s*\d+-\d+', '', text)
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text).strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
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

def prepare_legal_sentiment_data(json_path="14/json"):
    """
    The main function to orchestrate the data preparation pipeline.

    This function extracts data from JSON files, preprocesses the text,
    and saves the consolidated data to a single CSV file.
    
    Args:
        json_path (str): Path to the directory containing JSON files.
    """
    print("Step 1: Extracting and preprocessing legal case data...")
    law_data = extract_cases(json_path)
    
    if not law_data:
        print("No data extracted. Exiting.")
        return None
    
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
    
    print(f"\nData preparation is complete. Processed {len(df)} documents.")
    print("The dataset is ready for use.")
    return df

if __name__ == "__main__":
    prepare_legal_sentiment_data()
