import pandas as pd

def load_dataset(file_path):
    """Load raw dataset from CSV."""
    return pd.read_csv(file_path)

def clean_text(text):
    """Clean Tanglish text (e.g., remove special characters)."""
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

def preprocess_data(input_path, output_path):
    """Load, clean, and save processed data."""
    data = load_dataset(input_path)
    data['Tanglish'] = data['Tanglish'].apply(clean_text)
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/raw/tanglish_dataset.csv", "data/processed/processed_tanglish.csv")

