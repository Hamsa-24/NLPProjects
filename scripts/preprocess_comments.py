import pandas as pd
import re
import os
from nltk.corpus import stopwords

def clean_comment(comment):
    """Clean a single comment."""
    # Remove emojis
    comment = re.sub(r'[^\x00-\x7F]+', '', comment)
    # Remove special characters
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)
    # Convert to lowercase
    comment = comment.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    comment = " ".join(word for word in comment.split() if word not in stop_words)
    return comment

def preprocess_comments(input_file, output_file="data/processed/cleaned_comments.csv"):
    """Preprocess comments from the raw data."""
    df = pd.read_csv(input_file)
    df["Cleaned_Comment"] = df["Comment"].apply(clean_comment)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    preprocess_comments("data/raw/instagram_comments.csv")

