from transformers import pipeline
import pandas as pd
import os

def analyze_sentiment(input_file, output_file="data/processed/sentiment_results.csv"):
    """Perform sentiment analysis on Instagram comments."""
    # Load cleaned data
    df = pd.read_csv(input_file)

    # Load sentiment analysis model
    sentiment_analyzer = pipeline("sentiment-analysis")

    # Analyze sentiment
    df["Sentiment"] = df["Cleaned_Comment"].apply(lambda x: sentiment_analyzer(x)[0]["label"])

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Sentiment analysis results saved to {output_file}")

if __name__ == "__main__":
    analyze_sentiment("data/processed/cleaned_comments.csv")
