import pandas as pd
import matplotlib.pyplot as plt

def visualize_sentiment(input_file="data/processed/sentiment_results.csv"):
    """Visualize sentiment analysis results."""
    df = pd.read_csv(input_file)

    # Count sentiment labels
    sentiment_counts = df["Sentiment"].value_counts()

    # Plot
    sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
    plt.title("Sentiment Analysis Results")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    visualize_sentiment()
