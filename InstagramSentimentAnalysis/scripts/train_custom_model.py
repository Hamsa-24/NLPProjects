from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def train_custom_model(dataset_path="data/labeled/labeled_comments.csv", model_path="models/trained_model"):
    """Fine-tune a custom sentiment analysis model."""
    # Load dataset
    df = pd.read_csv(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    # Tokenize dataset
    def tokenize(batch):
        return tokenizer(batch["Comment"], padding=True, truncation=True)

    tokenized_data = df.map(tokenize, batched=True)

    # Define Trainer
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_data)

    # Train
    trainer.train()
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_custom_model()
