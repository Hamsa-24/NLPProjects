from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

def train_model(model_name, dataset_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("csv", data_files=dataset_path)

    # Tokenize data
    def preprocess_batch(batch):
        inputs = tokenizer(batch['Tanglish'], truncation=True, padding=True, max_length=128)
        outputs = tokenizer(batch['Tamil'], truncation=True, padding=True, max_length=128)
        return {'input_ids': inputs['input_ids'], 'labels': outputs['input_ids']}

    tokenized_dataset = dataset.map(preprocess_batch, batched=True)

    # Train the model (simplified training loop)
    model.train()
    # Save trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model("t5-small", "data/processed/processed_tanglish.csv", "models/trained_model/")
