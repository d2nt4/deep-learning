from datasets import load_dataset

def load_text_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = " ".join(dataset["train"]["text"])
    test_text = " ".join(dataset["test"]["text"])
    return train_text, test_text
