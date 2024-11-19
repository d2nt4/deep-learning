from transformers import AutoTokenizer

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer
