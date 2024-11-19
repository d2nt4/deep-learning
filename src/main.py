import torch
from model import build_transformer_model
from tokenizer import get_tokenizer
from data_loader import load_text_data
from train import train_model
from generate import generate_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    # Configurar modelo
    model = build_transformer_model(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        src_seq_len=128,
        tgt_seq_len=128,
        d_model=512,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        hidden_size=2048
    ).to(device)

    # Cargar datos
    train_text, _ = load_text_data()

    # Entrenar el modelo
    train_model(model, tokenizer, train_text, device)

    # Generar texto
    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt, device=device)
    print(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    main()
