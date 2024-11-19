import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.tokens[idx:idx + self.seq_len],
            self.tokens[idx + 1:idx + self.seq_len + 1]
        )

def train_model(model, tokenizer, train_text, device, epochs=3, batch_size=16, seq_len=128, lr=1e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    # Tokenizar datos
    train_tokens = tokenizer(train_text, return_tensors="pt")["input_ids"].squeeze(0)
    dataset = TextDataset(train_tokens, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model.decode(src, None, tgt, None)
            logits = model.linear(output)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
