import torch

def generate_text(model, tokenizer, prompt, max_length=50, device="cpu"):
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = tokens

    for _ in range(max_length):
        output = model.decode(generated, None, generated, None)
        logits = model.linear(output[:, -1, :])
        next_token = torch.argmax(logits, dim=-1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0])
