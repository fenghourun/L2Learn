import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) # (B, T) --> (B, T, D)
        x = self.block(x) # attention block
        return self.lm_head(x)


model = TinyTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# Training loop

for step in range(100):
    # Fake batch of token IDs
    x = torch.randint(0, 1000, (32, 16))  # (batch, seq)

    # Next-token targets
    y = x.roll(-1, dims=1)

    logits = model(x)

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step={step} loss={loss.item():.4f}")

model.eval()

with torch.no_grad():
    x = torch.tensor([[5, 12, 42, 17]])  # (1, T)

    logits = model(x)

    # logits shape: (1, T, vocab_size)
    next_token_logits = logits[:, -1, :]

    next_token = next_token_logits.argmax(dim=-1)

    print('Next token', next_token)
