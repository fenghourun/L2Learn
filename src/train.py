# train.py

from transformer import GPT
from data import TextDataset


def train():
    # TODO:
    # 1. load text
    # 2. build dataset
    # 3. init model
    # 4. training loop

    model = GPT(
        vocab_size=100,
        d_model=128,
        n_layers=2,
    )

    for step in range(1000):
        x, y = None, None  # TODO

        logits = model.forward(x)

        loss = None  # TODO cross entropy

        # TODO:
        # backward
        # update weights (SGD first)

        print(step, loss)


if __name__ == "__main__":
    train()
