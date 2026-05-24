# transformer.py

from nn import Linear, LayerNorm
from attention import SelfAttention


class TransformerBlock:
    """
    One transformer layer:
    Attention + MLP with residual connections.
    """

    def forward(self, x):
        """
        Args:
            x: sequence (T, d_model)

        Returns:
            transformed sequence

        Behavior:
            1. x1 = LayerNorm(x)
            2. x2 = SelfAttention(x1)
            3. x = x + x2 (residual)
            4. x3 = LayerNorm(x)
            5. x4 = MLP(x3)
            6. x = x + x4
        """
        pass


class GPT:
    """
    Full transformer language model.
    """

    def forward(self, tokens):
        """
        Args:
            tokens: list of token IDs

        Returns:
            logits over vocabulary for each position

        Steps:
            1. Convert tokens → embeddings
            2. Add positional embeddings
            3. Pass through transformer blocks
            4. Apply final normalization
            5. Project to vocabulary logits
        """
        pass
