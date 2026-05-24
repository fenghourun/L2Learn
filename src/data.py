# data.py


class TextDataset:
    """
    Simple character-level dataset.
    """

    def __init__(self, text, block_size):
        """
        Args:
            text: raw string corpus
            block_size: sequence length

        Behavior:
            - build vocabulary from unique characters
            - create encode/decode maps
        """
        pass

    def encode(self, text):
        """
        Convert string → list of token IDs.

        Behavior:
            - map each character to integer ID
        """
        pass

    def decode(self, tokens):
        """
        Convert token IDs → string.

        Behavior:
            - inverse mapping of encode
        """
        pass

    def get_batch(self):
        """
        Returns:
            (x, y) training batch

        Behavior:
            - randomly sample sequence of length block_size
            - x = tokens[i:i+n]
            - y = tokens[i+1:i+n+1]
        """
        pass
