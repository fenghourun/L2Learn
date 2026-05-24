# data.py

class TextDataset:
    def __init__(self, text, block_size):
        self.text = text
        self.block_size = block_size

        # TODO: build vocab (char-level first)

    def encode(self, text):
        # TODO
        pass

    def decode(self, tokens):
        # TODO
        pass

    def get_batch(self):
        # TODO:
        # return (x, y)
        pass
