# generate.py


def sample(logits, temperature=1.0):
    """
    Sample next token from logits.

    Behavior:
        - divide logits by temperature
        - apply softmax
        - randomly sample from distribution
    """
    pass


def generate(model, prompt, max_tokens):
    """
    Autoregressive text generation.

    Behavior:
        1. encode prompt
        2. loop:
            - run model
            - sample next token
            - append to sequence
        3. decode output
    """
    pass
