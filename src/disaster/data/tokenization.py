# TRANSCRIBED FROM notebooks/Final_notebook.ipynb — faithful to the original.

"""HuggingFace tokenization helper (max_len=256, padded, truncated)."""

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


def get_tokenizer(model_name: str):
    assert AutoTokenizer is not None, "pip install transformers"
    return AutoTokenizer.from_pretrained(model_name)


def encode(tokenizer, text: str, max_len: int = 256):
    """Tokenize a single string to fixed-length tensors (squeezed)."""
    enc = tokenizer(
        str(text), max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in enc.items()}
