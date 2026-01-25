import math

from src import data


def test_chunking_shapes():
    tokenizer = data.load_tokenizer("gpt2")
    seq_len = 8
    sample = {"text": ["Hello world."] * 5}
    tokenized = data._tokenize_function(sample, tokenizer)
    grouped = data._group_texts(tokenized, seq_len=seq_len)
    assert "input_ids" in grouped
    assert len(grouped["input_ids"]) > 0
    assert all(len(seq) == seq_len for seq in grouped["input_ids"])
