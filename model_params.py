from types import SimpleNamespace


def set_model_params(embedding_dim, hidden_dim, vocab_size, output_size=None, lr=0.1, epochs=10):
    if not output_size:
        output_size = vocab_size

    return SimpleNamespace(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        output_size=output_size,
        lr=lr,
        epochs=epochs
    )
