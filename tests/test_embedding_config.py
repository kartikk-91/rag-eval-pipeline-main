import pytest

from rag import config


class _Encoder:
    def __init__(self, maximum: int):
        config_type = type("Config", (), {"max_position_embeddings": maximum})
        self.auto_model = type("Backbone", (), {"config": config_type()})()
        self.tokenizer = lambda texts, **_: {
            "input_ids": [text.split() for text in texts]
        }


class _Model:
    def __init__(self, maximum: int):
        self.encoder = _Encoder(maximum)
        self.max_seq_length = 128

    def __getitem__(self, index):
        assert index == 0
        return self.encoder


def test_embedding_model_uses_configured_window_within_backbone_limit(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_SEQ_LENGTH", 256)
    model = _Model(maximum=512)

    assert config.configure_embedding_model(model) is model
    assert model.max_seq_length == 256


def test_embedding_model_rejects_window_larger_than_backbone(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_SEQ_LENGTH", 513)

    with pytest.raises(ValueError, match="encoder limit"):
        config.configure_embedding_model(_Model(maximum=512))


def test_embedding_model_rejects_unusable_window(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_SEQ_LENGTH", 63)

    with pytest.raises(ValueError, match="at least 64"):
        config.configure_embedding_model(_Model(maximum=512))


def test_index_validation_rejects_text_that_would_be_truncated(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_SEQ_LENGTH", 64)
    model = config.configure_embedding_model(_Model(maximum=512))

    with pytest.raises(ValueError, match="would be truncated"):
        config.validate_embedding_inputs(model, [" ".join(["word"] * 65)])


def test_index_validation_accepts_text_within_token_window(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_SEQ_LENGTH", 64)
    model = config.configure_embedding_model(_Model(maximum=512))

    assert config.validate_embedding_inputs(model, ["one two three"]) == 3
