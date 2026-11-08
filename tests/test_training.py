import numpy as np
import pytest
import torch

from reclab.data import load_partition, prepare
from reclab.train import Config, assess, fit


def test_deterministic_training(raw_data):
    prepare(raw_data)
    train = load_partition(raw_data, "train")
    valid = load_partition(raw_data, "valid")
    genres = np.load(raw_data / "genres.npy")
    config = Config(epochs=1)
    a, cost = fit(train, genres, 42, config)
    b, _ = fit(train, genres, 42, config)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert cost["optimizer_steps"] == 1
    result = assess(a, valid, train, 10)
    assert 0 <= result["recall@10"] <= 1
    assert result["normalized_entropy"] > 0


def test_invalid_config():
    with pytest.raises(ValueError):
        Config(epochs=100)


def test_test_is_opened_after_frozen_decision(raw_data, tmp_path, monkeypatch):
    import reclab.experiment as experiment

    prepare(raw_data)
    output = tmp_path / "experiment"
    original = experiment.load_partition
    accessed = []

    def guarded(root, name):
        if name == "test":
            assert (output / "decision.json").exists()
        accessed.append(name)
        return original(root, name)

    monkeypatch.setattr(experiment, "load_partition", guarded)
    result = experiment.run(raw_data, output, Config(epochs=1), seeds=(42,))
    assert accessed == ["train", "valid", "test"]
    assert (output / "index.html").exists()
    assert len(result["runs"]) == 5
    with pytest.raises(ValueError, match="fresh"):
        experiment.run(raw_data, output)
