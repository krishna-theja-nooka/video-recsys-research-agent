import numpy as np
import pytest

from reclab.data import download, load_partition, prepare


def test_license_required(tmp_path):
    with pytest.raises(ValueError, match="accept-license"):
        download(tmp_path)


def test_temporal_integrity(raw_data):
    manifest = prepare(raw_data)
    tr, va, te = [load_partition(raw_data, p) for p in ("train", "valid", "test")]
    assert tr[:, 3].max() < va[:, 3].min() <= va[:, 3].max() < te[:, 3].min()
    assert sum(p["raw"] for p in manifest["partitions"].values()) == 960
    assert not set(map(tuple, tr[:, :2])) & set(map(tuple, va[:, :2]))
    assert manifest == prepare(raw_data)


def test_checksum_rejects_modified_test(raw_data):
    prepare(raw_data)
    rows = load_partition(raw_data, "test")
    rows[0, 2] = 99
    np.save(raw_data / "test.npy", rows)
    with pytest.raises(ValueError, match="Integrity"):
        load_partition(raw_data, "test")


def test_cold_start_counted(raw_data):
    rows = np.loadtxt(raw_data / "u.data", dtype=int)
    rows[-1, 0] = 1000
    np.savetxt(raw_data / "u.data", rows, fmt="%d")
    assert prepare(raw_data)["partitions"]["test"]["excluded_cold_start"] == 1


def test_duplicate_rejected(raw_data):
    first = (raw_data / "u.data").read_text().splitlines()[0]
    with (raw_data / "u.data").open("a") as file:
        file.write(first + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(raw_data)


def test_ties_not_split(raw_data):
    rows = np.loadtxt(raw_data / "u.data", dtype=int)
    rows[:, 3] = (rows[:, 3] // 20) * 20
    np.savetxt(raw_data / "u.data", rows, fmt="%d")
    prepare(raw_data)
    tr, va = load_partition(raw_data, "train"), load_partition(raw_data, "valid")
    assert tr[:, 3].max() < va[:, 3].min()
