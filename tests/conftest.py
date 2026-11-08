import numpy as np
import pytest


@pytest.fixture
def raw_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(4)
    pairs = [(u, i) for u in range(1, 25) for i in range(1, 41)]
    rng.shuffle(pairs)
    rows = [[u, i, int(rng.integers(1, 6)), 100000 + t] for t, (u, i) in enumerate(pairs)]
    np.savetxt(root / "u.data", rows, fmt="%d", delimiter="\t")
    (root / "u.item").write_text(
        "\n".join(
            f"{i}|Synthetic {i}|01-Jan-2000||example|"
            + "|".join("1" if j == i % 19 else "0" for j in range(19))
            for i in range(1, 41)
        )
    )
    return root
