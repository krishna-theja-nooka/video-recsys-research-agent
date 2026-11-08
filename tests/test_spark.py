import os

import pytest

from reclab.data import prepare


@pytest.mark.skipif(os.getenv("RUN_SPARK_TESTS") != "1", reason="Opt-in Java/Spark integration")
def test_spark_python_split_parity(raw_data):
    python = prepare(raw_data)
    spark = prepare(raw_data, "spark")
    assert python["partitions"] == spark["partitions"]
    assert python["raw_sha256"] == spark["raw_sha256"]
    assert python["genres_sha256"] == spark["genres_sha256"]
    assert (raw_data / "spark-interactions").exists()
