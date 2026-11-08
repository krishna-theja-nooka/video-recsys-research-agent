"""Deterministic global-time splits; raw data stays local."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"


def digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def download(root, accept_license=False):
    if not accept_license:
        raise ValueError("Read GroupLens terms and supply --accept-license for research use.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ml-100k.zip"
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                with temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Corrupt dataset archive")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 mismatch; do not extract unverified data")
    with zipfile.ZipFile(path) as archive:
        # Explicit files only: no zip path traversal and no user demographics.
        for name in ("u.data", "u.item", "README"):
            (root / name).write_bytes(archive.read("ml-100k/" + name))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "download.json").write_text(json.dumps({"url": URL, "sha256": checksum}, indent=2))
    return checksum


def read_rows(root, engine="python"):
    path = Path(root) / "u.data"
    if engine == "python":
        rows = np.loadtxt(path, dtype=np.int64)
    elif engine == "spark":
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[2]").appName("RecLab ETL").getOrCreate()
        try:
            frame = spark.read.csv(
                str(path.resolve()), sep="\t", schema="user INT, item INT, rating INT, ts LONG"
            )
            from pyspark.sql import functions as F

            invalid = frame.filter(
                F.col("user").isNull()
                | F.col("item").isNull()
                | F.col("rating").isNull()
                | F.col("ts").isNull()
                | ~F.col("rating").between(1, 5)
                | (F.col("user") <= 0)
                | (F.col("item") <= 0)
                | (F.col("ts") <= 0)
            ).count()
            if invalid:
                raise ValueError("Invalid rows in Spark ingestion")
            if frame.groupBy("user", "item").count().filter("count > 1").count():
                raise ValueError("Duplicate user-item interactions")
            profile = (
                frame.agg(
                    F.count("*").alias("rows"),
                    F.countDistinct("user").alias("users"),
                    F.countDistinct("item").alias("items"),
                )
                .first()
                .asDict()
            )
            (Path(root) / "spark-profile.json").write_text(json.dumps(profile, indent=2))
            frame.write.mode("overwrite").parquet(
                str((Path(root) / "spark-interactions").resolve())
            )
            # This pilot then collects 100K rows for CPU training; not a distributed trainer.
            rows = np.array([tuple(r) for r in frame.collect()], dtype=np.int64)
        finally:
            spark.stop()
    else:
        raise ValueError("engine must be python or spark")
    if rows.ndim != 2 or rows.shape[1] != 4 or len(rows) < 10:
        raise ValueError("Expected user, item, rating, timestamp columns")
    if not np.isin(rows[:, 2], [1, 2, 3, 4, 5]).all() or (rows[:, [0, 1, 3]] <= 0).any():
        raise ValueError("Invalid interaction")
    if len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError("Duplicate user-item interactions")
    return rows[np.lexsort((rows[:, 1], rows[:, 0], rows[:, 3]))]


def prepare(root, engine="python"):
    root = Path(root)
    rows = read_rows(root, engine)
    t1, t2 = rows[int(len(rows) * 0.8), 3], rows[int(len(rows) * 0.9), 3]
    train = rows[rows[:, 3] < t1]
    valid = rows[(rows[:, 3] >= t1) & (rows[:, 3] < t2)]
    test = rows[rows[:, 3] >= t2]
    users = {int(v): k for k, v in enumerate(np.unique(train[:, 0]))}
    items = {int(v): k for k, v in enumerate(np.unique(train[:, 1]))}
    if not len(train) or not len(valid) or not len(test):
        raise ValueError("Need three nonempty temporal partitions")
    meta = {
        "raw_rows": len(rows),
        "raw_sha256": digest(rows),
        "engine": engine,
        "cutoffs": [int(t1), int(t2)],
        "users": len(users),
        "items": len(items),
        "positive_label": "rating >= 4",
        "partitions": {},
    }
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        retained = np.array([r for r in part if int(r[0]) in users and int(r[1]) in items])
        if not len(retained):
            raise ValueError(f"No warm-start rows in {name}")
        mapped = np.array(
            [[users[int(u)], items[int(i)], int(r), int(t)] for u, i, r, t in retained],
            dtype=np.int64,
        )
        np.save(root / f"{name}.npy", mapped, allow_pickle=False)
        meta["partitions"][name] = {
            "raw": len(part),
            "retained": len(mapped),
            "excluded_cold_start": len(part) - len(mapped),
            "sha256": digest(mapped),
        }
    genre_rows = {}
    for line in (root / "u.item").read_text(encoding="latin-1").splitlines():
        cols = line.split("|")
        genre_rows[int(cols[0])] = [j + 1 for j, flag in enumerate(cols[5:]) if flag == "1"]
    genres = np.zeros((len(items), 19), dtype=np.int64)
    for item, index in items.items():
        tokens = genre_rows[item] or [1]
        genres[index, : len(tokens)] = tokens
    genres = genres[:, : max(1, int((genres != 0).sum(1).max()))]
    np.save(root / "genres.npy", genres, allow_pickle=False)
    meta["genres_sha256"] = digest(genres)
    (root / "manifest.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_partition(root, name):
    if name not in {"train", "valid", "test"}:
        raise ValueError("Unknown partition")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = np.load(root / f"{name}.npy", allow_pickle=False)
    if digest(rows) != manifest["partitions"][name]["sha256"]:
        raise ValueError(f"Integrity check failed for {name}")
    return rows


def load_genres(root):
    root = Path(root)
    genres = np.load(root / "genres.npy", allow_pickle=False)
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(genres) != manifest["genres_sha256"]:
        raise ValueError("Genre integrity check failed")
    return genres
