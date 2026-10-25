"""Column-projected deterministic loading for data frames, CSV and Parquet."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import SchemaConfig


DataSource = pd.DataFrame | str | Path


@dataclass
class DatasetAudit:
    role: str
    source_type: str
    path: str | None
    format: str
    checksum_sha256: str | None
    full_size: int
    evaluated_size: int
    available_columns: list[str]
    projected_columns: list[str]
    sampling_method: str
    sampling_seed: int | None
    sample_index_sha256: str | None
    sampling_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    audit: DatasetAudit


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_columns(source: DataSource) -> list[str]:
    if isinstance(source, pd.DataFrame):
        return [str(item) for item in source.columns]
    path = Path(source).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return [str(item) for item in pd.read_csv(path, sep=separator, nrows=0).columns]
    raise ValueError(f"Unsupported data format: {suffix or '<none>'}")


def _selected_indices(full_size: int, maximum: int, seed: int) -> np.ndarray:
    if full_size <= maximum:
        return np.arange(full_size, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(full_size, size=maximum, replace=False).astype(np.int64))


def _index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def _sample_parquet(
    path: Path, columns: list[str], maximum: int, seed: int, batch_size: int
) -> tuple[pd.DataFrame, int, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    full_size = parquet.metadata.num_rows
    indices = _selected_indices(full_size, maximum, seed)
    if full_size <= maximum:
        table = parquet.read(columns=columns)
        return table.to_pandas(), full_size, indices
    pieces: list[pa.RecordBatch] = []
    offset = 0
    left = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        right = int(np.searchsorted(indices, offset + batch.num_rows, side="left"))
        if right > left:
            local = indices[left:right] - offset
            pieces.append(batch.take(pa.array(local)))
        offset += batch.num_rows
        left = right
        if left >= len(indices):
            break
    table = pa.Table.from_batches(pieces)
    return table.to_pandas(), full_size, indices


def _sample_csv(
    path: Path, columns: list[str], maximum: int, seed: int, batch_size: int
) -> tuple[pd.DataFrame, int, np.ndarray]:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    full_size = sum(len(chunk) for chunk in pd.read_csv(
        path, sep=separator, usecols=columns, chunksize=batch_size
    ))
    indices = _selected_indices(full_size, maximum, seed)
    if full_size <= maximum:
        return pd.read_csv(path, sep=separator, usecols=columns), full_size, indices
    pieces: list[pd.DataFrame] = []
    offset = 0
    left = 0
    for chunk in pd.read_csv(path, sep=separator, usecols=columns, chunksize=batch_size):
        right = int(np.searchsorted(indices, offset + len(chunk), side="left"))
        if right > left:
            local = indices[left:right] - offset
            pieces.append(chunk.iloc[local])
        offset += len(chunk)
        left = right
        if left >= len(indices):
            break
    return pd.concat(pieces, ignore_index=True), full_size, indices


def load_dataset(
    source: DataSource,
    role: str,
    schema: SchemaConfig,
    maximum_rows: int,
    seed: int,
    batch_size: int = 65536,
    calculate_checksum: bool = True,
) -> LoadedDataset:
    """Load only declared columns and disclose any deterministic sampling."""

    available = inspect_columns(source)
    projected = [item for item in schema.all_features if item in available]
    if not projected:
        raise ValueError(f"No declared schema features are available for role {role}")
    if isinstance(source, pd.DataFrame):
        full_size = len(source)
        indices = _selected_indices(full_size, maximum_rows, seed)
        frame = source.iloc[indices][projected].reset_index(drop=True).copy()
        path_text = None
        checksum = None
        source_type = "dataframe"
        format_name = "in_memory"
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            frame, full_size, indices = _sample_parquet(
                path, projected, maximum_rows, seed, batch_size
            )
            format_name = "parquet"
        elif suffix in {".csv", ".tsv"}:
            frame, full_size, indices = _sample_csv(
                path, projected, maximum_rows, seed, batch_size
            )
            format_name = "tsv" if suffix == ".tsv" else "csv"
        else:
            raise ValueError(f"Unsupported data format: {suffix or '<none>'}")
        path_text = str(path)
        checksum = sha256_file(path) if calculate_checksum else None
        source_type = "path"
    sampled = full_size > len(frame)
    audit = DatasetAudit(
        role=role,
        source_type=source_type,
        path=path_text,
        format=format_name,
        checksum_sha256=checksum,
        full_size=int(full_size),
        evaluated_size=int(len(frame)),
        available_columns=available,
        projected_columns=projected,
        sampling_method="uniform_without_replacement" if sampled else "full_data",
        sampling_seed=int(seed) if sampled else None,
        sample_index_sha256=_index_digest(indices),
        sampling_reason="configured row cap for bounded memory" if sampled else None,
    )
    return LoadedDataset(frame=frame, audit=audit)
