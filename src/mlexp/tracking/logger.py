from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("mlexp")


@dataclass
class ExperimentRecord:
    run_id: str
    model_name: str
    search_strategy: str
    evaluation_strategy: str
    best_params: dict[str, Any]
    mean_score: float
    std_score: float
    outer_scores: list[float]
    best_params_per_fold: list[dict[str, Any]]
    # {metric_name: {"mean": float, "std": float, "scores": [float, ...]}}
    additional_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentLogger:
    """
    Lightweight experiment tracker that writes JSON-line records to a log file.

    Designed to be drop-in compatible with MLflow's mental model:
      - start_run / end_run context
      - log_params, log_metrics
      - final record persisted to disk

    If you want to swap in real MLflow later, replace this class with an
    MLflow-backed implementation behind the same interface.
    """

    def __init__(self, log_dir: str | Path = "mlexp_runs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_log = self.log_dir / "runs.jsonl"
        self._current: ExperimentRecord | None = None

    # ------------------------------------------------------------------
    def log(self, record: ExperimentRecord) -> None:
        """Persist a completed ExperimentRecord."""
        with self._run_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(
            "run_id=%s  model=%s  %s=%.4f±%.4f",
            record.run_id,
            record.model_name,
            record.evaluation_strategy,
            record.mean_score,
            record.std_score,
        )

    def load_all(self) -> list[ExperimentRecord]:
        """Read all persisted runs from disk."""
        records: list[ExperimentRecord] = []
        if not self._run_log.exists():
            return records
        with self._run_log.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(ExperimentRecord(**data))
        return records

    def best_run(
        self, model_name: str | None = None
    ) -> ExperimentRecord | None:
        """Return the run with the highest mean_score, optionally filtered."""
        runs = self.load_all()
        if model_name:
            runs = [r for r in runs if r.model_name == model_name]
        if not runs:
            return None
        return max(runs, key=lambda r: r.mean_score)
