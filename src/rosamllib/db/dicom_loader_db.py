from __future__ import annotations

import os
import queue
import logging
import threading
from pathlib import Path
from typing import Iterable, Optional, Union, Dict, Any

import pydicom
from tqdm import tqdm
from rosamllib.db.db_nodes import DatasetNodeDB, DatasetRow
from rosamllib.utils.db_utils import (
    CORE_TAGS,
    _normalize_tag,
    build_tag_plan,
    start_pipeline,
    stop_pipeline,
    ParseJob,
    setup_module_logging,
)
from rosamllib.db.db_schema import init_schema

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

_STOP = object()


def _count_files(root: Union[str, Path], *, followlinks: bool = False) -> int:
    root = str(root)
    total = 0
    for _, _, files in os.walk(root, followlinks=followlinks):
        total += len(files)
    return total


def _iter_paths(root: Union[str, Path], *, followlinks: bool = False):
    root = str(root)
    for dirpath, _, files in os.walk(root, followlinks=followlinks):
        for fn in files:
            yield os.path.join(dirpath, fn)


def _load_config(config_or_path: Union[dict, str]) -> dict:
    if isinstance(config_or_path, dict):
        return dict(config_or_path)
    import json

    with open(config_or_path, "r", encoding="utf-8") as f:
        return json.load(f)


class DICOMLoaderDB:

    def __init__(self, dicom_path: Union[str, Path]):
        self.dicom_path = Path(dicom_path)
        if not self.dicom_path.exists() or not self.dicom_path.is_dir():
            raise FileNotFoundError(f"Not a directory: {self.dicom_path}")

        self.dataset: Optional[DatasetNodeDB] = None

    def load(self, config_or_path: Union[dict, str]) -> None:
        self.cfg: Dict[str, Any] = _load_config(config_or_path)

        # minimal required
        for k in ("db-url", "dataset-id"):
            if k not in self.cfg:
                raise ValueError(f"Config missing required key: {k}")

        # defaults
        self.cfg.setdefault("dataset-name", "Directory DICOMs")
        self.cfg.setdefault("source", None)
        self.cfg.setdefault("force-json-tags", [])
        self.cfg.setdefault("tags-to-index", [])
        self.cfg.setdefault("follow-symlinks", False)
        self.cfg.setdefault("queue-size", 2000)

        # concurrency knobs
        cpu = os.cpu_count() or 8
        self.cfg.setdefault("read-workers", min(32, cpu * 2))
        self.cfg.setdefault("parse-workers", max(4, cpu // 2))
        self.cfg.setdefault("write-commit-every", 500)

        # dcmread knobs
        self.cfg.setdefault("stop-before-pixels", True)
        self.cfg.setdefault("force", False)

        # filtering
        self.cfg.setdefault("include-modalities", [])
        self.cfg.setdefault("exclude-modalities", [])

        # post processing
        self.cfg.setdefault("rebuild-associations", True)

        dicom_root = self.dicom_path

        db_url = self.cfg["db-url"]
        dataset_id = self.cfg["dataset-id"]
        dataset_name = self.cfg["dataset-name"]
        force_json_tags = self.cfg["force-json-tags"]

        followlinks = bool(self.cfg["follow-symlinks"])
        stop_before_pixels = bool(self.cfg["stop-before-pixels"])
        force = bool(self.cfg["force"])

        include_modalities = set(self.cfg.get("include-modalities") or [])
        exclude_modalities = set(self.cfg.get("exclude-modalities") or [])

        source = self.cfg.get("source") or f"LOCAL:{dicom_root}"

        # --- Build tag plan (core tags + extras) ---
        default_tags = {_normalize_tag(t) for t in CORE_TAGS}
        default_tags = {t for t in default_tags if t}

        extras = set()
        for t in self.cfg.get("tags-to-index") or []:
            nt = _normalize_tag(t)
            if nt:
                extras.add(nt)

        tags_to_index = list(default_tags | extras)

        tag_plan = build_tag_plan(tags_to_index)

        engine = create_engine(db_url, future=True)
        init_schema(engine, tag_plan=tag_plan, force_json_tags=force_json_tags)

        SessionLocal = sessionmaker(bind=engine, future=True)

        # ensure dataset row exists
        with SessionLocal() as session:
            ds_row = session.get(DatasetRow, dataset_id)
            if ds_row is None:
                ds_row = DatasetRow(dataset_id=dataset_id, dataset_name=dataset_name)
                session.add(ds_row)
                session.commit()
            elif dataset_name and ds_row.dataset_name != dataset_name:
                ds_row.dataset_name = dataset_name
                session.commit()

        dataset_db = DatasetNodeDB.from_engine(engine, dataset_id)

        pipeline = start_pipeline(
            SessionLocal,
            parse_workers=int(self.cfg["parse-workers"]),
            write_commit_every=int(self.cfg["write-commit-every"]),
            tag_plan=tag_plan,
        )

        # directory streaming
        total_files = _count_files(dicom_root, followlinks=followlinks)
        logger.info("DICOMLoaderDB: discovered %d files under %s", total_files, dicom_root)

        path_q: queue.Queue = queue.Queue(maxsize=int(self.cfg["queue-size"]))
        progress_q: queue.Queue = queue.Queue()

        read_workers = int(self.cfg["read-workers"])

        def feeder():
            try:
                for fp in _iter_paths(dicom_root, followlinks=followlinks):
                    path_q.put(fp)
            finally:
                for _ in range(read_workers):
                    path_q.put(_STOP)

        def reader_worker():
            while True:
                fp = path_q.get()

                if fp is _STOP:
                    path_q.task_done()
                    return

                try:
                    # attempt read (counts even if fail)
                    try:
                        ds = pydicom.dcmread(
                            fp,
                            stop_before_pixels=stop_before_pixels,
                            force=force,
                        )
                    except Exception:
                        continue

                    mod = getattr(ds, "Modality", None)
                    if include_modalities and mod not in include_modalities:
                        continue
                    if exclude_modalities and mod in exclude_modalities:
                        continue

                    pipeline.parse_q.put(
                        ParseJob(
                            dcm=ds,
                            query_ae=str(source),
                            dataset_id=dataset_id,
                            filepath=str(fp),
                        )
                    )
                finally:
                    progress_q.put(1)
                    path_q.task_done()

        feeder_t = threading.Thread(target=feeder, daemon=True)
        reader_ts = [
            threading.Thread(target=reader_worker, daemon=True) for _ in range(read_workers)
        ]
        feeder_t.start()
        for t in reader_ts:
            t.start()

        # tqdm loop
        pbar = tqdm(total=total_files, desc="Ingest directory", unit="file")
        attempted = 0
        try:
            while attempted < total_files:
                try:
                    inc = progress_q.get(timeout=0.25)
                    attempted += inc
                    pbar.update(inc)
                    progress_q.task_done()
                except queue.Empty:
                    pass
        finally:
            pbar.close()

        # make sure producer queue drained
        path_q.join()

        for t in reader_ts:
            t.join()

        # stop pipeline (flush parse/write)
        stop_pipeline(pipeline)

        feeder_t.join()

        # post: build association edges
        if bool(self.cfg.get("rebuild-associations", True)):
            dataset_db.associate_dicoms(rebuild=True)

        try:
            engine.dispose()
        except Exception:
            pass

        self.dataset = dataset_db

    def get_modality_distribution(
        self,
        *,
        all_instance_level: bool = False,
        force_instance_level_modalities: Iterable[str] = (),
        unknown_label: str = "Unknown",
    ) -> dict[str, int]:
        """
        Convenience wrapper around DatasetNodeDB.get_modality_distribution().

        Raises
        ------
        RuntimeError
            If load() has not been called yet (self.dataset is None).
        """
        if self.dataset is None:
            raise RuntimeError(
                "DICOMLoaderDB.get_modality_distribution() called before load(). "
                "Run load(...) first so self.dataset is initialized."
            )

        return self.dataset.get_modality_distribution(
            all_instance_level=all_instance_level,
            force_instance_level_modalities=force_instance_level_modalities,
            unknown_label=unknown_label,
        )

    def get_patient(self, PatientID: str):
        return self._require_dataset().get_patient(PatientID)

    def get_study(self, StudyInstanceUID: str, *, PatientID: Optional[str] = None):
        return self._require_dataset().get_study(StudyInstanceUID, PatientID=PatientID)

    def get_series(
        self,
        SeriesInstanceUID: str,
        *,
        PatientID: Optional[str] = None,
        StudyInstanceUID: Optional[str] = None,
    ):
        return self._require_dataset().get_series(
            SeriesInstanceUID, PatientID=PatientID, StudyInstanceUID=StudyInstanceUID
        )

    def get_instance(self, sop_uid: str):
        return self._require_dataset().get_instance(sop_uid)

    def n_patients(self) -> int:
        return self._require_dataset().n_patients()

    def n_studies(self, **kwargs) -> int:
        return self._require_dataset().n_studies(**kwargs)

    def n_series(self, **kwargs) -> int:
        return self._require_dataset().n_series(**kwargs)

    def n_instances(self, **kwargs) -> int:
        return self._require_dataset().n_instances(**kwargs)

    def iter_patients(self):
        return self._require_dataset().iter_patients()

    def iter_studies(self):
        return self._require_dataset().iter_studies()

    def iter_series(self):
        return self._require_dataset().iter_series()

    def iter_instances(self):
        return self._require_dataset().iter_instances()

    def __iter__(self):
        return self.iter_patients()

    def __len__(self) -> int:
        return len(self._require_dataset())

    def _require_dataset(self) -> DatasetNodeDB:
        if self.dataset is None:
            raise RuntimeError(
                "DICOMLoaderDB dataset is not initialized. " "Call load(config_or_path) first."
            )
        return self.dataset

    @classmethod
    def configure_logging(
        cls, *, log_file_path=None, log_to_console=True, level=logging.INFO, **kw
    ):
        if log_file_path is None and not log_to_console:
            # No handlers desired; user might configure root logging externally.
            logging.getLogger(__name__).setLevel(level)
            return

        setup_module_logging(
            log_file_path=log_file_path or "dicom_loader_db.log",
            level=level,
            log_to_console=log_to_console,
            logger_name=__name__,  # <— must be supported
            **kw,
        )
