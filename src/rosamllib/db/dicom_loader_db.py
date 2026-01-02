from __future__ import annotations

import os
import queue
import logging
import threading
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict, Any

import pydicom
from tqdm import tqdm
from rosamllib.db.db_nodes import DatasetNodeDB
from rosamllib.utils.db_utils import (
    CORE_TAGS,
    _normalize_tag,
    build_tag_plan,
    start_pipeline,
    stop_pipeline,
    ParseJob,
    setup_module_logging,
)
from rosamllib.db.db_schema import (
    init_schema,
    DatasetRow,
    PatientRow,
    StudyRow,
    SeriesRow,
    InstanceRow,
)

from sqlalchemy import create_engine, select, and_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import Select


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

    def query(
        self,
        query_level: str = "INSTANCE",
        *,
        include: Optional[Iterable[str]] = None,
        case_insensitive: bool = False,
        sort_by: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        **filters,
    ) -> "pd.DataFrame":
        """
        Fast SQLite-first query using the instances table as the base.

        Design (SQLite):
        - Always filter from InstanceRow (fact table).
        - PATIENT/STUDY/SERIES use SQL DISTINCT on ID columns (no pandas dedup).
        - include columns can come from:
            * InstanceRow (INSTANCE level only, plus ID cols at higher levels)
            * PatientRow (e.g., PatientName)
            * StudyRow   (e.g., StudyDescription)  [only when StudyInstanceUID is in level]
            * SeriesRow  (e.g., SeriesDescription) [only when SeriesInstanceUID is in level]
        - For SERIES-level include of "Modality", we prefer SeriesRow.Modality.
        - Regex filtering is NOT supported for SQLite fast path.
        - Wildcards: "*" -> SQL LIKE "%", excludes NULLs by default.
        - None filter value means IS NULL.

        Parameters
        ----------
        query_level : {'PATIENT','STUDY','SERIES','INSTANCE'}
        include : optional iterable[str]
            Extra columns to include in output.
            Higher-level table columns are joined after DISTINCT IDs.
        case_insensitive : bool
            For string equality and LIKE.
        sort_by : optional iterable[str]
            Sort keys (only those in the final selected columns are applied).
        limit : optional int
            LIMIT in SQL.
        **filters
            Column filters applied on InstanceRow columns only (including dynamic tag columns).

        Returns
        -------
        pandas.DataFrame
        """

        if self.dataset is None:
            raise RuntimeError("Call load(config_or_path) first.")
        ds = self.dataset

        LEVEL_COLS = {
            "PATIENT": ["PatientID"],
            "STUDY": ["PatientID", "StudyInstanceUID"],
            "SERIES": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
            "INSTANCE": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"],
        }

        lvl = str(query_level).upper()
        if lvl not in LEVEL_COLS:
            raise ValueError(
                f"Invalid query_level '{query_level}'. Must be one of {list(LEVEL_COLS)}."
            )

        include = list(include or [])
        level_cols = LEVEL_COLS[lvl]

        # -------------------------
        # Helpers for filters
        # -------------------------
        inst_cols = InstanceRow.__table__.columns

        def _inst_col(name: str):
            if name not in inst_cols:
                raise KeyError(
                    f"Filter column {name!r} is not a column on InstanceRow. "
                    "SQLite-fast query() v1 supports filtering on instance columns only."
                )
            return inst_cols[name]

        def _like_pattern(s: str) -> str:
            # Escape LIKE metacharacters and convert '*' to '%'
            return (
                s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("*", "%")
            )

        def _cmp(expr, value):
            if value is None:
                return expr.is_(None)

            if isinstance(value, str) and "*" in value:
                pat = _like_pattern(value)
                if case_insensitive:
                    return func.lower(expr).like(func.lower(pat), escape="\\")
                return expr.like(pat, escape="\\")

            if case_insensitive and isinstance(value, str):
                return func.lower(expr) == value.lower()

            return expr == value

        def _compile_filter(expr, value):
            """
            Supports:
            - scalar
            - iterable -> IN
            - dict ops: in, notin, gt,gte,lt,lte, exists
            """
            if isinstance(value, dict):
                clauses = []

                if "RegEx" in value or "regex" in value:
                    raise NotImplementedError(
                        "RegEx filtering not supported for SQLite-fast query()."
                    )

                if "exists" in value:
                    ex = bool(value["exists"])
                    clauses.append(expr.is_not(None) if ex else expr.is_(None))

                if "in" in value:
                    clauses.append(expr.in_(list(value["in"] or [])))
                if "notin" in value:
                    clauses.append(expr.notin_(list(value["notin"] or [])))
                if "gt" in value:
                    clauses.append(expr > value["gt"])
                if "gte" in value:
                    clauses.append(expr >= value["gte"])
                if "lt" in value:
                    clauses.append(expr < value["lt"])
                if "lte" in value:
                    clauses.append(expr <= value["lte"])

                if not clauses:
                    raise ValueError(f"Unsupported filter dict for {expr}: {value}")
                return clauses

            if isinstance(value, (list, tuple, set)):
                return [expr.in_(list(value))]

            return [_cmp(expr, value)]

        # -------------------------
        # Build WHERE (instances)
        # -------------------------
        where_clauses = [InstanceRow.dataset_id == ds.dataset_id]
        for name, value in filters.items():
            expr = _inst_col(name)
            where_clauses.extend(_compile_filter(expr, value))

        # -------------------------
        # Classify include columns by preferred owner table
        # (prefer higher-level tables to avoid ambiguity)
        # -------------------------
        patient_cols = set(PatientRow.__table__.columns.keys())
        study_cols = set(StudyRow.__table__.columns.keys())
        series_cols = set(SeriesRow.__table__.columns.keys())
        inst_colnames = set(inst_cols.keys())

        def _preferred_owner(col: str) -> str:
            if col in patient_cols:
                return "patient"
            if col in study_cols:
                return "study"
            if col in series_cols:
                return "series"
            if col in inst_colnames:
                return "instance"
            raise KeyError(
                f"include column {col!r} not found on PatientRow/StudyRow/SeriesRow/InstanceRow."
            )

        include_patient: list[str] = []
        include_study: list[str] = []
        include_series: list[str] = []
        include_instance: list[str] = []

        for c in include:
            owner = _preferred_owner(c)

            # SPECIAL: SERIES-level "Modality" should come from SeriesRow if it exists.
            if lvl == "SERIES" and c == "Modality" and c in series_cols:
                owner = "series"

            if owner == "patient":
                include_patient.append(c)
            elif owner == "study":
                include_study.append(c)
            elif owner == "series":
                include_series.append(c)
            else:
                include_instance.append(c)

        # Disallow including instance-level non-ID columns at higher levels (ambiguous)
        if lvl != "INSTANCE":
            bad_inst = [c for c in include_instance if c not in level_cols]
            msg = (
                f"At query_level={lvl}, include columns {bad_inst} are instance-level columns. "
                "This is ambiguous because values can vary within a patient/study/series group. "
                "Include them only at query_level='INSTANCE' or include higher-level fields "
                "(PatientName/StudyDescription/SeriesDescription/Modality from SeriesRow, etc.)."
            )
            if bad_inst:
                raise ValueError(msg)

        # -------------------------
        # Build statement
        # -------------------------
        ids_exprs = [inst_cols[c].label(c) for c in level_cols]

        if lvl == "INSTANCE":
            # select IDs + any extra instance includes (excluding duplicate ID cols)
            extra_inst = [c for c in include_instance if c not in level_cols]
            select_exprs = ids_exprs + [inst_cols[c].label(c) for c in extra_inst]
            stmt: Select = select(*select_exprs).where(and_(*where_clauses))

        else:
            # DISTINCT IDs from instances
            ids_stmt: Select = select(*ids_exprs).where(and_(*where_clauses)).distinct()
            ids_sq = ids_stmt.subquery("ids")

            # base select from ids
            select_exprs = [ids_sq.c[c].label(c) for c in level_cols]
            stmt = select(*select_exprs).select_from(ids_sq)

            # Join patient columns (if requested)
            if include_patient:
                stmt = stmt.join(
                    PatientRow,
                    and_(
                        PatientRow.dataset_id == ds.dataset_id,
                        PatientRow.PatientID == ids_sq.c.PatientID,
                    ),
                    isouter=True,
                )
                for c in include_patient:
                    stmt = stmt.add_columns(getattr(PatientRow, c).label(c))

            # Join study columns (if requested)
            if include_study:
                if "StudyInstanceUID" not in level_cols:
                    raise ValueError(
                        f"Cannot include study columns {include_study} at level {lvl}."
                    )
                stmt = stmt.join(
                    StudyRow,
                    and_(
                        StudyRow.dataset_id == ds.dataset_id,
                        StudyRow.StudyInstanceUID == ids_sq.c.StudyInstanceUID,
                    ),
                    isouter=True,
                )
                for c in include_study:
                    stmt = stmt.add_columns(getattr(StudyRow, c).label(c))

            # Join series columns (if requested)
            if include_series:
                if "SeriesInstanceUID" not in level_cols:
                    raise ValueError(
                        f"Cannot include series columns {include_series} at level {lvl}."
                    )
                stmt = stmt.join(
                    SeriesRow,
                    and_(
                        SeriesRow.dataset_id == ds.dataset_id,
                        SeriesRow.SeriesInstanceUID == ids_sq.c.SeriesInstanceUID,
                    ),
                    isouter=True,
                )
                for c in include_series:
                    stmt = stmt.add_columns(getattr(SeriesRow, c).label(c))

        # -------------------------
        # ORDER BY / LIMIT
        # -------------------------
        sort_keys = list(sort_by) if sort_by else list(level_cols)

        # Only keep sort columns that exist in the final selected columns
        final_cols = list(stmt.selected_columns.keys())
        sort_keys = [c for c in sort_keys if c in final_cols]

        if sort_keys:
            order_exprs = [stmt.selected_columns[c] for c in sort_keys]
            stmt = stmt.order_by(*order_exprs)

        if limit is not None:
            stmt = stmt.limit(int(limit))

        # -------------------------
        # Execute
        # -------------------------
        with ds._session() as s:
            rows = s.execute(stmt).all()

        colnames = list(stmt.selected_columns.keys())
        return pd.DataFrame(rows, columns=colnames)

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

    @classmethod
    def list_datasets(cls, db_url: str) -> List[dict]:
        """
        Convenience wrapper for DatasetNodeDB.list_datasets(db_url).
        """
        return DatasetNodeDB.list_datasets(db_url)

    @classmethod
    def from_existing_dataset(
        cls,
        *,
        db_url: str,
        dataset_id: Optional[str] = None,
        seq_policy: str = "json",
    ) -> "DICOMLoaderDB":
        """
        Create a DICOMLoaderDB instance without ingesting a directory, by attaching
        an existing DatasetNodeDB.
        """
        obj = cls(dicom_path=".")
        obj.dataset = DatasetNodeDB.open_existing(
            db_url, dataset_id=dataset_id, seq_policy=seq_policy
        )
        return obj
