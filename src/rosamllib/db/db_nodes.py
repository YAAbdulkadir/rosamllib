from __future__ import annotations

from collections import deque
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Set,
    Union,
    ClassVar,
    Dict,
    ContextManager,
)

from sqlalchemy import create_engine, select, func, delete
from sqlalchemy.orm import Session, sessionmaker

from rosamllib.db.db_schema import (
    DatasetRow,
    PatientRow,
    StudyRow,
    SeriesRow,
    InstanceRow,
    SeriesReferenceRow,
    InstanceReferenceRow,
    init_schema,
    TagPlan,
    build_tag_plan,
    load_patient,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


SessionFactory = Callable[[], Session]


class JSONListFieldMixin:
    """
    Mixin for DB-node facades that need list-JSON field helpers.

    Subclasses must provide:
      - _ROW_CLS: the ORM row class (PatientRow, StudyRow, ...)
      - _LIST_FIELD_MAP: mapping from logical name -> ORM column name
      - _load_row(session: Session) -> row instance
      - _session() -> context manager yielding Session (already in your _DBNodeBase)
    """

    _ROW_CLS: ClassVar[type] = None
    _LIST_FIELD_MAP: ClassVar[Dict[str, str]] = {}

    # ---- internal helpers ----

    def _resolve_list_field(self, field_name: str) -> str:
        """
        Normalize field names so user can pass fields without '_json'.
        Accepts logical names like 'sources' and actual columns like 'sources_json'.
        """
        # Friendly name
        if field_name in self._LIST_FIELD_MAP:
            return self._LIST_FIELD_MAP[field_name]

        # Direct column name
        if field_name.endswith("_json") and field_name in self._ROW_CLS.__table__.columns:
            return field_name

        raise AttributeError(
            f"Unknown list field '{field_name}'. "
            f"Valid names: {list(self._LIST_FIELD_MAP.keys())}"
        )

    def _load_row(self, session: Session):
        """
        Must be implemented by subclasses to load the appropriate ORM row
        (e.g., by dataset_id + PatientID, or dataset_id + SOPInstanceUID).
        """
        raise NotImplementedError

    def _mutate_list_field(self, field_name: str, mutator) -> None:
        """
        Internal helper: load row, apply mutator(list), write back if changed.
        """
        db_field = self._resolve_list_field(field_name)

        with self._session() as s:  # _session provided by your _DBNodeBase
            row = self._load_row(s)
            if row is None:
                raise ValueError(f"{type(self).__name__}: backing row not found")

            current = getattr(row, db_field) or []
            lst = list(current)

            before = list(lst)
            mutator(lst)

            # No change -> no commit
            if lst == before:
                return

            setattr(row, db_field, lst)
            s.commit()

    # ---- public API ----

    def append(self, field_name: str, value, *, unique: bool = True) -> None:
        """
        Append a single value to a list JSON field.

        Examples:
            inst_db.append("sources", "ORTHANC")
            inst_db.append("referenced_sop_uids", sop_uid)
        """
        if value is None:
            return

        def mutator(lst):
            if unique and value in lst:
                return
            lst.append(value)

        self._mutate_list_field(field_name, mutator)

    def extend(self, field_name: str, values: Iterable, *, unique: bool = True) -> None:
        """
        Extend a list JSON field with multiple values.

        Examples:
            inst_db.extend("sources", ["MIM", "ECLIPSE"])
        """
        values = list(values or [])
        if not values:
            return

        def mutator(lst):
            for v in values:
                if v is None:
                    continue
                if unique and v in lst:
                    continue
                lst.append(v)

        self._mutate_list_field(field_name, mutator)

    def remove(self, field_name: str, value) -> None:
        """
        Remove a single value from a list JSON field.
        Does nothing if the value is not present.
        """
        if value is None:
            return

        def mutator(lst):
            if value in lst:
                lst.remove(value)

        self._mutate_list_field(field_name, mutator)

    def clear(self, field_name: str) -> None:
        """
        Clear a list JSON field (set to empty list).
        """

        def mutator(lst):
            lst.clear()

        self._mutate_list_field(field_name, mutator)


# Base mixin for DB nodes


class _DBNodeBase:
    """
    Common base for all DB-backed node facades.

    Holds:
        - dataset_id
        - a session factory
    """

    def __init__(self, dataset_id: str, session_factory: SessionFactory):
        self.dataset_id = dataset_id
        self._session_factory = session_factory

    def _session(self) -> ContextManager[Session]:
        """
        Open a new Session. Use as:
            with self._session() as s:
                ...

        """
        return self._session_factory()


# DatasetNodeDB


class DatasetNodeDB(_DBNodeBase, JSONListFieldMixin):
    """
    DB-backed facade for a dataset, identified by `dataset_id`.

    Use the classmethods:
        - list_datasets()
        - open_existing()
        - create_new()
        - open_or_create()
    to construct instances from a database URL.
    """

    def __init__(
        self,
        dataset_id: str,
        session_factory: SessionFactory,
        *,
        dataset_name: str | None = None,
        tag_plan: list[TagPlan] | None = None,
        seq_policy: str = "json",
    ) -> None:
        super().__init__(dataset_id, session_factory)
        self.dataset_name = dataset_name
        self.tag_plan: list[TagPlan] | None = tag_plan
        self.seq_policy = seq_policy

    # Internal helpers
    @classmethod
    def _make_engine_and_factory(cls, db_url: str):
        engine = create_engine(db_url, future=True)
        factory = sessionmaker(bind=engine, future=True)
        return engine, factory

    @classmethod
    def from_engine(cls, engine: "Engine", dataset_id: str) -> "DatasetNodeDB":
        """
        Construct from an existing SQLAlchemy Engine and dataset_id.
        Assumes schema is already initialized.
        """
        factory = sessionmaker(bind=engine, future=True)
        return cls.from_session_factory(factory, dataset_id)

    @classmethod
    def from_session_factory(
        cls,
        session_factory: SessionFactory,
        dataset_id: str,
        *,
        tag_plan: list[TagPlan] | None = None,
        seq_policy: str = "json",
    ) -> "DatasetNodeDB":
        with session_factory() as s:
            ds_row = s.get(DatasetRow, dataset_id)
            if ds_row is None:
                raise ValueError(f"Dataset {dataset_id!r} not found in database")
            return cls(
                dataset_id=ds_row.dataset_id,
                dataset_name=ds_row.dataset_name,
                session_factory=session_factory,
                tag_plan=tag_plan,
                seq_policy=seq_policy,
            )

    # High-level entry points

    @classmethod
    def list_datasets(cls, db_url: str) -> List[dict]:
        """
        List all datasets present in the given database.

        Returns a list of dicts with keys: 'dataset_id', 'dataset_name'.
        """
        engine, factory = cls._make_engine_and_factory(db_url)
        init_schema(engine)

        with factory() as s:
            rows = s.scalars(select(DatasetRow)).all()
            return [
                {
                    "dataset_id": r.dataset_id,
                    "dataset_name": r.dataset_name,
                }
                for r in rows
            ]

    @classmethod
    def open_existing(
        cls,
        db_url: str,
        *,
        dataset_id: Optional[str] = None,
        seq_policy: str = "json",
    ) -> "DatasetNodeDB":
        """
        Open an existing dataset in the database.

        - If dataset_id is given, require that it exists.
        - If dataset_id is None:
            * If exactly one dataset exists, open it.
            * If 0 or >1 exist, raise ValueError.
        """
        engine, factory = cls._make_engine_and_factory(db_url)
        init_schema(engine)

        with factory() as s:
            if dataset_id is None:
                existing = s.scalars(select(DatasetRow)).all()
                if len(existing) == 0:
                    raise ValueError(
                        "No datasets found in DB. "
                        "Specify dataset_id or use open_or_create() to create one."
                    )
                if len(existing) > 1:
                    ids = [d.dataset_id for d in existing]
                    raise ValueError(
                        "Multiple datasets found in DB; please specify dataset_id. "
                        f"Available dataset_ids: {ids}"
                    )
                dataset_id = existing[0].dataset_id

            ds_row = s.get(DatasetRow, dataset_id)
            if ds_row is None:
                raise ValueError(f"Dataset {dataset_id!r} not found in DB.")

        return cls.from_session_factory(factory, dataset_id, tag_plan=None, seq_policy=seq_policy)

    @classmethod
    def create_new(
        cls,
        db_url: str,
        *,
        dataset_id: str,
        dataset_name: Optional[str] = None,
        overwrite: bool = False,
        tags_to_index: list[tuple[int, int]] | None = None,
        seq_policy: str = "json",
    ) -> "DatasetNodeDB":
        """
        Create a new dataset entry in the DB and return a DatasetNodeDB.

        If tags_to_index is provided, the instances table is extended with
        extra columns corresponding to those tags (plus CORE_TAGS), via
        `build_tag_plan` + `init_schema(engine, tag_plan=...)`.

        If a dataset with the given ID already exists:
            - overwrite=False: raise ValueError.
            - overwrite=True: update its dataset_name, but keep all data.
              (Schema changes from tags_to_index are still applied globally.)
        """
        engine, factory = cls._make_engine_and_factory(db_url)

        # Build tag plan and initialize schema with dynamic columns
        tag_plan = build_tag_plan(tags_to_index or [])
        init_schema(engine, tag_plan=tag_plan, seq_policy=seq_policy)

        with factory() as s:
            ds_row = s.get(DatasetRow, dataset_id)
            if ds_row is not None:
                if not overwrite:
                    raise ValueError(
                        f"Dataset {dataset_id!r} already exists. "
                        "Use overwrite=True or choose a different dataset_id."
                    )
                # Overwrite name only (do not clear data)
                if dataset_name is not None and ds_row.dataset_name != dataset_name:
                    ds_row.dataset_name = dataset_name
                    s.commit()
            else:
                ds_row = DatasetRow(dataset_id=dataset_id, dataset_name=dataset_name)
                s.add(ds_row)
                s.commit()

        return cls.from_session_factory(
            factory,
            dataset_id,
            tag_plan=tag_plan,
            seq_policy=seq_policy,
        )

    @classmethod
    def open_or_create(
        cls,
        db_url: str,
        *,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        create_if_missing: bool = True,
        tags_to_index: list[tuple[int, int]] | None = None,
        seq_policy: str = "json",
    ) -> "DatasetNodeDB":
        """
        Open a dataset in the DB, creating it if needed.

        If tags_to_index is provided, we:
            - build a TagPlan
            - call init_schema(engine, tag_plan=...) to ensure extra columns exist

        IMPORTANT: dynamic columns are per-table, not per-dataset. So columns
        added here are visible to all datasets in the same DB.
        """
        engine, factory = cls._make_engine_and_factory(db_url)

        tag_plan = build_tag_plan(tags_to_index or [])

        init_schema(engine, tag_plan=tag_plan, seq_policy=seq_policy)

        with factory() as s:
            if dataset_id is None:
                existing = s.scalars(select(DatasetRow)).all()
                if len(existing) == 0:
                    if not create_if_missing:
                        raise ValueError(
                            "No datasets found in DB and create_if_missing=False. "
                            "Provide dataset_id or allow_creation."
                        )
                    # Create a default dataset
                    new_id = dataset_name or "default"
                    ds_row = DatasetRow(dataset_id=new_id, dataset_name=dataset_name)
                    s.add(ds_row)
                    s.commit()
                    dataset_id = new_id
                elif len(existing) == 1:
                    ds_row = existing[0]
                    dataset_id = ds_row.dataset_id
                    if dataset_name and ds_row.dataset_name != dataset_name:
                        ds_row.dataset_name = dataset_name
                        s.commit()
                else:
                    ids = [d.dataset_id for d in existing]
                    raise ValueError(
                        "Multiple datasets found in DB; please specify dataset_id. "
                        f"Available dataset_ids: {ids}"
                    )

            # dataset_id now set

            ds_row = s.get(DatasetRow, dataset_id)
            if ds_row is None:
                if not create_if_missing:
                    raise ValueError(
                        f"Dataset {dataset_id!r} not found and create_if_missing=False."
                    )
                ds_row = DatasetRow(dataset_id=dataset_id, dataset_name=dataset_name)
                s.add(ds_row)
                s.commit()
            else:
                # Optionally update name
                if dataset_name and ds_row.dataset_name != dataset_name:
                    ds_row.dataset_name = dataset_name
                    s.commit()

        return cls.from_session_factory(
            factory,
            dataset_id,
            tag_plan=tag_plan,
            seq_policy=seq_policy,
        )

    @property
    def get_or_create_patient(
        self,
        PatientID: str,
        PatientName: Optional[str] = None,
    ) -> "PatientNodeDB":
        """
        DB-backed analog of DatasetNode.get_or_create_patient.

        - If a patient with this ID exists in this dataset, return it.
          * If PatientName is provided and the existing PatientName is empty
            or different, it will be updated.
        - If it does not exist, create it.
        """
        with self._session() as s:
            prow = s.scalar(
                select(PatientRow).where(
                    PatientRow.dataset_id == self.dataset_id,
                    PatientRow.PatientID == PatientID,
                )
            )

            if prow is None:
                # Create new
                prow = PatientRow(
                    dataset_id=self.dataset_id,
                    PatientID=PatientID,
                    PatientName=PatientName,
                )
                s.add(prow)
                s.commit()
            else:
                if PatientName and prow.PatientName != PatientName:
                    prow.PatientName = PatientName
                    s.commit()

        return PatientNodeDB(
            dataset_id=self.dataset_id,
            PatientID=PatientID,
            session_factory=self._session_factory,
        )

    # navigation

    def get_patient(self, PatientID: str) -> Optional["PatientNodeDB"]:
        with self._session() as s:
            prow = s.scalar(
                select(PatientRow).where(
                    PatientRow.dataset_id == self.dataset_id,
                    PatientRow.PatientID == PatientID,
                )
            )
        if prow is None:
            return None
        return PatientNodeDB(
            dataset_id=self.dataset_id,
            PatientID=PatientID,
            session_factory=self._session_factory,
        )

    def get_study(
        self,
        StudyInstanceUID: str,
        *,
        PatientID: Optional[str] = None,
    ) -> Optional["StudyNodeDB"]:
        """
        DB-backed analog of DatasetNode.get_study.

        If PatientID is provided, we also filter by it (fast O(1)-ish).
        Otherwise we just search by StudyInstanceUID in this dataset.
        """
        with self._session() as s:
            if PatientID is not None:
                st = s.scalar(
                    select(StudyRow).where(
                        StudyRow.dataset_id == self.dataset_id,
                        StudyRow.StudyInstanceUID == StudyInstanceUID,
                        StudyRow.PatientID == PatientID,
                    )
                )
            else:
                st = s.scalar(
                    select(StudyRow).where(
                        StudyRow.dataset_id == self.dataset_id,
                        StudyRow.StudyInstanceUID == StudyInstanceUID,
                    )
                )

        if st is None:
            return None

        return StudyNodeDB(
            dataset_id=self.dataset_id,
            StudyInstanceUID=StudyInstanceUID,
            session_factory=self._session_factory,
        )

    def get_series(
        self,
        SeriesInstanceUID: str,
        *,
        PatientID: Optional[str] = None,
        StudyInstanceUID: Optional[str] = None,
    ) -> Optional["SeriesNodeDB"]:
        """
        DB-backed analog of DatasetNode.get_series.

        Uses optional hints PatientID / StudyInstanceUID to narrow the query,
        but falls back to dataset-wide search when they are not provided.
        """
        with self._session() as s:
            q = select(SeriesRow).where(
                SeriesRow.dataset_id == self.dataset_id,
                SeriesRow.SeriesInstanceUID == SeriesInstanceUID,
            )

            # If study hint is provided
            if StudyInstanceUID is not None:
                q = q.where(SeriesRow.StudyInstanceUID == StudyInstanceUID)

            # If patient hint is provided, join to StudyRow to filter
            if PatientID is not None:
                q = q.join(
                    StudyRow,
                    (StudyRow.dataset_id == SeriesRow.dataset_id)
                    & (StudyRow.StudyInstanceUID == SeriesRow.StudyInstanceUID),
                ).where(StudyRow.PatientID == PatientID)

            se = s.scalar(q)

        if se is None:
            return None

        return SeriesNodeDB(
            dataset_id=self.dataset_id,
            SeriesInstanceUID=SeriesInstanceUID,
            session_factory=self._session_factory,
        )

    def get_instance(self, sop_uid: str) -> Optional["InstanceNodeDB"]:
        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == sop_uid,
                )
            )
        if inst is None:
            return None
        return InstanceNodeDB(
            dataset_id=self.dataset_id,
            SOPInstanceUID=sop_uid,
            session_factory=self._session_factory,
        )

    def find_study(self, StudyInstanceUID: str) -> Optional["StudyNodeDB"]:
        """
        Fallback traversal-style finder (but uses a direct query).
        """
        with self._session() as s:
            st = s.scalar(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.StudyInstanceUID == StudyInstanceUID,
                )
            )
        if st is None:
            return None
        return StudyNodeDB(
            dataset_id=self.dataset_id,
            StudyInstanceUID=StudyInstanceUID,
            session_factory=self._session_factory,
        )

    def find_series(self, SeriesInstanceUID: str) -> Optional["SeriesNodeDB"]:
        """
        Fallback traversal-style finder for series.
        """
        with self._session() as s:
            se = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.SeriesInstanceUID == SeriesInstanceUID,
                )
            )
        if se is None:
            return None
        return SeriesNodeDB(
            dataset_id=self.dataset_id,
            SeriesInstanceUID=SeriesInstanceUID,
            session_factory=self._session_factory,
        )

    def iter_patients(self) -> Iterable["PatientNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(PatientRow).where(PatientRow.dataset_id == self.dataset_id)
            ).all()
        for prow in rows:
            yield PatientNodeDB(
                dataset_id=self.dataset_id,
                PatientID=prow.PatientID,
                session_factory=self._session_factory,
            )

    def iter_studies(self) -> Iterable["StudyNodeDB"]:
        with self._session() as s:
            rows = s.scalars(select(StudyRow).where(StudyRow.dataset_id == self.dataset_id)).all()
        for st in rows:
            yield StudyNodeDB(
                dataset_id=self.dataset_id,
                StudyInstanceUID=st.StudyInstanceUID,
                session_factory=self._session_factory,
            )

    def iter_series(self) -> Iterable["SeriesNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(SeriesRow).where(SeriesRow.dataset_id == self.dataset_id)
            ).all()
        for se in rows:
            yield SeriesNodeDB(
                dataset_id=self.dataset_id,
                SeriesInstanceUID=se.SeriesInstanceUID,
                session_factory=self._session_factory,
            )

    def iter_instances(self) -> Iterable["InstanceNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(InstanceRow).where(InstanceRow.dataset_id == self.dataset_id)
            ).all()
        for inst in rows:
            yield InstanceNodeDB(
                dataset_id=self.dataset_id,
                SOPInstanceUID=inst.SOPInstanceUID,
                session_factory=self._session_factory,
            )

    def materialize_patient(self, PatientID: str):
        with self._session() as s:
            return load_patient(s, self.dataset_id, PatientID)

    # aggregate helpers
    def get_modality_distribution(
        self,
        *,
        all_instance_level: bool = False,
        force_instance_level_modalities: Iterable[str] = (),
        unknown_label: str = "Unknown",
    ) -> dict[str, int]:
        """
        DB-backed modality distribution.

        Default behavior:
            - Count modalities at the *series* level for everything.

        Override behavior:
            - For modalities in `force_instance_level_modalities`, count at the *instance* level
            (i.e., number of instances with that modality).

        """
        forced = {m for m in (force_instance_level_modalities or ())}

        # COALESCE(NULLIF(Modality, ''), 'Unkown')
        series_mod = func.coalesce(func.nullif(SeriesRow.Modality, ""), unknown_label)
        inst_mod = func.coalesce(func.nullif(InstanceRow.Modality, ""), unknown_label)

        out: dict[str, int] = {}

        with self._session() as s:
            if all_instance_level:
                stmt = (
                    select(inst_mod.label("modality"), func.count().label("n"))
                    .where(InstanceRow.dataset_id == self.dataset_id)
                    .group_by(inst_mod)
                )
                return {str(m): int(n) for m, n in s.execute(stmt)}

            stmt_series = select(series_mod.label("modality"), func.count().label("n")).where(
                SeriesRow.dataset_id == self.dataset_id
            )
            if forced:
                stmt_series = stmt_series.where(series_mod.notin_(forced))

            stmt_series = stmt_series.group_by(series_mod)

            for modality, n in s.execute(stmt_series):
                out[str(modality)] = int(n)

            if forced:
                stmt_inst = (
                    select(inst_mod.label("modality"), func.count().label("n"))
                    .where(
                        InstanceRow.dataset_id == self.dataset_id,
                        inst_mod.in_(forced),
                    )
                    .group_by(inst_mod)
                )

                for modality, n in s.execute(stmt_inst):
                    # forced wins over series-level
                    out[str(modality)] = int(n)

        return out

    def get_referenced_nodes(
        self,
        node,
        modality=None,
        level="INSTANCE",
        recursive=True,
        include_start=False,
    ):
        return get_referenced_nodes(node, modality, level, recursive, include_start)

    def get_referencing_nodes(
        self,
        node,
        modality=None,
        level="INSTANCE",
        recursive=True,
        include_start=False,
    ):
        return get_referencing_nodes(node, modality, level, recursive, include_start)

    def get_frame_registered_nodes(
        self,
        node,
        *,
        level="SERIES",
        include_self=False,
        modality=None,
        dicom_files=None,
        derive_frame_from_references=True,
    ):
        return get_frame_registered_nodes(
            node,
            level=level,
            include_self=include_self,
            modality=modality,
            dicom_files=dicom_files,
            derive_frame_from_references=derive_frame_from_references,
        )

    def associate_dicoms(self, *, rebuild: bool = True) -> None:
        """
        DB-backed associate_dicoms for the *entire* dataset.
        """
        with self._session() as s:
            _associate_dicoms_db(
                s,
                dataset_id=self.dataset_id,
                PatientID=None,
                rebuild=rebuild,
            )

    def n_patients(self) -> int:
        with self._session() as s:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(PatientRow)
                    .where(PatientRow.dataset_id == self.dataset_id)
                )
                or 0
            )

    def n_studies(self, *, PatientID: Optional[str] = None) -> int:
        with self._session() as s:
            stmt = (
                select(func.count())
                .select_from(StudyRow)
                .where(StudyRow.dataset_id == self.dataset_id)
            )

            if PatientID is not None:
                stmt = stmt.where(StudyRow.PatientID == PatientID)

            return int(s.scalar(stmt) or 0)

    def n_series(
        self,
        *,
        PatientID: Optional[str] = None,
        StudyInstanceUID: Optional[str] = None,
    ) -> int:
        with self._session() as s:
            stmt = (
                select(func.count())
                .select_from(SeriesRow)
                .where(SeriesRow.dataset_id == self.dataset_id)
            )

            if StudyInstanceUID is not None:
                stmt = stmt.where(SeriesRow.StudyInstanceUID == StudyInstanceUID)

            if PatientID is not None:
                stmt = stmt.join(
                    StudyRow,
                    (StudyRow.dataset_id == SeriesRow.dataset_id)
                    & (StudyRow.StudyInstanceUID == SeriesRow.StudyInstanceUID),
                ).where(StudyRow.PatientID == PatientID)

        return int(s.scalar(stmt) or 0)

    def n_instances(
        self,
        *,
        PatientID: Optional[str] = None,
        StudyInstanceUID: Optional[str] = None,
        SeriesInstanceUID: Optional[str] = None,
        Modality: Optional[str] = None,
    ) -> int:
        with self._session() as s:
            stmt = (
                select(func.count())
                .select_from(InstanceRow)
                .where(InstanceRow.dataset_id == self.dataset_id)
            )

            if SeriesInstanceUID is not None:
                stmt = stmt.where(InstanceRow.SeriesInstanceUID == SeriesInstanceUID)

            if Modality is not None:
                stmt = stmt.where(InstanceRow.Modality == Modality)

            if StudyInstanceUID is not None or PatientID is not None:
                stmt = stmt.join(
                    SeriesRow,
                    (SeriesRow.dataset_id == InstanceRow.dataset_id)
                    & (SeriesRow.SeriesInstanceUID == InstanceRow.SeriesInstanceUID),
                )

            if StudyInstanceUID is not None:
                stmt = stmt.where(SeriesRow.StudyInstanceUID == StudyInstanceUID)

            if PatientID is not None:
                stmt = stmt.join(
                    StudyRow,
                    (StudyRow.dataset_id == SeriesRow.dataset_id)
                    & (StudyRow.StudyInstanceUID == SeriesRow.StudyInstanceUID),
                ).where(StudyRow.PatientID == PatientID)

        return int(s.scalar(stmt) or 0)

    def __iter__(self):
        """Iterate over PatientNodeDB objects in this dataset."""
        return self.iter_patients()

    def __len__(self) -> int:
        """
        Number of patients in this dataset.
        """
        with self._session() as s:
            n = s.scalar(
                select(func.count())
                .select_from(PatientRow)
                .where(PatientRow.dataset_id == self.dataset_id)
            )
        return int(n or 0)


# PatientNodeDB
class PatientNodeDB(_DBNodeBase, JSONListFieldMixin):
    _ROW_CLS = PatientRow
    _LIST_FIELD_MAP = {}

    def __init__(self, dataset_id: str, PatientID: str, session_factory: SessionFactory):
        super().__init__(dataset_id, session_factory)
        self.PatientID = PatientID

    def _load_row(self, session: Session) -> PatientRow | None:
        return session.scalar(
            select(PatientRow).where(
                PatientRow.dataset_id == self.dataset_id,
                PatientRow.PatientID == self.PatientID,
            )
        )

    @property
    def PatientName(self) -> Optional[str]:
        with self._session() as s:
            prow = s.scalar(
                select(PatientRow).where(
                    PatientRow.dataset_id == self.dataset_id,
                    PatientRow.PatientID == self.PatientID,
                )
            )
            return prow.PatientName if prow is not None else None

    @property
    def parent_dataset(self):
        """
        Return the DatasetNodeDB for this patient.
        """
        return DatasetNodeDB(
            dataset_id=self.dataset_id,
            session_factory=self._session_factory,
        )

    def get_or_create_study(
        self,
        StudyInstanceUID: str,
        StudyDescription: Optional[str] = None,
    ) -> "StudyNodeDB":
        """
        DB-backed analog of PatientNode.get_or_create_study.

        Creates or updates a StudyRow under this patient.
        """
        with self._session() as s:
            st = s.scalar(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.PatientID == self.PatientID,
                    StudyRow.StudyInstanceUID == StudyInstanceUID,
                )
            )

            if st is None:
                st = StudyRow(
                    dataset_id=self.dataset_id,
                    PatientID=self.PatientID,
                    StudyInstanceUID=StudyInstanceUID,
                    StudyDescription=StudyDescription,
                )
                s.add(st)
                s.commit()
            else:
                if StudyDescription and st.StudyDescription != StudyDescription:
                    st.StudyDescription = StudyDescription
                    s.commit()

        return StudyNodeDB(
            dataset_id=self.dataset_id,
            StudyInstanceUID=StudyInstanceUID,
            session_factory=self._session_factory,
        )

    # navigation
    def iter_studies(self) -> Iterable["StudyNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.PatientID == self.PatientID,
                )
            ).all()
        for st in rows:
            yield StudyNodeDB(
                dataset_id=self.dataset_id,
                StudyInstanceUID=st.StudyInstanceUID,
                session_factory=self._session_factory,
            )

    def get_study(self, StudyInstanceUID: str) -> Optional["StudyNodeDB"]:
        with self._session() as s:
            st = s.scalar(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.PatientID == self.PatientID,
                    StudyRow.StudyInstanceUID == StudyInstanceUID,
                )
            )
        if st is None:
            return None
        return StudyNodeDB(
            dataset_id=self.dataset_id,
            StudyInstanceUID=StudyInstanceUID,
            session_factory=self._session_factory,
        )

    def iter_series(self) -> Iterable["SeriesNodeDB"]:
        """
        All series under this patient (across all studies).
        """
        with self._session() as s:
            study_uids = s.scalars(
                select(StudyRow.StudyInstanceUID).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.PatientID == self.PatientID,
                )
            ).all()

            if not study_uids:
                return
            series_rows = s.scalars(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.StudyInstanceUID.in_(study_uids),
                )
            ).all()

        for se in series_rows:
            yield SeriesNodeDB(
                dataset_id=self.dataset_id,
                SeriesInstanceUID=se.SeriesInstanceUID,
                session_factory=self._session_factory,
            )

    def iter_intances(self) -> Iterable["InstanceNodeDB"]:
        """
        All instances under this patient (across all studies/series).
        """
        with self._session() as s:
            study_uids = s.scalars(
                select(StudyRow.StudyInstanceUID).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.PatientID == self.PatientID,
                )
            ).all()
            if not study_uids:
                return
            series_uids = s.scalars(
                select(SeriesRow.SeriesInstanceUID).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.StudyInstanceUID.in_(study_uids),
                )
            ).all()
            if not series_uids:
                return
            inst_rows = s.scalars(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SeriesInstanceUID.in_(series_uids),
                )
            ).all()

        for inst in inst_rows:
            yield InstanceNodeDB(
                dataset_id=self.dataset_id,
                SOPInstanceUID=inst.SOPInstanceUID,
                session_factory=self._session_factory,
            )

    def to_patient_node(self):
        with self._session() as s:
            return load_patient(s, self.dataset_id, self.PatientID)

    # DB-node version of get_nodes_for_patient
    def get_nodes(
        self,
        level: str = "SERIES",
        Modality: Optional[str] = None,
        uid: Optional[str] = None,
    ) -> List[Union["StudyNodeDB", "SeriesNodeDB", "InstanceNodeDB"]]:
        """
        DB-backed analog of get_nodes_for_patient, returning DB nodes.

        level: 'STUDY' | 'SERIES' | 'INSTANCE'
        Modality: filter by Modality for SERIES/INSTANCE levels (case-insensitive).
        uid:
          - STUDY   -> StudyInstanceUID
          - SERIES  -> SeriesInstanceUID
          - INSTANCE-> SOPInstanceUID
        """
        level = level.upper()
        if level not in {"STUDY", "SERIES", "INSTANCE"}:
            raise ValueError("level must be 'STUDY', 'SERIES', or 'INSTANCE'")

        mod_norm = Modality.upper() if Modality else None

        with self._session() as s:
            # STUDY level
            if level == "STUDY":
                q = select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.PatientID == self.PatientID,
                )
                if uid:
                    q = q.where(StudyRow.StudyInstanceUID == uid)
                rows = s.scalars(q).all()
                return [
                    StudyNodeDB(
                        dataset_id=self.dataset_id,
                        StudyInstanceUID=st.StudyInstanceUID,
                        session_factory=self._session_factory,
                    )
                    for st in rows
                ]

            # SERIES level
            study_uids_q = select(StudyRow.StudyInstanceUID).where(
                StudyRow.dataset_id == self.dataset_id,
                StudyRow.PatientID == self.PatientID,
            )
            study_uids = s.scalars(study_uids_q).all()
            if not study_uids:
                return []

            if level == "SERIES":
                q = select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.StudyInstanceUID.in_(study_uids),
                )
                if uid:
                    q = q.where(SeriesRow.SeriesInstanceUID == uid)
                if mod_norm:
                    q = q.where(func.upper(SeriesRow.Modality) == mod_norm)

                rows = s.scalars(q).all()
                return [
                    SeriesNodeDB(
                        dataset_id=self.dataset_id,
                        SeriesInstanceUID=se.SeriesInstanceUID,
                        _session_factory=self._session_factory,
                    )
                    for se in rows
                ]

            # INSTANCE level
            # First find series_uids under this patient
            series_uids_q = select(SeriesRow.SeriesInstanceUID).where(
                SeriesRow.dataset_id == self.dataset_id,
                SeriesRow.StudyInstanceUID.in_(study_uids),
            )
            series_uids = s.scalars(series_uids_q).all()
            if not series_uids:
                return []

            q = select(InstanceRow).where(
                InstanceRow.dataset_id == self.dataset_id,
                InstanceRow.SeriesInstanceUID.in_(series_uids),
            )
            if uid:
                q = q.where(InstanceRow.SOPInstanceUID == uid).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SeriesInstanceUID.in_(series_uids),
                )
            if mod_norm:
                q = q.where(func.upper(InstanceRow.Modality) == mod_norm)

            inst_rows = s.scalars(q).all()
            return [
                InstanceNodeDB(
                    dataset_id=self.dataset_id,
                    SOPInstanceUID=inst.SOPInstanceUID,
                    session_factory=self._session_factory,
                )
                for inst in inst_rows
            ]

    def get_referenced_nodes(
        self,
        node,
        modality=None,
        level="INSTANCE",
        recursive=True,
        include_start=False,
    ):
        return get_referenced_nodes(node, modality, level, recursive, include_start)

    def get_referencing_nodes(
        self,
        node,
        modality=None,
        level="INSTANCE",
        recursive=True,
        include_start=False,
    ):
        return get_referencing_nodes(node, modality, level, recursive, include_start)

    def get_frame_registered_nodes(
        self,
        node,
        *,
        level="SERIES",
        include_self=False,
        modality=None,
        dicom_files=None,
        derive_frame_from_references=True,
    ):
        return get_frame_registered_nodes(
            node,
            level=level,
            include_self=include_self,
            modality=modality,
            dicom_files=dicom_files,
            derive_frame_from_references=derive_frame_from_references,
        )

    def associate_dicoms(self, *, rebuild: bool = True) -> None:
        """
        DB-backed associate_dicoms scoped to this patient.
        Only edges whose *source* series/instances belong to this patient are rebuilt.
        """
        with self._session() as s:
            _associate_dicoms_db(
                s,
                dataset_id=self.dataset_id,
                PatientID=self.PatientID,
                rebuild=rebuild,
            )

    def __getatt__(self, name: str):
        """
        Fallback: expose columns on PatientRow as attributes.

        Called only if normal attribute lookup fails.
        """
        if name.startswith("_"):
            # don't try to resolve private/internal attrs dynamically
            raise AttributeError(name)

        with self._session() as s:
            row = self._load_row(s)
            if row is not None and hasattr(row, name):
                return getattr(row, name)

        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r} "
            "and it was not found on PatientRow."
        )

    def __iter__(self):
        """Iterate over StudyNodeDB objects for this patient."""
        return self.iter_studies()


# StudyNodeDB


class StudyNodeDB(_DBNodeBase, JSONListFieldMixin):
    _ROW_CLS = StudyRow
    _LIST_FIELD_MAP = {}

    def __init__(self, dataset_id: str, StudyInstanceUID: str, session_factory):
        super().__init__(dataset_id, session_factory)
        self.StudyInstanceUID = StudyInstanceUID

    def _load_row(self, session: Session) -> StudyRow | None:
        return session.scalar(
            select(StudyRow).where(
                StudyRow.dataset_id == self.dataset_id,
                StudyRow.StudyInstanceUID == self.StudyInstanceUID,
            )
        )

    @property
    def StudyDescription(self) -> Optional[str]:
        with self._session() as s:
            row = s.scalar(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.StudyInstanceUID == self.StudyInstanceUID,
                )
            )
            return row.StudyDescription if row else None

    @property
    def parent_patient(self) -> Optional["PatientNodeDB"]:
        with self._session() as s:
            st = s.scalar(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.StudyInstanceUID == self.StudyInstanceUID,
                )
            )
        if st is None:
            return None

        return PatientNodeDB(
            dataset_id=self.dataset_id,
            PatientID=st.PatientID,
            session_factory=self._session_factory,
        )

    @property
    def series(self) -> Dict[str, "SeriesNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.StudyInstanceUID == self.StudyInstanceUID,
                )
            ).all()

        return {
            se.SeriesInstanceUID: SeriesNodeDB(
                dataset_id=self.dataset_id,
                SeriesInstanceUID=se.SeriesInstanceUID,
                session_factory=self._session_factory,
            )
            for se in rows
        }

    def get_or_create_series(
        self,
        SeriesInstanceUID: str,
        Modality=None,
        SeriesDescription=None,
        FrameOfReferenceUID=None,
    ) -> "SeriesNodeDB":

        with self._session() as s:
            se = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.SeriesInstanceUID == SeriesInstanceUID,
                )
            )

            if se is None:
                se = SeriesRow(
                    dataset_id=self.dataset_id,
                    StudyInstanceUID=self.StudyInstanceUID,
                    SeriesInstanceUID=SeriesInstanceUID,
                    Modality=Modality,
                    SeriesDescription=SeriesDescription,
                    FrameOfReferenceUID=FrameOfReferenceUID,
                )
                s.add(se)
                s.commit()

            else:
                changed = False
                if Modality and se.Modality != Modality:
                    se.Modality = Modality
                    changed = True
                if SeriesDescription and se.SeriesDescription != SeriesDescription:
                    se.SeriesDescription = SeriesDescription
                    changed = True
                if FrameOfReferenceUID and se.FrameOfReferenceUID != FrameOfReferenceUID:
                    se.FrameOfReferenceUID = FrameOfReferenceUID
                    changed = True
                if changed:
                    s.commit()

        return SeriesNodeDB(
            dataset_id=self.dataset_id,
            SeriesInstanceUID=SeriesInstanceUID,
            session_factory=self._session_factory,
        )

    def iter_series(self):
        with self._session() as s:
            rows = s.scalars(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.StudyInstanceUID == self.StudyInstanceUID,
                )
            ).all()

        for se in rows:
            yield SeriesNodeDB(
                dataset_id=self.dataset_id,
                SeriesInstanceUID=se.SeriesInstanceUID,
                session_factory=self._session_factory,
            )

    def get_series(self, SeriesInstanceUID: str):
        with self._session() as s:
            se = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,  # FIXED
                    SeriesRow.StudyInstanceUID == self.StudyInstanceUID,
                    SeriesRow.SeriesInstanceUID == SeriesInstanceUID,
                )
            )
        if se is None:
            return None

        return SeriesNodeDB(
            dataset_id=self.dataset_id,
            SeriesInstanceUID=SeriesInstanceUID,
            session_factory=self._session_factory,
        )

    def update_fields(self, **fields) -> None:
        """
        Generic multi-field update on StudyRow.

        Example:
            study_db.update_fields(
                study_description="New desc",
                SomeDynamicTag="value",
            )
        """
        if not fields:
            return

        with self._session() as s:
            row = s.scalar(
                select(StudyRow).where(
                    StudyRow.dataset_id == self.dataset_id,
                    StudyRow.StudyInstanceUID == self.StudyInstanceUID,
                )
            )
            if row is None:
                raise ValueError(
                    f"Study {self.StudyInstanceUID!r} not foune " f"in dataset {self.dataset_id!r}"
                )

            changed = False
            for name, value in fields.items():
                if not hasattr(row, name):
                    raise AttributeError(
                        f"StudyRow has no attribute {name!r}; "
                        "check your db_schema or field name."
                    )
                if getattr(row, name) != value:
                    setattr(row, name, value)
                    changed = True

            if changed:
                s.commit()

    def __getattr__(self, name: str):
        """
        Fallback resolution:

        1) If attribute exists on StudyRow -> return it.
        2) Else, if parent patient exists and has it -> return that.
        """
        if name.startswith("_"):
            raise AttributeError(name)

        # 1) Try StudyRow
        with self._session() as s:
            row = self._load_row(s)
            if row is not None and hasattr(row, name):
                return getattr(row, name)

        # 2) Try parent patient
        parent = self.parent_patient
        if parent is not None and hasattr(parent, name):
            return getattr(parent, name)

        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r} "
            "and it was not found on StudyRow or PatientNodeDB."
        )

    def __iter__(self):
        """Iterate over SeriesNodeDB objects for this study."""
        return self.iter_series()


# SeriesNodeDB


class SeriesNodeDB(_DBNodeBase, JSONListFieldMixin):
    _ROW_CLS = SeriesRow
    _LIST_FIELD_MAP = {
        "instance_paths": "instance_paths_json",
        "referenced_sids": "referenced_sids_json",
        "referencing_sids": "referencing_sids_json",
    }

    def __init__(self, dataset_id: str, SeriesInstanceUID: str, session_factory):
        super().__init__(dataset_id, session_factory)
        self.SeriesInstanceUID = SeriesInstanceUID

    def _load_row(self, session: Session):
        return session.scalar(
            select(SeriesRow).where(
                SeriesRow.dataset_id == self.dataset_id,
                SeriesRow.SeriesInstanceUID == self.SeriesInstanceUID,
            )
        )

    @property
    def Modality(self) -> Optional[str]:
        with self._session() as s:
            se = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            )
            return se.Modality if se is not None else None

    @property
    def SeriesDescription(self) -> Optional[str]:
        with self._session() as s:
            se = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            )
            return se.SeriesDescription if se is not None else None

    @property
    def FrameOfReferenceUID(self) -> Optional[str]:
        with self._session() as s:
            row = self._load_row(s)
            return row.FrameOfReferenceUID if row is not None else None

    @property
    def instance_paths(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.instance_paths_json or []) if row is not None else []

    @property
    def referenced_sids(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.referenced_sids_json or []) if row is not None else []

    @property
    def referencing_sids(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.referencing_sids_json or []) if row is not None else []

    @property
    def referenced_series(self) -> list["SeriesNodeDB"]:
        """
        List of SeriesNodeDB that this series references.
        """
        return [
            SeriesNodeDB(
                dataset_id=self.dataset_id,
                SeriesInstanceUID=sid,
                session_factory=self._session_factory,
            )
            for sid in self.referenced_sids
        ]

    @property
    def referencing_series(self) -> list["SeriesNodeDB"]:
        return [
            SeriesNodeDB(
                dataset_id=self.dataset_id,
                SeriesInstanceUID=sid,
                session_factory=self._session_factory,
            )
            for sid in self.referencing_sids
        ]

    @property
    def frame_of_reference_registered(self):
        """
        Returns all SeriesNodeDB objects that share the same FrameOfReferenceUID.
        """
        my_fuid = self.FrameOfReferenceUID
        if not my_fuid:
            return []

        ds = self.parent_study.parent_patient.parent_dataset
        return [
            series
            for series in ds.iter_series()
            if series.FrameOfReferenceUID == my_fuid and series is not self
        ]

    @property
    def is_embedded_in_raw(self) -> bool:
        with self._session() as s:
            row = self._load_row(s)
            return bool(row.is_embedded_in_raw) if row is not None else False

    @property
    def raw_series_reference_uid(self) -> Optional[str]:
        with self._session() as s:
            row = self._load_row(s)
            return row.raw_series_ref_uid if row is not None else None

    @property
    def raw_series_reference(self) -> Optional["SeriesNodeDB"]:
        uid = self.raw_series_reference_uid
        if not uid:
            return None
        return SeriesNodeDB(
            dataset_id=self.dataset_id,
            SeriesInstanceUID=uid,
            session_factory=self._session_factory,
        )

    @property
    def SOPInstances(self) -> list[str]:
        with self._session() as s:
            rows = s.scalars(
                select(InstanceRow.SOPInstanceUID).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            ).all()
        return list(rows)

    @property
    def instances(self) -> Dict[str, "InstanceNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            ).all()

        return {
            inst_row.SOPInstanceUID: InstanceNodeDB(
                dataset_id=self.dataset_id,
                SOPInstanceUID=inst_row.SOPInstanceUID,
                session_factory=self._session_factory,
            )
            for inst_row in rows
        }

    @property
    def parent_study(self):
        with self._session() as s:
            se = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            )
        if se is None:
            return None

        return StudyNodeDB(
            dataset_id=self.dataset_id,
            StudyInstanceUID=se.StudyInstanceUID,
            session_factory=self._session_factory,
        )

    def get_or_create_instance(
        self,
        SOPInstanceUID,
        file_path,
        Modality=None,
        frame_of_reference_uids=None,
        sources=None,
    ):

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == SOPInstanceUID,
                )
            )

            if inst is None:
                inst = InstanceRow(
                    dataset_id=self.dataset_id,
                    SeriesInstanceUID=self.SeriesInstanceUID,
                    SOPInstanceUID=SOPInstanceUID,
                    file_path=file_path,
                    Modality=Modality,
                )
                if frame_of_reference_uids is not None:
                    inst.frame_of_reference_uids_json = frame_of_reference_uids
                if sources is not None:
                    inst.sources_json = sources

                s.add(inst)
                s.commit()

            else:
                changed = False

                if inst.SeriesInstanceUID != self.SeriesInstanceUID:
                    inst.SeriesInstanceUID = self.SeriesInstanceUID
                    changed = True

                if inst.file_path != file_path:
                    inst.file_path = file_path
                    changed = True

                if Modality and inst.Modality != Modality:
                    inst.Modality = Modality
                    changed = True

                if frame_of_reference_uids is not None:
                    if inst.frame_of_reference_uids_json != frame_of_reference_uids:
                        inst.frame_of_reference_uids_json = frame_of_reference_uids
                        changed = True

                if sources is not None:
                    if inst.sources_json != sources:
                        inst.sources_json = sources
                        changed = True

                if changed:
                    s.commit()

        return InstanceNodeDB(
            dataset_id=self.dataset_id,
            SOPInstanceUID=SOPInstanceUID,
            session_factory=self._session_factory,
        )

    def iter_instances(self):
        with self._session() as s:
            rows = s.scalars(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            ).all()

        for row in rows:
            yield InstanceNodeDB(
                dataset_id=self.dataset_id,
                SOPInstanceUID=row.SOPInstanceUID,
                session_factory=self._session_factory,
            )

    def get_instance(self, sop_instance_uid: str) -> Optional["InstanceNodeDB"]:
        with self._session() as s:
            row = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == sop_instance_uid,
                )
            )
        if row is None:
            return None

        return InstanceNodeDB(
            dataset_id=self.dataset_id,
            SOPInstanceUID=sop_instance_uid,
            session_factory=self._session_factory,
        )

    def update_fields(self, **fields) -> None:
        """
        Generic multi-field update on SeriesRow.
        """
        if not fields:
            return

        with self._session() as s:
            row = s.scalar(
                select(SeriesRow).where(
                    SeriesRow.dataset_id == self.dataset_id,
                    SeriesRow.SeriesInstanceUID == self.SeriesInstanceUID,
                )
            )
            if row is None:
                raise ValueError(
                    f"Series {self.SeriesInstanceUID!r} not found "
                    f"in dataset {self.dataset_id!r}"
                )

            changed = False
            for name, value in fields.items():
                if not hasattr(row, name):
                    raise AttributeError(
                        f"SeriesRow has no attribute {name!r}; "
                        "check your db_schema or field name."
                    )
                if getattr(row, name) != value:
                    setattr(row, name, value)
                    changed = True

            if changed:
                s.commit()

    def __getattr__(self, name: str):
        """
        Fallback resolution for SeriesNodeDB:

        1) SeriesRow
        2) parent StudyNodeDB
        3) parent PatientNodeDB (through study)
        """
        if name.startswith("_"):
            raise AttributeError(name)

        # 1) Try SeriesRow
        with self._session() as s:
            row = self._load_row(s)
            if row is not None and hasattr(row, name):
                return getattr(row, name)

        # 2) Try parent study
        study = self.parent_study
        if study is not None and hasattr(study, name):
            return getattr(study, name)

        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r} "
            "and it was not found on SeriesRow, StudyNodeDB, or PatientNodeDB."
        )

    def __iter__(self):
        """Iterate over InstanceNodeDB objects for this series."""
        return self.iter_instances()


# InstanceNodeDB


class InstanceNodeDB(_DBNodeBase, JSONListFieldMixin):
    _ROW_CLS = InstanceRow
    _LIST_FIELD_MAP = {
        "frame_of_reference_uids": "frame_of_reference_uids_json",
        "referenced_sop_uids": "referenced_sop_uids_json",
        "referenced_sids": "referenced_sids_json",
        "other_referenced_sids": "other_referenced_sids_json",
        "sources": "sources_json",
    }

    def __init__(self, dataset_id: str, SOPInstanceUID: str, session_factory):
        super().__init__(dataset_id, session_factory)
        self.SOPInstanceUID = SOPInstanceUID

    def _load_row(self, session: Session):
        return session.scalar(
            select(InstanceRow).where(
                InstanceRow.dataset_id == self.dataset_id,
                InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
            )
        )

    @property
    def Modality(self) -> Optional[str]:
        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            return inst.Modality if inst is not None else None

    @property
    def FilePath(self) -> Optional[str]:
        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            return inst.file_path if inst is not None else None

    @property
    def parent_series(self):
        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )

        if inst is None:
            return None

        return SeriesNodeDB(
            dataset_id=self.dataset_id,
            SeriesInstanceUID=inst.SeriesInstanceUID,
            session_factory=self._session_factory,
        )

    @property
    def FrameOfReferenceUIDs(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.frame_of_reference_uids_json or []) if row else []

    @property
    def referenced_sop_instance_uids(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.referenced_sop_uids_json or []) if row else []

    @property
    def referenced_sids(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.referenced_sids_json or []) if row else []

    @property
    def other_referenced_sids(self) -> list[str]:
        with self._session() as s:
            row = self._load_row(s)
            return list(row.other_referenced_sids_json or []) if row else []

    @property
    def referenced_series(self) -> list["SeriesNodeDB"]:
        sids = self.referenced_sids
        out = []
        for sid in sids:
            out.append(
                SeriesNodeDB(
                    dataset_id=self.dataset_id,
                    SeriesInstanceUID=sid,
                    session_factory=self._session_factory,
                )
            )
        return out

    @property
    def other_referenced_series(self) -> list["SeriesNodeDB"]:
        sids = self.other_referenced_sids
        out = []
        for sid in sids:
            out.append(
                SeriesNodeDB(
                    dataset_id=self.dataset_id,
                    SeriesInstanceUID=sid,
                    session_factory=self._session_factory,
                )
            )
        return out

    @property
    def referenced_instances(self) -> list["InstanceNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(InstanceReferenceRow.dst_SOPInstanceUID).where(
                    InstanceReferenceRow.dataset_id == self.dataset_id,
                    InstanceReferenceRow.src_SOPInstanceUID == self.SOPInstanceUID,
                )
            ).all()
        return [
            InstanceNodeDB(
                dataset_id=self.dataset_id,
                SOPInstanceUID=dst_uid,
                session_factory=self._session_factory,
            )
            for dst_uid in rows
        ]

    @property
    def referencing_instances(self) -> list["InstanceNodeDB"]:
        with self._session() as s:
            rows = s.scalars(
                select(InstanceReferenceRow.src_SOPInstanceUID).where(
                    InstanceReferenceRow.dataset_id == self.dataset_id,
                    InstanceReferenceRow.dst_SOPInstanceUID == self.SOPInstanceUID,
                )
            ).all()

        return [
            InstanceNodeDB(
                dataset_id=self.dataset_id,
                SOPInstanceUID=src_uid,
                session_factory=self._session_factory,
            )
            for src_uid in rows
        ]

    @property
    def references(self) -> list[str]:
        return self.referenced_sop_instance_uids + self.referenced_sids

    # Generic multi-field update
    def update_fields(self, **fields) -> None:
        """
        Update one or more mapped attributes on the underlying InstanceRow.

        Example
        -------
        inst_db.update_fields(
            Modality="RTDOSE",
            file_path=r"/new/path.dcm",
        )
        """
        if not fields:
            return

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            if inst is None:
                raise ValueError(
                    f"Instance {self.SOPInstanceUID!r} not found "
                    f"in dataset {self.dataset_id!r}"
                )
            changed = False
            for name, value in fields.items():
                if not hasattr(inst, name):
                    raise AttributeError(
                        f"InstanceRow has no attribute {name!r}; "
                        "check your db_schema or field name."
                    )
                if getattr(inst, name) != value:
                    setattr(inst, name, value)
                    changed = True

            if changed:
                s.commit()

    # Generic list-field appender
    def append_to_list_field(self, field_name: str, value, *, unique: bool = True) -> None:
        """
        Append `value` to a JSON/list field on InstanceRow.

        field_name must be something like:
          - 'sources_json'
          - 'frame_of_reference_uids_json'
          - 'referenced_sop_uids_json'
          - etc.
        """
        if value is None:
            return

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            if inst is None:
                raise ValueError(
                    f"Instance {self.SOPInstanceUID!r} not found "
                    f"in dataset {self.dataset_id!r}"
                )

            if not hasattr(inst, field_name):
                raise AttributeError(
                    f"InstanceRow has no attribute {field_name!r}; "
                    "check your db_schema or field name."
                )
            current = getattr(inst, field_name) or []
            lst = list(current)
            if unique and value in lst:
                return
            lst.append(value)
            setattr(inst, field_name, lst)
            s.commit()

    def append(self, field_name: str, value, *, unique: bool = True) -> None:
        """
        Append `value` to a list field (JSON array) on InstanceRow.

        User-friendly usage:
            inst.append("sources", new_source)
            inst.append("referenced_sop_uids", sop_uid)
            inst.append("frame_of_reference_uids", fo_uid)

        Also supports explicit DB column:
            inst.append("sources_json", new_source)
        """
        if value is None:
            return

        db_field = self._resolve_list_field(field_name)

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            if inst is None:
                raise ValueError(
                    f"Instance {self.SOPInstanceUID!r} not found in dataset {self.dataset_id!r}"
                )

            current = getattr(inst, db_field) or []
            lst = list(current)

            if unique and value in lst:
                return

            lst.append(value)
            setattr(inst, db_field, lst)

            s.commit()

    def extend(self, field_name: str, values, *, unique: bool = True) -> None:
        """
        Add multiple values to a list JSON field.
        values: any iterable (list, set, generator, etc.)
        """
        if not values:
            return

        db_field = self._resolve_list_field(field_name)

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            if inst is None:
                raise ValueError(
                    f"Instance {self.SOPInstanceUID!r} not found "
                    f"in dataset {self.dataset_id!r}"
                )

            lst = list(getattr(inst, db_field) or [])

            for v in values:
                if v is None:
                    continue
                if unique and v in lst:
                    continue
                lst.append(v)

            setattr(inst, db_field, lst)
            s.commit()

    def remove(self, field_name: str, value) -> None:
        """
        Remove a value from a list JSON field (no error if missing).
        """
        if value is None:
            return

        db_field = self._resolve_list_field(field_name)

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            if inst is None:
                raise ValueError(
                    f"Instance {self.SOPInstanceUID!r} not found "
                    f"in dataset {self.dataset_id!r}"
                )

            lst = list(getattr(inst, db_field) or [])
            if value not in lst:
                return

            lst.remove(value)
            setattr(inst, db_field, lst)
            s.commit()

    def clear(self, field_name: str) -> None:
        """
        Clear a list JSON field (set it to empty list).
        """
        db_field = self._resolve_list_field(field_name)

        with self._session() as s:
            inst = s.scalar(
                select(InstanceRow).where(
                    InstanceRow.dataset_id == self.dataset_id,
                    InstanceRow.SOPInstanceUID == self.SOPInstanceUID,
                )
            )
            if inst is None:
                raise ValueError(
                    f"Instance {self.SOPInstanceUID!r} not found "
                    f"in dataset {self.dataset_id!r}"
                )

            setattr(inst, db_field, [])
            s.commit()

    def __getattr__(self, name: str):
        """
        Fallback resolution for InstanceNodeDB:

        1) InstanceRow
        2) parent SeriesNodeDB
        3) parent StudyNodeDB (through series)
        4) parent PatientNodeDB (through study)
        """
        if name.startswith("_"):
            raise AttributeError(name)

        # 1) Try InstanceRow
        with self._session() as s:
            row = self._load_row(s)
            if row is not None and hasattr(row, name):
                return getattr(row, name)

        # 2) Try parent series
        series = self.parent_series
        if series is not None and hasattr(series, name):
            return getattr(series, name)

        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r} "
            "and it was not found on InstanceRow, SeriesNodeDB, StudyNodeDB, or PatientNodeDB."
        )


def _associate_dicoms_db(
    session: Session,
    dataset_id: str,
    *,
    PatientID: Optional[str] = None,
    rebuild: bool = True,
) -> None:
    """
    DB-backed analog of your in-memory associate_dicoms.

    Uses JSON fields on InstanceRow and updates:

      - instance_references (InstanceReferenceRow): instance→instance edges
      - series_references   (SeriesReferenceRow):   series→series edges
      - SeriesRow.referenced_sids_json / referencing_sids_json

    Scope
    -----
    - If PatientID is None:
        consider all series/instances in `dataset_id`.
    - If PatientID is given:
        consider only series/instances whose STUDY belongs to that PatientID.
        Only edges whose *source* is in this scope are deleted/rebuilt.
    """
    # 1) Series and instances in scope
    if PatientID is None:
        series_rows = session.scalars(
            select(SeriesRow).where(SeriesRow.dataset_id == dataset_id)
        ).all()
    else:
        series_rows = session.scalars(
            select(SeriesRow)
            .join(
                StudyRow,
                (StudyRow.dataset_id == SeriesRow.dataset_id)
                & (StudyRow.StudyInstanceUID == SeriesRow.StudyInstanceUID),
            )
            .where(
                SeriesRow.dataset_id == dataset_id,
                StudyRow.PatientID == PatientID,
            )
        ).all()

    if not series_rows:
        return

    series_uids = {se.SeriesInstanceUID for se in series_rows}

    inst_rows = session.scalars(
        select(InstanceRow).where(
            InstanceRow.dataset_id == dataset_id,
            InstanceRow.SeriesInstanceUID.in_(series_uids),
        )
    ).all()
    inst_uids = {inst.SOPInstanceUID for inst in inst_rows}

    # 2) Clear existing edges + series JSON for these sources
    if rebuild and inst_uids:
        # Instnace edges: remove edges whose SOURCE is in this scope
        session.execute(
            delete(InstanceReferenceRow).where(
                InstanceReferenceRow.dataset_id == dataset_id,
                InstanceReferenceRow.src_SOPInstanceUID.in_(inst_uids),
            )
        )

    if rebuild and series_uids:
        # Series edges: remove edges whose SOURCE is in this scope
        session.execute(
            delete(SeriesReferenceRow).where(
                SeriesReferenceRow.dataset_id == dataset_id,
                SeriesReferenceRow.src_SeriesInstanceUID.in_(series_uids),
            )
        )
        # Clear JSON per series in scope (we'll rebuild)
        for se in series_rows:
            se.referenced_sids_json = []
            se.referencing_sids_json = []

    # We'll accumulate series-level references and push them back at the end
    series_referenced: dict[str, set[str]] = {se.SeriesInstanceUID: set() for se in series_rows}
    series_referencing: dict[str, set[str]] = {se.SeriesInstanceUID: set() for se in series_rows}

    # Dedup sets for edge rows
    seen_inst_edges: set[tuple[str, str]] = set()
    seen_series_edges: set[tuple[str, str, str]] = set()  # (src_series, dst_series, kind)

    # 3) Rebuild edges from InstanceRow JSON fields
    for inst in inst_rows:
        src_sop = inst.SOPInstanceUID
        src_series = inst.SeriesInstanceUID

        # Instnace->instance edges
        for dst_sop in inst.referenced_sop_uids_json or []:
            if not dst_sop:
                continue
            dst_sop = str(dst_sop)
            if dst_sop not in inst_uids:
                # Referenced instance not present in this dataset/scope -> skip
                continue

            edge_key = (src_sop, dst_sop)
            if edge_key in seen_inst_edges:
                continue
            seen_inst_edges.add(edge_key)

            session.add(
                InstanceReferenceRow(
                    dataset_id=dataset_id,
                    src_SOPInstanceUID=src_sop,
                    dst_SOPInstanceUID=dst_sop,
                )
            )

        # Series->series edges
        for dst_sid in inst.referenced_sids_json or []:
            if not dst_sid:
                continue
            dst_sid = str(dst_sid)
            if dst_sid not in series_uids:
                # Referenced series not present -> skip
                continue

            edge_key = (src_series, dst_sid, "direct")
            if edge_key in seen_series_edges:
                continue
            seen_series_edges.add(edge_key)

            session.add(
                SeriesReferenceRow(
                    dataset_id=dataset_id,
                    src_SeriesInstanceUID=src_series,
                    dst_SeriesInstanceUID=dst_sid,
                    kind="direct",
                )
            )
            series_referenced.setdefault(src_series, set()).add(dst_sid)
            series_referencing.setdefault(dst_sid, set()).add(src_series)

        # Series-> series edges (other)
        for dst_sid in inst.other_referenced_sids_json or []:
            if not dst_sid:
                continue
            dst_sid = str(dst_sid)
            if dst_sid not in series_uids:
                continue

            edge_key = (src_series, dst_sid, "other")
            if edge_key in seen_series_edges:
                continue
            seen_series_edges.add(edge_key)

            session.add(
                SeriesReferenceRow(
                    dataset_id=dataset_id,
                    src_SeriesInstanceUID=src_series,
                    dst_SeriesInstanceUID=dst_sid,
                    kind="other",
                )
            )
            series_referenced.setdefault(src_series, set()).add(dst_sid)
            series_referencing.setdefault(dst_sid, set()).add(src_series)

    # 4) Push aggregated series-level JSON back into SeriesRow
    for se in series_rows:
        sid = se.SeriesInstanceUID
        se.referenced_sids_json = sorted(series_referenced.get(sid, set()))
        se.referencing_sids_json = sorted(series_referencing.get(sid, set()))
    session.commit()


def get_referenced_nodes(
    node,
    modality: Optional[Union[str, Iterable[str]]] = None,
    level: str = "INSTANCE",
    recursive: bool = True,
    include_start: bool = False,
):
    """
    Return nodes that are *referenced by* the given node.

    Works for:
        - SeriesNode / InstanceNode (in-memory)
        - SeriesNodeDB / InstanceNodeDB (DB-backed)

    Parameters
    ----------
    node : SeriesNode | InstanceNode | SeriesNodeDB | InstanceNodeDB
    modality : str or Iterable[str], optional
        Case-insensitive modality filter.
    level : {'INSTANCE', 'SERIES'}
        Whether to return instance- or series-level nodes.
    recursive : bool, default True
        If False, only direct neighbors are returned (depth=1).
    include_start : bool, default False
        Whether to include the starting node in the results (if it matches
        the level and modality).
    """
    from rosamllib.db.db_nodes import SeriesNodeDB, InstanceNodeDB

    def norm_modalities(m) -> Optional[Set[str]]:
        if m is None:
            return None
        if isinstance(m, str):
            return {m.upper()}
        return {str(x).upper() for x in m}

    def modality_ok(obj) -> bool:
        if wanted is None:
            return True
        mod = getattr(obj, "Modality", None)
        return (mod or "").upper() in wanted

    def maybe_add(n):
        if level == "INSTANCE" and isinstance(n, InstanceNodeDB) and modality_ok(n):
            out.append(n)
        elif level == "SERIES" and isinstance(n, SeriesNodeDB) and modality_ok(n):
            out.append(n)

    level = level.upper()
    if level not in {"INSTANCE", "SERIES"}:
        raise ValueError("level must be 'INSTANCE' or 'SERIES'")

    wanted = norm_modalities(modality)
    out = []
    seen: Set[int] = set()

    # BFS
    q = deque()
    # depth=0 is the start node; we still may include it if requested
    q.append((node, 0))
    if include_start:
        maybe_add(node)

    max_depth = None if recursive else 1

    while q:
        n, d = q.popleft()
        nid = id(n)
        if nid in seen:
            continue
        seen.add(nid)

        # collect (except the start if include_start=False)
        if d > 0:
            maybe_add(n)

        # stop expanding if we've hit the depth limit
        if max_depth is not None and d >= max_depth:
            continue

        # neighbors
        if isinstance(n, SeriesNodeDB):
            # 1) direct series->series links (e.g., REG/SEG edges resolved at series level)
            for s in getattr(n, "referenced_series", []) or []:
                q.append((s, d + 1))
            # 2) go through instances to follow instance-level links
            instances_attr = getattr(n, "instances", None)
            if isinstance(instances_attr, dict):
                inst_iter = instances_attr.values()
            else:
                iter_fn = getattr(n, "iter_instances", None)
                inst_iter = iter_fn() if callable(iter_fn) else []

            for inst in inst_iter:
                # instance->instance
                for ref in getattr(inst, "referenced_instances", []) or []:
                    q.append((ref, d + 1))
                # instance->series
                for s in getattr(inst, "referenced_series", []) or []:
                    q.append((s, d + 1))

        elif isinstance(n, InstanceNodeDB):
            # instance->instance
            for ref in getattr(n, "referenced_instances", []) or []:
                q.append((ref, d + 1))
            # instance->series
            for s in getattr(n, "referenced_series", []) or []:
                q.append((s, d + 1))

    # de-dup while preserving order (by id)
    seen_ids: Set[int] = set()
    deduped = []
    for x in out:
        xid = id(x)
        if xid not in seen_ids:
            seen_ids.add(xid)
            deduped.append(x)

    return deduped


def get_referencing_nodes(
    node,
    modality: Optional[Union[str, Iterable[str]]] = None,
    level: str = "INSTANCE",
    recursive: bool = True,
    include_start: bool = False,
):
    """
    Return nodes that share the same FrameOfReferenceUID as the given node.

    Parameters
    ----------
    node : SeriesNode | InstanceNode | SeriesNodeDB | InstanceNodeDB
        Anchor node. If an InstanceNode is provided, its parent SeriesNode is used.
    level : {'SERIES', 'INSTANCE'}, default 'SERIES'
        - 'SERIES': return SeriesNode peers in the same Frame of Reference (FoR).
        - 'INSTANCE': return InstanceNode peers from all series in the same FoR.
        Note: with 'INSTANCE' and include_self=False, result can be empty if
        there are no peer series (i.e., anchor is the only series in its FoR).
    include_self : bool, default False
        Include the anchor in the results:
        - 'SERIES': include the anchor series.
        - 'INSTANCE': include all instances from the anchor series.
    modality : str or Iterable[str], optional
        Case-insensitive modality filter.
        - For level='SERIES', filters by `series.Modality` (e.g., 'CT', 'MR').
        - For level='INSTANCE', filters by `instance.Modality` (e.g., 'CT', 'RTDOSE').
        You may pass a single string (e.g., "CT") or an iterable (e.g., ["CT","MR"]).

    Returns
    -------
    list[NodeLike]
        Peers in the same Frame of Reference, filtered by level/modality.

    Notes
    -----
    - This method prefers the precomputed `series.frame_of_reference_registered`
    filled during `_associate_dicoms`. If that list is empty, it falls back
    to scanning `self.dicom_files`.
    - Passing a single string for `modality` is supported and treated as a set
    with one element (e.g., "CT" -> {"CT"}).

    Examples
    --------
    >>> # Series peers (CT or MR) sharing the same FoR as a dose's CT
    >>> peers = loader.get_frame_registered_nodes(dose.parent_series,
    ...                                           level="SERIES",
    ...                                           modality=["CT","MR"])
    >>> # All RTDOSE instances within the same FoR (including the anchor series)
    >>> doses = loader.get_frame_registered_nodes(ct_series,
    ...                                           level="INSTANCE",
    ...                                           include_self=True,
    ...                                           modality="RTDOSE")
    """

    from rosamllib.db.db_nodes import SeriesNodeDB, InstanceNodeDB

    def norm_modalities(m) -> Optional[Set[str]]:
        if m is None:
            return None
        if isinstance(m, str):
            return {m.upper()}
        return {str(x).upper() for x in m}

    def modality_ok(obj) -> bool:
        if wanted is None:
            return True
        mod = getattr(obj, "Modality", None)
        return (mod or "").upper() in wanted

    def maybe_add(n):
        if level == "INSTANCE" and isinstance(n, InstanceNodeDB) and modality_ok(n):
            out.append(n)
        elif level == "SERIES" and isinstance(n, SeriesNodeDB) and modality_ok(n):
            out.append(n)

    def enqueue(nei, depth):
        if nei is None:
            return
        q.append((nei, depth))

    level = level.upper()
    if level not in {"INSTANCE", "SERIES"}:
        raise ValueError("level must be 'INSTANCE' or 'SERIES'")

    wanted = norm_modalities(modality)
    out = []
    seen: Set[int] = set()
    q = deque()
    q.append((node, 0))

    if include_start:
        maybe_add(node)

    max_depth = None if recursive else 1

    while q:
        n, d = q.popleft()
        nid = id(n)
        if nid in seen:
            continue
        seen.add(nid)

        # collect (except depth 0 unless include_start)
        if d > 0:
            maybe_add(n)

        # stop expanding if depth cap
        if max_depth is not None and d >= max_depth:
            continue

        # ---- incoming neighbors ----
        if isinstance(n, InstanceNodeDB):
            # instances that reference this instance
            for rin in getattr(n, "referencing_instances", []) or []:
                enqueue(rin, d + 1)
                # their parent series are also referrers at the series level
                enqueue(getattr(rin, "parent_series", None), d + 1)

            # series that reference this instance directly (if you maintain such a list)
            # Not standard in your model; typically we discover series via the instances above.

            # the instance's parent series might be referenced by other series;
            # climb to series and continue
            ps = getattr(n, "parent_series", None)
            if ps is not None:
                # series that reference this series (if populated)
                for rs in getattr(ps, "referencing_series", []) or []:
                    enqueue(rs, d + 1)

        elif isinstance(n, SeriesNodeDB):
            # series that reference this series (if populated)
            for rs in getattr(n, "referencing_series", []) or []:
                enqueue(rs, d + 1)

            # instances that reference any instance within this series
            for inst in getattr(n, "instances", {}).values():
                for rin in getattr(inst, "referencing_instances", []) or []:
                    enqueue(rin, d + 1)
                    enqueue(getattr(rin, "parent_series", None), d + 1)

    # stable de-dup by object id
    uniq_ids: Set[int] = set()
    deduped = []
    for x in out:
        xid = id(x)
        if xid not in uniq_ids:
            uniq_ids.add(xid)
            deduped.append(x)

    return deduped


def get_frame_registered_nodes(
    node,
    *,
    level: str = "SERIES",
    include_self: bool = False,
    modality: Optional[Union[str, Iterable[str]]] = None,
    dicom_files=None,
    derive_frame_from_references: bool = True,
):
    """
    Return nodes that share at least one effective FrameOfReferenceUID with the anchor.

    Effective FoR of a series is the union of:
    - series.FrameOfReferenceUID
    - (if derive_frame_from_references) any inst.FrameOfReferenceUIDs
    - (if derive_frame_from_references) FoR of any series referenced by its instances
    """
    from rosamllib.db.db_nodes import SeriesNodeDB, InstanceNodeDB

    def _wanted_set(m):
        if m is None:
            return None
        return {m.upper()} if isinstance(m, str) else {str(x).upper() for x in m}

    def _series_mod_ok(s):
        return wanted is None or (getattr(s, "Modality", None) or "").upper() in wanted

    def _inst_mod_ok(i):
        return wanted is None or (getattr(i, "Modality", None) or "").upper() in wanted

    def _effective_fors(series) -> set[str]:
        fors: set[str] = set()
        fo_direct = getattr(series, "FrameOfReferenceUID", None)
        if fo_direct:
            fors.add(str(fo_direct))
        if derive_frame_from_references:
            for inst in getattr(series, "instances", {}).values():
                for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                    if u:
                        fors.add(str(u))
                for rs in getattr(inst, "referenced_series", []) or []:
                    u = getattr(rs, "FrameOfReferenceUID", None)
                    if u:
                        fors.add(str(u))
        return fors

    lvl = str(level).upper()
    if lvl not in {"SERIES", "INSTANCE"}:
        raise ValueError("level must be 'SERIES' or 'INSTANCE'")

    wanted = _wanted_set(modality)

    # Anchor series + anchor FoR set
    anchor_series = (
        node if isinstance(node, SeriesNodeDB) else getattr(node, "parent_series", None)
    )
    anchor_fors: set[str] = set()
    if isinstance(node, InstanceNodeDB) and getattr(node, "FrameOfReferenceUIDs", None):
        anchor_fors |= {str(u) for u in (node.FrameOfReferenceUIDs or []) if u}
    if anchor_series and getattr(anchor_series, "FrameOfReferenceUID", None):
        anchor_fors.add(str(anchor_series.FrameOfReferenceUID))
    # If still empty and we’re allowed to derive, derive from anchor series’ instances
    if not anchor_fors and anchor_series and derive_frame_from_references:
        anchor_fors |= _effective_fors(anchor_series)

    if not anchor_fors:
        # no FoR context — return only self if requested
        if include_self:
            if lvl == "SERIES" and isinstance(node, SeriesNodeDB) and _series_mod_ok(node):
                return [node]
            if lvl == "INSTANCE" and isinstance(node, InstanceNodeDB) and _inst_mod_ok(node):
                return [node]
        return []

    # Collect peer series: intersection of effective FoRs with anchor_fors
    peer_series = []
    seen_sid: set[int] = set()

    # Prefer dicom_files when provided (covers RTSTRUCT/SEG/REG cases correctly)
    if dicom_files:
        for _pid, sdict in dicom_files.items():
            for s in sdict.values():
                if anchor_series is not None and s is anchor_series:
                    continue
                eff = _effective_fors(s)
                if eff and (eff & anchor_fors):
                    if id(s) not in seen_sid:
                        peer_series.append(s)
                        seen_sid.add(id(s))
    else:
        # Fallback to precomputed FoR neighbors (series-level only; may miss RTSTRUCT)
        if anchor_series:
            for s in list(getattr(anchor_series, "frame_of_reference_registered", []) or []):
                if id(s) in seen_sid:
                    continue
                # verify intersection using effective FoRs to avoid false negatives
                if _effective_fors(s) & anchor_fors:
                    peer_series.append(s)
                    seen_sid.add(id(s))

    if lvl == "SERIES":
        out = []
        if include_self and anchor_series and _series_mod_ok(anchor_series):
            out.append(anchor_series)
        out.extend([s for s in peer_series if _series_mod_ok(s)])
        # de-dup
        seen = set()
        dedup = []
        for x in out:
            if id(x) not in seen:
                seen.add(id(x))
                dedup.append(x)
        return dedup

    # INSTANCE level: return instances from anchor (optional) + peer series
    out_i = []

    def add_series(series):
        for inst in getattr(series, "instances", {}).values():
            if _inst_mod_ok(inst):
                out_i.append(inst)

    if include_self and anchor_series:
        add_series(anchor_series)
    for s in peer_series:
        add_series(s)

    seen = set()
    dedup = []
    for x in out_i:
        if id(x) not in seen:
            seen.add(id(x))
            dedup.append(x)
    return dedup
