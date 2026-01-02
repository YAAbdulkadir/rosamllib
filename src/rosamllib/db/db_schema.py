from __future__ import annotations
from typing import Optional, Dict, Mapping, Iterable
from dataclasses import dataclass
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag, tag_for_keyword, dictionary_VR, dictionary_VM
from sqlalchemy.types import TypeEngine
from sqlalchemy import (
    Column,
    Integer,
    Float,
    Date,
    DateTime,
    Time,
    ForeignKeyConstraint,
    String,
    Boolean,
    Text,
    JSON,
    PrimaryKeyConstraint,
    delete,
    select,
    ForeignKey,
    Index,
    inspect,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
)

from rosamllib.nodes import (
    DatasetNode,
    PatientNode,
    StudyNode,
    SeriesNode,
    InstanceNode,
)

import time
from dataclasses import field
from typing import List, Tuple


# Base + ORM models
class Base(DeclarativeBase):
    """Base for all ORM models."""

    pass


class DatasetRow(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class PatientRow(Base):
    __tablename__ = "patients"

    dataset_id: Mapped[str] = mapped_column(
        String, ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False
    )
    PatientID: Mapped[str] = mapped_column(String, nullable=False)
    PatientName: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    extras_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (PrimaryKeyConstraint("dataset_id", "PatientID", name="pk_patient"),)


class StudyRow(Base):
    __tablename__ = "studies"

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    PatientID: Mapped[str] = mapped_column(String, nullable=False)
    StudyInstanceUID: Mapped[str] = mapped_column(String, nullable=False)
    StudyDescription: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    extras_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("dataset_id", "StudyInstanceUID", name="pk_study"),
        ForeignKeyConstraint(
            ["dataset_id", "PatientID"],
            ["patients.dataset_id", "patients.PatientID"],
            ondelete="CASCADE",
        ),
        Index("ix_studies_dataset_patient", "dataset_id", "PatientID"),
    )


class SeriesRow(Base):
    __tablename__ = "series"

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    PatientID: Mapped[str] = mapped_column(String, nullable=False)
    SeriesInstanceUID: Mapped[str] = mapped_column(String, nullable=False)
    StudyInstanceUID: Mapped[str] = mapped_column(
        String,
        ForeignKey("studies.StudyInstanceUID"),
        nullable=False,
    )

    Modality: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    SeriesDescription: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    FrameOfReferenceUID: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    is_embedded_in_raw: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_series_ref_uid: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    instance_paths_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    referenced_sids_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    referencing_sids_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    extras_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("dataset_id", "SeriesInstanceUID", name="pk_series"),
        ForeignKeyConstraint(
            ["dataset_id", "StudyInstanceUID"],
            ["studies.dataset_id", "studies.StudyInstanceUID"],
            ondelete="CASCADE",
        ),
        Index("ix_series_dataset_patient", "dataset_id", "PatientID"),
        Index("ix_series_dataset_study", "dataset_id", "StudyInstanceUID"),
        Index("ix_series_dataset_modality", "dataset_id", "Modality"),
    )


class InstanceRow(Base):
    __tablename__ = "instances"

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    sources_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    PatientID: Mapped[str] = mapped_column(String, nullable=False)
    StudyInstanceUID: Mapped[str] = mapped_column(String, nullable=True)
    SeriesInstanceUID: Mapped[str] = mapped_column(
        String,
        ForeignKey("series.SeriesInstanceUID"),
        nullable=False,
    )
    SOPInstanceUID: Mapped[str] = mapped_column(String, nullable=False)
    SOPClassUID: Mapped[str] = mapped_column(String, nullable=True)
    Modality: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # SeriesInstanceUID: Mapped[str] = mapped_column(String, nullable=False)

    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    frame_of_reference_uids_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    referenced_sop_uids_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    referenced_sids_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    other_referenced_sids_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    extras_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("dataset_id", "SOPInstanceUID", name="pk_instance"),
        ForeignKeyConstraint(
            ["dataset_id", "SeriesInstanceUID"],
            ["series.dataset_id", "series.SeriesInstanceUID"],
            ondelete="CASCADE",
        ),
        Index("ix_instances_dataset_patient", "dataset_id", "PatientID"),
        Index("ix_instances_dataset_series", "dataset_id", "SeriesInstanceUID"),
        Index("ix_instances_dataset_study", "dataset_id", "StudyInstanceUID"),
        Index("ix_instances_dataset_modality", "dataset_id", "Modality"),
    )


class SeriesReferenceRow(Base):
    __tablename__ = "series_references"

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    src_SeriesInstanceUID: Mapped[str] = mapped_column(String, nullable=False)
    dst_SeriesInstanceUID: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "direct", "raw", ...

    __table_args__ = (
        PrimaryKeyConstraint(
            "dataset_id",
            "src_SeriesInstanceUID",
            "dst_SeriesInstanceUID",
            "kind",
            name="pk_seriesref",
        ),
        ForeignKeyConstraint(
            ["dataset_id", "src_SeriesInstanceUID"],
            ["series.dataset_id", "series.SeriesInstanceUID"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["dataset_id", "dst_SeriesInstanceUID"],
            ["series.dataset_id", "series.SeriesInstanceUID"],
            ondelete="CASCADE",
        ),
        Index("ix_seriesref_dataset_src", "dataset_id", "src_SeriesInstanceUID"),
        Index("ix_seriesref_dataset_dst", "dataset_id", "dst_SeriesInstanceUID"),
    )


class InstanceReferenceRow(Base):
    __tablename__ = "instance_references"

    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    src_SOPInstanceUID: Mapped[str] = mapped_column(String, nullable=False)
    dst_SOPInstanceUID: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "dataset_id", "src_SOPInstanceUID", "dst_SOPInstanceUID", name="pk_instanceref"
        ),
        ForeignKeyConstraint(
            ["dataset_id", "src_SOPInstanceUID"],
            ["instances.dataset_id", "instances.SOPInstanceUID"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["dataset_id", "dst_SOPInstanceUID"],
            ["instances.dataset_id", "instances.SOPInstanceUID"],
            ondelete="CASCADE",
        ),
        Index("ix_instref_dataset_src", "dataset_id", "src_SOPInstanceUID"),
        Index("ix_instref_dataset_dst", "dataset_id", "dst_SOPInstanceUID"),
    )


class MetaRow(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)


# Core, always-needed keywords
CORE_TAGS = [
    "SOPClassUID",
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "PatientID",
    "PatientName",
    "StudyDescription",
    "SeriesDescription",
    "FrameOfReferenceUID",
    "Modality",
]


@dataclass(frozen=True)
class TagPlan:
    tag: Tag
    name: str
    vr: str
    is_sq: bool


def _sa_type_for_tagplan(
    tp: TagPlan,
    *,
    force_json_tags: set[str] | None = None,
    type_overrides: Mapping[str, TypeEngine] | None = None,
) -> TypeEngine:
    """
    Decide the SQLAlchemy column type for a given TagPlan based on VR + VM
    (with optional overrides).

    - SQ or multi-valued (VM != '1') -> JSON
    - Integer-like VRs -> Integer
    - Float-like VRs   -> Float
    - DA / TM / DT     -> Date / Time / DateTime
    - Everything else  -> Text

    """
    force_json_tags = force_json_tags or set()
    type_overrides = type_overrides or {}

    # 1) Hard override wins
    if tp.name in type_overrides:
        return type_overrides[tp.name]

    # 2) Forced JSON
    if tp.name in force_json_tags or tp.is_sq:
        return JSON

    # 3) Default VR + VM based mapping
    vr = (tp.vr or "").upper()

    # VM string is like "1", "1-n", "2", "2-n", etc.
    vm = dictionary_VM(tp.tag)  # returns e.g. "1", "1-n", ...
    is_multi = vm is not None and vm != "1"

    # Sequences OR multi-valued -> JSON (we store lists)
    if tp.is_sq or is_multi:
        return JSON

    # Scalar VRs
    if vr in {"IS", "SL", "SS", "UL", "US"}:
        return Integer
    if vr in {"DS", "FL", "FD"}:
        return Float
    if vr == "DA":
        return Date
    if vr == "TM":
        return Time
    if vr == "DT":
        return DateTime

    # Everything else falls back to text
    return Text


def build_tag_plan(tags_to_index: list[tuple[int, int]]) -> list[TagPlan]:
    plan = []
    all_tags = []

    for k in CORE_TAGS:
        t = tag_for_keyword(k)
        if t:
            all_tags.append((Tag(t).group, Tag(t).element))
    if tags_to_index:
        for g, e in tags_to_index:
            all_tags.append((g, e))

    seen = set()
    uniq = []
    for ge in all_tags:
        if ge not in seen:
            seen.add(ge)
            uniq.append(ge)

    for g, e in uniq:
        t = Tag((g, e))
        vr = dictionary_VR(t)
        name = keyword_for_tag(t) or f"{g:04X},{e:04X}"
        is_sq = vr == "SQ"
        plan.append(TagPlan(tag=t, name=name, vr=vr, is_sq=is_sq))
    return plan


def init_schema(
    engine,
    tag_plan: list[TagPlan] | None = None,
    *,
    force_json_tags: Iterable[str] | None = None,
    type_overrides: Mapping[str, TypeEngine] | None = None,
    **kwargs,
) -> None:
    """
    Initialize or update the DB schema.

    If tag_plan is provided, ensure that InstanceRow has columns for each
    TagPlan.name.

    - By default, VR is mapped to an appropriate SQLAlchemy type.
    - `force_json_tags`: keywords to *force* into JSON columns.
    - `type_overrides`: finer-grained type control, e.g. {"DoseGridScaling": Float}.
    """

    force_json_tags = set(force_json_tags or [])
    type_overrides = dict(type_overrides or {})

    if tag_plan:
        existing_names = {col.name for col in InstanceRow.__table__.columns}
        for tp in tag_plan:
            col_name = tp.name
            if col_name in existing_names:
                continue

            coltype = _sa_type_for_tagplan(
                tp,
                force_json_tags=force_json_tags,
                type_overrides=type_overrides,
            )

            col = Column(col_name, coltype)
            InstanceRow.__table__.append_column(col)
            InstanceRow.__mapper__.add_property(col_name, col)

            existing_names.add(col_name)

    Base.metadata.create_all(engine)

    if not tag_plan:
        _reflect_and_map_instance_dynamic_columns(engine)

    with Session(engine) as session:
        existing = session.get(MetaRow, "schema_version")
        if existing is None:
            session.add(MetaRow(key="schema_version", value="1"))
            session.commit()


def _reflect_and_map_instance_dynamic_columns(engine) -> None:
    """
    Ensure InstanceRow has mapped properties for *existing DB columns*
    that aren't part of the base schema.
    """
    insp = inspect(engine)
    db_cols = insp.get_columns("instances")
    existing_names = {col.name for col in InstanceRow.__table__.columns}

    for c in db_cols:
        name = c["name"]
        if name in existing_names:
            continue

        coltype = c["type"]
        col = Column(name, coltype)

        InstanceRow.__table__.append_column(col)
        InstanceRow.__mapper__.add_property(name, col)
        existing_names.add(name)


# Save: DatasetNode -> DB


def _extras_from_node(node) -> dict:
    """
    Extract extras from a node that inherits _ExtensibleAttrs.
    """
    if hasattr(node, "iter_attrs"):
        return dict(node.iter_attrs())
    return {}


def save_dataset(session: Session, dataset: DatasetNode) -> None:
    """
    Persist a DatasetNode (and all children) into the database.

    This will remove any existing rows for the same dataset_id and replace them.
    Assumes ORM rows use DICOM-style names for core tags:
      - PatientRow.PatientID / PatientName
      - StudyRow.StudyInstanceUID / PatientID / StudyDescription
      - SeriesRow.SeriesInstanceUID / StudyInstanceUID / Modality /
        SeriesDescription / FrameOfReferenceUID
      - InstanceRow.SOPInstanceUID / SeriesInstanceUID / Modality / file_path / ...
    """
    dataset_id = dataset.dataset_id

    with session.begin():
        # 1) Delete existing rows for this dataset
        session.execute(
            delete(InstanceReferenceRow).where(InstanceReferenceRow.dataset_id == dataset_id)
        )
        session.execute(
            delete(SeriesReferenceRow).where(SeriesReferenceRow.dataset_id == dataset_id)
        )
        session.execute(delete(InstanceRow).where(InstanceRow.dataset_id == dataset_id))
        session.execute(delete(SeriesRow).where(SeriesRow.dataset_id == dataset_id))
        session.execute(delete(StudyRow).where(StudyRow.dataset_id == dataset_id))
        session.execute(delete(PatientRow).where(PatientRow.dataset_id == dataset_id))
        session.execute(delete(DatasetRow).where(DatasetRow.dataset_id == dataset_id))

        # 2) Insert dataset
        session.add(
            DatasetRow(
                dataset_id=dataset.dataset_id,
                dataset_name=dataset.dataset_name,
            )
        )

        # 3) Insert patients / studies / series / instances
        for patient in dataset:
            p_extras = _extras_from_node(patient)
            session.add(
                PatientRow(
                    dataset_id=dataset_id,
                    PatientID=patient.PatientID,
                    PatientName=patient.PatientName,
                    extras_json=p_extras or None,
                )
            )

            for study in patient:
                st_extras = _extras_from_node(study)
                session.add(
                    StudyRow(
                        dataset_id=dataset_id,
                        StudyInstanceUID=study.StudyInstanceUID,
                        PatientID=patient.PatientID,
                        StudyDescription=study.StudyDescription,
                        extras_json=st_extras or None,
                    )
                )

                for series in study:
                    se_extras = _extras_from_node(series)

                    session.add(
                        SeriesRow(
                            dataset_id=dataset_id,
                            SeriesInstanceUID=series.SeriesInstanceUID,
                            StudyInstanceUID=study.StudyInstanceUID,
                            Modality=series.Modality,
                            SeriesDescription=(
                                str(series.SeriesDescription) if series.SeriesDescription else None
                            ),
                            FrameOfReferenceUID=series.FrameOfReferenceUID,
                            is_embedded_in_raw=bool(series.is_embedded_in_raw),
                            raw_series_ref_uid=series.raw_series_reference_uid,
                            instance_paths_json=series.instance_paths or None,
                            referenced_sids_json=series.referenced_sids or None,
                            referencing_sids_json=series.referencing_sids or None,
                            extras_json=se_extras or None,
                        )
                    )

                    for inst in series:
                        inst_extras = _extras_from_node(inst)

                        session.add(
                            InstanceRow(
                                dataset_id=dataset_id,
                                SOPInstanceUID=inst.SOPInstanceUID,
                                SeriesInstanceUID=series.SeriesInstanceUID,
                                file_path=inst.FilePath,
                                Modality=inst.Modality,
                                frame_of_reference_uids_json=(inst.FrameOfReferenceUIDs or None),
                                referenced_sop_uids_json=(
                                    inst.referenced_sop_instance_uids or None
                                ),
                                referenced_sids_json=inst.referenced_sids or None,
                                other_referenced_sids_json=(inst.other_referenced_sids or None),
                                sources_json=inst.sources or None,
                                extras_json=inst_extras or None,
                            )
                        )

        # 4) Insert edge tables (series_references, instance_references)
        #    We assume associate_dicoms or other logic already populated
        #    referenced_series / referenced_instances on the nodes.

        for patient in dataset:
            for study in patient:
                for series in study:
                    # Series references
                    for target in getattr(series, "referenced_series", []):
                        kind = "direct"
                        if getattr(series, "raw_series_reference", None) is target:
                            # You can refine this logic as needed
                            kind = "raw"

                        session.add(
                            SeriesReferenceRow(
                                dataset_id=dataset_id,
                                src_SeriesInstanceUID=series.SeriesInstanceUID,
                                dst_SeriesInstanceUID=target.SeriesInstanceUID,
                                kind=kind,
                            )
                        )

                    # Instance references
                    for inst in series:
                        for target in getattr(inst, "referenced_instances", []):
                            session.add(
                                InstanceReferenceRow(
                                    dataset_id=dataset_id,
                                    src_SOPInstanceUID=inst.SOPInstanceUID,
                                    dst_SOPInstanceUID=target.SOPInstanceUID,
                                )
                            )


# Load: DB -> DatasetNode


def load_dataset(session, dataset_id: str) -> DatasetNode:
    """
    Reconstruct a DatasetNode (and all children) from the database.

    This version is strict: it will raise if any series/instance refers to a missing
    parent study/series, instead of silently skipping.
    """
    # 1) Dataset
    ds_row = session.get(DatasetRow, dataset_id)
    if ds_row is None:
        raise ValueError(f"Dataset {dataset_id!r} not found in database")

    ds = DatasetNode(dataset_id=ds_row.dataset_id, dataset_name=ds_row.dataset_name)

    # helper dicts (keyed only by UID; we are loading one dataset_id at a time)
    patients_by_id: Dict[str, PatientNode] = {}
    studies_by_uid: Dict[str, StudyNode] = {}
    series_by_uid: Dict[str, SeriesNode] = {}
    instances_by_uid: Dict[str, InstanceNode] = {}

    # 2) Patients
    patient_rows = session.scalars(
        select(PatientRow).where(PatientRow.dataset_id == dataset_id)
    ).all()

    for prow in patient_rows:
        p = PatientNode(
            patient_id=prow.PatientID,
            patient_name=prow.PatientName,
            parent_dataset=ds,
        )
        for k, v in (prow.extras_json or {}).items():
            p.set_attrs(**{k: v})

        ds.add_patient(p)
        patients_by_id[prow.PatientID] = p

    # 3) Studies
    study_rows = session.scalars(select(StudyRow).where(StudyRow.dataset_id == dataset_id)).all()

    for srow in study_rows:
        parent_patient = patients_by_id.get(srow.PatientID)
        if parent_patient is None:
            raise RuntimeError(
                f"StudyRow(StudyInstanceUID={srow.StudyInstanceUID!r}) refers to "
                f"missing PatientID={srow.PatientID!r}"
            )

        st = StudyNode(
            study_uid=srow.StudyInstanceUID,
            study_description=srow.StudyDescription,
            parent_patient=parent_patient,
        )
        for k, v in (srow.extras_json or {}).items():
            st.set_attrs(**{k: v})

        parent_patient.add_study(st)
        studies_by_uid[srow.StudyInstanceUID] = st

    # 4) Series
    series_rows = session.scalars(
        select(SeriesRow).where(SeriesRow.dataset_id == dataset_id)
    ).all()

    for srow in series_rows:
        parent_study = studies_by_uid.get(srow.StudyInstanceUID)
        if parent_study is None:
            raise RuntimeError(
                f"SeriesRow(SeriesInstanceUID={srow.SeriesInstanceUID!r}) refers to "
                f"missing StudyInstanceUID={srow.StudyInstanceUID!r}"
            )

        se = SeriesNode(
            series_uid=srow.SeriesInstanceUID,
            parent_study=parent_study,
        )
        se.Modality = srow.Modality
        se.SeriesDescription = srow.SeriesDescription
        se.FrameOfReferenceUID = srow.FrameOfReferenceUID
        se.is_embedded_in_raw = bool(srow.is_embedded_in_raw)
        se.raw_series_reference_uid = srow.raw_series_ref_uid

        se.instance_paths = list(srow.instance_paths_json or [])
        se.referenced_sids = list(srow.referenced_sids_json or [])
        se.referencing_sids = list(srow.referencing_sids_json or [])

        for k, v in (srow.extras_json or {}).items():
            se.set_attrs(**{k: v})

        parent_study.add_series(se)
        series_by_uid[srow.SeriesInstanceUID] = se

    # 5) Instances
    instance_rows = session.scalars(
        select(InstanceRow).where(InstanceRow.dataset_id == dataset_id)
    ).all()

    for irow in instance_rows:
        parent_series = series_by_uid.get(irow.SeriesInstanceUID)
        if parent_series is None:
            raise RuntimeError(
                f"InstanceRow(SOPInstanceUID={irow.SOPInstanceUID!r}) refers to "
                f"missing SeriesInstanceUID={irow.SeriesInstanceUID!r}"
            )

        inst = InstanceNode(
            SOPInstanceUID=irow.SOPInstanceUID,
            FilePath=irow.file_path,
            Modality=irow.Modality,
            parent_series=parent_series,
        )

        inst.FrameOfReferenceUIDs = list(irow.frame_of_reference_uids_json or [])
        inst.referenced_sop_instance_uids = list(irow.referenced_sop_uids_json or [])
        inst.referenced_sids = list(irow.referenced_sids_json or [])
        inst.other_referenced_sids = list(irow.other_referenced_sids_json or [])
        inst.sources = list(irow.sources_json or [])

        for k, v in (irow.extras_json or {}).items():
            inst.set_attrs(**{k: v})

        parent_series.add_instance(inst)
        instances_by_uid[irow.SOPInstanceUID] = inst

    # 6) Rebuild series edges
    series_ref_rows = session.scalars(
        select(SeriesReferenceRow).where(SeriesReferenceRow.dataset_id == dataset_id)
    ).all()

    for rrow in series_ref_rows:
        src = series_by_uid.get(rrow.src_SeriesInstanceUID)
        dst = series_by_uid.get(rrow.dst_SeriesInstanceUID)
        if src is None or dst is None:
            raise RuntimeError(
                f"SeriesReferenceRow(dataset_id={rrow.dataset_id!r}, "
                f"src_SeriesInstanceUID={rrow.src_SeriesInstanceUID!r}, "
                f"dst_SeriesInstanceUID={rrow.dst_SeriesInstanceUID!r}) refers to missing series"
            )

        src.referenced_series.append(dst)
        dst.referencing_series.append(src)

        if rrow.kind == "raw":
            src.raw_series_reference = dst
            src.is_embedded_in_raw = True

    # 7) Rebuild instance edges
    inst_ref_rows = session.scalars(
        select(InstanceReferenceRow).where(InstanceReferenceRow.dataset_id == dataset_id)
    ).all()

    for rrow in inst_ref_rows:
        src = instances_by_uid.get(rrow.src_SOPInstanceUID)
        dst = instances_by_uid.get(rrow.dst_SOPInstanceUID)
        if src is None or dst is None:
            raise RuntimeError(
                f"InstanceReferenceRow(dataset_id={rrow.dataset_id!r}, "
                f"src_SOPInstanceUID={rrow.src_SOPInstanceUID!r}, "
                f"dst_SOPInstanceUID={rrow.dst_SOPInstanceUID!r}) refers to missing instance"
            )

        src.referenced_instances.append(dst)
        dst.referencing_instances.append(src)

    from rosamllib.utils import associate_dicoms

    associate_dicoms(ds)

    return ds


@dataclass
class _Timers:
    t0: float = field(default_factory=time.perf_counter)
    marks: List[Tuple[str, float]] = field(default_factory=list)

    def mark(self, name: str) -> None:
        self.marks.append((name, time.perf_counter()))

    def report(self, *, header: str = "load_patient timing") -> Dict[str, float]:
        out: Dict[str, float] = {}
        last = self.t0
        print(f"\n=== {header} ===")
        for name, t in self.marks:
            dt = t - last
            out[name] = dt
            print(f"{name:28s} {dt:8.3f} s")
            last = t

        total = last - self.t0 if self.marks else (time.perf_counter() - self.t0)
        out["TOTAL"] = total
        print(f"{'TOTAL':28s} {total:8.3f} s")
        return out


def load_patient(session: Session, dataset_id: str, patient_id: str) -> PatientNode:
    """
    Reconstruct a PatientNode (and its children) from the database WITHOUT using
    the reference tables (SeriesReferenceRow / InstanceReferenceRow).

    Semantics (patient-local only):
      - Load all studies/series/instances that belong to (dataset_id, patient_id).
      - Materialize normal in-memory nodes.
      - Run associate_dicoms(ds), which uses per-row JSON fields
        (referenced_sids_json, referenced_sop_uids_json, etc.) to build links.

    Returns
    -------
    PatientNode
        The requested patient node (normal in-memory node). A mini DatasetNode is created
        containing exactly this patient.
    """
    timers = _Timers()

    # -------------------------
    # helpers
    # -------------------------
    def _chunks(seq, n: int = 950):
        lst = list(seq)
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # -------------------------
    # 0) Dataset + patient exist?
    # -------------------------
    ds_row = session.get(DatasetRow, dataset_id)
    if ds_row is None:
        raise ValueError(f"Dataset {dataset_id!r} not found in database")
    timers.mark("0a) dataset_row")

    prow = session.get(PatientRow, {"dataset_id": dataset_id, "PatientID": patient_id})
    if prow is None:
        raise ValueError(
            f"PatientID {patient_id!r} not found in dataset {dataset_id!r} in database"
        )
    timers.mark("0b) patient_row")

    # -------------------------
    # 1) Seed UIDs (patient-local)
    # -------------------------
    # Studies (UID-only)
    study_uids = set(
        session.scalars(
            select(StudyRow.StudyInstanceUID).where(
                StudyRow.dataset_id == dataset_id,
                StudyRow.PatientID == patient_id,
            )
        ).all()
    )
    timers.mark("1a) seed studies (uids)")

    # Series (UID-only)
    series_uids = set(
        session.scalars(
            select(SeriesRow.SeriesInstanceUID).where(
                SeriesRow.dataset_id == dataset_id,
                SeriesRow.PatientID == patient_id,
            )
        ).all()
    )
    timers.mark("1b) seed series (uids)")

    # Instances (UID-only)
    instance_uids = set(
        session.scalars(
            select(InstanceRow.SOPInstanceUID).where(
                InstanceRow.dataset_id == dataset_id,
                InstanceRow.PatientID == patient_id,
            )
        ).all()
    )
    timers.mark("1c) seed instances (uids)")

    # -------------------------
    # 2) Materialize mini DatasetNode
    # -------------------------
    ds = DatasetNode(dataset_id=ds_row.dataset_id, dataset_name=ds_row.dataset_name)

    patients_by_id: Dict[str, PatientNode] = {}
    studies_by_uid: Dict[str, StudyNode] = {}
    series_by_uid: Dict[str, SeriesNode] = {}
    instances_by_uid: Dict[str, InstanceNode] = {}

    def _ensure_patient(pid: str) -> PatientNode:
        pnode = patients_by_id.get(pid)
        if pnode is not None:
            return pnode

        row = session.execute(
            select(PatientRow.PatientID, PatientRow.PatientName, PatientRow.extras_json).where(
                PatientRow.dataset_id == dataset_id,
                PatientRow.PatientID == pid,
            )
        ).one_or_none()

        if row is None:
            # If DB is inconsistent, still create a placeholder patient node.
            p = PatientNode(patient_id=pid, patient_name=None, parent_dataset=ds)
            ds.add_patient(p)
            patients_by_id[pid] = p
            return p

        pid2, pname2, pextras2 = row
        p = PatientNode(patient_id=pid2, patient_name=pname2, parent_dataset=ds)
        for k, v in (pextras2 or {}).items():
            p.set_attrs(**{k: v})
        ds.add_patient(p)
        patients_by_id[pid2] = p
        return p

    def _ensure_study(study_uid: str) -> StudyNode:
        st = studies_by_uid.get(study_uid)
        if st is not None:
            return st

        row = session.execute(
            select(
                StudyRow.PatientID,
                StudyRow.StudyInstanceUID,
                StudyRow.StudyDescription,
                StudyRow.extras_json,
            ).where(
                StudyRow.dataset_id == dataset_id,
                StudyRow.StudyInstanceUID == study_uid,
            )
        ).one_or_none()

        if row is None:
            raise RuntimeError(
                f"StudyInstanceUID={study_uid!r} referenced but "
                f"missing in DB (dataset_id={dataset_id!r})."
            )

        st_pid, st_uid, st_desc, st_extras = row
        parent_patient = _ensure_patient(st_pid)

        st = StudyNode(
            study_uid=st_uid,
            study_description=st_desc,
            parent_patient=parent_patient,
        )
        for k, v in (st_extras or {}).items():
            st.set_attrs(**{k: v})
        parent_patient.add_study(st)
        studies_by_uid[st_uid] = st
        return st

    def _ensure_series(series_uid: str) -> SeriesNode:
        se = series_by_uid.get(series_uid)
        if se is not None:
            return se

        srow = session.scalar(
            select(SeriesRow).where(
                SeriesRow.dataset_id == dataset_id,
                SeriesRow.SeriesInstanceUID == series_uid,
            )
        )
        if srow is None:
            raise RuntimeError(
                f"SeriesInstanceUID={series_uid!r} referenced but "
                f"missing in DB (dataset_id={dataset_id!r})."
            )

        parent_study = studies_by_uid.get(srow.StudyInstanceUID)
        if parent_study is None:
            parent_study = _ensure_study(srow.StudyInstanceUID)

        se = SeriesNode(series_uid=srow.SeriesInstanceUID, parent_study=parent_study)
        se.Modality = srow.Modality
        se.SeriesDescription = srow.SeriesDescription
        se.FrameOfReferenceUID = srow.FrameOfReferenceUID
        se.is_embedded_in_raw = bool(srow.is_embedded_in_raw)
        se.raw_series_reference_uid = srow.raw_series_ref_uid

        se.instance_paths = list(srow.instance_paths_json or [])
        se.referenced_sids = list(srow.referenced_sids_json or [])
        se.referencing_sids = list(srow.referencing_sids_json or [])

        for k, v in (srow.extras_json or {}).items():
            se.set_attrs(**{k: v})

        parent_study.add_series(se)
        series_by_uid[srow.SeriesInstanceUID] = se
        return se

    # -------------------------
    # 2a) Patient (single)
    # -------------------------
    # # Load only needed columns (avoid full ORM row)
    # pid, pname, pextras = session.execute(
    #     select(PatientRow.PatientID, PatientRow.PatientName, PatientRow.extras_json).where(
    #         PatientRow.dataset_id == dataset_id,
    #         PatientRow.PatientID == patient_id,
    #     )
    # ).one()
    p = _ensure_patient(patient_id)
    timers.mark("2a) load patient (cols)")

    # p = PatientNode(patient_id=pid, patient_name=pname, parent_dataset=ds)
    # for k, v in (pextras or {}).items():
    #     p.set_attrs(**{k: v})
    # ds.add_patient(p)
    # patients_by_id[pid] = p
    timers.mark("2b) build PatientNode")

    # -------------------------
    # 2b) Studies
    # -------------------------
    if study_uids:
        # Need PatientID too (paranoia / strictness), plus description + extras
        study_rows = session.execute(
            select(
                StudyRow.PatientID,
                StudyRow.StudyInstanceUID,
                StudyRow.StudyDescription,
                StudyRow.extras_json,
            ).where(
                StudyRow.dataset_id == dataset_id,
                StudyRow.StudyInstanceUID.in_(study_uids),
            )
        ).all()
    else:
        study_rows = []
    timers.mark("2c) load studies (cols)")

    for st_pid, st_uid, st_desc, st_extras in study_rows:
        # materialize under the actual patient of the study
        parent_patient = _ensure_patient(st_pid)
        if st_uid in studies_by_uid:
            continue

        # if st_pid != patient_id:
        #     # patient-local semantics: do not materialize other patients
        #     continue

        st = StudyNode(
            study_uid=st_uid,
            study_description=st_desc,
            parent_patient=p,
        )
        for k, v in (st_extras or {}).items():
            st.set_attrs(**{k: v})
        parent_patient.add_study(st)
        studies_by_uid[st_uid] = st
    timers.mark("2d) build StudyNodes")

    # -------------------------
    # 2c) Series
    # -------------------------
    # We use many series fields; full ORM row is fine, but you can column-select if desired.
    if series_uids:
        series_rows = session.scalars(
            select(SeriesRow).where(
                SeriesRow.dataset_id == dataset_id,
                SeriesRow.SeriesInstanceUID.in_(series_uids),
                SeriesRow.PatientID == patient_id,
            )
        ).all()
    else:
        series_rows = []
    timers.mark("2e) load series rows")

    for srow in series_rows:
        parent_study = studies_by_uid.get(srow.StudyInstanceUID)
        # if parent_study is None:
        #     raise RuntimeError(
        #         f"SeriesRow(SeriesInstanceUID={srow.SeriesInstanceUID!r}) refers to "
        #         f"missing StudyInstanceUID={srow.StudyInstanceUID!r} (patient-local load)"
        #     )

        if parent_study is None:
            parent_study = _ensure_study(srow.StudyInstanceUID)

        if srow.SeriesInstanceUID in series_by_uid:
            continue

        se = SeriesNode(series_uid=srow.SeriesInstanceUID, parent_study=parent_study)
        se.Modality = srow.Modality
        se.SeriesDescription = srow.SeriesDescription
        se.FrameOfReferenceUID = srow.FrameOfReferenceUID
        se.is_embedded_in_raw = bool(srow.is_embedded_in_raw)
        se.raw_series_reference_uid = srow.raw_series_ref_uid

        se.instance_paths = list(srow.instance_paths_json or [])
        se.referenced_sids = list(srow.referenced_sids_json or [])
        se.referencing_sids = list(srow.referencing_sids_json or [])

        for k, v in (srow.extras_json or {}).items():
            se.set_attrs(**{k: v})

        parent_study.add_series(se)
        series_by_uid[srow.SeriesInstanceUID] = se
    timers.mark("2f) build SeriesNodes")

    # -------------------------
    # 2d) Instances (ONLY needed columns)
    # -------------------------
    instance_rows = []
    if instance_uids:
        for chunk in _chunks(instance_uids):
            instance_rows.extend(
                session.execute(
                    select(
                        InstanceRow.SOPInstanceUID,
                        InstanceRow.SeriesInstanceUID,
                        InstanceRow.Modality,
                        InstanceRow.file_path,
                        InstanceRow.frame_of_reference_uids_json,
                        InstanceRow.referenced_sop_uids_json,
                        InstanceRow.referenced_sids_json,
                        InstanceRow.other_referenced_sids_json,
                        InstanceRow.sources_json,
                        InstanceRow.extras_json,
                    ).where(
                        InstanceRow.dataset_id == dataset_id,
                        InstanceRow.SOPInstanceUID.in_(chunk),
                        InstanceRow.PatientID == patient_id,
                    )
                ).all()
            )
    timers.mark("2g) load instances (needed cols)")

    for (
        sop,
        series_uid,
        modality,
        file_path,
        for_uids_json,
        ref_sop_json,
        ref_sids_json,
        other_ref_sids_json,
        sources_json,
        extras_json,
    ) in instance_rows:
        parent_series = series_by_uid.get(series_uid)
        # if parent_series is None:
        #     raise RuntimeError(
        #         f"InstanceRow(SOPInstanceUID={sop!r}) refers to "
        #         f"missing SeriesInstanceUID={series_uid!r} (patient-local load)"
        #     )

        if parent_series is None:
            parent_series = _ensure_series(series_uid)

        if sop in instances_by_uid:
            continue

        inst = InstanceNode(
            SOPInstanceUID=sop,
            FilePath=file_path,
            Modality=modality,
            parent_series=parent_series,
        )

        inst.FrameOfReferenceUIDs = list(for_uids_json or [])
        inst.referenced_sop_instance_uids = list(ref_sop_json or [])
        inst.referenced_sids = list(ref_sids_json or [])
        inst.other_referenced_sids = list(other_ref_sids_json or [])
        inst.sources = list(sources_json or [])

        for k, v in (extras_json or {}).items():
            inst.set_attrs(**{k: v})

        parent_series.add_instance(inst)
        instances_by_uid[sop] = inst
    timers.mark("2h) build InstanceNodes")

    # -------------------------
    # 3) Associate using JSON reference fields
    # -------------------------
    from rosamllib.utils import associate_dicoms

    associate_dicoms(ds)
    timers.mark("3) associate_dicoms")

    out = patients_by_id.get(patient_id)
    if out is None:
        raise RuntimeError(
            f"PatientID={patient_id!r} was not materialized (dataset_id={dataset_id!r})."
        )

    # timers.report(header=f"load_patient(patient-local, dataset={dataset_id}, patient=...)")
    return out
