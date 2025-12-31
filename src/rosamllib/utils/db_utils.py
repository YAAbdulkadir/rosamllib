from pathlib import Path
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from rosamllib.utils import parse_vr_value
from rosamllib.utils.dicom_utils import (
    get_referenced_sop_instance_uids,
    extract_rtstruct_for_uids,
)
import warnings
from rosamllib.readers import REGReader, DICOMRawReader
from rosamllib.db.db_nodes import PatientRow, StudyRow, SeriesRow, InstanceRow
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pydicom.tag import Tag
from itertools import chain
from pydicom.datadict import keyword_for_tag, tag_for_keyword, dictionary_VR
from sqlalchemy import select
from sqlalchemy.orm import Session
from queue import Queue
from sqlalchemy.exc import OperationalError, IntegrityError, SQLAlchemyError

warnings.filterwarnings(
    "ignore",
    message=r"Invalid value for VR",
)
warnings.filterwarnings("ignore", message=r"The value")


logger = logging.getLogger(__name__)


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

IMAGE_TAGS_NEEDED = [
    "ImageType",
    "StudyDate",
    "SeriesDate",
    "StudyTime",
    "SeriesTime",
    "Manufacturer",
    "ReferringPhysicianName",
    "StationName",
    "PhysiciansOfRecord",
    "OperatorsName",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "SliceThickness",
    "PatientPosition",
    "StudyID",
    "SeriesNumber",
    "ImageOrientationPatient",
    "FrameOfReferenceUID",
    "Rows",
    "Columns",
    "PixelSpacing",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
]

NETWORK_LOCK = threading.Lock()
MIN_NET_INTERVAL_S = 0.0


@dataclass
class ParseJob:
    dcm: object
    query_ae: str
    dataset_id: str
    filepath: str = ""


@dataclass
class WriteJob:
    inst_dicts: dict
    embedded: list
    dataset_id: str


@dataclass
class Pipeline:
    parse_q: Queue
    write_q: Queue
    parser_threads: list[threading.Thread]
    writer_t: threading.Thread


def start_pipeline(
    session_factory, *, parse_workers: int, write_commit_every: int, tag_plan
) -> Pipeline:
    parse_q: Queue = Queue(maxsize=128)
    write_q: Queue = Queue(maxsize=2000)

    writer_t = threading.Thread(
        target=_db_writer_worker,
        args=(write_q, session_factory),
        kwargs={"commit_every": write_commit_every, "tag_plan": tag_plan},
        daemon=True,
    )
    writer_t.start()

    parser_threads: list[threading.Thread] = []
    for _ in range(max(1, parse_workers)):
        t = threading.Thread(target=_parser_worker, args=(parse_q, write_q, tag_plan), daemon=True)
        t.start()
        parser_threads.append(t)

    return Pipeline(
        parse_q=parse_q, write_q=write_q, parser_threads=parser_threads, writer_t=writer_t
    )


def stop_pipeline(p: Pipeline):
    # stop parsers
    for _ in p.parser_threads:
        p.parse_q.put(_STOP)
    p.parse_q.join()

    # stop writer
    p.write_q.put(_STOP)
    p.write_q.join()

    # join threads
    for t in p.parser_threads:
        t.join(timeout=2.0)
    p.writer_t.join(timeout=2.0)


_STOP = object()


def _parser_worker(parse_q: Queue, write_q: Queue, tag_plan):
    while True:
        job = parse_q.get()
        try:
            if job is _STOP:
                return  # task_done will happen in finally

            dcm = job.dcm
            embedded_instances_list = []

            mod = getattr(dcm, "Modality", None)
            fp = getattr(job, "filepath", "") or ""

            if mod in ["RTPLAN", "RTSTRUCT"]:
                inst_dicts = process_standard_dicom(dcm, fp, job.query_ae, tag_plan, None)
                if mod == "RTPLAN":
                    if hasattr(dcm, "FractionGroupSequence"):
                        extract_nested_tags(dcm.FractionGroupSequence[0], inst_dicts)
                    if hasattr(dcm, "BeamSequence"):
                        inst_dicts.update(group_sequence_item_values(dcm.BeamSequence))
                if mod == "RTSTRUCT":
                    inst_dicts.update({"ROIName": get_structure_names(dcm)})

            elif mod == "REG":
                inst_dicts = process_reg_file(dcm, fp, job.query_ae, tag_plan, None)

            elif mod == "RAW":
                inst_dicts, embedded_instances_list = process_raw_file(
                    dcm, fp, job.query_ae, tag_plan, None
                )

            elif mod in ["CT", "MR", "PT", "RTDOSE"]:
                inst_dicts = process_standard_dicom(dcm, fp, job.query_ae, tag_plan, None)

            elif mod == "RTIMAGE":
                inst_dicts = process_other_file(dcm, fp, job.query_ae, tag_plan, None)

            elif mod == "RTRECORD":
                inst_dicts = process_other_file(dcm, fp, job.query_ae, tag_plan, None)
                extract_nested_tags(dcm, inst_dicts)

            else:
                inst_dicts = process_other_file(dcm, fp, job.query_ae, tag_plan, None)

            write_q.put(
                WriteJob(
                    inst_dicts=inst_dicts,
                    embedded=embedded_instances_list,
                    dataset_id=job.dataset_id,
                )
            )

        except Exception:
            logger.exception(
                "Parser failed: AE=%s Modality=%s SOP=%s",
                getattr(job, "query_ae", None),
                getattr(getattr(job, "dcm", None), "Modality", None),
                getattr(getattr(job, "dcm", None), "SOPInstanceUID", None),
            )
        finally:
            parse_q.task_done()


def _commit_with_retry(session, max_retries: int = 6, base_sleep_s: float = 0.2):
    """
    Commit with retry/backoff for transient SQLite lock errors.
    Rolls back on any failure.
    """
    for attempt in range(max_retries + 1):
        try:
            session.commit()
            return
        except OperationalError as exc:
            session.rollback()
            msg = str(exc).lower()
            if "database is locked" in msg or "locked" in msg:
                time.sleep(base_sleep_s * (2**attempt))
                continue
            raise


def _db_writer_worker(write_q: Queue, session_factory, *, commit_every: int = 500, tag_plan=None):
    n = 0
    with session_factory() as session:
        while True:
            job = write_q.get()
            try:
                if job is _STOP:
                    # final flush before exit
                    _commit_with_retry(session)
                    return  # task_done in finally

                build_node_db(job.inst_dicts, job.dataset_id, session, tag_plan=tag_plan)
                for embedded_inst in job.embedded or []:
                    build_node_db(embedded_inst, job.dataset_id, session, tag_plan=tag_plan)

                n += 1
                if n % commit_every == 0:
                    _commit_with_retry(session)

            except IntegrityError:
                session.rollback()
                logger.exception("DB writer: IntegrityError (rolled back)")
            except OperationalError:
                session.rollback()
                logger.exception("DB writer: OperationalError (rolled back)")
            except SQLAlchemyError:
                session.rollback()
                logger.exception("DB writer: SQLAlchemyError (rolled back)")
            except Exception:
                session.rollback()
                logger.exception("DB writer: unexpected error (rolled back)")
            finally:
                write_q.task_done()


def setup_module_logging(
    log_file_path: Union[str, Path] = "dicom_mapper.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 1,
    log_to_console: bool = False,
    logger_name: Optional[str] = None,
    logger_obj: Optional[logging.Logger] = None,
    propagage: bool = False,
):
    """
    Configure a rotating file handler for the specified logger.

    Use either:
      - logger_obj=<logging.Logger>, OR
      - logger_name="some.module.name"
    If neither is given, defaults to this module's logger.
    """
    if logger_obj is not None:
        target_logger = logger_obj
    elif logger_name is not None:
        target_logger = logging.getLogger(logger_name)
    else:
        target_logger = logger

    target_logger.setLevel(level)
    target_logger.propagate = propagage

    log_file_path = str(log_file_path)

    # Avoid duplicates: check for a rotating handler pointing at the same file
    for h in target_logger.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if getattr(h, "baseFilename", None) == str(Path(log_file_path).resolve()):
                    return
            except Exception:
                # If we can't compare safely, fall back to not adding duplicates
                return

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    target_logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        target_logger.addHandler(ch)

    target_logger.info(
        (
            "setup_module_logging: rotating file handler initialized "
            "at %s (max_bytes=%d, backup_count=%d)"
        ),
        log_file_path,
        max_bytes,
        backup_count,
    )


@dataclass(frozen=True)
class TagPlan:
    tag: Tag
    name: str
    vr: str
    is_sq: bool
    # parser: Any


def get_metadata(ds, plan: list[TagPlan], seq_policy: str = "json") -> Dict[str, Any]:
    sop = getattr(ds, "SOPInstanceUID", None)
    logger.debug(
        "get_metadata: extracting metadata for SOPInstanceUID=%s with seq_policy=%s",
        sop,
        seq_policy,
    )
    out = {
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
        "Modality": getattr(ds, "Modality", None),
        "FrameOfReferenceUID": getattr(ds, "FrameOfReferenceUID", None),
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
        "StudyDescription": getattr(ds, "StudyDescription", ""),
        "PatientID": getattr(ds, "PatientID", None),
        "PatientName": str(getattr(ds, "PatientName", "")),
    }

    for tp in plan:
        try:
            if tp.tag in ds:
                val = ds[tp.tag].value
                if tp.is_sq:
                    if seq_policy == "drop":
                        continue
                    elif seq_policy == "len":
                        out[tp.name] = len(val) if val is not None else 0
                    else:
                        out[tp.name] = ds[tp.tag].to_json() if val else None
                else:
                    out[tp.name] = parse_vr_value(tp.vr, val)
            else:
                if tp.is_sq:
                    if seq_policy == "drop":
                        continue
                    elif seq_policy == "len":
                        out[tp.name] = 0
                    else:
                        out[tp.name] = None
                else:
                    out[tp.name] = None
        except Exception as exc:
            logger.warning(
                "get_metadata: failed to parse tag %s (%s) for SOP=%s: %s",
                tp.name,
                tp.tag,
                sop,
                exc,
            )
            out[tp.name] = None
    update = {
        "SeriesDescription": str(getattr(ds, "SeriesDescription", "")),
    }
    out.update(update)
    logger.debug(
        (
            "get_metadata: extracted core meta for SOPInstanceUID=%s "
            "PatientID=%s SeriesInstanceUID=%s Modality=%s"
        ),
        out.get("SOPInstanceUID"),
        out.get("PatientID"),
        out.get("SeriesInstanceUID"),
        out.get("Modality"),
    )
    return out


def extract_nested_tags(
    ds: Dataset,
    out: dict,
    *,
    prefix: str = "",
):
    """
    Recursively extract all DICOM elements (including nested sequences),
    using the DICOM keyword as the key and string-cast value.
    """
    for elem in ds:
        # Skip elements without a keyword (rare but possible)
        if elem.keyword:
            key = elem.keyword
        else:
            continue

        if elem.VR == "SQ" and isinstance(elem.value, Sequence):
            for item in elem.value:
                extract_nested_tags(item, out)
        else:
            out[key] = parse_vr_value(elem.VR, elem.value)


def group_sequence_item_values(seq: Sequence) -> dict[str, list]:
    """
    For a multi-item DICOM Sequence, aggregate each element keyword across items.
    Only includes TOP-LEVEL (non-sequence) elements from each item.
    Values are cast to str.
    """
    grouped: dict[str, list[str]] = {}

    for item in seq:  # each item is a pydicom.Dataset
        for elem in item:
            if elem.VR == "SQ":
                continue  # skip nested sequences in this version

            if not elem.keyword:
                continue

            grouped.setdefault(elem.keyword, []).append(parse_vr_value(elem.VR, elem.value))

    return grouped


def get_structure_names(ds: Dataset) -> List[str]:
    """
    Extract all ROI names from an RTSTRUCT dataset.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        A DICOM dataset, typically an RT Structure Set (RTSTRUCT).
        If it contains a `StructureSetROISequence`, each item is expected
        to have an `ROIName` attribute.

    Returns
    -------
    list[str]
        A list of ROIName strings in the order they appear in
        `StructureSetROISequence`. Returns an empty list if the sequence
        is missing or empty.
    """
    if hasattr(ds, "StructureSetROISequence") and ds.StructureSetROISequence:
        return [str(structure.ROIName) for structure in ds.StructureSetROISequence]
    return []


def process_standard_dicom(ds, filepath, source, tag_plan, seq_policy):
    modality = getattr(ds, "Modality", None)
    sop = getattr(ds, "SOPInstanceUID", None)
    logger.debug(
        "process_standard_dicom: SOPInstanceUID=%s Modality=%s " "source=%s filepath=%s",
        sop,
        modality,
        source,
        filepath,
    )
    metadata = get_metadata(ds, tag_plan, seq_policy)
    instance_dict = {"FilePath": filepath, "source": source, **metadata}

    if modality in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
        refs_map = get_referenced_sop_instance_uids(ds)
        instance_dict["ReferencedSOPInstanceUIDs"] = list(chain.from_iterable(refs_map.values()))
        if modality == "RTSTRUCT":
            instance_dict["RTStructFoRUIDs"] = extract_rtstruct_for_uids(ds)

    return instance_dict


def process_other_file(ds, filepath, source, tag_plan, seq_policy):
    sop = getattr(ds, "SOPInstanceUID", None)
    modality = getattr(ds, "Modality", None)
    logger.debug(
        "process_other_file: SOPInstanceUID=%s Modality=%s source=%s filepath=%s",
        sop,
        modality,
        source,
        filepath,
    )
    metadata = get_metadata(ds, tag_plan, seq_policy)
    instance_dict = {"FilePath": filepath, "source": source, **metadata}
    refs_map = get_referenced_sop_instance_uids(ds)
    instance_dict["ReferencedSOPInstanceUIDs"] = list(chain.from_iterable(refs_map.values()))
    return instance_dict


def process_reg_file(ds, filepath, source, tag_plan, seq_policy):
    logger.debug(
        "process_reg_file: starting REG processing for SOPInstanceUID=%s",
        getattr(ds, "SOPInstanceUID", None),
    )
    reg = REGReader(ds).read()
    metadata = get_metadata(reg, tag_plan, seq_policy)
    instance_dict = {
        "FilePath": filepath,
        "source": source,
        "ReferencedSeriesUIDs": reg.get_fixed_image_info()["SeriesInstanceUID"],
        "OtherReferencedSeriesUIDs": reg.get_moving_image_info()["SeriesInstanceUID"],
        **metadata,
    }

    return instance_dict


def process_raw_file(ds, filepath, source, tag_plan, seq_policy):
    logger.debug(
        "process_raw_file: starting RAW processing for SOPInstanceUID=%s",
        getattr(ds, "SOPInstanceUID", None),
    )
    raw_reader = DICOMRawReader(ds)
    raw = raw_reader.read()
    metadata = get_metadata(raw, tag_plan, seq_policy)
    instance_dict = {
        "FilePath": filepath,
        "source": source,
        **metadata,
    }
    embedded_instances = []
    try:
        embedded_datasets = raw.get_embedded_datasets()

        for embedded_ds in embedded_datasets:
            embedded_metadata = get_metadata(embedded_ds, tag_plan, seq_policy)
            embedded_instance_dict = {
                **embedded_metadata,
                "FilePath": filepath,
                "source": source,
                "is_embedded_in_raw": True,
                "raw_series_reference_uid": instance_dict["SeriesInstanceUID"],
            }
            if embedded_instance_dict["Modality"] in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
                refs_map = get_referenced_sop_instance_uids(embedded_ds)
                embedded_instance_dict["ReferencedSOPInstanceUIDs"] = list(
                    chain.from_iterable(refs_map.values())
                )

            elif embedded_instance_dict["Modality"] == "REG":
                embedded_reg = REGReader(embedded_ds).read()
                embedded_instance_dict["ReferencedSeriesUIDs"] = (
                    embedded_reg.get_fixed_image_info()["SeriesInstanceUID"]
                )
                embedded_instance_dict["OtherReferencedSeriesUIDs"] = (
                    embedded_reg.get_moving_image_info()["SeriesInstanceUID"]
                )
            embedded_instances.append(embedded_instance_dict)
    except Exception:
        pass

    return instance_dict, embedded_instances


def _normalize_tag(tag):
    """
    Normalize to a (group, element) tuple of ints.
    Accepts DICOM keyword ('PatientID') or any Tag-like input.
    """
    try:
        if isinstance(tag, tuple) and len(tag) == 2:
            # allow ('0008', '0018') or (0x0008, 0x0018) etc.
            g = int(tag[0], 16) if isinstance(tag[0], str) else int(tag[0])
            e = int(tag[1], 16) if isinstance(tag[1], str) else int(tag[1])
            return (g, e)
        # keyword
        t = Tag(tag_for_keyword(str(tag)))
        if t is None:
            raise ValueError(f"Unknown keyword '{tag}'")
        return (t.group, t.element)
    except Exception:
        # print(f"Unknown tag/keyword '{tag}' ignored.")
        return None


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


def _extend_unique_list(base, values):
    """
    Helper: extend a list with unique values, handling scalar vs iterable.
    Returns a new list.
    """
    if base is None:
        base = []
    out = list(base)
    if not values:
        return out

    if isinstance(values, (list, tuple, set)):
        for v in values:
            if v is None:
                continue
            if v not in out:
                out.append(v)
    else:
        if values is not None and values not in out:
            out.append(values)
    return out


def build_node_db(inst_dict: Dict[str, Any], dataset_id: str, session: Session, tag_plan) -> None:
    """
    DB-backed version of build_node.

    Creates / updates PatientRow, StudyRow, SeriesRow, InstanceRow
    and populates JSON list fields + extras_json from inst_dict.
    """
    sop_instance_uid = inst_dict["SOPInstanceUID"]
    patient_id = inst_dict.get("PatientID")
    patient_name = inst_dict.get("PatientName")
    # print(type(patient_name))
    study_uid = inst_dict.get("StudyInstanceUID")
    study_desc = inst_dict.get("StudyDescription")
    series_uid = inst_dict.get("SeriesInstanceUID")
    series_desc = inst_dict.get("SeriesDescription")
    modality = (inst_dict.get("Modality") or "").upper()
    filepath = inst_dict.get("FilePath")
    source = inst_dict.get("source")

    logger.debug(
        "build_node_db: SOP=%s Modality=%s PatientID=%s StudyUID=%s SeriesUID=%s source=%s",
        sop_instance_uid,
        modality,
        patient_id,
        study_uid,
        series_uid,
        source,
    )

    # ------------------------
    # 1) Patient
    # ------------------------
    if patient_id is None:
        # extremely unlikely, but don't blow up the pipeline
        logger.warning("build_node_db: missing PatientID for SOP=%s", sop_instance_uid)
        return
    else:
        prow = session.scalar(
            select(PatientRow).where(
                PatientRow.dataset_id == dataset_id,
                PatientRow.PatientID == patient_id,
            )
        )
        if prow is None:
            prow = PatientRow(
                dataset_id=dataset_id,
                PatientID=patient_id,
                PatientName=patient_name,
            )
            session.add(prow)
        else:
            if patient_name and prow.PatientName != patient_name:
                prow.PatientName = patient_name

    # ------------------------
    # 2) Study
    # ------------------------
    if study_uid is not None:
        srow = session.scalar(
            select(StudyRow).where(
                StudyRow.dataset_id == dataset_id,
                StudyRow.StudyInstanceUID == study_uid,
            )
        )
        if srow is None:
            srow = StudyRow(
                dataset_id=dataset_id,
                StudyInstanceUID=study_uid,
                PatientID=patient_id,
                StudyDescription=study_desc,
            )
            session.add(srow)
        else:
            if study_desc and not srow.StudyDescription:
                srow.StudyDescription = study_desc

    # ------------------------
    # 3) Series
    # ------------------------
    if series_uid is None:
        logger.warning(
            "build_node_db: missing SeriesInstanceUID for SOP=%s; "
            "skipping series/instance creation",
            sop_instance_uid,
        )
        return

    ser = session.scalar(
        select(SeriesRow).where(
            SeriesRow.dataset_id == dataset_id,
            SeriesRow.SeriesInstanceUID == series_uid,
        )
    )
    if ser is None:
        ser = SeriesRow(
            dataset_id=dataset_id,
            PatientID=patient_id,
            SeriesInstanceUID=series_uid,
            StudyInstanceUID=study_uid,
            Modality=modality or None,
            SeriesDescription=series_desc,
            FrameOfReferenceUID=inst_dict.get("FrameOfReferenceUID"),
            is_embedded_in_raw=bool(inst_dict.get("is_embedded_in_raw", False)),
            raw_series_ref_uid=inst_dict.get("raw_series_reference_uid"),
            instance_paths_json=None,
            referenced_sids_json=None,
            referencing_sids_json=None,
            extras_json=None,
        )
        session.add(ser)
    else:
        # fill in missing metadata, but don't clobber existing
        if modality and not ser.Modality:
            ser.Modality = modality
        if series_desc and not ser.SeriesDescription:
            ser.SeriesDescription = series_desc
        if not ser.FrameOfReferenceUID:
            ser.FrameOfReferenceUID = inst_dict.get("FrameOfReferenceUID")
        if inst_dict.get("is_embedded_in_raw"):
            if not ser.is_embedded_in_raw:
                ser.is_embedded_in_raw = True
            raw_ref = inst_dict.get("raw_series_reference_uid")
            if raw_ref and not ser.raw_series_ref_uid:
                ser.raw_series_ref_uid = raw_ref

    # Optionally track instance file paths at series level
    if filepath:
        ser.instance_paths_json = _extend_unique_list(ser.instance_paths_json, filepath)

    # ------------------------
    # 4) Instance
    # ------------------------
    inst = session.scalar(
        select(InstanceRow).where(
            InstanceRow.dataset_id == dataset_id,
            InstanceRow.SOPInstanceUID == sop_instance_uid,
        )
    )

    if inst is None:
        inst = InstanceRow(
            dataset_id=dataset_id,
            SOPInstanceUID=sop_instance_uid,
            SeriesInstanceUID=series_uid,
            StudyInstanceUID=study_uid,
            PatientID=patient_id,
            file_path=filepath or "",
            Modality=modality or None,
            frame_of_reference_uids_json=None,
            referenced_sop_uids_json=None,
            referenced_sids_json=None,
            other_referenced_sids_json=None,
            sources_json=None,
            extras_json=None,
        )
        session.add(inst)

    # ------------------------
    # 5) Populate JSON list fields
    # ------------------------

    # RTSTRUCT: FrameOfReferenceUIDs may come from RTStructFoRUIDs
    if modality == "RTSTRUCT":
        rtstruct_fors = inst_dict.get("RTStructFoRUIDs")
        inst.frame_of_reference_uids_json = _extend_unique_list(
            inst.frame_of_reference_uids_json,
            rtstruct_fors,
        )

    # Referenced SOPs (RTPLAN/RTDOSE/RTSTRUCT/RTRECORD etc.)
    refs = inst_dict.get("ReferencedSOPInstanceUIDs")
    if refs:
        inst.referenced_sop_uids_json = _extend_unique_list(
            inst.referenced_sop_uids_json,
            refs,
        )

    # REG / SEG: referenced series UIDs + other referenced series UIDs
    if modality in {"REG", "SEG"}:
        ref_sid = inst_dict.get("ReferencedSeriesUIDs")
        if ref_sid:
            inst.referenced_sids_json = _extend_unique_list(
                inst.referenced_sids_json,
                ref_sid,
            )
        if modality == "REG":
            other_sid = inst_dict.get("OtherReferencedSeriesUIDs")
            if other_sid:
                inst.other_referenced_sids_json = _extend_unique_list(
                    inst.other_referenced_sids_json,
                    other_sid,
                )

    # Sources (AEs)
    if source:
        inst.sources_json = _extend_unique_list(inst.sources_json, source)

    # ------------------------
    # 6) Persist DICOM-tag columns from tag_plan onto InstanceRow
    # ------------------------

    inst_cols = set(InstanceRow.__table__.columns.keys())

    for tp in tag_plan:
        col_name = tp.name
        # We already handle some explicitly; skipping is optional but avoids double-work.
        if col_name not in inst_cols:
            continue
        if col_name not in inst_dict:
            continue

        value = inst_dict[col_name]
        # Let SQLAlchemy / the DB driver handle type conversion based on column type.
        setattr(inst, col_name, value)
