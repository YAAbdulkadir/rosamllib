import argparse
import json
import threading
import time
import logging
import pandas as pd
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from rosamllib.networking import QueryRetrieveSCU, StoreSCP
from rosamllib.utils import parse_vr_value
from rosamllib.utils.dicom_utils import (
    get_referenced_sop_instance_uids,
    extract_rtstruct_for_uids,
)
from tqdm import tqdm
import warnings
from rosamllib.readers import REGReader, DICOMRawReader
from rosamllib.db.db_nodes import DatasetNodeDB
from rosamllib.db.db_schema import (
    init_schema,
    DatasetRow,
    PatientRow,
    StudyRow,
    SeriesRow,
    InstanceRow,
)
from dataclasses import dataclass
from typing import Any, Dict, List
from pydicom.tag import Tag
from itertools import chain
from pydicom.datadict import keyword_for_tag, tag_for_keyword, dictionary_VR
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from rosamllib.constants import CLASS_UID_BY_MODALITY
from queue import Queue
from sqlalchemy.exc import OperationalError, IntegrityError, SQLAlchemyError

warnings.filterwarnings(
    "ignore",
    message=r"Invalid value for VR",
)
warnings.filterwarnings("ignore", message=r"The value")

logger = logging.getLogger(__name__)

NETWORK_LOCK = threading.Lock()
MIN_NET_INTERVAL_S = 0.0


@dataclass
class ParseJob:
    dcm: object
    query_ae: str
    dataset_id: str


@dataclass
class WriteJob:
    inst_dicts: dict
    embedded: list
    dataset_id: str


_STOP = object()


@dataclass
class Pipeline:
    parse_q: Queue
    write_q: Queue
    parser_threads: list[threading.Thread]
    writer_t: threading.Thread


def start_pipeline(session_factory, *, parse_workers: int, write_commit_every: int) -> Pipeline:
    parse_q: Queue = Queue(maxsize=2000)
    write_q: Queue = Queue(maxsize=2000)

    writer_t = threading.Thread(
        target=_db_writer_worker,
        args=(write_q, session_factory),
        kwargs={"commit_every": write_commit_every},
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


def _parser_worker(parse_q: Queue, write_q: Queue, tag_plan):
    while True:
        job = parse_q.get()
        try:
            if job is _STOP:
                return  # task_done will happen in finally

            dcm = job.dcm
            embedded_instances_list = []

            mod = getattr(dcm, "Modality", None)

            if mod in ["RTPLAN", "RTSTRUCT"]:
                inst_dicts = process_standard_dicom(dcm, "", job.query_ae, tag_plan, None)
                if mod == "RTPLAN":
                    if hasattr(dcm, "FractionGroupSequence"):
                        extract_nested_tags(dcm.FractionGroupSequence[0], inst_dicts)
                    if hasattr(dcm, "BeamSequence"):
                        inst_dicts.update(group_sequence_item_values(dcm.BeamSequence))
                if mod == "RTSTRUCT":
                    inst_dicts.update({"ROIName": get_structure_names(dcm)})

            elif mod == "REG":
                inst_dicts = process_reg_file(dcm, "", job.query_ae, tag_plan, None)

            elif mod == "RAW":
                inst_dicts, embedded_instances_list = process_raw_file(
                    dcm, "", job.query_ae, tag_plan, None
                )

            elif mod in ["CT", "MR", "PT", "RTDOSE"]:
                inst_dicts = process_standard_dicom(dcm, "", job.query_ae, tag_plan, None)

            elif mod == "RTIMAGE":
                inst_dicts = process_other_file(dcm, "", job.query_ae, tag_plan, None)

            elif mod == "RTRECORD":
                inst_dicts = process_other_file(dcm, "", job.query_ae, tag_plan, None)
                extract_nested_tags(dcm, inst_dicts)

            else:
                inst_dicts = process_other_file(dcm, "", job.query_ae, tag_plan, None)

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


def _db_writer_worker(write_q: Queue, session_factory, *, commit_every: int = 500):
    n = 0
    with session_factory() as session:
        while True:
            job = write_q.get()
            try:
                if job is _STOP:
                    # final flush before exit
                    _commit_with_retry(session)
                    return  # task_done in finally

                build_node_db(job.inst_dicts, job.dataset_id, session)
                for embedded_inst in job.embedded or []:
                    build_node_db(embedded_inst, job.dataset_id, session)

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
    log_file_path: str = "dicom_mapper.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 1,
    log_to_console: bool = False,
):
    """
    Configure a rotating file handler for this module's logger.

    Parameters
    ----------
    log_file_path : str
        Path to the log file (rotation will use .1, .2, ...).
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    max_bytes : int
        Maximum size in bytes before rotation.
    backup_count : int
        Number of rotated log files to keep.
    log_to_console : bool
        If True, also log to stderr.
    """
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
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
    logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.info(
        (
            "setup_module_logging: rotating file handler initialized "
            "at %s (max_bytes=%d, backup_count=%d)"
        ),
        log_file_path,
        max_bytes,
        backup_count,
    )


def load_config(config_path: str) -> dict[str, Any]:
    logger.info("load_config: loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "remote_aes" not in cfg or not cfg["remote_aes"]:
        logger.error("load_config: 'remote_aes' missing or empty in config")
        raise ValueError("Config must define at least one remote AE in 'remote_aes'")

    logger.info(
        "load_config: loaded %d remote AEs, scu_ae_title=%s",
        len(cfg["remote_aes"]),
        cfg.get("scu_ae_title"),
    )
    return cfg


def configure_scu_from_config(cfg: dict[str, Any]) -> tuple[QueryRetrieveSCU, list[str]]:
    scu_ae_title = cfg.get("scu_ae_title")
    logger.info("configure_scu_from_config: initializing SCU with AE title %s", scu_ae_title)

    scu = QueryRetrieveSCU(scu_ae_title)

    ae_names: list[str] = []
    for ae_cfg in cfg["remote_aes"]:
        name = ae_cfg["name"]
        ae_title = ae_cfg["ae_title"]
        ip = ae_cfg["ip"]
        port = int(ae_cfg["port"])
        logger.info(
            "configure_scu_from_config: adding remote AE name=%s ae_title=%s ip=%s port=%d",
            name,
            ae_title,
            ip,
            port,
        )
        scu.add_remote_ae(name, ae_title, ip, port)
        ae_names.append(name)

    return scu, ae_names


def load_patient_ids_from_csv(csv_path: str) -> list[str]:
    """
    Load unique PatientIDs from a CSV file with a 'PatientID' column.
    Empty/missing PatientIDs are skipped.
    """
    logger.info("load_patient_ids_from_csv: loading PatientIDs from %s", csv_path)

    df = pd.read_csv(csv_path, dtype=str)

    if "PatientID" not in df.columns:
        msg = f"CSV {csv_path!r} must contain a 'PatientID' column"
        logger.error("load_patient_ids_from_csv: %s", msg)
        raise ValueError(msg)

    pid_series = df["PatientID"].astype(str).str.strip()

    pid_series = pid_series.replace("", pd.NA).dropna()

    patient_ids = pid_series.drop_duplicates().tolist()

    logger.info(
        "load_patient_ids_from_csv: loaded %d unique PatientIDs from %s " "(raw_rows=%d)",
        len(patient_ids),
        csv_path,
        len(df),
    )
    return patient_ids


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
        "PatientName": getattr(ds, "PatientName", None),
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


def build_node_db(inst_dict: Dict[str, Any], dataset_id: str, session: Session) -> None:
    """
    DB-backed version of build_node.

    Creates / updates PatientRow, StudyRow, SeriesRow, InstanceRow
    and populates JSON list fields + extras_json from inst_dict.
    """
    sop_instance_uid = inst_dict["SOPInstanceUID"]
    patient_id = inst_dict.get("PatientID")
    patient_name = inst_dict.get("PatientName")
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


def query_all_series(patient_id, query_ae, scu):
    logger.info(
        "query_all_series: querying SERIES for PatientID=%s on AE=%s",
        patient_id,
        query_ae,
    )
    series_ds = Dataset()
    series_ds.PatientID = patient_id
    series_ds.StudyInstanceUID = ""
    series_ds.SeriesInstanceUID = ""
    series_ds.SOPInstanceUID = ""
    series_ds.Modality = ""
    series_ds.QueryRetrieveLevel = "SERIES"
    series_results = scu.c_find(query_ae, series_ds)
    logger.info(
        "query_all_series: received %d series for PatientID=%s from AE=%s",
        len(series_results),
        patient_id,
        query_ae,
    )
    return series_results


def create_node_map(
    series_results,
    query_ae,
    staging_ae,
    dataset_id: str,
    scu,
    parse_q: Queue,
    exclude_modalities=None,
):
    """
    Network ops are serialized (no simultaneous AE load).
    Parsing happens concurrently (thread pool).
    DB writes happen in a single writer thread (safe for SQLite).
    """
    if exclude_modalities is None:
        exclude_modalities = []

    if not series_results:
        return

    pid = series_results[0].PatientID
    logger.info(
        "create_node_map: start building node map from %s series for PatientID=%s from AE=%s",
        len(series_results),
        pid,
        query_ae,
    )

    # -------------------------
    # NETWORK PRODUCER (serialized)
    # -------------------------
    for series_result in tqdm(
        series_results,
        total=len(series_results),
        desc=f"{pid} | {query_ae}",
        unit="series",
        position=1,
        leave=False,
    ):
        if series_result.Modality in exclude_modalities:
            continue

        try:
            inst_ds = series_result
            inst_ds.SOPInstanceUID = ""
            inst_ds.SOPClassUID = ""
            inst_ds.StudyDescription = ""
            inst_ds.SeriesDescription = ""
            inst_ds.QueryRetrieveLevel = "IMAGE"

            # --- RTPLAN / RTSTRUCT / REG / RAW (C-FIND then C-MOVE each instance) ---
            if series_result.Modality in ["RTPLAN", "RTSTRUCT", "REG", "RAW"]:
                inst_results = scu.c_find(query_ae, inst_ds) or []
                if inst_results:
                    for inst_result in inst_results:
                        StoreSCP.drain_received(timeout_s=0.0, idle_grace_s=0.0)
                        move_status = scu.c_move(query_ae, inst_result, staging_ae)
                        logger.debug(
                            "create_node_map: C-MOVE status for SOP=%s: %s",
                            getattr(inst_result, "SOPInstanceUID", None),
                            f"0x{int(move_status.status):04X}",
                        )
                        received = StoreSCP.drain_received(idle_grace_s=0.0)
                        if not received:
                            logger.warning(
                                "create_node_map: no objects received for move of SOP=%s",
                                getattr(inst_result, "SOPInstanceUID", "<unknown>"),
                            )
                            continue

                        for dcm in received:
                            parse_q.put(
                                ParseJob(dcm=dcm, query_ae=query_ae, dataset_id=dataset_id)
                            )

            # --- CT / MR / PT (sniff first slice; copy IMAGE_TAGS_NEEDED;
            # enqueue each inst_result) ---
            elif series_result.Modality in ["CT", "MR", "PT"]:
                inst_results = scu.c_find(query_ae, inst_ds) or []
                if inst_results:
                    first_inst = inst_results[0]
                    StoreSCP.drain_received(timeout_s=0.0, idle_grace_s=0.0)
                    scu.c_move(query_ae, first_inst, staging_ae)
                    sniff = StoreSCP.drain_received(idle_grace_s=0.0)
                    dcm0 = sniff[0] if sniff else None
                    if not dcm0:
                        logger.warning(
                            "create_node_map: sniff move returned no dataset for "
                            "SeriesInstanceUID=%s",
                            getattr(series_result, "SeriesInstanceUID", None),
                        )

                    # We only compute dcm0_inst once
                    dcm0_inst = (
                        process_standard_dicom(dcm0, "", query_ae, tag_plan, None) if dcm0 else {}
                    )

                    update_values = {}
                    for key, value in (dcm0_inst or {}).items():
                        if key in IMAGE_TAGS_NEEDED:
                            update_values[key] = value

                    for inst_result in inst_results:
                        inst_result.FrameOfReferenceUID = (
                            getattr(dcm0, "FrameOfReferenceUID", None) if dcm0 else None
                        )
                        inst_result.SeriesDescription = (
                            str(getattr(dcm0, "SeriesDescription", "")) if dcm0 else ""
                        )
                        inst_result.StudyDescription = (
                            str(getattr(dcm0, "StudyDescription", "")) if dcm0 else None
                        )
                        inst_result.PatientName = (
                            str(getattr(dcm0, "PatientName", "")) if dcm0 else None
                        )
                        inst_result.Modality = series_result.Modality
                        inst_result.update(update_values)

                        # enqueue the *C-FIND result* (metadata only), like before
                        parse_q.put(
                            ParseJob(dcm=inst_result, query_ae=query_ae, dataset_id=dataset_id)
                        )

            # --- RTDOSE (C-FIND only; enqueue each inst_result) ---
            elif series_result.Modality == "RTDOSE":
                inst_ds.StudyDescription = ""

                referenced_plan_sequence = Sequence()
                referenced_plan_ds = Dataset()
                referenced_plan_ds.ReferencedSOPClassUID = ""
                referenced_plan_ds.ReferencedSOPInstanceUID = ""
                referenced_plan_sequence.append(referenced_plan_ds)
                inst_ds.ReferencedRTPlanSequence = referenced_plan_sequence

                ref_struct_seq = Sequence()
                ref_struct_seq_ds = Dataset()
                ref_struct_seq_ds.ReferencedSOPClassUID = CLASS_UID_BY_MODALITY["RTSTRUCT"]
                ref_struct_seq_ds.ReferencedSOPInstanceUID = ""
                ref_struct_seq.append(ref_struct_seq_ds)
                inst_ds.ReferencedStructureSetSequence = ref_struct_seq

                inst_ds.DoseSummationType = ""

                inst_results = scu.c_find(query_ae, inst_ds) or []
                for inst_result in inst_results:
                    inst_result.Modality = "RTDOSE"
                    parse_q.put(
                        ParseJob(dcm=inst_result, query_ae=query_ae, dataset_id=dataset_id)
                    )

            # --- RTIMAGE (C-FIND only; enqueue each inst_result) ---
            elif series_result.Modality == "RTIMAGE":
                inst_ds.StudyDescription = ""

                referenced_plan_sequence = Sequence()
                referenced_plan_ds = Dataset()
                referenced_plan_ds.ReferencedSOPClassUID = ""
                referenced_plan_ds.ReferencedSOPInstanceUID = ""
                referenced_plan_sequence.append(referenced_plan_ds)
                inst_ds.ReferencedRTPlanSequence = referenced_plan_sequence

                inst_results = scu.c_find(query_ae, inst_ds) or []
                for inst_result in inst_results:
                    inst_result.Modality = "RTIMAGE"
                    parse_q.put(
                        ParseJob(dcm=inst_result, query_ae=query_ae, dataset_id=dataset_id)
                    )

            # --- RTRECORD (C-FIND only; patch ReferencedRTPlanSequence; enqueue) ---
            elif series_result.Modality == "RTRECORD":
                inst_ds.StudyDescription = ""
                inst_ds.ReferencedSOPClassUID = ""
                inst_ds.ReferencedSOPInstanceUID = ""
                inst_ds.TreatmentDate = ""
                inst_ds.TreatmentTime = ""
                inst_ds.TreatmentSessionBeamSequence = ""

                inst_results = scu.c_find(query_ae, inst_ds) or []
                for inst_result in inst_results:
                    inst_result.Modality = "RTRECORD"
                    if hasattr(inst_result, "ReferencedSOPInstanceUID"):
                        referenced_plan = Dataset()
                        referenced_plan.ReferencedSOPClassUID = inst_result.ReferencedSOPClassUID
                        referenced_plan.ReferencedSOPInstanceUID = (
                            inst_result.ReferencedSOPInstanceUID
                        )
                        inst_result.ReferencedRTPlanSequence = Sequence([referenced_plan])

                    parse_q.put(
                        ParseJob(dcm=inst_result, query_ae=query_ae, dataset_id=dataset_id)
                    )

            # --- generic catch-all (C-FIND only; enqueue each inst_result) ---
            else:
                inst_results = scu.c_find(query_ae, inst_ds) or []
                for inst_result in inst_results:
                    # keep modality from series_result if you want it explicit
                    inst_result.Modality = getattr(
                        series_result, "Modality", getattr(inst_result, "Modality", None)
                    )
                    parse_q.put(
                        ParseJob(dcm=inst_result, query_ae=query_ae, dataset_id=dataset_id)
                    )

        except Exception:
            logger.exception(
                "create_node_map: error for PatientID=%s, SeriesInstanceUID=%s",
                pid,
                getattr(series_result, "SeriesInstanceUID", None),
            )

    logger.info(
        "create_node_map: finished building DB node map for PatientID=%s (AE=%s)",
        pid,
        query_ae,
    )


def load_existing_patient_ids(session_factory, dataset_id: str) -> set[str]:
    with session_factory() as session:
        rows = session.execute(
            select(PatientRow.PatientID).where(PatientRow.dataset_id == dataset_id)
        ).all()
    return {r[0] for r in rows if r[0]}


def run_mapper(config_path, patients_list_path, log_file_path=None):
    """
    Runs the full DICOM mapping workflow:
      - configure SCU + StoreSCP
      - initialize DB + tag plan
      - start ONE parse/write pipeline for the whole run
      - iterate patients x AEs:
          - query SERIES
          - enqueue IMAGE-level work into pipeline via create_node_map(...)
      - stop pipeline (flush + commit)
      - run DB-backed association
      - clean shutdown (SCP + engine)
    """
    if not log_file_path:
        log_file_path = "dicom_mapper.log"
    setup_module_logging(
        log_file_path=log_file_path,
        level=logging.INFO,
        max_bytes=20 * 1024 * 1024,
        backup_count=1,
        log_to_console=False,
    )

    logger.info("main: starting DICOM query/retrieve + DB-backed node map build workflow")

    cfg = load_config(config_path)

    scu, aes_list = configure_scu_from_config(cfg)

    scu.configure_logging(
        log_to_file=True,
        log_to_console=False,
        log_file_path=f"qr_scu_{datetime.today().strftime('%Y-%m-%d')}.log",
        json_logs=False,
    )

    # Start the scp
    scp_cfg = cfg.get("store_scp", {})
    scp_name = scp_cfg.get("name")
    scp_ae_title = scp_cfg.get("ae_title")
    scp_ip = scp_cfg.get("ip", "0.0.0.0")
    scp_port = int(scp_cfg.get("port"))

    logger.info(
        "main: starting StoreSCP aet=%s ip=%s port=%d",
        scp_ae_title,
        scp_ip,
        scp_port,
    )
    scp = StoreSCP(scp_ae_title, scp_ip, scp_port)
    scp.configure_logging(
        log_to_file=True,
        log_to_console=False,
        log_file_path=f"store_scp_{datetime.today().strftime('%Y-%m-%d')}.log",
    )
    scp.add_custom_function_store(StoreSCP.stage_received_dcms)

    scp.register_sop_class(
        "1.2.246.352.70.1.70", "VarianRTPlanStorage"
    )  # Ethos RTPLAN SOPClassUID
    scp.register_sop_class(
        "1.2.246.352.70.1.71", "VarianRTRecordStorage"
    )  # Ethos RTRECORD SOPClassUID
    scp.add_registered_presentation_context("VarianRTRecordStorage")
    scp.add_registered_presentation_context("VarianRTPlanStorage")

    scp.start(block=False)

    # Build tag plan (core tags + user-specified tags)
    default_tags = {_normalize_tag(tag) for tag in CORE_TAGS}
    default_tags = {t for t in default_tags if t}  # drop Nones

    index_tags = cfg["tags-to-index"]
    extra_tags: set[tuple[int, int]] = set()
    if index_tags:
        for t in index_tags:
            norm = _normalize_tag(t)
            if norm:
                extra_tags.add(norm)

    tags_to_index = list(default_tags | extra_tags)

    global tag_plan
    tag_plan = build_tag_plan(tags_to_index)
    logger.info(
        "main: built TagPlan with %d tags (core=%d, extras=%d)",
        len(tag_plan),
        len(default_tags),
        len(extra_tags),
    )

    force_json_tags = cfg["force-json-tags"]
    engine = None
    pipeline = None

    try:
        dataset_id = cfg.get("dataset-id", "DS-001")
        dataset_name = cfg.get("dataset-name", "Mapping DICOMs")
        db_url = cfg.get("db-url")
        logger.info("main: initializing DB at %s", db_url)
        engine = create_engine(db_url, future=True)
        init_schema(engine, tag_plan=tag_plan, force_json_tags=force_json_tags)
        SessionLocal = sessionmaker(bind=engine, future=True)
        dataset_db = DatasetNodeDB.from_engine(engine, dataset_id)

        # Ensure DatasetRow exists / update name if needed
        with SessionLocal() as session:
            ds_row = session.get(DatasetRow, dataset_id)
            if ds_row is None:
                ds_row = DatasetRow(dataset_id=dataset_id, dataset_name=dataset_name)
                session.add(ds_row)
                session.commit()
                logger.info(
                    "main: created new DatasetRow id=%s name=%s",
                    dataset_id,
                    dataset_name,
                )
            else:
                if dataset_name and ds_row.dataset_name != dataset_name:
                    ds_row.dataset_name = dataset_name
                    session.commit()
                    logger.info(
                        "main: updated DatasetRow id=%s name=%s",
                        dataset_id,
                        dataset_name,
                    )

        patients_list = load_patient_ids_from_csv(patients_list_path)

        skip_existing = cfg.get("skip-existing", True)

        existing_patient_ids: set[str] = set()
        if skip_existing:
            existing_patient_ids = load_existing_patient_ids(SessionLocal, dataset_id)
            logger.info(
                "main: loaded %d existing PatientIDs for skip-existing", len(existing_patient_ids)
            )

        # Start one pipeline for the entire run
        parse_workers = int(cfg.get("parse-workers", 4))
        write_commit_every = int(cfg.get("write-commit-every", 500))

        pipeline = start_pipeline(
            SessionLocal,
            parse_workers=parse_workers,
            write_commit_every=write_commit_every,
        )

        logger.info(
            "main: will process %d patients into dataset_id=%s in DB=%s",
            len(patients_list),
            dataset_id,
            dataset_name,
        )

        exclude_patients = cfg.get("exclude-patients", [])
        exclude_modalities = cfg.get("exclude-modalities", [])

        # Main loop: Patient x AEs
        for patient_id in tqdm(patients_list, desc="Patients", unit="patient"):
            if patient_id in exclude_patients:
                continue

            if skip_existing and patient_id in existing_patient_ids:
                continue

            for query_ae in aes_list:
                logger.info(
                    "main: processing PatientID=%s on AE=%s",
                    patient_id,
                    query_ae,
                )
                series_results = query_all_series(patient_id, query_ae, scu)
                if not series_results:
                    logger.info(
                        "main: no series returned for PatientID=%s on AE=%s",
                        patient_id,
                        query_ae,
                    )
                    continue
                create_node_map(
                    series_results=series_results,
                    query_ae=query_ae,
                    staging_ae=scp_name,
                    dataset_id=dataset_id,
                    scu=scu,
                    parse_q=pipeline.parse_q,
                    exclude_modalities=exclude_modalities,
                )

        # Stop pipeline (flush parse+write queues)
        stop_pipeline(pipeline)
        pipeline = None

        # After all patients are ingested, run DB-backed association to populate edge tables
        logger.info("main: running DB-backed associate_dicoms for entire dataset")

        dataset_db.associate_dicoms(rebuild=True)
        logger.info("main: DB-backed associate_dicoms completed")

        logger.info("main: workflow complete")

    finally:
        # Best-effort pipeline cleanup (if an exception interrupted the run)
        if pipeline is not None:
            try:
                stop_pipeline(pipeline)
            except Exception:
                logger.exception("main: failed stopping pipeline cleanly")

        # Stop SCP
        try:
            scp.stop()
        except Exception:
            pass

        # Dispose engine
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM AE mapper / DB-backed node builder")
    parser.add_argument(
        "--config",
        type=str,
        default="dicom_mapper_config.json",
        help="Path to JSON config file with AE definitions",
    )
    parser.add_argument(
        "--patients-csv",
        type=str,
        default="patients.csv",
        help="CSV file containing a 'PatientID' column",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=f"dicom_mapper_{datetime.today().strftime('%Y-%m-%d')}.log",
        help="Path to rotating log file for this module",
    )

    args = parser.parse_args()
    run_mapper(args.config, args.patients_csv, args.log_file)
