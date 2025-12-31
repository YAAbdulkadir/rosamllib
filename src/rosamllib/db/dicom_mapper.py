import argparse
import json
import logging
import pandas as pd
from datetime import datetime
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from rosamllib.networking import QueryRetrieveSCU, StoreSCP

from tqdm import tqdm
import warnings
from rosamllib.db.db_nodes import DatasetNodeDB
from rosamllib.db.db_schema import (
    init_schema,
    DatasetRow,
    PatientRow,
)
from typing import Any
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from rosamllib.constants import CLASS_UID_BY_MODALITY
from queue import Queue
from rosamllib.utils.db_utils import (
    CORE_TAGS,
    IMAGE_TAGS_NEEDED,
    ParseJob,
    start_pipeline,
    stop_pipeline,
    setup_module_logging,
    process_standard_dicom,
    build_tag_plan,
    _normalize_tag,
)

warnings.filterwarnings(
    "ignore",
    message=r"Invalid value for VR",
)
warnings.filterwarnings("ignore", message=r"The value")

logger = logging.getLogger(__name__)


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
        dataset_db = DatasetNodeDB.from_engine(engine, dataset_id)

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
            tag_plan=tag_plan,
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
