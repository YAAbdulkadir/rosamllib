import json
import time
import logging
import pandas as pd
from logging import StreamHandler, FileHandler, Formatter, Handler
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import deque
from pydicom.tag import Tag, BaseTag
from pydicom.datadict import dictionary_VR, keyword_for_tag, tag_for_keyword
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.presentation import build_context
from pynetdicom.sop_class import (
    Verification,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelMove,
)
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEG2000,
    RLELossless,
    JPEGBaseline8Bit,
)
from rosamllib.constants import VR_TO_DTYPE
from rosamllib.utils import (
    validate_entry,
    parse_vr_value,
    _ContextFilter,
    build_formatter,
    make_rotating_file_handler,
    _dedupe_handlers,
    attach_pynetdicom_to_logger,
)

_DEFAULT_TS = [ExplicitVRLittleEndian, ImplicitVRLittleEndian]


@dataclass
class MoveResult:
    status: Optional[int]
    completed: int = 0
    failed: int = 0
    warning: int = 0
    remaining: int = 0
    error_comment: Optional[str] = None


@dataclass
class FindResult:
    status: Optional[int] = None
    matches: List[Dataset] = field(default_factory=list)
    error_comment: Optional[str] = None

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, indx):
        return self.matches[indx]

    def __contains__(self, item):
        return item in self.matches


class QueryRetrieveSCU:
    """
    A DICOM Query/Retrieve SCU for managing C-FIND, C-MOVE, C-STORE, and C-ECHO requests.

    The `QueryRetrieveSCU` class provides a flexible interface for querying and retrieving
    DICOM datasets from remote AEs (Application Entities) using standard DICOM Query/Retrieve
    operations. It supports association management, query result parsing, and custom logging.

    Parameters
    ----------
    ae_title : str
        The AE Title for this SCU instance.
    acse_timeout : int, optional
        Timeout for association requests, in seconds (default: 120).
    dimse_timeout : int, optional
        Timeout for DIMSE operations, in seconds (default: 121).
    network_timeout : int, optional
        Timeout for network operations, in seconds (default: 122).
    logger : logging.Logger, optional
        An optional logger instance. If not provided, a default logger is configured.

    Attributes
    ----------
    ae : pynetdicom.AE
        The Application Entity instance that manages DICOM associations and operations.
    remote_entities : dict
        A dictionary of configured remote AEs with their connection details.
    logger : logging.Logger
        Logger used for logging messages and errors.

    Examples
    --------
    Create a `QueryRetrieveSCU` instance and perform a C-ECHO operation:

    >>> from pydicom.dataset import Dataset
    >>> from rosamllib.networking import QueryRetrieveSCU
    >>> scu = QueryRetrieveSCU(ae_title="MY_SCU")
    >>> scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    >>> scu.c_echo("remote1")

    Perform a C-FIND operation with a sample query dataset:

    >>> query = Dataset()
    >>> query.PatientID = "12345"
    >>> query.QueryRetrieveLevel = "STUDY"
    >>> results = scu.c_find("remote1", query)
    >>> print(results)

    Perform a C-MOVE operation to retrieve studies to another AE:

    >>> destination_ae = "STORAGE_AE"
    >>> scu.c_move("remote1", query, destination_ae)

    Convert C-FIND results to a Pandas DataFrame:

    >>> df = scu.convert_results_to_df(results, query)
    >>> print(df)
    """

    def __init__(
        self,
        ae_title: str,
        acse_timeout: int = 120,
        dimse_timeout: int = 121,
        network_timeout: int = 122,
        logger: Optional[logging.Logger] = None,
    ):
        if not validate_entry(ae_title, "AET"):
            raise ValueError("Invalid AE Title.")

        self.ae_title = ae_title

        self.ae = AE(self.ae_title)
        self.ae.acse_timeout = acse_timeout
        self.ae.dimse_timeout = dimse_timeout
        self.ae.network_timeout = network_timeout
        self.remote_entities: Dict[str, Dict] = {}  # Remote AEs

        self.ae.add_requested_context(Verification)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)

        self._pc_lru = deque()  # abstract syntax UIDs in MRU order
        self._pc_ts = {}  # abstract syntax UID -> set of transfer syntaxes
        self._pc_cap = 120  # headroom below 128-limit for QR + Verification

        # Configure logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

    def add_remote_ae(self, name: str, ae_title: str, host: str, port: int):
        """Add a remote AE to the dictionary of managed AEs."""
        if not (
            validate_entry(ae_title, "AET")
            and validate_entry(port, "Port")
            and validate_entry(host, "IP")
        ):
            raise ValueError("Invalid input for AE Title, Host, or Port.")

        if name in self.remote_entities:
            self.logger.warning(f"AE '{name}' already exists. Overwriting AE info.")
        self.remote_entities[name] = {
            "ae_title": ae_title,
            "host": host,
            "port": port,
        }
        self.logger.info(f"Added remote AE '{name}': {ae_title}@{host}:{port}")

    def add_extended_negotiation(self, ae_name: str, ext_neg_items: List):
        """Add extended negotiation items to the remote AE.
        The ext_neg_items parameter should be a list of extended negotiation objects
        (e.g., SOPClassExtendedNegotiation, AsynchronousOperationsWindowNegotiation).
        """
        if ae_name not in self.remote_entities:
            raise ValueError(
                f"Remote AE '{ae_name}' not found. Add it with `add_remote_ae` first."
            )

        rm_ae = self.remote_entities[ae_name]
        rm_ae["ext_neg"] = ext_neg_items

    @contextmanager
    def association_context(self, ae_name: str):
        """Context manager for establishing and releasing an association."""
        assoc = self._establish_association(ae_name)
        try:
            if assoc and assoc.is_established:
                yield assoc
            else:
                yield None
        finally:
            if assoc and assoc.is_established:
                assoc.release()

    def _ensure_requested_context(self, sop_class_uid, ts_list=None):
        """Merge TS for SOP class; evict true LRU if at capacity; idempotent."""
        ts = set(ts_list or _DEFAULT_TS)

        if sop_class_uid in self._pc_ts:
            # merge TS + replace requested context cleanly
            self._pc_ts[sop_class_uid] |= ts
            for c in list(self.ae.requested_contexts):
                if str(c.abstract_syntax) == str(sop_class_uid):
                    self.ae.requested_contexts.remove(c)
                    break
            ctx = build_context(sop_class_uid, list(self._pc_ts[sop_class_uid]))
            self.ae.add_requested_context(ctx.abstract_syntax, ctx.transfer_syntax)
            try:
                self._pc_lru.remove(sop_class_uid)
            except ValueError:
                pass
            self._pc_lru.appendleft(sop_class_uid)
            return

        # new abstract syntax
        if len(self._pc_lru) >= self._pc_cap:
            evict = self._pc_lru.pop()
            self._pc_ts.pop(evict, None)
            for c in list(self.ae.requested_contexts):
                if str(c.abstract_syntax) == str(evict):
                    self.ae.requested_contexts.remove(c)
                    break

        self._pc_ts[sop_class_uid] = ts
        ctx = build_context(sop_class_uid, list(ts))
        self.ae.add_requested_context(ctx.abstract_syntax, ctx.transfer_syntax)
        self._pc_lru.appendleft(sop_class_uid)

    def _maybe_add_image_ts(self, sop_class_uid):
        image_ts = _DEFAULT_TS + [JPEG2000Lossless, JPEG2000, JPEGBaseline8Bit, RLELossless]
        self._ensure_requested_context(sop_class_uid, image_ts)

    def _establish_association(self, ae_name: str, retry_count: int = 3, delay: int = 5):
        """Helper method to establish an association with a remote AE, with retry logic."""
        if ae_name not in self.remote_entities:
            raise ValueError(
                f"Remote AE '{ae_name}' not found. Add it with `add_remote_ae` first."
            )

        rm_ae = self.remote_entities[ae_name]
        ext_neg = rm_ae.get("ext_neg", [])

        for attempt in range(retry_count):
            try:
                assoc = self.ae.associate(
                    rm_ae["host"],
                    rm_ae["port"],
                    ae_title=rm_ae["ae_title"],
                    ext_neg=ext_neg,
                )
                if assoc.is_established:
                    return assoc
            except Exception as e:
                self.logger.error(
                    f"Association attempt {attempt + 1} failed with AE '{ae_name}': {e}"
                )
                time.sleep(delay)

        self.logger.error(f"Failed to associate with AE '{ae_name}' after {retry_count} attempts.")
        return None

    def c_echo(self, ae_name: str):
        """Launch a C-ECHO request to verify connectivity with a remote AE."""
        extra = self._op_ctx(ae_name, "C-ECHO")
        t0 = time.perf_counter()
        with self.association_context(ae_name) as assoc:
            if not assoc:
                self.logger.error("Failed to associate.", extra=extra)
                return False
            self.log_accepted_contexts(assoc)
            self.logger.info("Association established. Sending C-ECHO...", extra=extra)
            status = assoc.send_c_echo()
            extra["status_hex"] = hex(getattr(status, "Status", 0xFFFF))
            extra["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            if getattr(status, "Status", None) == 0x0000:
                self.logger.info("C-ECHO successful.", extra=extra)
                return True
            else:
                self.logger.error(f"C-ECHO failed. Status={status}", extra=extra)
                return False

    def c_find(self, ae_name: str, query: Dataset) -> FindResult:
        """Perform a C-FIND request using the provided query Dataset."""
        extra = self._op_ctx(ae_name, "C-FIND", dataset=query)
        t0 = time.perf_counter()

        result = FindResult()

        with self.association_context(ae_name) as assoc:
            if not assoc:
                msg = f"Failed to associate with AE '{ae_name}'."
                self.logger.error(msg, extra=extra)
                result.error_comment = msg
                extra["status_hex"] = None
                extra["matches"] = 0
                extra["duration_ms"] = int((time.perf_counter() - t0) * 1000)
                return result

            self.log_accepted_contexts(assoc)
            self.logger.info("Association established. Sending C-FIND...", extra=extra)

            matches: List[Dataset] = []
            last_status = None

            # results: List[Dataset] = []
            # last_status = None
            responses = assoc.send_c_find(query, StudyRootQueryRetrieveInformationModelFind)
            for status, identifier in responses:
                last_status = status
                if status and status.Status in (0xFF00, 0xFF01):  # Pending
                    if identifier is not None:
                        matches.append(identifier)

            result.matches = matches
            if last_status is not None:
                result.status = getattr(last_status, "Status", None)
                if hasattr(last_status, "ErrorComment"):
                    result.error_comment = str(last_status.ErrorComment)

            extra["status_hex"] = hex(result.status) if result.status is not None else None
            extra["matches"] = len(matches)
            extra["duration_ms"] = int((time.perf_counter() - t0) * 1000)

            if result.status == 0x0000:
                self.logger.info(f"C-FIND completed: {len(matches)} matches.", extra=extra)
            else:
                self.logger.warning(
                    f"C-FIND finished with status={last_status}; matches={len(matches)}.",
                    extra=extra,
                )

            return result

    def c_move(self, source_ae: str, query: Dataset, destination_ae: str) -> MoveResult:
        """Perform a C-MOVE request to move studies to a specified AE."""
        extra = self._op_ctx(source_ae, "C-MOVE", dataset=query, destination_ae=destination_ae)
        t0 = time.perf_counter()
        result = MoveResult(status=None)

        with self.association_context(source_ae) as assoc:
            if not assoc:
                msg = f"Failed to associate with AE '{source_ae}."
                self.logger.error(msg, extra=extra)
                result.error_comment = msg
                extra.update(
                    {
                        "status_hex": None,
                        "completed": result.completed,
                        "failed": result.failed,
                        "warning": result.warning,
                        "remaining": result.remaining,
                        "duration_ms": int((time.perf_counter() - t0) * 1000),
                    }
                )
                return result
            self.log_accepted_contexts(assoc)
            self.logger.info(
                f"Association established. Sending C-MOVE to '{destination_ae}'...", extra=extra
            )
            result.status = 0xFFF

            for status, _ in assoc.send_c_move(
                query, destination_ae, StudyRootQueryRetrieveInformationModelMove
            ):
                if not status:
                    continue
                result.status = status.Status
                # Sub-op counters commonly present in C-MOVE respones
                for fld in (
                    "NumberOfRemainingSuboperations",
                    "NumberOfCompletedSuboperations",
                    "NumberOfFailedSuboperations",
                    "NumberOfWarningSuboperations",
                ):
                    if hasattr(status, fld):
                        val = int(getattr(status, fld))
                        if fld.endswith("RemainingSuboperations"):
                            result.remaining = val
                        elif fld.endswith("CompletedSuboperations"):
                            result.completed = val
                        elif fld.endswith("FailedSuboperations"):
                            result.failed = val
                        elif fld.endswith("WarningSuboperations"):
                            result.warning = val
                if hasattr(status, "ErrorComment"):
                    result.error_comment = str(status.ErrorComment)

            extra.update(
                {
                    "status_hex": hex(result.status) if result.status is not None else None,
                    "completed": result.completed,
                    "failed": result.failed,
                    "warning": result.warning,
                    "remaining": result.remaining,
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                }
            )

            if result.status == 0x0000:
                self.logger.info("C-MOVE completed.", extra=extra)
            else:
                self.logger.error("C-MOVE finished with non-success status.", extra=extra)
            return result

    def c_store(self, ae_name: str, dataset: Dataset):
        """Perform a C-STORE request to store a dataset to a remote AE."""
        for req in ("SOPClassUID", "SOPInstanceUID"):
            if not getattr(dataset, req, None):
                raise ValueError(f"C-STORE dataset missing required attribute: {req}")

        extra = self._op_ctx(ae_name, "C-STORE", dataset=dataset)
        t0 = time.perf_counter()

        # Ensure storage PC exists
        self._ensure_requested_context(dataset.SOPClassUID, _DEFAULT_TS)

        with self.association_context(ae_name) as assoc:
            if not assoc:
                # self.logger.error(f"Failed to associate with {ae_name}.")
                self.logger.error("Failed to associate.", extra=extra)
                return None
            self.log_accepted_contexts(assoc)

            # Guard: ensure peer accepted the Storage PC we need
            accepted = any(
                pc.abstract_syntax == dataset.SOPClassUID for pc in assoc.accepted_contexts
            )
            if not accepted:
                acc_list = [
                    (pc.abstract_syntax.name, [str(ts) for ts in pc.transfer_syntax])
                    for pc in assoc.accepeted_contexts
                ]
                self.logger.error(
                    "No accepted Storage presentation context for SOP Class.",
                    extra={**extra, "accepted_pcs": acc_list},
                )
                return None
            self.logger.info("Association established. Sending C-STORE...", extra=extra)
            status = assoc.send_c_store(dataset)

            extra["status_hex"] = hex(getattr(status, "Status", 0xFFFF))
            extra["duration_ms"] = int((time.perf_counter() - t0) * 1000)

            if getattr(status, "Status", None) == 0x0000:
                self.logger.info("C-STORE successful.", extra=extra)
            else:
                self.logger.error(f"C-STORE failed. Status={status}", extra=extra)
            return status

    @staticmethod
    def convert_results_to_df(
        results: Union[List[Dataset] | FindResult], query_dataset: Dataset
    ) -> pd.DataFrame:
        """
        Convert C-FIND results to a pandas DataFrame.

        Parameters
        ----------
        results :
            Either:
            - Iterable of pydicom.Dataset (old behavior), or
            - An object with a `.matches` attribute (e.g., FindResult) containing the list of
            Datasets.
        query_dataset : Dataset
            The original C-FIND query dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per match. Columns include:
            - All attributes requested in `query_dataset` (by keyword where possible)
            - Any additional attributes present in the result datasets.
        """
        if hasattr(results, "matches"):
            matches = getattr(results, "matches")
        else:
            matches = results

        if not matches:
            # Return empty DF with columns matching requested keys (by keyword where possible)
            cols = []
            for tag in query_dataset.keys():
                kw = keyword_for_tag(tag)
                cols.append(kw if kw else int(tag))
            return pd.DataFrame(columns=cols)

        metadata_list = [QueryRetrieveSCU._get_metadata(r, query_dataset) for r in matches]
        df = pd.DataFrame(metadata_list)

        for col in df.columns:
            # Determine VR by tag or keyword
            vr = None
            try:
                if isinstance(col, (int, BaseTag)):
                    vr = dictionary_VR(Tag(col))
                else:
                    tag = tag_for_keyword(col)
                    if tag is not None:
                        vr = dictionary_VR(Tag(tag))
            except (KeyError, TypeError, ValueError):
                vr = None

            dtype = VR_TO_DTYPE.get(vr, object)
            if dtype == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
            elif dtype == "time":
                # df[col] = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce").dt.time
                pass
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                # Avoid pandas warnings with nullable types
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    pass
        return df

    @staticmethod
    def _get_metadata(result_dataset: Dataset, query_dataset: Dataset) -> Dict:
        """
        Build a metadata dict for a sinlge C-FIND result.

        Includes both:
            - Tags present in the query_dataset
            - Any extra tags present in result_dataset
        """
        md: Dict = {}

        # Union of tags from query and result
        all_tags = set(query_dataset.keys()) | set(result_dataset.keys())

        # for tag in list(query_dataset.keys()):
        for tag in all_tags:
            vr = dictionary_VR(tag)
            value = result_dataset[tag].value if tag in result_dataset else None
            key = keyword_for_tag(tag) or int(tag)

            if isinstance(value, Sequence) and vr == "SQ":
                try:
                    # pydicom 3.0+: Dataset.json() exists; else fallback
                    md[key] = (
                        result_dataset[tag].to_json()
                        if hasattr(result_dataset[tag], "to_json")
                        else json.dumps([ds.to_json_dict() for ds in value])
                    )
                except Exception:
                    md[key] = None
            else:
                md[key] = parse_vr_value(vr, value)
        return md

    def _op_ctx(
        self,
        ae_name: str,
        op: str,
        *,
        dataset: Dataset | None = None,
        destination_ae: str | None = None,
    ):
        rm = self.remote_entities.get(ae_name, {})
        ctx = {
            "op": op,
            "ae_name": ae_name,
            "remote_host": rm.get("host"),
            "remote_port": rm.get("port"),
            "sop_class": (
                str(getattr(dataset, "SOPClassUID", None)) if dataset is not None else None
            ),
            "study_uid": (
                getattr(dataset, "StudyInstanceUID", None) if dataset is not None else None
            ),
            "series_uid": (
                getattr(dataset, "SeriesInstanceUID", None) if dataset is not None else None
            ),
            "sop_uid": getattr(dataset, "SOPInstanceUID", None) if dataset is not None else None,
        }
        if destination_ae:
            ctx["destination_ae"] = destination_ae
        return ctx

    def set_logger(self, new_logger: logging.Logger):
        """Set a new logger for the class, overriding the existing one."""
        self.logger = new_logger

    def add_log_handler(self, handler: logging.Handler):
        """Add an additional handler to the existing logger."""
        self.logger.addHandler(handler)

    def remove_log_handler(self, handler: logging.Handler):
        """
        Remove a specific handler from the logger.

        Parameters
        ----------
        handler : logging.Handler
            The handler to be removed.
        """
        self.logger.removeHandler(handler)

    def clear_log_handlers(self):
        """
        Remove all handlers from the logger.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def configure_logging(
        self,
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file_path: str = "qr_scu.log",
        log_level: int = logging.INFO,
        formatter: Optional[Formatter] = None,
        json_logs: bool = False,
        rotate: bool = True,
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        static_context: Optional[Dict] = None,
    ):
        """Configure logging with console and/or file handlers.

        Parameters
        ----------
        log_to_console : bool
            Whether to log to the console.
        log_to_file : bool
            Whether to log to a file.
        log_file_path : str
            The path to the log file if `log_to_file` is True.
        log_level : int
            The logging level (e.g., logging.INFO, logging.DEBUG).
        formatter : Optional[Formatter]
            A custom formatter for the log messages. If None, a default formatter is used.

        Returns
        -------
        Dict[str, logging.Handler]
            The handlers that were added, keyed by "console" and/or "file".
        """
        # formatter
        if formatter is None:
            formatter = build_formatter(human=not json_logs)

        # attach a context filter so evey log carries these fields
        ctx = static_context or {"component": "QueryRetrieveSCU"}
        # avoid adding the same filter multiple times
        if not any(isinstance(f, _ContextFilter) for f in self.logger.filters):
            self.logger.addFilter(_ContextFilter(**ctx))

        self.logger.setLevel(log_level)
        self.logger.propagate = False  # avoid duplicate logs via root

        added: Dict[str, Handler] = {}

        # Console
        if log_to_console:
            if not any(isinstance(h, StreamHandler) for h in self.logger.handlers):
                ch = StreamHandler()
                ch.setLevel(log_level)
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
                added["console"] = ch
        else:
            for h in list(self.logger.handlers):
                if isinstance(h, StreamHandler):
                    self.logger.removeHandler(h)

        # File
        if log_to_file:
            # ensure only one file handler for the given path
            existing = next(
                (
                    h
                    for h in self.logger.handlers
                    if isinstance(h, FileHandler)
                    and getattr(h, "baseFilename", "") == log_file_path
                ),
                None,
            )
            if existing is None:
                if rotate:
                    fh = make_rotating_file_handler(
                        log_file_path,
                        log_level,
                        formatter,
                        max_bytes=max_bytes,
                        backup_count=backup_count,
                    )
                else:
                    fh = FileHandler(log_file_path)
                    fh.setLevel(log_level)
                    fh.setFormatter(formatter)
                self.logger.addHandler(fh)
                added["file"] = fh
        else:
            for h in list(self.logger.handlers):
                if isinstance(h, FileHandler):
                    self.logger.removeHandler(h)

        _dedupe_handlers(self.logger)
        return added

    def enable_wire_debug(self, enable: bool = True, level: int = logging.DEBUG):
        """Enable/disable verbose pynetdicom + route through our logger."""
        attach_pynetdicom_to_logger(enable, level)

    def set_log_level(self, level: int):
        """Dynamic level change."""
        self.logger.setLevel(level)
        for h in self.logger.handlers:
            h.setLevel(level)

    def log_accepted_contexts(self, assoc) -> None:
        rows = []
        for pc in assoc.accepted_contexts:
            ts_list = ", ".join(str(ts) for ts in pc.transfer_syntax)
            rows.append(f"{pc.abstract_syntax.name} [{ts_list}]")
        self.logger.debug("Accepted presentation contexts:\n  - " + "\n  - ".join(rows))

    def close_log_handlers(self):
        for h in list(self.logger.handlers):
            try:
                h.flush()
                h.close()
            finally:
                self.logger.removeHandler(h)
