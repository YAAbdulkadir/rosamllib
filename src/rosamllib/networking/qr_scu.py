import time
import logging
import pandas as pd
from logging import StreamHandler, FileHandler, Formatter
from typing import List, Dict, Optional
from contextlib import contextmanager
from pydicom.tag import Tag
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
from rosamllib.constants import VR_TO_DTYPE
from rosamllib.utils import (
    validate_entry,
    parse_vr_value,
)


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
        with self.association_context(ae_name) as assoc:
            if assoc:
                self.logger.info(f"Association established with {ae_name}. Sending C-ECHO...")
                status = assoc.send_c_echo()
                if status.Status == 0x0000:
                    self.logger.info(f"C-ECHO with '{ae_name}' successful.")
                    return True
                else:
                    self.logger.error(f"C-ECHO with '{ae_name}' failed. Status: {status}")
                    return False
            else:
                self.logger.error(f"Failed to associate with {ae_name}.")
                return False

    def c_find(self, ae_name: str, query: Dataset) -> Optional[List[Dataset]]:
        """Perform a C-FIND request using the provided query Dataset."""
        with self.association_context(ae_name) as assoc:
            if assoc:
                self.logger.info(f"Association established with {ae_name}. Sending C-FIND...")
                results = []
                responses = assoc.send_c_find(query, StudyRootQueryRetrieveInformationModelFind)
                for status, identifier in responses:
                    if status and status.Status in (0xFF00, 0xFF01):
                        results.append(identifier)
                return results
            else:
                self.logger.error(f"Failed to associate with {ae_name}.")
                return None

    def c_move(self, ae_name: str, query: Dataset, destination_ae: str):
        """Perform a C-MOVE request to move studies to a specified AE."""
        with self.association_context(ae_name) as assoc:
            if assoc:
                self.logger.info(
                    f"Association established with {ae_name}. "
                    f"Sending C-MOVE to '{destination_ae}'..."
                )
                responses = assoc.send_c_move(
                    query,
                    destination_ae,
                    StudyRootQueryRetrieveInformationModelMove,
                )
                for status, _ in responses:
                    if status.Status == 0x0000:
                        self.logger.info(f"C-MOVE successful to AE '{destination_ae}'.")
                        return status
                    elif status.Status == 0xFF00:
                        pass
                    else:
                        self.logger.error(f"C-MOVE failed. Status: {status}")
                        return status
            else:
                self.logger.error(f"Failed to associate with {ae_name}.")

    def c_store(self, ae_name: str, dataset: Dataset):
        """Perform a C-STORE request to store a dataset to a remote AE."""
        context = build_context(dataset.SOPClassUID)
        if not any(
            ctx.abstract_syntax == context.abstract_syntax for ctx in self.ae.requested_contexts
        ):
            try:
                self.ae.add_requested_context(context)
            except ValueError:
                self.ae.requested_contexts.pop()
                self.ae.add_requested_context(context)

        with self.association_context(ae_name) as assoc:
            if assoc:
                self.logger.info(f"Association established with {ae_name}. Sending C-STORE...")
                status = assoc.send_c_store(dataset)
                if status.Status == 0x0000:
                    self.logger.info(f"C-STORE with '{ae_name}' successful.")
                else:
                    self.logger.error(f"C-STORE with '{ae_name}' failed. Status: {status}")
                return status
            else:
                self.logger.error(f"Failed to associate with {ae_name}.")

    @staticmethod
    def convert_results_to_df(results, query_dataset):
        metadata_list = []
        for result in results:
            metadata_list.append(QueryRetrieveSCU._get_metadata(result, query_dataset))

        metadata_df = pd.DataFrame(metadata_list)

        for col in metadata_df.columns:
            try:
                vr = dictionary_VR(Tag(tag_for_keyword(col)))
            except TypeError:
                vr = dictionary_VR(Tag(col))
            dtype = VR_TO_DTYPE.get(vr, object)
            if dtype == "date":
                metadata_df[col] = pd.to_datetime(metadata_df[col], errors="coerce")
            elif dtype == "time":
                metadata_df[col] = pd.to_datetime(
                    metadata_df[col], format="%H:%M:%S", errors="coerce"
                ).dt.time
            elif dtype == "datetime":
                metadata_df[col] = pd.to_datetime(metadata_df[col], errors="coerce")
            else:
                metadata_df[col] = metadata_df[col].astype(dtype, errors="ignore")
        return metadata_df

    @staticmethod
    def _get_metadata(result_dataset, query_dataset):
        metadata = {}
        all_tags = list(query_dataset.keys())
        for tag in all_tags:
            vr = dictionary_VR(tag)
            value = result_dataset[tag].value if tag in result_dataset else None
            if isinstance(value, Sequence) and vr == "SQ":
                metadata[keyword_for_tag(tag) or tag] = (
                    (result_dataset[tag].to_json()) if value else None
                )
            else:
                metadata[keyword_for_tag(tag) or tag] = parse_vr_value(vr, value)
        return metadata

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
        log_file_path: str = "store_scp.log",
        log_level: int = logging.INFO,
        formatter: Optional[Formatter] = None,
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
        """
        if formatter is None:
            formatter = Formatter("%(levelname).1s: %(asctime)s: %(name)s: %(message)s")

        console_handler = next(
            (h for h in self.logger.handlers if isinstance(h, StreamHandler)), None
        )

        if log_to_console:
            if not console_handler:
                console_handler = StreamHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        else:
            if console_handler:
                self.logger.removeHandler(console_handler)

        file_handler = next(
            (
                h
                for h in self.logger.handlers
                if isinstance(h, FileHandler) and h.baseFilename == log_file_path
            ),
            None,
        )
        if log_to_file:
            if not file_handler:
                file_handler = FileHandler(log_file_path)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        else:
            if file_handler:
                self.logger.removeHandler(file_handler)
