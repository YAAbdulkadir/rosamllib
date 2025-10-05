"""
Class module for DICOM SCP
"""

import logging
import time
import pynetdicom.sop_class as sop_class
from logging import StreamHandler, FileHandler, Formatter, Handler
from typing import Optional, Callable, List, Dict
from pydicom.dataset import Dataset
from pynetdicom import AE, StoragePresentationContexts, evt, register_uid
from pynetdicom.sop_class import Verification
from pynetdicom.service_class import StorageServiceClass
from rosamllib.utils import (
    validate_entry,
    attach_pynetdicom_to_logger,
    build_formatter,
    _ContextFilter,
    make_rotating_file_handler,
    _dedupe_handlers,
)


# Status constants (common DICOM codes)
STATUS_SUCCESS = 0x0000
STATUS_OUT_OF_RESOURCES = 0xA700
STATUS_DATASET_MISMATCH = 0xA900
STATUS_CANNOT_UNDERSTAND = 0xC000  # general processing failure


def _mask(s: str | None, keep: int = 6) -> str | None:
    """Mask potentially identifying strings in logs (keep first `keep` chars)."""
    if not s or len(s) <= keep:
        return s
    return s[:keep] + "..."


def _ctx_from_event(event, op: str, mask_phi: bool = True):
    """Build structured logging context from a pynetdicom event."""
    assoc = getattr(event, "assoc", None)
    dset = getattr(event, "dataset", None)
    requestor = getattr(assoc, "requestor", None)
    acceptor = getattr(assoc, "acceptor", None)

    calling_ae = requestor.ae_title if requestor else None
    called_ae = acceptor.ae_title if acceptor else None
    remote = getattr(event, "address", None)
    hostport = f"{remote[0]}:{remote[1]}" if remote else None

    def get(attr: str):
        return getattr(dset, attr, None) if dset is not None else None

    val = (lambda x: _mask(x)) if mask_phi else (lambda x: x)

    return {
        "op": op,
        "calling_ae": calling_ae,
        "called_ae": called_ae,
        "remote_addr": hostport,
        "modality": get("Modality"),
        "sop_class": str(get("SOPClassUID")),
        "sop_uid": val(get("SOPInstanceUID")),
        "study_uid": val(get("StudyInstanceUID")),
        "series_uid": val(get("SeriesInstanceUID")),
    }


class StoreSCP:
    """
    A DICOM Storage SCP (Service Class Provider) for handling C-STORE requests.

    The `StoreSCP` class implements a DICOM Storage SCP that listens for incoming
    C-STORE requests, stores the received DICOM files, and supports configurable event
    handlers for custom behavior during association establishment, storage, and closure.
    It also supports registering custom SOP Class UIDs and adding presentation contexts.

    Parameters
    ----------
    aet : str
        The Application Entity (AE) title to use for this SCP.
    ip : str
        The IP address to bind the SCP server.
    port : int
        The port number to listen for incoming connections.
    acse_timeout : int, optional
        The timeout for association requests, in seconds (default: 120).
    dimse_timeout : int, optional
        The timeout for DIMSE operations, in seconds (default: 121).
    network_timeout : int, optional
        The timeout for network operations, in seconds (default: 122).
    logger : logging.Logger, optional
        An optional logger instance. If not provided, a default logger is configured.

    Attributes
    ----------
    ae : pynetdicom.AE
        The Application Entity instance that handles DICOM associations and operations.
    handlers : list of tuple
        The event handlers registered for different DICOM events.
    custom_functions_open : list of callable
        Custom functions to execute during association establishment.
    custom_functions_store : list of callable
        Custom functions to execute during C-STORE requests.
    custom_functions_close : list of callable
        Custom functions to execute during association closure.
    logger : logging.Logger
        Logger used for logging messages and errors.

    Examples
    --------
    Create a basic StoreSCP instance and start the server.

    >>> from rosamllib.networking import StoreSCP
    >>> scp = StoreSCP(aet="MY_SCP", ip="127.0.0.1", port=11112)
    >>> scp.start(block=True)

    Add a custom function to log the PatientID during C-STORE requests.

    >>> def custom_store_handler(event):
    ...     print(f"Received Patient ID: {event.dataset.PatientID}")
    >>> scp.add_custom_function_store(custom_store_handler)

    Register a custom SOP Class and start the server.

    >>> scp.register_sop_class("1.2.246.352.70.1.70", "VarianRTPlanStorage")
    >>> scp.add_registered_presentation_context("VarianRTPlanStorage")
    >>> scp.start()
    """

    def __init__(
        self,
        aet: str,
        ip: str,
        port: int,
        acse_timeout: int = 120,
        dimse_timeout: int = 121,
        network_timeout: int = 122,
        logger: Optional[logging.Logger] = None,
        mask_phi_logs: bool = False,
    ):
        """Initialize the SCP to handle store requests.

        Parameters
        ----------
        aet : str
            The AE title to use.
        ip : str
            The IP address to use.
        port : int
            The port number to use.
        acse_timeout : int, optional
            The ACSE timeout value, by default 120
        dimse_timeout : int, optional
            The DIMSE timeout value, by default 121
        network_timeout : int, optional
            The network timeout value, by default 122
        logger : logging.Logger, optional
            The logger instance to use, by default None
        """
        if not (
            validate_entry(aet, "AET")
            and validate_entry(ip, "IP")
            and validate_entry(port, "Port")
        ):
            raise ValueError("Invalid input for AE Title, Host, or Port.")

        self.scpAET = aet
        self.scpIP = ip
        self.scpPort = port

        self.ae = AE(self.scpAET)
        # Add the supported presentation context (All Storage Contexts)
        self.ae.supported_contexts = StoragePresentationContexts
        self.ae.add_supported_context(Verification)

        # Set timeouts
        self.ae.acse_timeout = acse_timeout
        self.ae.dimse_timeout = dimse_timeout
        self.ae.network_timeout = network_timeout

        self._server = None
        self._server_running = None

        # Configure logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # Set the event handlers
        self.set_handlers()

        # Custom functions to be run during handle_open
        self.custom_functions_open: List[Callable[[evt.Event], None]] = []
        # Custom functions to be run during handle_store
        self.custom_functions_store: List[Callable[[evt.Event], None]] = []
        # Custom functions to be run during handle_close
        self.custom_functions_close: List[Callable[[evt.Event], None]] = []

        self.logger.info(
            f"StoreSCP initialized with AE Title: "
            f"{self.scpAET}, IP: {self.scpIP}, Port: {self.scpPort}"
        )

        self._mask_phi_logs = mask_phi_logs

    def is_running(self) -> bool:
        return self._server_running

    def handle_open(self, event):
        """Log association establishments.

        Parameters
        ----------
        event : `events.Event`
            A DICOM association establishment event
        """
        extra = _ctx_from_event(event, "ASSOC-OPEN", mask_phi=self._mask_phi_logs)
        self.logger.info("Association opened.", extra=extra)

        # Run custom functions
        for func in self.custom_functions_open:
            try:
                self.logger.debug(
                    f"Running custom function {func.__name__} in handle_open", extra=extra
                )
                func(event)
            except Exception as e:
                self.logger.error(
                    f"Error in handle_open callback {func.__name__}: {e}", extra=extra
                )

    def handle_close(self, event):
        """Log when association is disconnected.

        Parameters
        ----------
        event : `events.Event`
            A DICOM association close event
        """

        extra = _ctx_from_event(event, "ASSOC-CLOSE", mask_phi=self._mask_phi_logs)
        self.logger.info("Association closed.", extra=extra)

        # Run custom functions
        for func in self.custom_funcions_close:
            try:
                self.logger.debug(
                    f"Running custom function {func.__name__} in handle_close.", extra=extra
                )
                func(event)
            except Exception as e:
                self.logger.error(
                    f"Error in handle_close callback {func.__name__}: {e}", extra=extra
                )

    def handle_store(self, event) -> Dataset:
        """Handle incoming C-STORE requests.

        Parameters
        ----------
        event : `events.Event`
            A DICOM C-STORE request
        Returns
        -------
        Dataset
            The status message to respond with
        """
        t0 = time.perf_counter()
        extra = _ctx_from_event(event, "C-STORE", mask_phi=True)

        try:
            # Run custom functions
            for func in list(self.custom_functions_store):
                try:
                    self.logger.debug(
                        f"Running custom store function {func.__name__}", extra=extra
                    )
                    func(event)
                except Exception as e:
                    self.logger.error(
                        f"Custom store function {func.__name__} failed: {e}", extra=extra
                    )

            dur = int((time.perf_counter() - t0) * 1000)
            self.logger.info(f"C-STORE OK in {dur} ms.", extra={**extra, "duration_ms": dur})

            status_ds = Dataset()
            status_ds.Status = 0x0000
            return status_ds
        except Exception as e:
            self.logger.error(f"Error handling C-STORE request: {e}")
            status_ds = Dataset()
            status_ds.Status = 0xC000
            return status_ds

    def set_handlers(self):
        """Set event handlers for this SCP."""

        self.logger.info("Setting up event handlers for StoreSCP.")
        self.handlers = []
        self.handlers.append((evt.EVT_CONN_OPEN, self.handle_open))
        self.handlers.append((evt.EVT_CONN_CLOSE, self.handle_close))
        self.handlers.append((evt.EVT_C_STORE, self.handle_store))
        self.handlers.append((evt.EVT_REQUESTED, self._on_assoc_requested))
        self.handlers.append((evt.EVT_ACCEPTED, self._on_assoc_requested))
        self.handlers.append((evt.EVT_REJECTED, self._on_assoc_rejected))
        self.handlers.append((evt.EVT_ABORTED, self._on_abort))
        self.handlers.append((evt.EVT_C_STORE, self.handle_store))

    def _on_assoc_requested(self, event):
        extra = _ctx_from_event(event, "ASSOC-REQ", mask_phi=self._mask_phi_logs)
        # Include proposed presentation contexts at DEBUG for diagnostics
        if self.logger.isEnabledFor(logging.DEBUG):
            pcs = event.requestor.requested_contexts
            lines = []
            for pc in pcs:
                ts = ", ".join(str(ts) for ts in pc.transfer_syntax)
                lines.append(f"{pc.abstract_syntax.name} [{ts}]")
            if lines:
                self.logger.debug(
                    "Proposed presentation contexts:\n - " + "\n  - ".join(lines), extra=extra
                )
        self.logger.info("Association requested.", extra=extra)

    def _on_assoc_accepted(self, event):
        extra = _ctx_from_event(event, "ASSOC-ACC", mask_phi=self._mask_phi_logs)
        # Log accepted PCs at DEBUG
        if self.logger.isEnabledFor(logging.DEBUG):
            pcs = event.acceptor.accepted_contexts
            lines = []
            for pc in pcs:
                ts = ", ".join(str(ts) for ts in pc.transfer_syntax)
                lines.append(f"{pc.abstract_syntax.name} [{ts}]")
            if lines:
                self.logger.debug(
                    "Accepted presentation contexts:\n  - " + "\n  - ".join(lines), extra=extra
                )
        self.logger.info("Association accepted.", extra=extra)

    def _on_assoc_rejected(self, event):
        extra = _ctx_from_event(event, "ASSOC-REJ", mask_phi=self._mask_phi_logs)
        # pynetdicom provides result, source, reason
        details = dict(
            result=getattr(event, "result", None),
            source=getattr(event, "source", None),
            reason=getattr(event, "reason", None),
        )
        self.logger.error(f"Association rejected: {details}", extra=extra)

    def _on_abort(self, event):
        extra = _ctx_from_event(event, "ASSOC-ABRT", mask_phi=self._mask_phi_logs)
        self.logger.error("Association aborted.", extra=extra)

    def _on_c_echo(self, event):
        extra = _ctx_from_event(event, "C-ECHO", mask_phi=self._mask_phi_logs)
        self.logger.info("C-ECHO received.", extra=extra)
        return 0x0000

    def start(self, block: bool = False):
        if self._server_running:
            self.logger.warning(
                f"SCP already running at {self.scpIP}:{self.scpPort} "
                f"(AET={self.scpAET}); ignoring start().",
                extra={
                    "op": "SCP-START",
                    "called_ae": self.scpAET,
                    "remote_addr": f"{self.scpIP}:{self.scpPort}",
                },
            )
            return

        self.logger.info(
            f"Starting SCP {self.scpAET} on {self.scpIP}:{self.scpPort}",
            extra={
                "op": "SCP-START",
                "called_ae": self.scpAET,
                "remote_addr": f"{self.scpIP}:{self.scpPort}",
            },
        )
        try:
            self._server = self.ae.start_server(
                (self.scpIP, self.scpPort), block=block, evt_handlers=self.handlers
            )
            # If we got here without exception, consider it running
            self._server_running = True
            if not block:
                self.logger.info(
                    "SCP started (background).",
                    extra={"op": "SCP-START", "called_ae": self.scpAET, "alive": True},
                )
        except Exception as e:
            self._server = None
            self._server_running = False
            self.logger.error(
                f"Could not start SCP: {e}", extra={"op": "SCP-START", "called_ae": self.scpAET}
            )

    def stop(self):
        """Stop the DICOM SCP server (idempotent)."""
        if not self._server_running:
            self.logger.info(
                "SCP stop requested but server was not running.",
                extra={"op": "SCP-STOP", "called_ae": self.scpAET},
            )
            return

        self.logger.info("Stopping SCP…", extra={"op": "SCP-STOP", "called_ae": self.scpAET})
        try:
            if self._server:
                # Works across pynetdicom versions that return a server handle
                try:
                    self._server.shutdown()
                except AttributeError:
                    # Older versions: fall back to AE.shutdown()
                    pass
            self.ae.shutdown()  # safe to call regardless
        except Exception as e:
            self.logger.error(
                f"Exception during shutdown: {e}",
                extra={"op": "SCP-STOP", "called_ae": self.scpAET},
            )
        finally:
            self._server = None
            self._server_running = False
            self.logger.info("SCP stopped.", extra={"op": "SCP-STOP", "called_ae": self.scpAET})

    def add_custom_function_store(self, func: Callable[[evt.Event], None]):
        """
        Add a custom function to be run during `handle_store`.

        Parameters
        ----------
        func : Callable[[evt.Event], None]
            A custom function that takes an event as a parameter and returns None.

        Examples
        --------
        >>> def custom_store_function(event):
        ...     print(f"Custom store function called for Patient ID: {event.dataset.PatientID}")
        >>> scp = StoreSCP(aet='MY_SCP', ip='127.0.0.1', port=11112)
        >>> scp.add_custom_function_store(custom_store_function)
        """
        self.custom_functions_store.append(func)
        self.logger.info(f"Custom store function '{func.__name__}' added.")

    def add_custom_function_open(self, func: Callable[[evt.Event], None]):
        """
        Add a custom function to be run during `handle_open`.

        Parameters
        ----------
        func : Callable[[evt.Event], None]
            A custom function that takes an event as a parameter and returns None.

        Examples
        --------
        >>> def custom_open_function(event):
        ...     print(f"Custom open function called for remote address: {event.address}")
        >>> scp = StoreSCP(aet='MY_SCP', ip='127.0.0.1', port=11112)
        >>> scp.add_custom_function_open(custom_open_function)
        """
        self.custom_functions_open.append(func)
        self.logger.info(f"Custom open function '{func.__name__}' added.")

    def add_custom_function_close(self, func: Callable[[evt.Event], None]):
        """
        Add a custom function to be run during `handle_close`.

        Parameters
        ----------
        func : Callable[[evt.Event], None]
            A custom function that takes an event as a parameter and returns None.

        Examples
        --------
        >>> def custom_close_function(event):
        ...     print(f"Custom close function called for remote address: {event.address}")
        >>> scp = StoreSCP(aet='MY_SCP', ip='127.0.0.1', port=11112)
        >>> scp.add_custom_function_close(custom_close_function)
        """
        self.custom_functions_close.append(func)
        self.logger.info(f"Custom close function '{func.__name__}' added.")

    def remove_custom_function_store(self, func: Callable[[evt.Event], None]):
        """
        Remove a custom function from the `handle_store` custom functions list.

        Parameters
        ----------
        func : Callable[[evt.Event], None]
            The function to be removed.

        Raises
        ------
        ValueError
            If the function is not found in the custom functions list.
        """
        try:
            self.custom_functions_store.remove(func)
            self.logger.info(f"Custom store function '{func.__name__}' removed.")
        except ValueError:
            self.logger.error(f"Custom store function '{func.__name__}' not found.")

    def remove_custom_function_open(self, func: Callable[[evt.Event], None]):
        """
        Remove a custom function from the `handle_open` custom functions list.

        Parameters
        ----------
        func : Callable[[evt.Event], None]
            The function to be removed.

        Raises
        ------
        ValueError
            If the function is not found in the custom functions list.
        """
        try:
            self.custom_functions_open.remove(func)
            self.logger.info(f"Custom open function '{func.__name__}' removed.")
        except ValueError:
            self.logger.error(f"Custom open function '{func.__name__}' not found.")

    def remove_custom_function_close(self, func: Callable[[evt.Event], None]):
        """
        Remove a custom function from the `handle_close` custom functions list.

        Parameters
        ----------
        func : Callable[[evt.Event], None]
            The function to be removed.

        Raises
        ------
        ValueError
            If the function is not found in the custom functions list.
        """
        try:
            self.custom_functions_close.remove(func)
            self.logger.info(f"Custom close function '{func.__name__}' removed.")
        except ValueError:
            self.logger.error(f"Custom close function '{func.__name__}' not found.")

    def clear_custom_functions_store(self):
        """Clear all custom functions for the `handle_store` event."""
        self.custom_functions_store.clear()
        self.logger.info("All custom store functions cleared.")

    def clear_custom_functions_open(self):
        """Clear all custom functions for the `handle_open` event."""
        self.custom_functions_open.clear()
        self.logger.info("All custom open functions cleared.")

    def clear_custom_functions_close(self):
        """Clear all custom functions for the `handle_close` event."""
        self.custom_functions_close.clear()
        self.logger.info("All custom close functions cleared.")

    def register_sop_class(self, sop_class_uid: str, keyword: str):
        """
        Register a custom SOP Class UID if not already registered.

        Parameters
        ----------
        sop_class_uid : str
            The SOP Class UID to register.
        keyword : str
            The keyword for the SOP Class.

        Examples
        --------
        >>> scp = StoreSCP(aet='MY_SCP', ip='127.0.0.1', port=11112)
        >>> scp.register_sop_class('1.2.246.352.70.1.70', 'VarianRTPlanStorage')
        >>> scp.add_registered_presentation_context('VarianRTPlanStorage')
        """
        # Check if the SOP Class is already registered
        if not hasattr(sop_class, keyword):
            # Register the private SOP Class UID with the StorageServiceClass
            register_uid(uid=sop_class_uid, keyword=keyword, service_class=StorageServiceClass)
            self.logger.info(
                f"Registered custom SOP Class UID: {sop_class_uid} with keyword: {keyword}."
            )
            # Import the newly registered SOP Class
            new_sop_class = getattr(sop_class, keyword)
            self.ae.add_supported_context(new_sop_class)
        else:
            self.logger.debug(f"SOP Class with keyword '{keyword}' is already registered.")

    def add_registered_presentation_context(self, keyword: str):
        """
        Add a registered SOP Class UID to the supported presentation contexts.

        Parameters
        ----------
        keyword : str
            The keyword of the SOP Class to add.

        Raises
        ------
        ValueError
            If the SOP Class with the specified keyword is not registered.

        Examples
        --------
        >>> scp = StoreSCP(aet='MY_SCP', ip='127.0.0.1', port=11112)
        >>> scp.register_sop_class('1.2.246.352.70.1.70', 'VarianRTplanStorage')
        >>> scp.add_registered_presentation_context('VarianRTPlanStorage')
        """
        # Check if the SOP Class is registered
        if hasattr(sop_class, keyword):
            sop_class_instance = getattr(sop_class, keyword)
            self.ae.add_supported_context(sop_class_instance)
            self.logger.info(
                f"Added presentation context for SOP Class with keyword: '{keyword}'."
            )
        else:
            self.logger.error(
                "Failed to add presentation context: "
                f"SOP Class with keyword '{keyword}' is not registered."
            )
            raise ValueError(
                f"The SOP Class with keyword '{keyword}' is not registered."
                f" Please use `register_sop_class` to add the custom SOP Class."
            )

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
        Dict[str, logging.Handlers]
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

    def enable_wire_debu(self, enable: bool = True, level: int = logging.DEBUG):
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

    def _log_accepted_contexts_debug(self, assoc):
        """Emit accepted PCs at DEBUG level (handy for diagnosing rejections)."""
        if not self.logger.isEnabledFor(logging.DEBUG) or not assoc:
            return
        lines = []
        for pc in assoc.accepted_contexts:
            ts_list = ", ".join(str(ts) for ts in pc.transfer_syntax)
            lines.append(f"{pc.abstract_syntax.name} [{ts_list}]")
        if lines:
            self.logger.debug("Accepted presentation contexts:\n  - " + "\n  - ".join(lines))
