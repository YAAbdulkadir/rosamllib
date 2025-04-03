"""
Class module for DICOM SCP
"""

import logging
import pynetdicom.sop_class as sop_class
from logging import StreamHandler, FileHandler, Formatter
from typing import Optional, Callable, List
from pydicom.dataset import Dataset
from pynetdicom import AE, StoragePresentationContexts, evt, register_uid
from pynetdicom.sop_class import Verification
from pynetdicom.service_class import StorageServiceClass
from rosamllib.utils import validate_entry


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

    def handle_open(self, event):
        """Log association establishments.

        Parameters
        ----------
        event : `events.Event`
            A DICOM association establishment event
        """
        msg = f"Connected with remote at {event.address}"
        self.logger.info(msg)

        # Run custom functions
        for func in self.custom_functions_open:
            try:
                self.logger.debug(f"Running custom function {func.__name__} in handle_open")
                func(event)
            except Exception as e:
                self.logger.error(
                    f"Error running custom function {func.__name__} in handle_open: {e}"
                )

    def handle_close(self, event):
        """Log when association is disconnected.

        Parameters
        ----------
        event : `events.Event`
            A DICOM association close event
        """
        msg = f"Disconnected from remote at {event.address}"
        self.logger.info(msg)

        # Run custom functions
        for func in self.custom_functions_close:
            try:
                self.logger.debug(f"Running custom funcion {func.__name__} in handle_close.")
                func(event)
            except Exception as e:
                self.logger.error(
                    f"Error running custom function {func.__name__} in handle_close: {e}"
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
        msg = f"Received {event.dataset.Modality} with SOPInstanceUID={event.dataset.Modality}."
        self.logger.debug(msg)
        try:
            # Run custom functions
            for func in self.custom_functions_store:
                try:
                    self.logger.debug(f"Running custom function {func.__name__}  in handle_store")
                    func(event)
                except Exception as e:
                    self.logger.error(
                        f"Error running custom function {func.__name__} in handle_store: {e}"
                    )

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

    def start(self, block: bool = False):
        """Start the DICOM SCP server.

        Parameters
        ----------
        block : bool, optional
            Whether to block the thread that called this method, by default False
        """

        try:
            msg = f"Started SCP with AE Title {self.scpAET} on port {self.scpPort}"
            self.logger.info(msg)

            self.ae.start_server(
                (self.scpIP, self.scpPort), block=block, evt_handlers=self.handlers
            )

        except Exception as e:
            self.logger.error(f"Could not start SCP. {e}")

    def stop(self):
        """Stop the DICOM SCP server."""
        try:
            self.ae.shutdown()
            msg = "Stopped SCP"
            self.logger.info(msg)
        except Exception as e:
            msg = f"Exception raised while stopping SCP. {e}"
            self.logger.error(msg)

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
