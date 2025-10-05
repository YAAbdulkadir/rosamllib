import json
import logging
from logging import Handler
from logging.handlers import RotatingFileHandler
import re
from typing import Union
from ipaddress import ip_address


class _ContextFilter(logging.Filter):
    """Attach static context to all log records created by this logger."""

    def __init__(self, **static_ctx):
        super().__init__()
        self.static_ctx = static_ctx

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in self.static_ctx.items():
            setattr(record, k, v)
        # Ensure fields exist even if not provided in `extra=`
        for k in (
            "op",
            "ae_name",
            "remote_host",
            "remote_port",
            "sop_class",
            "study_uid",
            "series_uid",
            "sop_uid",
            "status_hex",
        ):
            if not hasattr(record, k):
                setattr(record, k, None)
        return True


class JsonFormatter(logging.Formatter):
    """Compact JSON logs for machines (SIEMs, Elastic, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        # base fields
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # useful runtime fields
        for k in (
            "op",
            "ae_name",
            "remote_host",
            "remote_port",
            "sop_class",
            "study_uid",
            "series_uid",
            "sop_uid",
            "status_hex",
        ):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        # exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _dedupe_handlers(logger: logging.Logger) -> None:
    """Avoid duplicate handlers when configure_logging is called repeatedly."""
    seen = set()
    unique: list[Handler] = []
    for h in logger.handlers:
        key = (type(h), getattr(h, "baseFilename", None), getattr(h, "stream", None))
        if key not in seen:
            seen.add(key)
            unique.append(h)
    logger.handlers = unique


def build_formatter(human: bool = True) -> logging.Formatter:
    if human:
        # 1-line, timestamp, level initial, logger short name
        return logging.Formatter(
            fmt="%(levelname).1s:%(asctime)s:%(name)s: %(message)s "
            "[op=%(op)s ae=%(ae_name)s host=%(remote_host)s:%(remote_port)s "
            "sop=%(sop_class)s study=%(study_uid)s series=%(series_uid)s sopi=%(sop_uid)s]"
        )
    return JsonFormatter()


def make_rotating_file_handler(
    path: str,
    level: int,
    formatter: logging.Formatter,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> RotatingFileHandler:
    fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    return fh


def attach_pynetdicom_to_logger(enable: bool, level: int = logging.DEBUG) -> None:
    """Route pynetdicom’s logs through standard logging (instead of print)."""
    log = logging.getLogger("pynetdicom")
    log.handlers.clear()
    log.setLevel(level if enable else logging.WARNING)
    log.propagate = True  # bubble up into your root/package logger


def validate_ae_title(ae_title: str) -> bool:
    """
    Validate a DICOM AE (Application Entity) Title.

    Parameters
    ----------
    ae_title : str
        The AE Title to validate.

    Returns
    -------
    bool
        True if the AE Title is valid, False otherwise.

    Notes
    -----
    - AE Titles must be between 1 and 16 characters long.
    - Allowed characters are uppercase letters (A-Z), digits (0-9),
      space, underscore (_), dash (-), and period (.).
    """
    if not (1 <= len(ae_title) <= 16):
        return False

    if not re.match(r"^[A-Z0-9 _\-.]+$", ae_title):
        return False

    return True


def validate_host(host: str) -> bool:
    """
    Validate a host address (IP or hostname).

    Parameters
    ----------
    host : str
        The host address to validate. This can be an IP address or hostname.

    Returns
    -------
    bool
        True if the host address is valid, False otherwise.

    Notes
    -----
    - For IP addresses, both IPv4 and IPv6 are supported.
    - Hostnames must be alphanumeric, may include hyphens, and
      must not exceed 253 characters.
    """
    try:
        # Try to parse as an IP address
        ip_address(host)
        return True
    except ValueError:
        # If not an IP address, validate as hostname
        if len(host) > 253:
            return False
        if re.match(r"^[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*$", host):
            return True
    return False


def validate_port(port: int) -> bool:
    """
    Validate a port number.

    Parameters
    ----------
    port : int
        The port number to validate.

    Returns
    -------
    bool
        True if the port number is valid, False otherwise.

    Notes
    -----
    - Valid port numbers are integers between 1 and 65535.
    """
    return 1 <= port <= 65535


def validate_entry(input_text: Union[str, int], entry_type: str) -> bool:
    """Checks whether a text input from the user contains invalid characters.

    Parameters
    ----------
    input_text : Union[str, int]
        The text input to a given field.
    entry_type : str
        The type of field where the text was input. The different
        types are:
        * AET
        * Port
        * IP

    Returns
    -------
    bool
        Whether the input was valid or not.
    """
    if entry_type == "AET":
        return validate_ae_title(input_text)
    elif entry_type == "IP":
        return validate_host(input_text)
    elif entry_type == "Port":
        return validate_port(input_text)

    else:
        return False
