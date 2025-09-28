import re
from typing import Union
from ipaddress import ip_address


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
