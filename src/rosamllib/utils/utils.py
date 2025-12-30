from typing import Any, Dict, List, Union
import pandas as pd
import warnings
from pydicom.datadict import dictionary_VR
from pydicom.multival import MultiValue
from rosamllib.constants import VR_TO_DTYPE
from functools import wraps
from pydicom.valuerep import DA, TM, DT
from datetime import date as _Date, time as _Time, datetime as _DateTime


def query_df(df: pd.DataFrame, **filters: Union[str, List[Any], Dict[str, Any]]) -> pd.DataFrame:
    """
    Filters a Pandas DataFrame based on a set of conditions, including wildcards (*, ?),
    ranges, lists, regular expressions, and inverse regular expressions. Supports escaping
    for literal wildcards.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to query.
    **filters : dict
        A set of filter conditions passed as keyword arguments. Each key is a column
        name, and its value is a condition. Supported conditions:
        - Exact match: {"column": "value"}
        - Wildcards: {"column": "value*"} or {"column": "val?e"}
          (* matches multiple characters, ? matches one character).
        - Ranges: {"column": {"gte": min_value, "lte": max_value}}
        - Regular expressions: {"column": {"RegEx": "pattern"}}
        - Inverse regular expressions: {"column": {"NotRegEx": "pattern"}}
        - Escaped wildcards: {"column": "val\\*e"} to match the literal `*` or `?`.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame based on the conditions provided.

    Notes
    -----
    - If the filter value contains `*`, it will be treated as a wildcard matching zero
      or more characters. Similarly, `?` will match exactly one character.
    - To match the literal characters `*` or `?`, escape them with a backslash (\\),
      e.g., `{"column": "value\\*"}`

    Examples
    --------
    # Sample DataFrame
    >>> data = {
    ...     "PatientID": ["123", "456", "789", "101", "121"],
    ...     "StudyDate": ["2023-01-01", "2023-02-15", "2023-03-01", "2023-04-20", "2023-05-10"],
    ...     "Age": [30, 45, 29, 60, 35],
    ... }
    >>> df = pd.DataFrame(data)

    # Example 1: Wildcard and exact match
    >>> filters = {"PatientID": ["1*", "456"]}
    >>> query_df(df, **filters)
      PatientID StudyDate  Age
    0       123 2023-01-01   30
    3       101 2023-04-20   60
    4       121 2023-05-10   35
    1       456 2023-02-15   45

    # Example 2: Date range
    >>> filters = {"StudyDate": {"gte": "2023-03-01"}}
    >>> query_df(df, **filters)
      PatientID StudyDate  Age
    2       789 2023-03-01   29
    3       101 2023-04-20   60
    4       121 2023-05-10   35
    """

    def _apply_condition(column: str, condition: Any) -> pd.Series:
        """
        Applies a single condition to a column of the DataFrame.

        Parameters
        ----------
        column : str
            The column to apply the condition on.
        condition : Any
            The condition to apply (exact match, wildcard, range, RegEx, etc.).

        Returns
        -------
        pd.Series
            A boolean mask indicating the rows that match the condition.
        """

        def process_literal(value: str) -> str:
            """
            Process escaped literals for wildcards.

            Parameters
            ----------
            value : str
                The input string potentially containing escaped literals.

            Returns
            -------
            str
                A regex-safe pattern with escaped wildcards handled.
            """
            return (
                value.replace(r"\*", r"\x1B")  # Temporarily replace \* with \x1B
                .replace(r"\?", r"\x1C")  # Temporarily replace \? with \x1C
                .replace("*", ".*")  # Convert * to regex wildcard
                .replace("?", ".")  # Convert ? to regex wildcard
                .replace(r"\x1B", r"\*")  # Restore literal *
                .replace(r"\x1C", r"\?")  # Restore literal ?
            )

        # Exact match or wildcard
        if isinstance(condition, str):
            if "*" in condition or "?" in condition:  # Wildcard filtering
                pattern = process_literal(condition)
                return df[column].astype(str).str.match(f"^{pattern}$", na=False)
            else:  # Exact match
                return df[column] == condition

        # Complex filtering
        elif isinstance(condition, dict):
            mask = pd.Series(True, index=df.index)
            for op, value in condition.items():
                if op == "RegEx":  # RegEx matching
                    if not isinstance(value, str):
                        raise ValueError("RegEx operator requires a string pattern.")
                    mask &= df[column].astype(str).str.contains(value, na=False)
                elif op == "NotRegEx":  # Inverse RegEx matching
                    if not isinstance(value, str):
                        raise ValueError("NotRegEx operator requires a string pattern.")
                    mask &= ~df[column].astype(str).str.contains(value, na=False)
                elif isinstance(value, str) and ("*" in value or "?" in value):
                    # Convert wildcard to regex pattern
                    pattern = process_literal(value)
                    if op == "eq":  # Equal with wildcard
                        mask &= df[column].astype(str).str.match(f"^{pattern}$", na=False)
                    elif op == "neq":  # Not equal with wildcard
                        mask &= ~df[column].astype(str).str.match(f"^{pattern}$", na=False)
                    else:
                        raise ValueError(
                            f"Operator '{op}' does not support wildcards in range filters."
                        )
                else:
                    if op == "gte":  # Greater than or equal to
                        mask &= df[column] >= value
                    elif op == "lte":  # Less than or equal to
                        mask &= df[column] <= value
                    elif op == "gt":  # Greater than
                        mask &= df[column] > value
                    elif op == "lt":  # Less than
                        mask &= df[column] < value
                    elif op == "eq":  # Equal
                        mask &= df[column] == value
                    elif op == "neq":  # Not equal
                        mask &= df[column] != value
                    else:
                        raise ValueError(f"Unsupported operator '{op}' in range filter.")
            return mask

        # List of values
        elif isinstance(condition, list):
            return df[column].isin(condition)

        raise ValueError(f"Unsupported condition type for column '{column}'.")

    filtered_df = df.copy()

    for column, condition in filters.items():
        if column not in filtered_df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        if isinstance(condition, list):  # Multiple conditions for the same column
            combined_mask = pd.Series(False, index=filtered_df.index)
            for sub_condition in condition:
                combined_mask |= _apply_condition(column, sub_condition)
            filtered_df = filtered_df.loc[
                combined_mask.reindex(filtered_df.index, fill_value=False)
            ]
        else:
            mask = _apply_condition(column, condition)
            filtered_df = filtered_df.loc[mask.reindex(filtered_df.index, fill_value=False)]

    return filtered_df


def _parse_int_like(value: Any):
    if isinstance(value, MultiValue):
        return [int(v) for v in value]
    return int(value)


def _parse_float_like(value: Any):
    if isinstance(value, MultiValue):
        return [float(v) for v in value]
    return float(value)


def _parse_str_like(value: Any):
    if isinstance(value, MultiValue):
        return [str(v) for v in value]
    return str(value)


def _da_to_date(v) -> _Date:
    """
    Convert a DICOM DA (or DA-like string) to datetime.date.
    DIOCM DA is 'YYYYMMDD' (optionally shorter, but we assume full).
    """
    if not isinstance(v, DA):
        v = DA(v)
    s = str(v)
    year = int(s[0:4])
    month = int(s[4:6]) if len(s) >= 6 else 1
    day = int(s[6:8]) if len(s) >= 8 else 1

    return _Date(year, month, day)


def _tm_to_time(v) -> _Time:
    """
    Convert a DICOM TM (or TM-like string) to datetime.time.

    DICOM TM is 'HHMMSS.frac', or shorter (HHMM, HH).
    We parse what we have and default missing fields to zero.
    """
    if not isinstance(v, TM):
        v = TM(v)
    s = str(v)

    # Split off fractional seconds if present
    if "." in s:
        main, frac = s.split(".", 1)
    else:
        main, frac = s, ""

    hh = int(main[0:2]) if len(main) >= 2 else 0
    mm = int(main[2:4]) if len(main) >= 4 else 0
    ss = int(main[4:6]) if len(main) >= 6 else 0

    micro = 0
    if frac:
        # up to 6 digits of fractional seconds -> microseconds
        frac = (frac + "000000")[:6]
        micro = int(frac)

    return _Time(hh, mm, ss, micro)


def _dt_to_datetime(v) -> _DateTime:
    """
    Convert a DICOM DT (or DT-like string) to datetime.datetime.

    We first try pydicom's DT(...) helper to get a Python datetime via .datetime;
    if that fails, we fall back to a simple manual parse of 'YYYYMMDDHHMMSS.frac'.
    """
    if not isinstance(v, DT):
        v = DT(v)

    s = str(v)

    # Basic date/time parts
    year = int(s[0:4])
    month = int(s[4:6]) if len(s) >= 6 else 1
    day = int(s[6:8]) if len(s) >= 8 else 1

    hh = int(s[8:10]) if len(s) >= 10 else 0
    mm = int(s[10:12]) if len(s) >= 12 else 0
    ss = int(s[12:14]) if len(s) >= 14 else 0

    micro = 0
    # fractional seconds and/or offsets may follow; we just handle fraction
    rest = s[14:]
    if rest.startswith("."):
        frac = rest[1:]
        # strip anything after a potential '+' or '-' timezone offset
        for sep in ("+", "-"):
            if sep in frac:
                frac = frac.split(sep, 1)[0]
                break
        frac = (frac + "000000")[:6]
        if frac.strip("0"):
            micro = int(frac)

    return _DateTime(year, month, day, hh, mm, ss, micro)


def _parse_da_like(value: Any):
    if isinstance(value, MultiValue):
        return [_da_to_date(v) for v in value]
    return _da_to_date(value)


def _parse_tm_like(value: Any):
    if isinstance(value, MultiValue):
        return [_tm_to_time(v) for v in value]
    return _tm_to_time(value)


def _parse_dt_like(value: Any):
    if isinstance(value, MultiValue):
        return [_dt_to_datetime(v) for v in value]
    return _dt_to_datetime(value)


def parse_vr_value(vr: str, value: Any):
    """
    Parses DICOM tag values based on VR.

    Parameters
    ----------
    vr : str
        The VR of the DICOM tag.
    value : Any
        The raw value of the DICOM tag (string, MultiValue, or already-parsed type).

    Returns
    -------
    Any
        Parsed value in the appropriate type (e.g., date, time, datetime, int, float, str),
        or the original value if parsing is not applicable or fails.
    """
    # Treat None / emtpy string as-is
    if value in (None, "", b""):
        if vr == "PN":
            return ""
        if vr in ["DA", "DT", "TM"]:
            return None
        return value

    vr = (vr or "").upper()

    try:
        # Date/Time types
        if vr == "DA":
            return _parse_da_like(value)

        elif vr == "TM":
            return _parse_tm_like(value)

        elif vr == "DT":
            return _parse_dt_like(value)

        # Integer like
        elif vr in ["IS", "SL", "SS", "UL", "US"]:
            return _parse_int_like(value)

        # Float like
        elif vr in ["DS", "FL", "FD"]:
            return _parse_float_like(value)

        # String like
        elif vr in {"LO", "SH", "ST", "LT", "CS", "UI", "AE", "AS", "UC", "UR", "UT", "PN"}:
            return _parse_str_like(value)

    except Exception:
        # If any of the parsing steps above fails, fall back to original value
        return None

    # Fallbackl for VRs we haven't explicitly mapped: store as string
    return _parse_str_like(value)


def get_pandas_column_dtype(tag):
    """
    Determines the Pandas dtype for a given DICOM tag based on its VR.

    Parameters
    ----------
    tag : tuple
        The DICOM tag in (group, element) format.

    Returns
    -------
    type or str
        The corresponding Pandas dtype, or `object` if the VR is unknown.
    """
    try:
        vr = dictionary_VR(tag)
        return VR_TO_DTYPE.get(vr, object)
    except KeyError:
        return object


def get_running_env():
    try:
        from IPython import get_ipython
        import sys

        shell = get_ipython()
        if shell is None:
            return "script"  # Running in a regular script

        # Check if running in a Jupyter environment
        if "ipykernel" in sys.modules:
            # Check if JupyterLab or Jupyter Notebook
            from jupyter_server.serverapp import list_running_servers

            if any("lab" in server["url"] for server in list_running_servers()):
                return "jupyterlab"
            return "jupyter_notebook"
    except Exception:
        return "script"  # Fallback to script mode


def deprecated(replacement: str, remove_in: str = ""):
    msg_tail = f" and will be removed in {remove_in}" if remove_in else ""

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{fn.__qualname__} is deprecated{msg_tail}; use {replacement} instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

        return wrapper

    return deco
