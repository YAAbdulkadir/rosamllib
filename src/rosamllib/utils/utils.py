from typing import Any, Dict, List, Union
import pandas as pd
import warnings
from datetime import datetime
from pydicom.datadict import dictionary_VR
from pydicom.multival import MultiValue
from rosamllib.constants import VR_TO_DTYPE
from functools import wraps


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


def parse_vr_value(vr, value):
    """
    Parses DICOM tag values based on VR.

    Parameters
    ----------
    vr : str
        The VR of the DICOM tag.
    value : str
        The raw value of the DICOM tag.

    Returns
    -------
    Parsed value in the appropriate type (e.g., date, time).
    """
    if value:
        if vr == "DA":
            try:
                return datetime.strptime(value, "%Y%m%d").date()
            except ValueError:
                return None
        elif vr == "TM":
            try:
                return datetime.strptime(value, "%H%M%S.%f").time()
            except ValueError:
                try:
                    return datetime.strptime(value, "%H%M%S").time()
                except ValueError:
                    return None
        elif vr == "DT":
            try:
                return datetime.strptime(value, "%Y%m%d%H%M%S.%f")
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y%m%d%H%M%S")
                except ValueError:
                    return None
        elif vr in ["IS", "SL", "SS", "UL", "US"]:
            try:
                if isinstance(value, MultiValue):
                    return [int(v) for v in value]
                else:
                    return int(value)
            except ValueError:
                return None
        elif vr in ["DS", "FL", "FD"]:
            try:
                if isinstance(value, MultiValue):
                    return [float(v) for v in value]
                else:
                    return float(value)
            except ValueError:
                return None
        elif vr == "LO":
            try:
                if isinstance(value, MultiValue):
                    return [str(v) for v in value]
                else:
                    return str(value)
            except ValueError:
                return None

    return value


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
