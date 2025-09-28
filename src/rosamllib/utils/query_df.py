import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Union
import pandas as pd
import numpy as np
import datetime as _dt

# -----------------------------
# Operator registry (extensible)
# -----------------------------
_OPS: Dict[str, Callable[[pd.Series, Any, Dict[str, Any]], pd.Series]] = {}
_COMPARATORS = {"eq", "neq", "lt", "lte", "gt", "gte", "between", "outside"}
# any custom ops whose values should NOT be coerced to datetime/time
_CUSTOM_NO_COERCE = {"in_last_days", "time_between"}
_COERCE_TEMPORAL_OPS = {
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "nin",
    "between",
    "time_between",
}


def register_op(name: str, fn: Callable[[pd.Series, Any, Dict[str, Any]], pd.Series]) -> None:
    """Register a custom operator usable in query_df via {"op": value}."""
    _OPS[name] = fn


# -----------------------------
# Helpers
# -----------------------------
_wild_escape_back = {"\\*": "\x1b", "\\?": "\x1c"}
_wild_unescape_back = {"\x1b": r"\*", "\x1c": r"\?"}


def _wildcard_to_regex(s: str) -> str:
    # preserve escaped wildcards, then translate, then restore
    for k, v in _wild_escape_back.items():
        s = s.replace(k, v)
    s = s.replace("*", ".*").replace("?", ".")
    for k, v in _wild_unescape_back.items():
        s = s.replace(k, v)
    return f"^{s}$"


def _coerce_str(s: pd.Series, case_insensitive: bool) -> pd.Series:
    out = s.astype("string", copy=False)
    return out.str.lower() if case_insensitive else out


def _is_iterable_nonstring(x: Any) -> bool:
    return isinstance(x, (list, tuple, set))


def _iterable_contains(series: pd.Series, needle: Any) -> pd.Series:
    # True if needle is in a container cell (list/tuple/set); scalar cells -> False unless equal
    def contains(x):
        if _is_iterable_nonstring(x):
            return needle in x
        return x == needle

    return series.apply(contains)


def _infer_temporal_kind(series: pd.Series):
    """Return 'datetime' | 'date' | 'time' | None by inspecting dtype/values."""
    # pandas datetime64
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # object-dtype: inspect first non-null
    for v in series.dropna():
        if isinstance(v, _dt.datetime):
            return "datetime"
        if isinstance(v, _dt.date) and not isinstance(v, _dt.datetime):
            return "date"
        if isinstance(v, _dt.time):
            return "time"
        break
    return None


def _parse_temporal_literal(kind: str, value):
    """Parse a single literal to the appropriate temporal type based on 'kind'."""
    if value is None:
        return None
    if kind == "datetime":
        # Let pandas handle many formats; keep tz if provided
        return pd.to_datetime(value, errors="raise", utc=False)
    if kind == "date":
        if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
            return value
        return pd.to_datetime(value, errors="raise").date()
    if kind == "time":
        if isinstance(value, _dt.time):
            return value
        # Accept HH:MM[:SS[.ffffff]]
        s = str(value)
        # Try fast path %H:%M:%S
        fmts = ["%H:%M:%S", "%H:%M"]
        for fmt in fmts:
            try:
                return _dt.datetime.strptime(s, fmt).time()
            except Exception:
                pass
        # Last resort: let pandas parse, then extract time
        return pd.to_datetime(s, errors="raise").time()
    return value


def _parse_days_like(v):
    """Return (days:int, now) supporting int, '7d', pd.Timedelta, datetime.timedelta, or dict."""
    now = None

    if isinstance(v, dict):
        days = int(v.get("days", 0))
        now = v.get("now")
        return days, now

    if isinstance(v, (int, float)):
        return int(v), None

    if isinstance(v, str):
        td = pd.to_timedelta(v, errors="coerce")
        if td is not pd.NaT:
            return int(td / pd.Timedelta(days=1)), None
        return int(v), None  # e.g. "7"

    if isinstance(v, _dt.timedelta):
        return int(v.days), None

    if isinstance(v, pd.Timedelta):
        return int(v / pd.Timedelta(days=1)), None

    raise TypeError(f"in_last_days expects int/str/timedelta/dict, got {type(v).__name__}")


def _maybe_coerce_temporal(series: pd.Series, value):
    """Coerce only values of comparison-like ops to the series' temporal kind."""
    kind = _infer_temporal_kind(series)
    if not kind:
        return value

    # lists/tuples: coerce each item
    if isinstance(value, (list, tuple)):
        return type(value)(_parse_temporal_literal(kind, v) for v in value)

    # dict of operators: only coerce for comparison ops
    if isinstance(value, dict):
        out = {}
        for op, v in value.items():
            if op in _COERCE_TEMPORAL_OPS:
                out[op] = _maybe_coerce_temporal(series, v)
            else:
                # e.g. in_last_days, RegEx, approx, exists/missing/contains/callable, custom ops…
                out[op] = v
        return out

    # scalar
    return _parse_temporal_literal(kind, value)


def _extract_time_component(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.time

    # object: normalize datetime -> time, leave time as-is
    def _to_time(x):
        if isinstance(x, _dt.datetime):
            return x.time()
        if isinstance(x, _dt.time):
            return x
        return None

    return s.apply(_to_time) if pd.api.types.is_object_dtype(s) else s


def _coerce_time_literal(x):
    if isinstance(x, _dt.time):
        return x
    if isinstance(x, _dt.datetime):
        return x.time()
    # accept "HH:MM[:SS]"
    s = str(x)
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return _dt.datetime.strptime(s, fmt).time()
        except Exception:
            pass
    # last resort: pandas
    return pd.to_datetime(s, errors="raise").time()


def _truncate_to_date(x):
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    return pd.to_datetime(x, errors="raise").date()


def _dot_get(obj: Any, path: str) -> Any:
    """
    Safely traverse dict/list/obj using dot-and-index syntax: 'a.b[0].c'.
    Works for scalar cells that contain nested structures.
    """
    if path == "" or obj is None:
        return obj
    cur = obj
    # split into tokens like ["a", "b[0]", "c"]
    tokens = re.split(r"\.", path)
    for tok in tokens:
        m = re.match(r"^([^\[]+)(\[(\d+)\])?$", tok)
        if not m:
            return None
        key, _, idx = m.groups()
        # dict / attr
        if isinstance(cur, Mapping):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None) if hasattr(cur, key) else None
        if cur is None:
            return None
        # optional list index
        if idx is not None:
            i = int(idx)
            if isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
                if 0 <= i < len(cur):
                    cur = cur[i]
                else:
                    return None
            else:
                return None
    return cur


def _extract_col(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Supports 'Column', or 'Column.subkey[0].field' for nested data inside the cell.
    If 'column' has a dot, we split at the first token and traverse inside each cell.
    """
    if column in df.columns:
        return df[column]
    # try Column.something
    root = column.split(".", 1)[0]
    if root in df.columns:
        subpath = column[len(root) :].lstrip(".")
        if not subpath:
            return df[root]
        return df[root].apply(lambda x: _dot_get(x, subpath))
    raise KeyError(f"Column '{column}' not found")


# -----------------------------
# Built-in operators
# -----------------------------
def _op_regex(s: pd.Series, pattern: str, ctx: Dict[str, Any]) -> pd.Series:
    s2 = _coerce_str(s, ctx["case_insensitive"])
    flags = re.IGNORECASE if ctx["case_insensitive"] else 0
    return s2.str.contains(pattern, na=False, regex=True, flags=flags)


def _op_notregex(s: pd.Series, pattern: str, ctx: Dict[str, Any]) -> pd.Series:
    return ~_op_regex(s, pattern, ctx)


def _op_eq(s: pd.Series, value: Any, ctx: Dict[str, Any]) -> pd.Series:
    if isinstance(value, str) and ("*" in value or "?" in value):
        pattern = _wildcard_to_regex(value)
        return _coerce_str(s, ctx["case_insensitive"]).str.match(pattern, na=False)
    if _is_iterable_nonstring(value):
        # treat as IN
        return s.isin(list(value))
    return s == value


def _op_neq(s: pd.Series, value: Any, ctx: Dict[str, Any]) -> pd.Series:
    return ~_op_eq(s, value, ctx)


def _op_in(s: pd.Series, values: Iterable[Any], ctx: Dict[str, Any]) -> pd.Series:
    return s.isin(list(values))


def _op_nin(s: pd.Series, values: Iterable[Any], ctx: Dict[str, Any]) -> pd.Series:
    return ~s.isin(list(values))


def _op_gte(s: pd.Series, v: Any, ctx: Dict[str, Any]) -> pd.Series:
    return s >= v


def _op_lte(s: pd.Series, v: Any, ctx: Dict[str, Any]) -> pd.Series:
    return s <= v


def _op_gt(s: pd.Series, v: Any, ctx: Dict[str, Any]) -> pd.Series:
    return s > v


def _op_lt(s: pd.Series, v: Any, ctx: Dict[str, Any]) -> pd.Series:
    return s < v


def _op_exists(s: pd.Series, _: Any, ctx: Dict[str, Any]) -> pd.Series:
    return s.notna()


def _op_missing(s: pd.Series, _: Any, ctx: Dict[str, Any]) -> pd.Series:
    return s.isna()


def _op_contains(s: pd.Series, needle: Any, ctx: Dict[str, Any]) -> pd.Series:
    # If cells are containers, check membership; else fallback to substring for strings
    def contains(x):
        if _is_iterable_nonstring(x):
            try:
                return needle in x
            except Exception:
                return False
        if isinstance(x, str) and isinstance(needle, str):
            a = x.lower() if ctx["case_insensitive"] else x
            b = needle.lower() if ctx["case_insensitive"] else needle
            return b in a
        return False

    return s.apply(contains)


def _op_callable(s: pd.Series, fn: Callable[[Any], bool], ctx: Dict[str, Any]) -> pd.Series:
    return s.apply(lambda x: bool(fn(x)))


def _op_approx(s: pd.Series, spec: Mapping[str, Any], ctx: Dict[str, Any]) -> pd.Series:
    """
    Approximate numeric equality:
      {"approx": {"value": 1.23, "atol": 1e-3, "rtol": 1e-4}}
    """
    v = spec.get("value", None)
    if v is None:
        return pd.Series(False, index=s.index)
    atol = float(spec.get("atol", ctx["default_atol"]))
    rtol = float(spec.get("rtol", ctx["default_rtol"]))
    s_num = pd.to_numeric(s, errors="coerce")
    return pd.Series(np.isclose(s_num.values, float(v), rtol=rtol, atol=atol), index=s.index)


def _op_time_between(s: pd.Series, v, ctx: dict) -> pd.Series:
    """
    v must be [start, end]; start/end are time-like (str/'HH:MM[:SS]' or datetime.time).
    Works for Series of dtype datetime64 (uses .dt.time) or time objects in object dtype.
    Wrap-around supported: if start > end, window spans midnight.
    """
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError("time_between expects a 2-tuple/list: [start, end]")

    start, end = _coerce_time_literal(v[0]), _coerce_time_literal(v[1])

    # Pull out time-of-day to compare
    stimes = _extract_time_component(s)

    # If column is datetime64 => stimes is .dt.time already (okay).
    # If column is object of datetime.time, comparisons work.
    # If column is date (no time), it's not applicable.
    if pd.api.types.is_datetime64_any_dtype(s):
        series_time = stimes
    else:
        # object dtype: make sure items are times
        if not stimes.dropna().empty and not isinstance(stimes.dropna().iloc[0], _dt.time):
            raise ValueError("time_between is only valid for time/datetime columns.")
        series_time = stimes

    if start <= end:
        return series_time.apply(lambda t: (t is not None) and (start <= t <= end))
    else:
        # wrap-around (e.g., 22:00..06:00)
        return series_time.apply(lambda t: (t is not None) and (t >= start or t <= end))


def _op_in_last_days(s: pd.Series, v, ctx: dict) -> pd.Series:
    """
    v: int/float days, '7d'/'48h', datetime.timedelta, pd.Timedelta,
       or dict {'days': N, 'now': <datetime|str|Timestamp>}.
    Works for datetime (tz-aware or naive) and date columns.
    """

    # --- parse "days" and optional "now" ---
    def _parse_days_like(v_):
        if isinstance(v_, dict):
            return int(v_.get("days", 0)), v_.get("now")
        if isinstance(v_, (int, float)):
            return int(v_), None
        if isinstance(v_, str):
            td = pd.to_timedelta(v_, errors="coerce")
            if td is not pd.NaT:
                return int(td / pd.Timedelta(days=1)), None
            return int(v_), None
        if isinstance(v_, _dt.timedelta):
            return int(v_.days), None
        if isinstance(v_, pd.Timedelta):
            return int(v_ / pd.Timedelta(days=1)), None
        raise TypeError(f"in_last_days expects int/str/timedelta/dict, got {type(v_).__name__}")

    days, now = _parse_days_like(v)

    # default "now" if not provided
    if now is None:
        now = ctx.get("now", _dt.datetime.now())

    # --- datetime columns ---
    if pd.api.types.is_datetime64_any_dtype(s):
        tz = getattr(s.dt, "tz", None)  # None for naive; tzinfo for aware
        now_ts = pd.Timestamp(now)
        if tz is not None:
            # series is tz-aware: localize/convert 'now' to same tz
            if now_ts.tz is None:
                now_ts = now_ts.tz_localize(tz)
            else:
                now_ts = now_ts.tz_convert(tz)
        else:
            # series is tz-naive: ensure 'now' is naive too
            if now_ts.tz is not None:
                now_ts = now_ts.tz_convert(None).tz_localize(None)
        cutoff = now_ts - pd.Timedelta(days=days)
        return s >= cutoff

    # --- date-like columns ---
    def _to_date(x):
        if isinstance(x, _dt.datetime):
            return x.date()
        if isinstance(x, _dt.date):
            return x
        ts = pd.to_datetime(x, errors="coerce")
        return ts.date() if pd.notna(ts) else None

    now_d = _to_date(now)
    cutoff_d = now_d - _dt.timedelta(days=days)

    if pd.api.types.is_object_dtype(s):
        return s.apply(_to_date) >= cutoff_d
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.date >= cutoff_d
    return s.apply(_to_date) >= cutoff_d


# Register built-ins
register_op("RegEx", _op_regex)
register_op("NotRegEx", _op_notregex)
register_op("eq", _op_eq)
register_op("neq", _op_neq)
register_op("in", _op_in)
register_op("nin", _op_nin)
register_op("gte", _op_gte)
register_op("lte", _op_lte)
register_op("gt", _op_gt)
register_op("lt", _op_lt)
register_op("exists", _op_exists)
register_op("missing", _op_missing)
register_op("contains", _op_contains)
register_op("callable", _op_callable)
register_op("approx", _op_approx)
register_op("time_between", _op_time_between)
register_op("in_last_days", _op_in_last_days)


# -----------------------------
# Main: generic query_df
# -----------------------------
def query_df(
    df: pd.DataFrame,
    *,
    case_insensitive: bool = False,
    default_atol: float = 1e-6,
    default_rtol: float = 1e-6,
    **filters: Union[
        Any,  # scalar equals
        List[Any],  # OR of conditions
        Mapping[str, Any],  # operator dict: {"gte": 3, "lt": 10}
        Callable[[Any], bool],  # row-wise predicate for the column
    ],
) -> pd.DataFrame:
    """
    Filter a Pandas DataFrame with expressive, extensible conditions.

    This generic helper supports exact matches, wildcards, regex, range comparisons,
    container membership, approximate numeric equality, callable predicates, temporal
    comparisons (dates/times/datetimes), and nested traversal via dot-paths into
    structured cells (e.g., dicts/lists or small objects).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to filter.
    case_insensitive : bool, default False
        If True, string equality and wildcard/regex operations are performed in
        a case-insensitive manner. (Numeric and temporal comparisons are unaffected.)
    default_atol : float, default 1e-6
        Default absolute tolerance for the ``approx`` operator.
    default_rtol : float, default 1e-6
        Default relative tolerance for the ``approx`` operator.
    **filters :
        Column → condition mapping. Each value can be:
          * **Scalar**: equality (with wildcard support for strings, e.g. ``"CT*"``).
          * **List**: OR across entries (each entry may itself be a scalar, operator dict,
            or callable).
          * **Operator dict**: any combination of the built-in (or registered) operators
            below, e.g. ``{"gte": 3, "lt": 10}``.
          * **Callable**: a predicate ``fn(cell) -> bool`` applied per-cell.

        Supported built-in operators
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ``{"eq": v}``              equality (also the default for a scalar)
        ``{"neq": v}``             inequality
        ``{"in": [v1, v2]}``       membership
        ``{"nin": [v1, v2]}``      non-membership
        ``{"gte": v}``             >= comparison
        ``{"lte": v}``             <= comparison
        ``{"gt": v}``              >  comparison
        ``{"lt": v}``              <  comparison
        ``{"RegEx": pattern}``     regex contains (use anchors for full match)
        ``{"NotRegEx": pattern}``  negated regex contains
        ``{"contains": x}``        container membership or substring (strings only)
        ``{"exists": True}``       non-missing values
        ``{"missing": True}``      missing values
        ``{"approx": {"value": v, "atol": a, "rtol": r}}``  numeric closeness
        ``{"time_between": ["HH:MM[:SS]", "HH:MM[:SS]"]}``  time-of-day window
        ``{"in_last_days": n}``    keep rows with date/datetime within last *n* days.
                                   Also accepts ``{"days": n, "now": <datetime|date>}``.

        Notes on temporal values
        ^^^^^^^^^^^^^^^^^^^^^^^^
        If a column is date/time/datetime typed (or object column holding such values),
        scalar/list/operands are coerced to the appropriate temporal type where possible.
        For timezone-aware datetimes, ``in_last_days`` compares against a timezone-aware
        ``now`` if provided; otherwise uses local ``datetime.now()`` and will align dtypes.

        Dot-path extraction
        ^^^^^^^^^^^^^^^^^^^
        When a column name includes dots and optional list indices (e.g.
        ``"meta.series[0].uid"``), the function traverses nested structures inside
        each cell (dict-like / object attributes / list indexing) to extract values.

    Returns
    -------
    pandas.DataFrame
        The subset of ``df`` that satisfies all provided filters (AND across columns).

    Raises
    ------
    KeyError
        If a referenced column (or root column of a dot-path) is not present.
    ValueError
        For malformed operator specifications or unsupported operators.

    See Also
    --------
    register_op : Register custom operators usable in ``query_df``.
    pandas.DataFrame.query : Pandas' expression-based row filtering.

    Examples
    --------
    Basic equality and wildcard
    >>> query_df(df, Modality="CT")
    >>> query_df(df, PatientID="12*")

    Ranges and lists
    >>> query_df(df, Age={"gte": 18, "lt": 65})
    >>> query_df(df, Modality=["CT", "MR"])

    Regex / inverse regex (case-insensitive)
    >>> query_df(df, case_insensitive=True, PatientName={"RegEx": r"^smith\\b"})

    Time-of-day window (wrap-around supported)
    >>> query_df(df, SeriesTime={"time_between": ["22:00", "06:00"]})

    Last 7 days (date/datetime columns)
    >>> query_df(df, AcqDateTime={"in_last_days": 7})

    Nested traversal
    >>> query_df(df, **{"meta.series[0].uid": {"eq": "1.2.3"}})
    """

    ctx = {
        "case_insensitive": bool(case_insensitive),
        "default_atol": float(default_atol),
        "default_rtol": float(default_rtol),
    }

    def apply_one(series: pd.Series, cond: Any) -> pd.Series:
        # List => OR across entries (each entry can be scalar/op-dict/callable)
        if isinstance(cond, list):
            if not cond:
                return pd.Series(False, index=series.index)
            m = pd.Series(False, index=series.index)
            for c in cond:
                m |= apply_one(series, c)
            return m

        # Callable predicate
        if callable(cond):
            return _op_callable(series, cond, ctx)

        # Dict => operator bag
        if isinstance(cond, Mapping):
            m = pd.Series(True, index=series.index)
            for op, spec in cond.items():
                fn = _OPS.get(op)
                if fn is None:
                    raise ValueError(f"Unsupported operator: {op}")
                m &= fn(series, spec, ctx)
            return m

        # Scalar => equality w/ wildcard support & container membership fallback
        if isinstance(cond, str) and ("*" in cond or "?" in cond):
            return _op_eq(series, cond, ctx)
        # If series has container cells, try membership
        if series.apply(_is_iterable_nonstring).any():
            return _iterable_contains(series, cond)
        return series == cond

    # Build global mask
    mask = pd.Series(True, index=df.index)
    for col, condition in filters.items():
        s = _extract_col(df, col)  # supports dot-paths for nested cell content
        condition = _maybe_coerce_temporal(s, condition)
        mask &= apply_one(s, condition)

    return df.loc[mask]
