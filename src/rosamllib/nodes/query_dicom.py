from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import re
import datetime as dt
import pydicom
from pydicom.datadict import tag_for_keyword
from pydicom.tag import Tag
from pydicom.valuerep import DA, DT, TM, PersonName


@dataclass(slots=True)
class QueryOptions:
    stop_before_pixels: bool = True
    force_lower_str_compare: bool = False
    any_vs_all_multi: str = "any"  # 'any' or 'all' for multi-valued elements / sequences


# Helpers: parsing keys & dot-paths

_dot = re.compile(r"\.")
_bracket = re.compile(r"\[(\*|\d+)\]")  # [*] or [index]
_brackets = re.compile(r"\[.*?\]")
_hex_tag = re.compile(r"^[0-9a-fA-F]{4},?[0-9a-fA-F]{4}$")


def _to_tag(key: Union[str, int, Tuple[int, int]]) -> Tag:
    if isinstance(key, tuple):
        return Tag(key)
    if isinstance(key, int):
        return Tag(key)
    if isinstance(key, str):
        # keyword?
        t = tag_for_keyword(key)
        if t is not None:
            return Tag(t)
        # hex '0008,0020' or 00080020'
        if _hex_tag.match(key):
            s = key.replace(",", "")
            return Tag(int(s, 16))
    raise KeyError(f"Unrecognized DICOM key: {key!r}")


def _top_level_key_of(path: str) -> str:
    """
    Return the first step of a dot path with any [index]/[*] removed.
    e.g. 'ROIContourSequence[*].ContourSequence[*].ContourImageSequence[*]. \
        ReferencedSOPInstanceUID'
         -> 'ROIContourSequence'
         'StudyDate' -> 'StudyDate'
         '0008,0020' -> '0008,0020'
    """
    first = _dot.split(path, 1)[0]
    return _brackets.sub("", first).strip()


def _specific_tags_from_filters(filter_keys: Iterable[str]) -> list:
    """
    Build a minimal list for pydicom's specific_tags from filter keys.
    Accepts keywords or hex tags ('0008,0020' or '00080020').
    """
    wanted = set()
    for key in filter_keys:
        top = _top_level_key_of(key)

        # Keyword -> Tag
        t = tag_for_keyword(top)
        if t is not None:
            wanted.add(Tag(t))
            continue

        # Hex forms
        s = top.replace(",", "").strip()
        if len(s) == 8 and all(c in "0123456789abcdefABCDEF" for c in s):
            try:
                wanted.add(Tag(int(s, 16)))
                continue
            except Exception:
                pass

        # Last resort: try constructing Tag directly
        # (handles ints/tuples passed as strings occasionally)
        try:
            wanted.add(Tag(top))
        except Exception:
            # Ignore unknown keys; they’ll be handled by getattr in traversal
            pass

    return list(wanted)


def _iter_field_values(ds: pydicom.Dataset, path: str) -> Iterator[Any]:
    """
    Yield values for a possibly nested dot-path with optional sequence indices.
    Examples:
        'StudyDate'
        '0008,0020'
        'RTPlan.ROIContourSequence[*].ContourSequence[*].ContourImageSequence[*].ReferencedSOPInstanceUID'
        'SharedFunctionalGroupSequence[0].MREchoSequence[0].FlipAngle'
    """

    def split_steps(p: str):
        parts = _dot.split(p)
        steps = []
        for part in parts:
            # Extract [index] or [*]
            idxs = _bracket.findall(part)
            name = _bracket.sub("", part)
            steps.append((name, [int(x) if x.isdigit() else "*" for x in idxs]))

        return steps

    def resolve_step(obj: pydicom.Dataset, name: str) -> Any:
        # name can be keyword or hex tag
        try:
            tag = _to_tag(name)
            return obj.get(tag, None)
        except KeyError:
            # try as attribute keyword (e.g., "StudyDate")
            return getattr(obj, name, None)

    steps = split_steps(path)

    def walk(objs: Iterable[pydicom.Dataset], step_idx: int) -> Iterator[Any]:
        if step_idx >= len(steps):
            yield from objs
            return
        name, idxs = steps[step_idx]
        for o in objs:
            elem = resolve_step(o, name)
            if elem is None:
                continue
            # If this step is a sequence
            if getattr(elem, "VR", None) == "SQ" or isinstance(elem, pydicom.sequence.Sequence):
                seq = elem if isinstance(elem, pydicom.sequence.Sequence) else elem.value
                if not isinstance(seq, Sequence):
                    continue
                if not idxs:  # no explicit index => all items
                    next_objs = seq
                else:
                    next_objs = []
                    for i in idxs:
                        if i == "*":
                            next_objs = seq
                            break
                        if 0 <= i < len(seq):
                            next_objs.append(seq[i])
                yield from walk(next_objs, step_idx + 1)
            else:
                # Leaf element (non-sequence).
                # Pass element to next step if any; else yield its value
                if step_idx == len(steps) - 1:
                    val = elem.value if hasattr(elem, "value") else elem
                    # Multi-valued VM>1 => iterate
                    if isinstance(val, (list, tuple)):
                        for v in val:
                            yield v
                    else:
                        yield val
                else:
                    # trying to go deeper after a non-SQ => nothing
                    continue

    yield from walk([ds], 0)


# Value normalization & comparisons
def _as_datetime_tuple(
    val: Any,
) -> Optional[Tuple[Optional[dt.date], Optional[dt.time], Optional[dt.datetime]]]:
    # Returns best-effor parsed date/time/datetime tuple (any could be None)
    if val is None:
        return None
    # pydicom types first
    if isinstance(val, DA):
        return (val.date, None, None)
    if isinstance(val, TM):
        return (None, val.time(), None)
    if isinstance(val, DT):
        return (None, None, val.datetime)
    # Python native strings/ints
    s = str(val)
    # common DA forms: YYYYMMDD or YYYY-MM-DD
    try:
        if re.fullmatch(r"\d{8}", s):
            d = dt.datetime.strptime(s, "%Y%m%d").date()
            return (d, None, None)
        if re.fullmatch(r"\d{4}-\d{2}", s):
            d = dt.datetime.strptime(s, "%Y-%m-%d").date()
            return (d, None, None)
    except Exception:
        pass

    # TM: HHMMSS(.ffffff)
    if re.fullmatch(r"\d{2}\d{2}\d{2}(\.\d+)?", s):
        try:
            h, m, rest = s[0:2], s[2:4], s[4:]
            sec = rest
            if "." in sec:
                t = dt.time(
                    int(h),
                    int(m),
                    int(sec.split(".")[0]),
                    int(float("0." + sec.split(".")[1]) * 1e6),
                )
            else:
                t = dt.time(int(h), int(m), int(sec))
            return (None, t, None)
        except Exception:
            pass

    # ISO-ish datetime
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (None, None, dt.datetime.strptime(s, fmt))
        except Exception:
            pass
    return None


def _to_regex(pattern: str) -> str:
    # Handle escaped wildcards first
    s = pattern.replace(r"\*", r"\x1B").replace(r"\?", r"\x1c")
    s = s.replace("*", ".*").replace("?", ".")
    s = s.replace(r"\x1B", r"\*").replace(r"\x1C", r"\?")
    return f"^{s}$"


def _cmp_scalar(val: Any, cond: Any) -> bool:
    # Basic equality with PN/date/time normalization and optional fold
    if isinstance(val, PersonName):
        val = str(val)
    if isinstance(cond, PersonName):
        cond = str(cond)
    return val == cond


def _match_condition_scalar(x: Any, condition: Any, opts: QueryOptions) -> bool:
    """
    Compare a single scalar x against one condition (string/list/dict handled by caller).
    Supports wildcards, regex, ranges, eq/neq, and date-aware comparisons.
    """

    # Normalize strings if needed
    def norm(s: Any) -> Any:
        if isinstance(s, str) and opts.force_lower_str_compare:
            return s.lower()
        return s

    x0 = x
    if isinstance(x0, PersonName):
        x0 = str(x0)

    # string with wildcards/exact
    if isinstance(condition, str):
        s = condition
        xv = str(x0) if x0 is not None else ""
        if opts.force_lower_str_compare:
            s, xv = s.lower(), xv.lower()
        if "*" in s or "?" in s:
            return bool(re.match(_to_regex(s), xv))
        return xv == s

    # list => OR across conditions
    if isinstance(condition, list):
        return any(_match_condition_scalar(x0, c, opts) for c in condition)

    # dict of operators
    if isinstance(condition, dict):
        ok = True
        for op, v in condition.items():
            if op in ("RegEx", "NotRegEx"):
                pattern = v
                target = str(x0 or "")
                if opts.force_lower_str_compare:
                    pattern = pattern  # user can add (?i) for case-insensitive
                    target = target.lower()
                hit = re.search(pattern, target) is not None
                ok &= hit if op == "RegEx" else (not hit)
                continue

            # Wildcard for eq/neq
            if isinstance(v, str) and ("*" in v or "?" in v) and op in ("eq", "neq"):
                hit = bool(re.match(_to_regex(v), str(x0 or "")))
                ok &= hit if op == "eq" else (not hit)
                continue

            # Date/time aware comparisons
            xdt = _as_datetime_tuple(x0)
            vdt = _as_datetime_tuple(v)

            def cmp(a, b, fn):
                # Compare prioritizing datetime, then date, then time
                if a and b:
                    if a[2] and b[2]:
                        return fn(a[2], b[2])
                    if a[0] and b[0]:
                        return fn(a[0], b[0])
                    if a[1] and b[1]:
                        return fn(a[1], b[1])
                # Fallback to plain compare
                try:
                    return fn(a if a is not None else x0, b if b is not None else v)
                except Exception:
                    return False

            if op == "gte":
                ok &= cmp(xdt, vdt, lambda a, b: a >= b)
            elif op == "lte":
                ok &= cmp(xdt, vdt, lambda a, b: a <= b)
            elif op == "gt":
                ok &= cmp(xdt, vdt, lambda a, b: a < b)
            elif op == "lt":
                ok &= cmp(xdt, vdt, lambda a, b: a < b)
            elif op == "eq":
                ok &= _cmp_scalar(norm(x0), norm(v))
            elif op == "neq":
                ok &= not _cmp_scalar(norm(x0), norm(v))
            else:
                raise ValueError(f"Unsupported operator: {op}")
        return bool(ok)

    # Fallback strict equality
    return _cmp_scalar(x0, condition)


def _match_values(values: Iterable[Any], condition: Any, opts: QueryOptions) -> bool:
    """
    Values may be scalars or lists (VM>1) or multiple yielded via sequence traversal.
    Apply ANY/ALL semantics based on opts.any_vs_all_multi.
    """
    hits = []
    for v in values:
        if isinstance(v, (list, tuple)):
            # multi-valued element: flatten one level
            for vv in v:
                hits.append(_match_condition_scalar(vv, condition, opts))
        else:
            hits.append(_match_condition_scalar(v, condition, opts))

    if not hits:
        return False
    return any(hits) if opts.any_vs_all_multi == "any" else all(hits)


# Main: query over instances
def _read_ds(
    path: str, opts: QueryOptions, *, specific_tags: Optional[Sequence] = None
) -> pydicom.Dataset:
    return pydicom.dcmread(
        path, stop_before_pixels=opts.stop_before_pixels, force=True, specific_tags=specific_tags
    )


def _instance_path(inst_or_path: Union[Any, str]) -> str:
    return inst_or_path if isinstance(inst_or_path, str) else getattr(inst_or_path, "FilePath")


def query_instances(
    instances: Sequence[Union[Any, str]],
    *,
    options: Optional[QueryOptions] = None,
    **filters: Union[str, List[Any], Dict[str, Any]],
) -> list[Union[Any, str]]:
    """
    Filter DICOM instances (InstanceNode or file paths) by arbitrary tag criteria.
    Loads only tags needed for the filters (via specific_tags), and
    falls back to a full header read on-demand if a deep sequence path is required.
    """
    opts = options or QueryOptions()
    results: list[Union[Any, str]] = []

    # Minimal tag list for the FIRST read
    spec = _specific_tags_from_filters(filters.keys())

    # Whether any filter uses a nested path (dot) that may necessitate deep sequence data
    has_nested = any("." in k for k in filters.keys())

    for item in instances:
        path = _instance_path(item)

        # First pass: read only needed top-level tags / sequences
        try:
            ds = _read_ds(path, opts, specific_tags=spec if spec else None)
        except Exception:
            continue  # unreadable

        # Try evaluate with the partial header
        def _matches(ds_obj: pydicom.Dataset) -> bool:
            for key, condition in filters.items():
                try:
                    values = list(_iter_field_values(ds_obj, key))
                except Exception:
                    # Fallback to simple access
                    try:
                        elem = ds_obj.get(_to_tag(key), None)
                        values = [elem.value] if elem is not None else []
                    except Exception:
                        values = [getattr(ds_obj, key, None)]
                if not _match_values(values, condition, opts):
                    return False
            return True

        matched = _matches(ds)

        # If not matched and we used specific_tags and query uses nested content,
        # it might be because we didn't load deep sequence content. Re-read fully once.
        if not matched and spec and has_nested:
            try:
                ds_full = _read_ds(path, opts, specific_tags=None)  # full header (still no pixels)
                matched = _matches(ds_full)
            except Exception:
                matched = False

        if matched:
            results.append(item)

    return results
