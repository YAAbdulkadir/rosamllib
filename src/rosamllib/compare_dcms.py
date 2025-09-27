from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Union, Optional, DefaultDict
from collections import defaultdict
import argparse
import json
import html
from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from pydicom.dataelem import DataElement
from pydicom.tag import Tag


# --------------------------- Defaults ---------------------------

BULK_KEYWORDS_DEFAULT = {
    "PixelData",
    "FloatPixelData",
    "DoubleFloatPixelData",
    "WaveformData",
    "OverlayData",
    "CurveData",
    "AudioSampleData",
    "EncapsulatedDocument",
    "SpectroscopyData",
    "LargePaletteColorLookupTableData",
}
BULK_VRS_DEFAULT = {"OB", "OD", "OF", "OL", "OW", "UN"}  # UN often holds raw bytes


# --------------------------- Options & Report ---------------------------


@dataclass
class Tolerance:
    abs_tol: float = 0.0  # absolute numeric tolerance


@dataclass
class DiffOptions:
    ignore_private: bool = True
    ignore_bulk: bool = True
    bulk_keywords: set = None
    bulk_vrs: set = None
    ignore_tokens: List[str] = None
    numeric_tol: Tolerance = field(default_factory=Tolerance)
    case_insensitive_strings: bool = False

    # Sequence key matching
    sequence_keys: Dict[str, List[str]] = None
    sequence_fallback: str = "order"

    # NEW: collect “ok” (match) rows, with a safety cap
    collect_all_matches: bool = True
    max_ok_rows: int = 50000  # raise/lower as you like
    # Optional: only collect matches for these paths/keywords/tags (empty = all)
    show_matches_for: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.bulk_keywords is None:
            self.bulk_keywords = set(BULK_KEYWORDS_DEFAULT)
        if self.bulk_vrs is None:
            self.bulk_vrs = set(BULK_VRS_DEFAULT)
        if self.ignore_tokens is None:
            self.ignore_tokens = []
        if self.sequence_keys is None:
            self.sequence_keys = {}


@dataclass
class Diff:
    path: str
    left: Any
    right: Any
    note: str = "mismatch"
    severity: str = "diff"  # "diff" | "warn" | "info"


class DiffReport:
    def __init__(self):
        self.diffs: List[Diff] = []
        self.left_meta: Dict[str, Any] = {}
        self.right_meta: Dict[str, Any] = {}

    def add(self, path, left, right, note="mismatch", severity="diff"):
        self.diffs.append(Diff(path, left, right, note, severity))

    def to_dict(self):
        by_sev = {
            "diff": sum(d.severity == "diff" for d in self.diffs),
            "warn": sum(d.severity == "warn" for d in self.diffs),
            "info": sum(d.severity == "info" for d in self.diffs),
            "ok": sum(d.severity == "ok" for d in self.diffs),  # NEW
        }
        return {
            "left": self.left_meta,
            "right": self.right_meta,
            "diffs": [asdict(d) for d in self.diffs],
            "summary": {"total": len(self.diffs), "by_severity": by_sev},
        }

    def to_text(self) -> str:
        lines = [f"Diffs: {len(self.diffs)}"]
        for d in self.diffs:
            lines.append(f"- {d.path}: {repr(d.left)} != {repr(d.right)} [{d.note}]")
        return "\n".join(lines)

    def write_html(self, output_path: str, title: str = "DICOM Comparison Report") -> str:
        page = self.to_html(title=title)
        Path(output_path).write_text(page, encoding="utf-8")
        return output_path

    def to_html(self, title: str = "DICOM Comparison Report") -> str:
        tree = _build_tree(self.diffs)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        left_meta = self.left_meta or {}
        right_meta = self.right_meta or {}
        summary = self.to_dict()["summary"]

        css = """
    :root{
    --bg:#0b0f14; --panel:#121822; --muted:#a7b0be; --text:#e6edf3;
    --border:#1f2630; --red:#ef4444; --amber:#f59e0b; --blue:#3b82f6; --green:#10b981;
    }
    *{box-sizing:border-box}
    body{margin:0;padding:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
    .header{padding:18px 20px;border-bottom:1px solid var(--border);background:linear-gradient(180deg,#111827,transparent)}
    .title{font-size:20px;margin:0 0 6px 0}
    .meta{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted)}
    .badges{display:flex;gap:8px;margin-top:10px}
    .badge{padding:4px 8px;border-radius:999px;background:#1c2533;border:1px solid var(--border);color:var(--muted)}
    .badge b{color:var(--text)}
    .container{padding:16px 20px}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;margin:12px 0;overflow:hidden}
    .card-header{padding:10px 12px;display:flex;align-items:center;gap:10px;cursor:pointer;border-bottom:1px solid var(--border)}
    .card-header:hover{background:#0f1520}
    .chev{transition:transform .15s ease}
    details[open] > summary .chev{transform:rotate(90deg)}
    .group-title{font-weight:600}
    .rows{width:100%;border-collapse:collapse}
    .rows th, .rows td{padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top}
    .rows th{position:sticky;top:0;background:var(--panel);z-index:1;text-align:left;color:var(--muted);font-weight:600}
    .sev{font-weight:600}
    .sev.diff{color:var(--red)}
    .sev.warn{color:var(--amber)}
    .sev.info{color:var(--blue)}
    .sev.ok{color:var(--green)}
    .note{color:var(--muted)}
    .path, .kv{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;word-break:break-all}
    .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 16px 0}
    input[type="search"]{flex:1;min-width:220px;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:#0e1420;color:var(--text)}
    .toggle{display:flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid var(--border);border-radius:999px;background:#0e1420;cursor:pointer}
    .btn{padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:#0e1420;color:var(--text);cursor:pointer}
    .btn:hover{background:#0f1520}
    .meta-table{width:auto;border-collapse:collapse;margin-top:10px}
    .meta-table td{padding:4px 8px;border-bottom:1px solid var(--border);color:var(--muted)}
    .footer{color:var(--muted);text-align:center;padding:16px 0;border-top:1px solid var(--border)}
    """

        js = """
    function filterAll(){
    const q = document.getElementById('filter').value.toLowerCase();
    const showDiff = document.getElementById('f-diff').checked;
    const showWarn = document.getElementById('f-warn').checked;
    const showInfo = document.getElementById('f-info').checked;
    const showOk   = document.getElementById('f-ok').checked;

    const rows = document.querySelectorAll('tbody tr[data-all]');
    rows.forEach(tr=>{
        const sev = tr.getAttribute('data-sev');
        const txt = tr.getAttribute('data-all') || '';
        const sevOk = (sev==='diff'&&showDiff)||(sev==='warn'&&showWarn)||(sev==='info'&&showInfo)||(sev==='ok'&&showOk);
        const qOk = !q || txt.toLowerCase().indexOf(q) !== -1;
        tr.style.display = (sevOk && qOk) ? '' : 'none';
    });
    }

    function presetAll(){ document.getElementById('f-diff').checked=true; document.getElementById('f-warn').checked=true; document.getElementById('f-info').checked=true; document.getElementById('f-ok').checked=true; filterAll(); }
    function presetDiffs(){ document.getElementById('f-diff').checked=true; document.getElementById('f-warn').checked=true; document.getElementById('f-info').checked=true; document.getElementById('f-ok').checked=false; filterAll(); }
    function presetNonMatches(){ document.getElementById('f-diff').checked=true; document.getElementById('f-warn').checked=true; document.getElementById('f-info').checked=false; document.getElementById('f-ok').checked=false; filterAll(); }

    document.addEventListener('DOMContentLoaded', ()=>{
    ['filter','f-diff','f-warn','f-info','f-ok'].forEach(id=>{
        document.getElementById(id).addEventListener('input', filterAll);
        document.getElementById(id).addEventListener('change', filterAll);
    });
    filterAll();
    });
    """

        def meta_table(meta: Dict[str, Any]) -> str:
            if not meta:
                return ""
            rows = "".join(
                f"<tr><td>{_escape(k)}</td><td class='kv'>{_escape(v)}</td></tr>"
                for k, v in meta.items()
            )
            return f"<table class='meta-table'>{rows}</table>"

        def render_rows(rows: List[Diff]) -> str:
            if not rows:
                return ""
            trs = []
            for d in rows:
                data_all = _escape(f"{d.path} {d.note} {d.severity} {d.left} {d.right}")
                trs.append(
                    f"""
    <tr data-all="{data_all}" data-sev="{_escape(d.severity)}">
    <td class="sev {d.severity}">{_escape(d.severity)}</td>
    <td class="path">{_escape(d.path)}</td>
    <td class="kv">{_escape(d.left)}</td>
    <td class="kv">{_escape(d.right)}</td>
    <td class="note">{_escape(d.note)}</td>
    </tr>"""
                )
            return f"""
    <table class="rows">
    <thead><tr><th>Severity</th><th>Path</th><th>Left</th><th>Right</th><th>Note</th></tr></thead>
    <tbody>{''.join(trs)}</tbody>
    </table>
    """

        def render_node(node: _TreeNode, level: int = 0) -> str:
            # Skip artificial root’s header; render its children only
            inner = []
            # Rows at this node (for exact-level fields)
            if node.rows:
                inner.append(render_rows(node.rows))
            # Children (subpaths) as nested <details>

            for name, child in node.children.items():
                stats = _agg_counts(child)  # subtree totals
                nested = render_node(child, level + 1)
                inner.append(
                    f"""
            <details class="card">
            <summary class="card-header">
                <span class="chev">▶</span>
                <span class="group-title">{_escape(name)}</span>
                <span class="count" style="color:var(--muted)">
                (rows: {stats['rows']} — 
                <span class="sev diff">{stats['diff']}</span> /
                <span class="sev warn">{stats['warn']}</span> /
                <span class="sev info">{stats['info']}</span> /
                <span class="sev ok">{stats['ok']}</span>)
                </span>
            </summary>
            {nested}
            </details>"""
                )
            return "".join(inner)

        # Build page
        groups_html = render_node(tree)

        return f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>{_escape(title)}</title>
    <style>{css}</style>
    </head>
    <body>
    <div class="header">
        <div class="title">{_escape(title)}</div>
        <div class="meta">
        <div><b>Generated:</b> {_escape(stamp)}</div>
        <div><b>Total rows:</b> {summary['total'] + (0)}</div>
        </div>
        <div class="badges">
        <div class="badge">diff: <b>{summary['by_severity'].get('diff',0)}</b></div>
        <div class="badge">warn: <b>{summary['by_severity'].get('warn',0)}</b></div>
        <div class="badge">info: <b>{summary['by_severity'].get('info',0)}</b></div>
        <div class="badge">ok:   <b>{summary['by_severity'].get('ok',0)}</b></div>
        </div>
        <div class="meta" style="margin-top:10px;">
        <div><div><b>Left</b></div>{meta_table(left_meta)}</div>
        <div><div><b>Right</b></div>{meta_table(right_meta)}</div>
        </div>
    </div>

    <div class="container">
        <div class="toolbar">
        <input id="filter" type="search" placeholder="Filter (path, values, note)…"/>
        <label class="toggle"><input id="f-diff" type="checkbox" checked/> diff</label>
        <label class="toggle"><input id="f-warn" type="checkbox" checked/> warn</label>
        <label class="toggle"><input id="f-info" type="checkbox" checked/> info</label>
        <label class="toggle"><input id="f-ok"   type="checkbox" checked/> ok</label>
        <button class="btn" onclick="presetAll()">All</button>
        <button class="btn" onclick="presetDiffs()">Diffs only</button>
        <button class="btn" onclick="presetNonMatches()">Non-matches</button>
        </div>

        {groups_html if groups_html else "<div class='card'><div class='card-header'><b>No rows found.</b></div></div>"}
    </div>

    <div class="footer">compare_dcms — HTML report</div>
    <script>{js}</script>
    </body>
    </html>"""


# --------------------------- Helpers ---------------------------


def _agg_counts(node: "_TreeNode") -> Dict[str, int]:
    out = {"rows": 0, "diff": 0, "warn": 0, "info": 0, "ok": 0}
    # rows at this node
    out["rows"] += len(node.rows)
    for r in node.rows:
        if r.severity in out:
            out[r.severity] += 1
    # recurse
    for child in node.children.values():
        sub = _agg_counts(child)
        for k in out:
            out[k] += sub[k]
    return out


def _should_emit_ok(path: str, elem: Optional[DataElement], opts: DiffOptions) -> bool:
    if not opts.collect_all_matches:
        return False
    if opts.show_matches_for:
        keys = {path}
        if elem:
            if elem.keyword:
                keys.add(elem.keyword)
            keys.add(tag_hex(elem.tag))
        return any(tok in keys for tok in opts.show_matches_for)
    return True  # no allowlist => collect all


def _group_prefix(path: str) -> str:
    """
    The first segment before the first '.' is used as the group.
    Examples:
      "BeamSequence.item[key=('1',)].ControlPointSequence[0].GantryAngle" -> "BeamSequence"
      "PatientName" -> "PatientName"
      "(0010,0010)" -> "(0010,0010)"
    """
    return path.split(".", 1)[0]


def _group_diffs_by_prefix(diffs: List["Diff"]) -> Dict[str, List["Diff"]]:
    groups: Dict[str, List["Diff"]] = {}
    for d in diffs:
        g = _group_prefix(d.path)
        groups.setdefault(g, []).append(d)
    # sort each group by path for stable rendering
    for g in groups:
        groups[g].sort(key=lambda x: x.path)
    return dict(sorted(groups.items(), key=lambda kv: kv[0].lower()))


def _escape(s: Any) -> str:
    return html.escape(str(s), quote=True)


def tag_hex(tag: Union[Tag, int, tuple]) -> str:
    t = Tag(tag)
    return f"({int(t.group):04x},{int(t.element):04x})".lower()


def elem_display_value(e: DataElement) -> str:
    if e is None:
        return "None"
    if e.VR == "SQ":
        return f"Sequence[{len(e.value)}]"
    v = e.value
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, tuple)) and len(v) > 16:
        return f"[len={len(v)}]"
    return str(v)


def path_for(elem: DataElement, fallback_tag: Tag, prefix: str) -> str:
    # Prefer keyword if available, else tag hex
    name = elem.keyword if elem and elem.keyword else tag_hex(fallback_tag)
    return prefix + name


def is_ignored(elem: DataElement, path: str, opts: DiffOptions) -> bool:
    if elem is None:
        return False
    if opts.ignore_private and elem.tag.is_private:
        return True
    keyset = {elem.keyword or "", tag_hex(elem.tag), path}
    return any(tok in keyset for tok in opts.ignore_tokens)


def is_bulk(elem: DataElement, opts: DiffOptions) -> bool:
    if elem is None:
        return False
    if elem.VR in opts.bulk_vrs:
        return True
    if (elem.keyword or "") in opts.bulk_keywords:
        return True
    return False


def try_float(x) -> Tuple[bool, float]:
    try:
        return True, float(x)
    except Exception:
        return False, 0.0


def values_equal(a: Any, b: Any, tol: float, casefold: bool) -> bool:
    ok_a, fa = try_float(a)
    ok_b, fb = try_float(b)
    if ok_a and ok_b:
        return abs(fa - fb) <= tol
    if isinstance(a, str) and isinstance(b, str):
        return (a.lower() == b.lower()) if casefold else (a == b)
    return a == b


# -------- Sequence key selection & mapping --------


def key_spec_for_sequence(
    path: str, elem: Optional[DataElement], opts: DiffOptions
) -> Optional[List[str]]:
    """
    Resolve which keys to use for matching items in this sequence.
    Precedence: exact path -> sequence keyword -> tag hex.
    `path` should be the sequence path without a trailing dot.
    """
    candidates = [path]
    if elem and elem.keyword:
        candidates.append(elem.keyword)
    if elem:
        candidates.append(tag_hex(elem.tag))
    for c in candidates:
        if c in opts.sequence_keys:
            return opts.sequence_keys[c]
    return None


def item_key_tuple(item: Dataset, key_fields: List[str]) -> Tuple:
    """Build a tuple key from top-level fields in a sequence item."""
    return tuple(getattr(item, k, None) for k in key_fields)


def map_items_by_key(
    seq: List[Dataset], key_fields: List[str]
) -> Tuple[DefaultDict[Tuple, List[int]], bool]:
    """Map key tuple -> list of indices; return (map, all_items_have_all_keys?)."""
    m: DefaultDict[Tuple, List[int]] = defaultdict(list)
    all_complete = True
    for i, it in enumerate(seq):
        kt = item_key_tuple(it, key_fields)
        if any(v is None for v in kt):
            all_complete = False
        m[kt].append(i)
    return m, all_complete


# --------------------------- Core Comparison ---------------------------


def compare_files(
    left_path: str, right_path: str, opts: DiffOptions = DiffOptions()
) -> DiffReport:
    ds1 = pydicom.dcmread(left_path, stop_before_pixels=False, force=True)
    ds2 = pydicom.dcmread(right_path, stop_before_pixels=False, force=True)
    rep = compare_datasets(ds1, ds2, opts)
    rep.left_meta = basic_meta(ds1, left_path)
    rep.right_meta = basic_meta(ds2, right_path)
    return rep


def basic_meta(ds: Dataset, path: str) -> Dict[str, Any]:
    return {
        "path": path,
        "SOPClassUID": getattr(ds, "SOPClassUID", None),
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
        "Modality": getattr(ds, "Modality", None),
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
    }


def compare_datasets(ds1: Dataset, ds2: Dataset, opts: DiffOptions) -> DiffReport:
    rep = DiffReport()
    _compare_level(ds1, ds2, opts, rep, prefix="")
    return rep


def _compare_level(ds1: Dataset, ds2: Dataset, opts: DiffOptions, rep: DiffReport, prefix: str):
    # All tags present in either dataset at this level
    tags = {e.tag for e in ds1} | {e.tag for e in ds2}
    for t in sorted(tags):
        e1 = ds1.get(t)
        e2 = ds2.get(t)
        path = path_for(e1 if e1 else e2, t, prefix)

        # Ignore?
        if (e1 and is_ignored(e1, path, opts)) or (e2 and is_ignored(e2, path, opts)):
            continue

        # Presence check
        if e1 is None or e2 is None:
            rep.add(
                path,
                elem_display_value(e1),
                elem_display_value(e2),
                note="element presence differs",
            )
            continue

        # Bulk?
        if opts.ignore_bulk and (is_bulk(e1, opts) or is_bulk(e2, opts)):
            continue

        # VR mismatch?
        if e1.VR != e2.VR:
            rep.add(path + ".VR", e1.VR, e2.VR, note="VR differs")
            if e1.VR != "SQ" and e2.VR != "SQ":
                continue

        # Sequence vs non-sequence
        if e1.VR == "SQ" or e2.VR == "SQ":
            if (e1.VR != "SQ") or (e2.VR != "SQ"):
                rep.add(path, e1.VR, e2.VR, note="one is sequence, other is not")
                continue
            seq_path_for_lookup = path  # no trailing dot
            _compare_sequence(
                e1.value,
                e2.value,
                opts,
                rep,
                prefix=path + ".",
                elem=e1,
                seq_lookup_path=seq_path_for_lookup,
            )
            continue

        # Value(s)
        _compare_values(e1, e2, opts, rep, path)


def _compare_sequence(
    seq1,
    seq2,
    opts: DiffOptions,
    rep: DiffReport,
    prefix: str,
    elem: Optional[DataElement] = None,
    seq_lookup_path: Optional[str] = None,
):

    seq_lookup_path = seq_lookup_path or (prefix[:-1] if prefix.endswith(".") else prefix)
    key_fields = key_spec_for_sequence(seq_lookup_path, elem, opts)

    if not key_fields:
        if len(seq1) != len(seq2):
            rep.add(prefix + "length", len(seq1), len(seq2), note="sequence length differs")
        else:
            # NEW: ok row when lengths match (helps the “show everything” view)
            if (
                _should_emit_ok(prefix + "length", elem, opts)
                and sum(d.severity == "ok" for d in rep.diffs) < opts.max_ok_rows
            ):
                rep.add(prefix + "length", len(seq1), len(seq2), note="match", severity="ok")
        n = min(len(seq1), len(seq2))
        for i in range(n):
            _compare_level(seq1[i], seq2[i], opts, rep, prefix=f"{prefix}[{i}].")
        return

    # Key-based matching (unchanged) ...
    map1, all1 = map_items_by_key(seq1, key_fields)
    map2, all2 = map_items_by_key(seq2, key_fields)

    if not (all1 and all2) and opts.sequence_fallback == "order":
        if len(seq1) != len(seq2):
            rep.add(
                prefix + "length",
                len(seq1),
                len(seq2),
                note="sequence length differs (fallback=order)",
            )
        else:
            if (
                _should_emit_ok(prefix + "length", elem, opts)
                and sum(d.severity == "ok" for d in rep.diffs) < opts.max_ok_rows
            ):
                rep.add(prefix + "length", len(seq1), len(seq2), note="match", severity="ok")
        n = min(len(seq1), len(seq2))
        for i in range(n):
            _compare_level(seq1[i], seq2[i], opts, rep, prefix=f"{prefix}[{i}].")
        rep.add(
            prefix + "keymatch",
            f"keys={key_fields}",
            "incomplete",
            note="fallback=order",
            severity="info",
        )
        return

    if not (all1 and all2):
        rep.add(
            prefix + "keymatch",
            f"keys={key_fields}",
            "incomplete",
            note="incomplete keys present",
            severity="warn",
        )

    keys1 = set(map1.keys())
    keys2 = set(map2.keys())
    for k in sorted(keys1 - keys2):
        rep.add(
            f"{prefix}item[key={k}]", "present", "absent", note="unmatched item (right missing)"
        )
    for k in sorted(keys2 - keys1):
        rep.add(
            f"{prefix}item[key={k}]", "absent", "present", note="unmatched item (left missing)"
        )

    for k in sorted(keys1 & keys2):
        idxs1 = map1[k]
        idxs2 = map2[k]
        if len(idxs1) != len(idxs2):
            rep.add(
                f"{prefix}item[key={k}].count",
                len(idxs1),
                len(idxs2),
                note="duplicate count differs",
            )
        else:
            # NEW: ok row when duplicate counts match for this key
            if (
                _should_emit_ok(f"{prefix}item[key={k}].count", elem, opts)
                and sum(d.severity == "ok" for d in rep.diffs) < opts.max_ok_rows
            ):
                rep.add(
                    f"{prefix}item[key={k}].count",
                    len(idxs1),
                    len(idxs2),
                    note="match",
                    severity="ok",
                )
        n = min(len(idxs1), len(idxs2))
        for j in range(n):
            _compare_level(
                seq1[idxs1[j]], seq2[idxs2[j]], opts, rep, prefix=f"{prefix}item[key={k}]."
            )


def _compare_values(
    e1: DataElement, e2: DataElement, opts: DiffOptions, rep: DiffReport, path: str
):
    v1, v2 = e1.value, e2.value

    # Multi-valued
    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        if len(v1) != len(v2):
            rep.add(path + ".VM", len(v1), len(v2), note="value multiplicity differs")
            return
        all_ok = True
        for i, (a, b) in enumerate(zip(v1, v2)):
            if not values_equal(a, b, opts.numeric_tol.abs_tol, opts.case_insensitive_strings):
                rep.add(
                    f"{path}[{i}]", a, b, note=f"value differs (tol={opts.numeric_tol.abs_tol})"
                )
                all_ok = False
        if (
            all_ok
            and _should_emit_ok(path, e1, opts)
            and sum(d.severity == "ok" for d in rep.diffs) < opts.max_ok_rows
        ):
            rep.add(path, v1, v2, note="match", severity="ok")
        return

    # Scalar
    if values_equal(v1, v2, opts.numeric_tol.abs_tol, opts.case_insensitive_strings):
        if (
            _should_emit_ok(path, e1, opts)
            and sum(d.severity == "ok" for d in rep.diffs) < opts.max_ok_rows
        ):
            rep.add(path, v1, v2, note="match", severity="ok")
        return

    rep.add(path, v1, v2, note=f"value differs (tol={opts.numeric_tol.abs_tol})")


class _TreeNode:
    __slots__ = ("name", "children", "rows")

    def __init__(self, name: str):
        self.name = name
        self.children: Dict[str, _TreeNode] = {}
        self.rows: List[Diff] = []


def _insert_row(root: _TreeNode, diff: "Diff"):
    parts = diff.path.split(".")
    node = root
    for p in parts[:-1]:  # all but last segment
        if p not in node.children:
            node.children[p] = _TreeNode(p)
        node = node.children[p]
    # attach the row at the full path level (last segment name kept for display in table)
    node.rows.append(diff)


def _build_tree(diffs: List["Diff"]) -> _TreeNode:
    root = _TreeNode("root")
    for d in diffs:
        _insert_row(root, d)
    return root


# --------------------------- YAML Loader (optional) ---------------------------


def load_yaml_config(path: str, base_opts: DiffOptions | None = None) -> DiffOptions:
    """
    Load YAML config into DiffOptions. Requires PyYAML.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config. Install with `pip install pyyaml`."
        ) from e

    cfg = yaml.safe_load(Path(path).read_text())

    collect_all = bool(cfg.get("collect_all_matches", base_opts.collect_all_matches))
    max_ok = int(cfg.get("max_ok_rows", base_opts.max_ok_rows))
    show_for = list(cfg.get("show_matches_for", base_opts.show_matches_for or []))
    if base_opts is None:
        base_opts = DiffOptions()

    ignore_private = bool(cfg.get("ignore_private", base_opts.ignore_private))
    ignore_bulk = bool(cfg.get("ignore_bulk", base_opts.ignore_bulk))
    ci = bool(cfg.get("case_insensitive_strings", base_opts.case_insensitive_strings))
    tol = float(cfg.get("numeric_tol", base_opts.numeric_tol.abs_tol))
    ignore_list = list(cfg.get("ignore", base_opts.ignore_tokens or []))

    # sequence_keys: map str -> list[str]
    seq_keys: Dict[str, List[str]] = dict(base_opts.sequence_keys or {})
    for k, v in (cfg.get("sequence_keys") or {}).items():
        seq_keys[str(k)] = [str(x) for x in v]

    seq_fallback = str(cfg.get("sequence_fallback", base_opts.sequence_fallback)).lower()
    if seq_fallback not in {"order", "report"}:
        seq_fallback = "order"

    return DiffOptions(
        ignore_private=ignore_private,
        ignore_bulk=ignore_bulk,
        ignore_tokens=ignore_list,
        numeric_tol=Tolerance(abs_tol=tol),
        case_insensitive_strings=ci,
        sequence_keys=seq_keys,
        sequence_fallback=seq_fallback,
        bulk_keywords=base_opts.bulk_keywords,
        bulk_vrs=base_opts.bulk_vrs,
        collect_all_matches=collect_all,
        max_ok_rows=max_ok,
        show_matches_for=show_for,
    )


# --------------------------- CLI ---------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Compare two DICOMs (deep, modality-agnostic).")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--config", type=str, help="YAML config path for options and sequence keys")
    p.add_argument(
        "--no-ignore-private",
        dest="ignore_private",
        action="store_false",
        help="Do not ignore private tags (default: ignore).",
    )
    p.add_argument(
        "--no-ignore-bulk",
        dest="ignore_bulk",
        action="store_false",
        help="Do not ignore bulk/byte VRs (default: ignore).",
    )
    p.add_argument(
        "--float-tol", type=float, default=0.0, help="Absolute numeric tolerance (e.g., 1e-6)."
    )
    p.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Tag keyword, (gggg,eeee), or full path to ignore. Repeatable.",
    )
    p.add_argument("--ci-strings", action="store_true", help="Case-insensitive string comparison.")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    p.add_argument("--html", type=str, help="Write HTML report to this file")
    p.add_argument(
        "--title", type=str, default="DICOM Comparison Report", help="Title for HTML report"
    )

    return p.parse_args()


def _cli():
    args = _parse_args()
    # Start with CLI opts (act as defaults)
    opts = DiffOptions(
        ignore_private=args.ignore_private,
        ignore_bulk=args.ignore_bulk,
        ignore_tokens=args.ignore,
        numeric_tol=Tolerance(abs_tol=args.float_tol),
        case_insensitive_strings=args.ci_strings,
    )
    # Apply YAML if provided (overrides/extends)
    if args.config:
        opts = load_yaml_config(args.config, base_opts=opts)

    rep = compare_files(args.left, args.right, opts)

    if args.html:
        out = rep.write_html(args.html, title=args.title)
        # still print JSON or text if requested, otherwise print a short confirmation
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2, default=str))
        else:
            print(f"HTML report written to: {out}")
        return

    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(rep.to_text())


if __name__ == "__main__":
    _cli()
