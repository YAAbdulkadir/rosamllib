from __future__ import annotations
from dataclasses import dataclass, asdict, field
import re
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
from pydicom.datadict import tag_for_keyword, dictionary_VR


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

# --------------------------- Themes ---------------------------
ThemeTokens = Dict[str, str]

THEME_PRESETS: Dict[str, ThemeTokens] = {
    # 1:1 with your dicom-headers tokens (trimmed comments)
    "light": {
        "bg": "#ffffff",
        "fg": "#111111",
        "panel": "#f9fafb",
        "border": "#e5e7eb",
        "muted": "#6b7280",
        "accent": "#2563eb",
        "danger": "#dc2626",
        "shadow": "0 1px 2px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.1)",
        "radius": "12px",
        "scrollbarTrack": "#f3f4f6",
        "scrollbarThumb": "#cbd5e1",
        "scrollbarThumbHover": "#94a3b8",
        "scrollbarThumbActive": "#64748b",
    },
    "midnightSlate": {
        "bg": "#0f1115",
        "fg": "#e5e7eb",
        "panel": "#111827",
        "border": "#1f2937",
        "muted": "#94a3b8",
        "accent": "#60a5fa",
        "danger": "#f87171",
        "shadow": "0 1px 2px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.6)",
        "radius": "12px",
        "scrollbarTrack": "#161b22",
        "scrollbarThumb": "#3a3f4b",
        "scrollbarThumbHover": "#4b5563",
        "scrollbarThumbActive": "#64748b",
    },
    "dark": {
        "bg": "#1E1E1E",
        "panel": "#252526",
        "fg": "#D4D4D4",
        "border": "#333333",
        "muted": "#A6A6A6",
        "accent": "#007ACC",
        "accentAlt": "#569CD6",
        "danger": "#F14C4C",
        "shadow": "0 1px 2px rgba(0,0,0,0.35), 0 2px 6px rgba(0,0,0,0.45)",
        "radius": "12px",
        "success": "#89D185",
        "warning": "#CCA700",
        "info": "#9CDCFE",
        "selection": "#264F78",
        "scrollbarTrack": "#2a2a2a",
        "scrollbarThumb": "#5a5a5a",
        "scrollbarThumbHover": "#6b6b6b",
        "scrollbarThumbActive": "#808080",
        "link": "#3794FF",
        "codeKeyword": "#C586C0",
        "codeString": "#CE9178",
        "codeNumber": "#B5CEA8",
    },
    "oled": {
        "bg": "#000000",
        "panel": "#0A0A0A",
        "fg": "#E6E6E6",
        "border": "#1A1A1A",
        "muted": "#9AA0A6",
        "accent": "#5EA0FF",
        "danger": "#FF6B6B",
        "shadow": "none",
        "radius": "12px",
        "selection": "#13324A",
        "scrollbarTrack": "#0A0A0A",
        "scrollbarThumb": "#242424",
        "scrollbarThumbHover": "#2E2E2E",
        "scrollbarThumbActive": "#3A3A3A",
        "link": "#58A6FF",
    },
    "dim": {
        "bg": "#121417",
        "panel": "#171A1E",
        "fg": "#D8DEE9",
        "border": "#242933",
        "muted": "#9AA7B2",
        "accent": "#7AA2F7",
        "danger": "#EE6D85",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
        "radius": "12px",
        "selection": "#243246",
        "scrollbarTrack": "#1A1E24",
        "scrollbarThumb": "#2C3440",
        "scrollbarThumbHover": "#354054",
        "scrollbarThumbActive": "#41506A",
        "link": "#8AADF4",
    },
    "highContrast": {
        "bg": "#000000",
        "panel": "#0E0E0E",
        "fg": "#FFFFFF",
        "border": "#FFFFFF",
        "muted": "#C0C0C0",
        "accent": "#00FFFF",
        "danger": "#FF0033",
        "shadow": "0 0 0 2px rgba(255,255,255,.2)",
        "radius": "0px",
        "selection": "#00FFFF33",
        "scrollbarTrack": "#000000",
        "scrollbarThumb": "#FFFFFF",
        "scrollbarThumbHover": "#E5E5E5",
        "scrollbarThumbActive": "#CCCCCC",
        "link": "#66FFFF",
    },
    "nord": {
        "bg": "#2E3440",
        "panel": "#3B4252",
        "fg": "#ECEFF4",
        "border": "#434C5E",
        "muted": "#D8DEE9",
        "accent": "#88C0D0",
        "danger": "#BF616A",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
        "radius": "12px",
        "selection": "#4C566A",
        "scrollbarTrack": "#3B4252",
        "scrollbarThumb": "#4C566A",
        "scrollbarThumbHover": "#5B677A",
        "scrollbarThumbActive": "#6B7A91",
        "link": "#81A1C1",
        "codeKeyword": "#81A1C1",
        "codeString": "#A3BE8C",
        "codeNumber": "#B48EAD",
    },
    "dracula": {
        "bg": "#282A36",
        "panel": "#1E1F29",
        "fg": "#F8F8F2",
        "border": "#44475A",
        "muted": "#CED1E6",
        "accent": "#BD93F9",
        "danger": "#FF5555",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
        "radius": "12px",
        "selection": "#44475A",
        "scrollbarTrack": "#1E1F29",
        "scrollbarThumb": "#3B3E51",
        "scrollbarThumbHover": "#4A4E66",
        "scrollbarThumbActive": "#5A5F7B",
        "link": "#8BE9FD",
        "codeKeyword": "#FF79C6",
        "codeString": "#F1FA8C",
        "codeNumber": "#BD93F9",
    },
    "solarizedLight": {
        "bg": "#FDF6E3",
        "panel": "#F5EAD0",
        "fg": "#586E75",
        "border": "#E5DCC5",
        "muted": "#93A1A1",
        "accent": "#268BD2",
        "danger": "#DC322F",
        "shadow": "0 1px 2px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.1)",
        "radius": "12px",
        "selection": "#DDE8C6",
        "scrollbarTrack": "#EFE6D1",
        "scrollbarThumb": "#D6CCB6",
        "scrollbarThumbHover": "#C5BBA6",
        "scrollbarThumbActive": "#B4AA95",
        "link": "#268BD2",
        "codeKeyword": "#859900",
        "codeString": "#2AA198",
        "codeNumber": "#CB4B16",
    },
    "solarizedDark": {
        "bg": "#002B36",
        "panel": "#073642",
        "fg": "#EAEAEA",
        "border": "#0B3742",
        "muted": "#93A1A1",
        "accent": "#268BD2",
        "danger": "#DC322F",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
        "radius": "12px",
        "selection": "#0E4A57",
        "scrollbarTrack": "#073642",
        "scrollbarThumb": "#0F4B57",
        "scrollbarThumbHover": "#145969",
        "scrollbarThumbActive": "#1A697C",
        "link": "#43A6DD",
        "codeKeyword": "#B58900",
        "codeString": "#2AA198",
        "codeNumber": "#CB4B16",
    },
}


def _resolve_theme_tokens(theme: str | None, custom: Optional[ThemeTokens]) -> ThemeTokens:
    """Return a ThemeTokens dict from built-ins or a provided custom mapping."""
    theme = theme or "dark"
    if custom:
        return custom
    if theme in THEME_PRESETS:
        return THEME_PRESETS[theme]
    # default
    return THEME_PRESETS["dark"]


def _theme_tokens_to_css_vars(tok: ThemeTokens) -> str:
    """
    Maps your ThemeTokens -> CSS variables used by the report.
    We also compute sensible fallbacks for extras not provided.
    """
    # Basic requireds
    bg = tok.get("bg", "#0b0f14")
    panel = tok.get("panel", "#121822")
    text = tok.get("fg", "#e6edf3")
    border = tok.get("border", "#1f2630")
    muted = tok.get("muted", "#a7b0be")

    # Semantic accents
    red = tok.get("danger", "#ef4444")
    amber = tok.get("warning", "#f59e0b")
    blue = tok.get("info", tok.get("accent", "#3b82f6"))
    green = tok.get("success", "#10b981")

    # UI niceties
    shadow = tok.get("shadow", "0 1px 2px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.1)")
    radius = tok.get("radius", "12px")
    selection = tok.get("selection", None)
    link = tok.get("link", None)

    # Scrollbars
    sTrack = tok.get("scrollbarTrack", "#161b22")
    sThumb = tok.get("scrollbarThumb", "#3a3f4b")
    sThumbH = tok.get("scrollbarThumbHover", "#4b5563")
    sThumbA = tok.get("scrollbarThumbActive", "#64748b")

    # Internal surfaces (derive if missing)
    search = tok.get("search", panel)
    chip = tok.get("chip", panel)
    row_hover = tok.get("rowHover", None) or _rgba_mix(panel, 0.08)

    code_kw = tok.get("codeKeyword", None)
    code_str = tok.get("codeString", None)
    code_num = tok.get("codeNumber", None)

    lines = [
        ":root{",
        f"--bg:{bg}; --panel:{panel}; --muted:{muted}; --text:{text}; --border:{border};",
        f"--red:{red}; --amber:{amber}; --blue:{blue}; --green:{green};",
        f"--row-hover:{row_hover}; --search:{search}; --chip:{chip};",
        f"--shadow:{shadow}; --radius:{radius};",
        f"--scrollbar-track:{sTrack}; --scrollbar-thumb:{sThumb}; --scrollbar-thumb-h:{sThumbH}; --scrollbar-thumb-a:{sThumbA};",
    ]
    if selection:
        lines.append(f"--selection:{selection};")
    if link:
        lines.append(f"--link:{link};")
    if code_kw:
        lines.append(f"--code-kw:{code_kw};")
    if code_str:
        lines.append(f"--code-str:{code_str};")
    if code_num:
        lines.append(f"--code-num:{code_num};")
    lines.append("}")
    # scrollbar styling (webkit-only; fine for static reports)
    lines.append(
        """
*{box-sizing:border-box}
*::-webkit-scrollbar{width:12px;height:12px}
*::-webkit-scrollbar-track{background:var(--scrollbar-track)}
*::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb);border-radius:8px;border:2px solid var(--scrollbar-track)}
*::-webkit-scrollbar-thumb:hover{background:var(--scrollbar-thumb-h)}
*::-webkit-scrollbar-thumb:active{background:var(--scrollbar-thumb-a)}
::selection{ background: var(--selection, rgba(0,0,0,0.15)); }
a{ color: var(--link, inherit); }
"""
    )
    return "\n".join(lines)


def _rgba_mix(hex_or_panel: str, alpha: float) -> str:
    # Cheap “overlay” fallback: returns an rgba with given alpha using white/black heuristic
    # If you ever want better mixing, you can replace this with a real hex->rgb blend.
    # Here we simply reuse panel with slight transparency on hover look.
    return "rgba(255,255,255,0.06)" if alpha >= 0.06 else "rgba(255,255,255,0.04)"


@dataclass
class _Theme:
    bg: str
    fg: str
    panel: str
    border: str
    muted: str
    accent: str
    danger: str
    shadow: str
    radius: str
    # optional
    success: str = "#10b981"
    warning: str = "#f59e0b"
    info: str = "#60a5fa"
    chip: str = None
    search: str = None
    row_hover: str = None
    scrollbar_track: str = None
    scrollbar_thumb: str = None
    scrollbar_thumb_hover: str = None
    scrollbar_thumb_active: str = None

    def to_css_vars(self) -> str:
        # derive a few if not provided
        chip = self.chip or self.panel
        search = self.search or self.bg
        row_hover = self.row_hover or self.panel
        # map to the variables your CSS uses
        return (
            ":root{"
            f"--bg:{self.bg};--panel:{self.panel};--muted:{self.muted};--fg:{self.fg};--text:{self.fg};"
            f"--border:{self.border};--red:{self.danger};--amber:{self.warning};--green:{self.success};--blue:{self.info};"
            f"--chip:{chip};--search:{search};--row-hover:{row_hover};"
            f"--scrollbar-track:{self.scrollbar_track or self.panel};"
            f"--scrollbar-thumb:{self.scrollbar_thumb or self.border};"
            f"--scrollbar-thumb-hover:{self.scrollbar_thumb_hover or self.muted};"
            f"--scrollbar-thumb-active:{self.scrollbar_thumb_active or self.fg};"
            "}"
        )


def _theme_catalog() -> dict:
    return {
        "light": _Theme(
            bg="#ffffff",
            fg="#111111",
            panel="#f9fafb",
            border="#e5e7eb",
            muted="#6b7280",
            accent="#2563eb",
            danger="#dc2626",
            shadow="0 1px 2px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.1)",
            radius="12px",
            info="#60a5fa",
            success="#10b981",
            warning="#f59e0b",
            chip="#f3f4f6",
            search="#ffffff",
            row_hover="#f3f4f6",
            scrollbar_track="#f3f4f6",
            scrollbar_thumb="#cbd5e1",
            scrollbar_thumb_hover="#94a3b8",
            scrollbar_thumb_active="#64748b",
        ),
        "midnightSlate": _Theme(
            bg="#0f1115",
            fg="#e5e7eb",
            panel="#111827",
            border="#1f2937",
            muted="#94a3b8",
            accent="#60a5fa",
            danger="#f87171",
            shadow="0 1px 2px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.6)",
            radius="12px",
            info="#60a5fa",
            success="#10b981",
            warning="#f59e0b",
            chip="#161b22",
            search="#0f1115",
            row_hover="#0f1520",
            scrollbar_track="#161b22",
            scrollbar_thumb="#3a3f4b",
            scrollbar_thumb_hover="#4b5563",
            scrollbar_thumb_active="#64748b",
        ),
        "dark": _Theme(
            bg="#1E1E1E",
            panel="#252526",
            fg="#D4D4D4",
            border="#333333",
            muted="#A6A6A6",
            accent="#007ACC",
            danger="#F14C4C",
            shadow="0 1px 2px rgba(0,0,0,0.35), 0 2px 6px rgba(0,0,0,0.45)",
            radius="12px",
            success="#89D185",
            warning="#CCA700",
            info="#9CDCFE",
            chip="#2a2a2a",
            search="#1E1E1E",
            row_hover="#202124",
            scrollbar_track="#2a2a2a",
            scrollbar_thumb="#5a5a5a",
            scrollbar_thumb_hover="#6b6b6b",
            scrollbar_thumb_active="#808080",
        ),
        "oled": _Theme(
            bg="#000000",
            panel="#0A0A0A",
            fg="#E6E6E6",
            border="#1A1A1A",
            muted="#9AA0A6",
            accent="#5EA0FF",
            danger="#FF6B6B",
            shadow="none",
            radius="12px",
            chip="#0A0A0A",
            search="#000000",
            row_hover="#0f0f0f",
            scrollbar_track="#0A0A0A",
            scrollbar_thumb="#242424",
            scrollbar_thumb_hover="#2E2E2E",
            scrollbar_thumb_active="#3A3A3A",
        ),
        "dim": _Theme(
            bg="#121417",
            panel="#171A1E",
            fg="#D8DEE9",
            border="#242933",
            muted="#9AA7B2",
            accent="#7AA2F7",
            danger="#EE6D85",
            shadow="0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
            radius="12px",
            chip="#1A1E24",
            search="#121417",
            row_hover="#161A20",
            scrollbar_track="#1A1E24",
            scrollbar_thumb="#2C3440",
            scrollbar_thumb_hover="#354054",
            scrollbar_thumb_active="#41506A",
        ),
        "highContrast": _Theme(
            bg="#000000",
            panel="#0E0E0E",
            fg="#FFFFFF",
            border="#FFFFFF",
            muted="#C0C0C0",
            accent="#00FFFF",
            danger="#FF0033",
            shadow="0 0 0 2px rgba(255,255,255,.2)",
            radius="0px",
            chip="#000000",
            search="#000000",
            row_hover="#111",
            scrollbar_track="#000000",
            scrollbar_thumb="#FFFFFF",
            scrollbar_thumb_hover="#E5E5E5",
            scrollbar_thumb_active="#CCCCCC",
        ),
        "nord": _Theme(
            bg="#2E3440",
            panel="#3B4252",
            fg="#ECEFF4",
            border="#434C5E",
            muted="#D8DEE9",
            accent="#88C0D0",
            danger="#BF616A",
            shadow="0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
            radius="12px",
            chip="#3B4252",
            search="#2E3440",
            row_hover="#343B4B",
            scrollbar_track="#3B4252",
            scrollbar_thumb="#4C566A",
            scrollbar_thumb_hover="#5B677A",
            scrollbar_thumb_active="#6B7A91",
        ),
        "dracula": _Theme(
            bg="#282A36",
            panel="#1E1F29",
            fg="#F8F8F2",
            border="#44475A",
            muted="#CED1E6",
            accent="#BD93F9",
            danger="#FF5555",
            shadow="0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
            radius="12px",
            chip="#1E1F29",
            search="#282A36",
            row_hover="#232433",
            scrollbar_track="#1E1F29",
            scrollbar_thumb="#3B3E51",
            scrollbar_thumb_hover="#4A4E66",
            scrollbar_thumb_active="#5A5F7B",
        ),
        "solarizedLight": _Theme(
            bg="#FDF6E3",
            panel="#F5EAD0",
            fg="#586E75",
            border="#E5DCC5",
            muted="#93A1A1",
            accent="#268BD2",
            danger="#DC322F",
            shadow="0 1px 2px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.1)",
            radius="12px",
            chip="#EFE6D1",
            search="#FDF6E3",
            row_hover="#EDE3C8",
            scrollbar_track="#EFE6D1",
            scrollbar_thumb="#D6CCB6",
            scrollbar_thumb_hover="#C5BBA6",
            scrollbar_thumb_active="#B4AA95",
        ),
        "solarizedDark": _Theme(
            bg="#002B36",
            panel="#073642",
            fg="#EAEAEA",
            border="#0B3742",
            muted="#93A1A1",
            accent="#268BD2",
            danger="#DC322F",
            shadow="0 1px 2px rgba(0,0,0,.35), 0 2px 6px rgba(0,0,0,.45)",
            radius="12px",
            chip="#073642",
            search="#002B36",
            row_hover="#09323E",
            scrollbar_track="#073642",
            scrollbar_thumb="#0F4B57",
            scrollbar_thumb_hover="#145969",
            scrollbar_thumb_active="#1A697C",
        ),
    }


def _resolve_theme(name: str) -> _Theme:
    name = (name or "dark").strip()
    # allow 'system' to fall back to dark for now
    if name == "system":
        name = "dark"
    catalog = _theme_catalog()
    return catalog.get(name, catalog["dark"])


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

    def to_model(self, *, skip_sequence_length: bool = True) -> Dict[str, Any]:
        """
        Build a normalized, presentation-friendly model:
          model = {
            'meta': {'left': {...}, 'right': {...}, 'generated': '...'},
            'summary': {'total': int, 'by_severity': {...}},
            'groups': [
               {'name': 'BeamSequence', 'counts': {...}, 'rows': [Row, ...], 'children': [Group, ...]}
            ]
          }

        Row shape:
          {
            'path': 'BeamSequence[0].GantryAngle',
            'keyword': 'GantryAngle',
            'group': 'BeamSequence',
            'left': '...', 'right': '...',
            'note': 'value differs',
            'severity': 'diff'
          }
        """
        # 1) optionally drop sequence-length diffs globally
        rows: List["Diff"] = (
            [d for d in self.diffs if not str(d.path).endswith(".length")]
            if skip_sequence_length
            else list(self.diffs)
        )

        # 2) summary
        summary = {
            "total": len(rows),
            "by_severity": {
                "diff": sum(d.severity == "diff" for d in rows),
                "warn": sum(d.severity == "warn" for d in rows),
                "info": sum(d.severity == "info" for d in rows),
                "ok": sum(d.severity == "ok" for d in rows),
            },
        }

        # 3) normalize/flatten rows for table renderers
        flat_rows = []
        for d in rows:
            flat_rows.append(
                {
                    "path": d.path,
                    "keyword": self._last_segment(d.path),
                    "group": self._first_segment(d.path),
                    "left": self._preview_cell(d.left),
                    "right": self._preview_cell(d.right),
                    "note": str(d.note or ""),
                    "severity": d.severity,
                    "tags": self._maybe_tag(self._last_segment(d.path)),
                    "vm": None,
                    "vr": None,
                }
            )

        # 4) hierarchical groups (first segment of path)
        groups = self._group_hierarchy(flat_rows)

        # 5) meta
        meta = {
            "left": self.left_meta or {},
            "right": self.right_meta or {},
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return {"meta": meta, "summary": summary, "groups": groups}

    def to_json(self, *, indent: int = 2, skip_sequence_length: bool = True) -> str:
        return json.dumps(
            self.to_model(skip_sequence_length=skip_sequence_length), indent=indent, default=str
        )

    def write_json(
        self, output_path: str, *, indent: int = 2, skip_sequence_length: bool = True
    ) -> str:
        data = self.to_json(indent=indent, skip_sequence_length=skip_sequence_length)
        Path(output_path).write_text(data, encoding="utf-8")
        return output_path

    def write_html(
        self, output_path: str, title: str = "DICOM Comparison Report", theme: str = "dark"
    ) -> str:
        page = self.to_html(title=title, theme=theme)
        Path(output_path).write_text(page, encoding="utf-8")
        return output_path

    def to_html(self, title: str = "DICOM Comparison Report", theme: str = "dark") -> str:
        tokens = _resolve_theme(theme)
        css_vars = tokens.to_css_vars()

        def _rgba(hex_color: str, alpha: float) -> str:
            h = hex_color.lstrip("#")
            if len(h) == 3:  # short #rgb
                r, g, b = (int(h[i] * 2, 16) for i in range(3))
            else:
                r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
            a = max(0.0, min(1.0, float(alpha)))
            return f"rgba({r},{g},{b},{a})"

        diff_bg = _rgba(tokens.danger, 0.18)  # mismatch background
        warn_bg = _rgba(getattr(tokens, "warning", "#f59e0b"), 0.3)
        ok_bg = _rgba(getattr(tokens, "success", "#10b981"), 0.3)

        # === build tree, skipping *.length rows ===
        rows = [d for d in self.diffs if not str(d.path).endswith(".length")]

        root: _Node = _Node(name="root")
        for d in rows:
            parts = self._split_path(d.path)
            if parts and parts[-1] == "VR" and len(parts) >= 2:
                parent = root.ensure_path(parts[:-1])
                parent.vr_left = "" if d.left is None else str(d.left)
                parent.vr_right = "" if d.right is None else str(d.right)
                parent.has_vr_diff_flag = d.left != d.right
                continue
            node = root.ensure_path(parts)
            node.rows.append(d)

        # try to infer VR if not present (best effort)
        def infer_vr(n: "_Node"):
            if (n.vr_left or n.vr_right) or n.is_item_row() or not n.name:
                return
            kw = n.keyword_candidate()
            if not kw:
                return
            try:
                tg = tag_for_keyword(kw)
                if tg is None:
                    return
                vr = dictionary_VR(tg)
                if vr:
                    n.vr_left = vr
                    n.vr_right = vr
            except Exception:
                pass

        root.walk(infer_vr)

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        left_meta = self.left_meta or {}
        right_meta = self.right_meta or {}
        summary = {
            "total": len(rows),
            "diff": sum(d.severity == "diff" for d in rows),
            "warn": sum(d.severity == "warn" for d in rows),
            "info": sum(d.severity == "info" for d in rows),
            "ok": sum(d.severity == "ok" for d in rows),
        }

        # ---------- CSS ----------
        css = f"""
        {css_vars}
        *{{box-sizing:border-box}}
        html,body{{height:100%}}
        body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}}

        .header{{padding:18px 20px;border-bottom:1px solid var(--border);
        background:linear-gradient(180deg, var(--panel), transparent)}}
        .title{{font-size:20px;margin:0 0 6px 0}}
        .meta{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted)}}
        .badges{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}}
        .badge{{padding:4px 8px;border-radius:999px;background:var(--chip);
        border:1px solid var(--border);color:var(--muted)}}
        .badge b{{color:var(--text)}}
        .meta-table{{width:auto;border-collapse:collapse;margin-top:10px}}
        .meta-table td{{padding:4px 8px;border-bottom:1px solid var(--border);color:var(--muted)}}

        .container{{padding:16px 20px}}
        .toolbar{{position:sticky;top:0;z-index:10;background:var(--bg);padding:10px 0 12px;
        display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--border)}}
        input[type="search"]{{flex:1;min-width:240px;padding:8px 10px;border-radius:8px;border:1px solid var(--border);
        background:var(--search);color:var(--text)}}
        .btn{{padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:var(--search);
        color:var(--text);cursor:pointer}}
        .btn:hover{{background:var(--row-hover)}}
        .toggle{{display:flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid var(--border);
        border-radius:999px;background:var(--search);color:var(--muted);cursor:pointer}}
        .toggle input{{accent-color:var(--blue);}}

        /* Scroll shells */
        .gridwrap{{border:1px solid var(--border);border-radius:{tokens.radius};overflow:hidden;width:100%;background:var(--bg);display:flex;flex-direction:column;box-shadow:{tokens.shadow}}}
        .hscroll-top{{height:14px;overflow-x:auto;overflow-y:hidden}}
        .hscroll-spacer{{height:1px}}
        .tablewrap{{overflow:auto;min-height:260px;max-height:70vh}}

        /* Table look */
        .rows{{border-collapse:collapse;width:max-content;min-width:100%}}
        .thCell{{ text-align:left; padding:4px 6px; border-bottom:1px solid var(--border);
        position:sticky; top:0; z-index:1; background:var(--panel); color:var(--muted);
        box-shadow:0 1px 0 0 var(--border) inset; line-height:1.25 }}
        .tdCell{{ padding:3px 6px; border-bottom:1px solid var(--border); vertical-align:top; line-height:1.25; user-select:text; position:relative }}
        .mono{{ font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace }}
        .kwcell{{ display:inline-flex; align-items:center; gap:6px }}
        .twistyWrap{{ width:18px; height:18px; display:inline-flex }}
        .twistyBtn{{ width:18px; height:18px; line-height:14px; border-radius:4px; border:1px solid var(--border);
        background:var(--panel); color:var(--fg); cursor:pointer; user-select:none; text-align:center; padding:0 }}
        .valCell{{ white-space:nowrap }}

        /* value state colors */
        .valCell {{ border-radius: 4px; padding: 1px 4px; }}
        .cell-missing {{ background:#000; color:#fff; }}
        .cell-diff {{ background:{diff_bg}; color:var(--fg); }}
        .cell-warn {{ background:{warn_bg}; color:var(--fg); }}
        .cell-ok {{ background:{ok_bg}; color:var(--fg); }}

        /* copy pill */
        .copypill{{ position:absolute; right:8px; top:4px; font-size:11px; padding:2px 6px; border-radius:6px;
        border:1px solid var(--border); background:var(--panel); color:var(--muted); pointer-events:none; opacity:0; transition:opacity .12s }}
        .tdCell.copying .copypill{{ opacity:1 }}

        /* Column sizing */
        .col-tight{{ width:1%; }}
        .col-vr{{ width:52px; }}
        .col-flex{{ width:auto; }}

        .thCell, .tdCell{{ white-space:nowrap; }}
        .mono{{ white-space:nowrap; }}
        .kwcell .kwtext{{ white-space:nowrap; }}
        .valCell{{ white-space:nowrap; max-width:40vw; }}

        /* Themed scrollbars */
        /* Firefox */
        *{{scrollbar-width:thin; scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);}}
        /* WebKit (Chrome/Edge/Safari) */
        *::-webkit-scrollbar{{width:12px;height:12px;}}
        *::-webkit-scrollbar-track{{background:var(--scrollbar-track);}}
        *::-webkit-scrollbar-thumb{{background:var(--scrollbar-thumb);border-radius:8px;border:2px solid transparent;background-clip:content-box;}}
        *::-webkit-scrollbar-thumb:hover{{background:var(--scrollbar-thumb-hover);background-clip:content-box;}}
        *::-webkit-scrollbar-thumb:active{{background:var(--scrollbar-thumb-active);background-clip:content-box;}}

        .footer{{color:var(--muted);text-align:center;padding:16px 0;border-top:1px solid var(--border);margin-top:16px}}

        .export{{position:relative;}}
        .menu{{position:absolute; top:100%; right:0; background:var(--panel);
        border:1px solid var(--border); border-radius:8px; padding:6px;
        display:flex; flex-direction:column; gap:6px; min-width:160px; box-shadow:var(--shadow,0 6px 18px rgba(0,0,0,.35));
        }}
        .menu[hidden]{{display:none;}}
        .menuitem{{
            padding:6px 10px; text-align:left; border:1px solid var(--border);
            border-radius:6px; background:var(--search); color:var(--text); cursor:pointer;
        }}
        .menuitem:hover{{ background:var(--row-hover);}}
        """

        # ---------- table body ----------
        export_rows: List[Dict[str, Any]] = []
        body_html = self._render_tree(root, depth=0, export_rows=export_rows)
        left_meta_payload = dict(left_meta)
        right_meta_payload = dict(right_meta)
        if "path" in left_meta_payload:
            left_meta_payload["path"] = str(left_meta_payload["path"])
        if "path" in right_meta_payload:
            right_meta_payload["path"] = str(right_meta_payload["path"])

        payload = {
            "title": title,
            "generated": stamp,
            "summary": summary,
            "left_meta": left_meta_payload,
            "right_meta": right_meta_payload,
            "rows": export_rows,
        }
        # print(left_meta)
        export_json = json.dumps(payload, ensure_ascii=False)

        # body_html = self._render_tree(root, depth=0)

        # ---------- JS (PLAIN RAW STRING, not f-string) ----------
        js = r"""
    const NUMERIC_VRS=/^(US|SS|UL|SL|FL|FD|IS|DS)$/i, BINARY_VRS=/^(OB|OW|OF|OD|OL|UN)$/i, SEQ_VR=/^SQ$/i;
    const NUM_RE=/[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/;
    function stripCtl(s){return String(s??'').replace(/[\x00-\x1F\x7F]/g,'')}
    function splitBackslash(raw){const s=stripCtl(String(raw??'')); return s.includes('\\')? s.split('\\').map(t=>t.trim()) : [s.trim()]}
    function isBinaryVR(vr){return !!vr && BINARY_VRS.test(vr)}
    function inferVM(vr, raw, byteLength) {
    const s=String(raw??''); if(SEQ_VR.test(vr)) return 0; if(isBinaryVR(vr)) return 1;
    if(/^(US|SS)$/i.test(vr) && byteLength && byteLength%2===0) return Math.max(1, byteLength/2);
    if(/^(UL|SL|FL)$/i.test(vr) && byteLength && byteLength%4===0) return Math.max(1, byteLength/4);
    if(/^FD$/i.test(vr)       && byteLength && byteLength%8===0) return Math.max(1, byteLength/8);
    return s.includes('\\') ? s.split('\\').length : 1;
    }
    function parseValuesByVR(vr, raw, vm) {
    const parts=splitBackslash(raw); const use=parts.slice(0, vm>0?vm:parts.length);
    if(NUMERIC_VRS.test(vr)) return use.map(p=> (NUM_RE.test(p)? Number(p): stripCtl(p)));
    if(isBinaryVR(vr)) return [];
    return use.map(p=> stripCtl(p));
    }

    // ===== copy pill =====
    function attachCopy(td, text) {
    td.classList.add('copyable');
    td.style.cursor='copy';
    let pill=null, hideTO=null;
    td.addEventListener('click', async () => {
        try {
        if(navigator.clipboard?.writeText) await navigator.clipboard.writeText(text||'');
        else {
            const ta=document.createElement('textarea'); ta.value=text||''; ta.style.position='fixed'; ta.style.opacity='0';
            document.body.appendChild(ta); ta.focus(); ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
        }
        if(!pill) { pill=document.createElement('span'); pill.className='copypill'; pill.textContent='Copied'; td.appendChild(pill); }
        td.classList.add('copying');
        clearTimeout(hideTO); hideTO=setTimeout(()=> td.classList.remove('copying'), 900);
        } catch(_) {}
    });
    td.title='Click to copy';
    }

    // ===== render value cells (truncate + array preview) =====
    function renderValueCell(td) {
    const vr=td.getAttribute('data-vr')||'';
    const raw=td.getAttribute('data-raw')||'';
    const blen=parseInt(td.getAttribute('data-length')||'')||null;
    const missing = td.getAttribute('data-missing')==='1';
    if(missing){ td.textContent=''; attachCopy(td,''); return; }
    if(SEQ_VR.test(vr)){ attachCopy(td, td.textContent); return; }
    if(isBinaryVR(vr)){ td.textContent='[binary]'; attachCopy(td,'[binary]'); return; }

    const vmAttr = td.getAttribute('data-vm');
    const vm = vmAttr? parseInt(vmAttr): inferVM(vr, raw, blen);
    const vals = parseValuesByVR(vr, raw, vm);
    const asStrings = vals.map(v => String(v));

    if(vm>1 || asStrings.length>1){
        td.style.whiteSpace='nowrap';
        const wrap=document.createElement('span'); wrap.style.fontFamily='ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
        const openBtn = document.createElement('button');
        function render(expanded){
        wrap.textContent='[' + (expanded? asStrings.join(', '): asStrings.slice(0,12).join(', ')) + ']';
        if(!expanded && asStrings.length>12){
            openBtn.textContent='+'+(asStrings.length-12)+' more';
            openBtn.style.border='1px solid var(--border)'; openBtn.style.background='var(--panel)'; openBtn.style.color='var(--fg)';
            openBtn.style.borderRadius='6px'; openBtn.style.padding='0 6px'; openBtn.style.marginLeft='6px'; openBtn.style.cursor='pointer';
            openBtn.title='Show all values';
            if(!openBtn.onclick) openBtn.onclick=()=>{ render(true); };
            td.replaceChildren(wrap, document.createTextNode(' '), openBtn);
        } else if(expanded && asStrings.length>12){
            const lessBtn=document.createElement('button');
            lessBtn.textContent='Show less';
            lessBtn.style.border='1px solid var(--border)'; lessBtn.style.background='var(--panel)'; lessBtn.style.color='var(--muted)';
            lessBtn.style.borderRadius='6px'; lessBtn.style.padding='0 6px'; lessBtn.style.marginLeft='6px'; lessBtn.style.cursor='pointer';
            lessBtn.title='Collapse';
            lessBtn.onclick=()=>{ render(false); };
            td.replaceChildren(wrap, document.createTextNode(' '), lessBtn);
        } else {
            td.replaceChildren(wrap);
        }
        }
        render(false);
        attachCopy(td, '['+asStrings.join(', ')+']');
        return;
    }

    const one = (asStrings[0] ?? '').slice(0, 10000);  // hard ceiling
    td.style.whiteSpace='nowrap';
    td.textContent = one;
    attachCopy(td, one);
    }

    // ===== expand/collapse + filter + top scrollbar sync =====
    (function(){
    const by=(s,r=document)=>Array.from(r.querySelectorAll(s));

    const table=document.getElementById('diffTable');
    const wrap=document.getElementById('tableWrap');
    const topScroll=document.getElementById('topScroll');
    const topSpacer=document.getElementById('topSpacer');
    const filterInput=document.getElementById('filterInput');

    // Checkboxes may be absent in older reports — guard them.
    const fDiffEl=document.getElementById('fDiff');
    const fWarnEl=document.getElementById('fWarn');
    const fOkEl  =document.getElementById('fOk');

    function syncTopWidth(){
        if(!table) return;
        topSpacer.style.width = table.scrollWidth + 'px';
    }
    if(wrap && topScroll){
        wrap.addEventListener('scroll', ()=>{ topScroll.scrollLeft = wrap.scrollLeft; });
        topScroll.addEventListener('scroll', ()=>{ wrap.scrollLeft = topScroll.scrollLeft; });
    }
    if(table && wrap){
        new ResizeObserver(syncTopWidth).observe(table);
        new ResizeObserver(syncTopWidth).observe(wrap);
        window.addEventListener('load', syncTopWidth);
    }

    function setRowVisibility(){
        if(!table) return;
        const rows=by('tbody > tr', table);
        const openAtDepth={};
        rows.forEach(r=>{
        const depth=+r.getAttribute('data-depth')||0;
        let visible=true;
        for(let d=0; d<depth; d++){ if(openAtDepth[d]===false){ visible=false; break; } }
        r.style.display = visible ? '' : 'none';
        const isGroup=r.getAttribute('data-seq')==='1';
        if(isGroup){
            const btn=r.querySelector('.twistyBtn');
            const open=btn && btn.dataset.open==='1';
            openAtDepth[depth]=(open!==false);
        }
        });
    }
    function toggleBranch(tr,open){
        const btn=tr.querySelector('.twistyBtn'); if(!btn) return;
        btn.dataset.open=open?'1':'0'; btn.textContent=open?'–':'+';
        setRowVisibility();
    }

    by('.twistyBtn', table).forEach(btn=>{
        const tr=btn.closest('tr');
        btn.addEventListener('click',e=>{
        e.stopPropagation();
        const nowOpen = btn.dataset.open!=='1';
        toggleBranch(tr, nowOpen);
        });
    });

    const expandAllBtn=document.getElementById('expandAll');
    const collapseAllBtn=document.getElementById('collapseAll');
    if(expandAllBtn) expandAllBtn.addEventListener('click', ()=>{
        by('.twistyBtn', table).forEach(btn=>{btn.dataset.open='1'; btn.textContent='–';});
        setRowVisibility();
    });
    if(collapseAllBtn) collapseAllBtn.addEventListener('click', ()=>{
        by('.twistyBtn', table).forEach(btn=>{btn.dataset.open='0'; btn.textContent='+';});
        setRowVisibility();
    });

    const norm = s => (s||'').toLowerCase().replace(/\s+/g,'');

    // Safe accessors for checkboxes (default ON if missing)
    const diffOn = () => !fDiffEl || !!fDiffEl.checked;
    const warnOn = () => !fWarnEl || !!fWarnEl.checked;
    const okOn   = () => !fOkEl   || !!fOkEl.checked;

    function leafMatchesSeverity(sev) {
        return (sev === 'diff' && diffOn())
            || (sev === 'warn' && warnOn())
            || ((sev === 'ok' || sev === 'info') && okOn());
    }
    function headerAggMatches(agg) {
        return (agg === 'diff' && diffOn())
            || (agg === 'warn' && warnOn())
            || (agg === 'ok'   && okOn());
    }

    function applyFilters() {
        if(!table) return;
        const q = norm(filterInput?.value?.trim?.() ?? '');

        // Filter leaf rows by text + severity
        const leafRows = by('tbody > tr[data-seq="0"]', table);
        leafRows.forEach(tr => {
        const sev = tr.getAttribute('data-sev') || '';
        const blob = tr.getAttribute('data-all') || tr.textContent || '';
        const textOk = !q || norm(blob).includes(q);
        const sevOk = leafMatchesSeverity(sev);
        const pass = textOk && sevOk;
        tr.style.visibility = pass ? 'visible' : 'collapse';
        tr.dataset.pass = pass ? '1' : '0';
        });

        // Headers: visible if any descendant leaf visible; otherwise check aggregate + text
        const headerRows = by('tbody > tr[data-seq="1"]', table);
        const onlyOk = okOn() && !diffOn() && !warnOn();

        function setHeaderTint(tr, cls){
            // reset both headerVal cells to desired cls (or base if '')
            const tds = Array.from(tr.querySelectorAll('td.headerVal'));
            const classes = ['cell-ok', 'cell-warn', 'cell-diff', 'cell-missing'];
            tds.forEach(td=>{
            classes.forEach(c=> td.classList.remove(c));
            if (cls) td.classList.add(cls);
            });
        }
        headerRows.forEach(tr => {
            const myDepth = +(tr.getAttribute('data-depth') || '0');
            const agg = tr.getAttribute('data-agg') || 'warn';
            const base = tr.getAttribute('data-basecls') || '';
            let anyVisibleChild = false;

            // scan forward until a row with depth <= myDepth (end of subtree)
            for (let nxt = tr.nextElementSibling; nxt; nxt = nxt.nextElementSibling) {
                const nd = +(nxt.getAttribute('data-depth') || '0');
                if (nd <= myDepth) break;
                if (nxt.getAttribute('data-seq') === '0' && nxt.style.visibility !== 'collapse') {
                anyVisibleChild = true;
                break;
                }
        }
        if (anyVisibleChild) {
            tr.style.visibility = 'visible';
            tr.dataset.pass = '1';
            // If we're in Match-only mode, force the header tint to green.
            // Otherwise show its base (aggregate-derived) tint.
            setHeaderTint(tr, onlyOk ? 'cell-ok' : base);
        } else {
            // Text-match for header itself (use its own text content) + severity
            const blob = tr.textContent || '';
            const textOk = !q || norm(blob).includes(q);
            const sevOk = headerAggMatches(agg);
            const pass = textOk && sevOk;
            tr.style.visibility = pass ? 'visible' : 'collapse';
            tr.dataset.pass = pass ? '1' : '0';
            setHeaderTint(tr, base);
        }
        });

        // Recompute open/closed visibility wrt hierarchy (display none)
        setRowVisibility();
    }

    function $(id){ return document.getElementById(id); }

    function safeFileName(s){
        return (s||'report').replace(/[\/\\:*?"<>|]+/g,'_').slice(0,120);
    }
    function nowStamp(){
        const d = new Date();
        const pad = (n)=>String(n).padStart(2,'0');
        return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
    }
    function download(filename, mime, content){
        const blob = new Blob([content], {type:mime});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        setTimeout(()=> URL.revokeObjectURL(a.href), 1500);
    }
    function getExportPayload(){
        try {
            const el = $('exportData');
            return JSON.parse(el?.textContent || '{}');
        } catch(e){ return {}; }
    }

    // Convert export rows → CSV; if you want filtered-only, pass a filtered array
    function toCSV(rows){
    const headers = ["Keyword","Tag","VR","Severity","Note","Left","Right","Path"];
    const esc = (v)=> `"${String(v??'').replace(/"/g,'""')}"`;
    const lines = [headers.map(esc).join(",")];
    rows.forEach(r=>{
        if(r.type === 'leaf'){
        lines.push([r.keyword, r.tag, r.vr, r.severity, r.note, r.left_raw, r.right_raw, r.path].map(esc).join(","));
        }
    });
    return lines.join("\n");
    }

    // If you want to export only what's currently visible, grab visible leaf rows:
    function visibleLeafRowsToJsonBase(){
    const by = (s,r=document)=>Array.from(r.querySelectorAll(s));
    const table = $('diffTable');
    const leaves = by('tbody > tr[data-seq="0"]', table);
    const set = new Set();
    leaves.forEach(tr=>{
        if(tr.style.visibility === 'collapse') return;
        const path = tr.getAttribute('data-path') || '';
        if(path) set.add(path);
    });
    return set;
    }

    function buildExportRowsVisibleOnly(payload){
    const visiblePaths = visibleLeafRowsToJsonBase();
    if(!visiblePaths.size) return [];
    return (payload.rows||[]).filter(r => r.type==='leaf' && visiblePaths.has(r.path));
    }

    // ===== Wire export menu =====
    (function(){
    const btn = $('exportBtn');
    const menu = $('exportMenu');
    const title = document.title || 'DICOM_Comparison_Report';
    const base = safeFileName(title) + '_' + nowStamp();

    function toggleMenu(show){
        if(show===undefined) show = menu.hasAttribute('hidden');
        if(show) menu.removeAttribute('hidden'); else menu.setAttribute('hidden','');
    }
    btn.addEventListener('click', (e)=>{ e.stopPropagation(); toggleMenu(); });
    document.addEventListener('click', ()=> toggleMenu(false));

    // Choose whether to export *all rows* or only *visible*.
    // Flip this flag per your preference:
    const EXPORT_VISIBLE_ONLY = true;

    function pickRows(payload){
        if(EXPORT_VISIBLE_ONLY) return buildExportRowsVisibleOnly(payload);
        return (payload.rows||[]).filter(r => r.type==='leaf');
    }

    $('expCsv').addEventListener('click', ()=>{
        const payload = getExportPayload();
        const rows = pickRows(payload);
        const csv = toCSV(rows);
        download(base + '.csv', 'text/csv;charset=utf-8', csv);
        toggleMenu(false);
    });

    $('expJson').addEventListener('click', ()=>{
        const payload = getExportPayload();
        const rows = pickRows(payload);
        const out = {
        title: payload.title,
        generated: payload.generated,
        summary: payload.summary,
        left_meta: payload.left_meta,
        right_meta: payload.right_meta,
        rows
        };
        download(base + '.json', 'application/json;charset=utf-8', JSON.stringify(out, null, 2));
        toggleMenu(false);
    });

    $('expHtml').addEventListener('click', ()=>{
        // Save the current page as standalone HTML (with current filters applied)
        const html = '<!doctype html>\n' + document.documentElement.outerHTML;
        download(base + '.html', 'text/html;charset=utf-8', html);
        toggleMenu(false);
    });

    if(filterInput) filterInput.addEventListener('input', applyFilters);
    [fDiffEl, fWarnEl, fOkEl].forEach(cb => { if(cb) cb.addEventListener('change', applyFilters); });

    // Initial render
    by('td.valCell', table).forEach(td => renderValueCell(td));
    by('.twistyBtn', table).forEach(btn=>{btn.dataset.open='1'; btn.textContent='–';});
    setRowVisibility();
    applyFilters();
    })();
    })();

    """

        # ---------- HTML skeleton with TOP SCROLLBAR ----------
        def meta_table(meta: Dict[str, Any]) -> str:
            if not meta:
                return ""
            rows = "".join(
                f"<tr><td>{self._e(k)}</td><td class='mono'>{self._e(v)}</td></tr>"
                for k, v in meta.items()
            )
            return f"<table class='meta-table'>{rows}</table>"

        # Final page (this is the ONLY f-string; it just injects css/js strings)
        return f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>{self._e(title)}</title>
    <style>{css}</style>
    </head>
    <body>
    <div class="header">
        <div class="title">{self._e(title)}</div>
        <div class="meta">
        <div><b>Generated:</b> {self._e(stamp)}</div>
        <div><b>Total rows:</b> {summary['total']}</div>
        </div>
        <div class="badges">
        <div class="badge">diff: <b>{summary['diff']}</b></div>
        <div class="badge">warn: <b>{summary['warn']}</b></div>
        <div class="badge">info: <b>{summary['info']}</b></div>
        <div class="badge">ok:   <b>{summary['ok']}</b></div>
        </div>
        <div class="meta" style="margin-top:10px;gap:40px">
        <div><div><b>Left</b></div>{meta_table(left_meta)}</div>
        <div><div><b>Right</b></div>{meta_table(right_meta)}</div>
        </div>
    </div>

    <div class="container">
        <div class="toolbar">
        <input id="filterInput" type="search" placeholder="Filter (keyword, tag, VR, value)…" />
        <label class="toggle"><input type="checkbox" id="fDiff" checked/> Diff</label>
        <label class="toggle"><input type="checkbox" id="fWarn" checked/>Warn</label>
        <label class="toggle"><input type="checkbox" id="fOk" checked/>Match</label>
        <div class="export">
        <button class="btn" id="exportBtn">Export ▾</button>
        <div class="menu" id="exportMenu" hidden>
            <button class="menuitem" id="expHtml">Save as HTML</button>
            <button class="menuitem" id="expCsv">Save as CSV</button>
            <button class="menuitem" id="expJson">Save as JSON</button>
        </div>
        </div>
        <button class="btn" id="expandAll">Expand all</button>
        <button class="btn" id="collapseAll">Collapse all</button>
        </div>

        <div class="gridwrap">
        <!-- TOP sticky horizontal scrollbar -->
        <div class="hscroll-top" id="topScroll"><div class="hscroll-spacer" id="topSpacer"></div></div>

        <!-- Main scroll area -->
        <div class="tablewrap" id="tableWrap">
            <table class="rows" id="diffTable">
            <colgroup>
                <col class="col-tight kw-col"/>
                <col class="col-tight tag-col"/>
                <col class="col-vr"/>
                <col class="col-flex"/>
                <col class="col-flex"/>
            </colgroup>
            <thead>
                <tr>
                <th class="thCell">Keyword</th>
                <th class="thCell">Tag</th>
                <th class="thCell">VR</th>
                <th class="thCell">Left</th>
                <th class="thCell">Right</th>
                </tr>
            </thead>
            <tbody>
                {body_html}
            </tbody>
            </table>
        </div>
        </div>
    </div>

    <div class="footer">compare_dcms — HTML report</div>
    <script id="exportData" type="application/json">{export_json}</script>
    <script>{js}</script>
    </body>
    </html>"""

    def _prepare_cell_payload(self, val: Any) -> Dict[str, Any]:
        """
        Normalize Python-side values for the JS renderer:
        - lists/tuples/MultiValue -> backslash-joined string; vm = len(...)
        - bytes/bytearray         -> raw '', length=len(...), JS shows [binary] if VR says so
        - everything else         -> str(val), vm=1
        """
        vm = 1
        blen = ""
        # pydicom MultiValue behaves like list; include numpy arrays too if present
        is_seq = isinstance(val, (list, tuple))
        try:
            # pydicom MultiValue duck-typing
            from pydicom.multival import MultiValue  # type: ignore

            if isinstance(val, MultiValue):
                is_seq = True
        except Exception:
            pass
        try:
            import numpy as np  # type: ignore

            if isinstance(val, np.ndarray):
                is_seq = True
                val = val.tolist()
        except Exception:
            pass

        if val is None:
            return {"raw": "", "vm": 0, "blen": ""}

        if isinstance(val, (bytes, bytearray)):
            blen = str(len(val))
            return {"raw": "", "vm": 1, "blen": blen}

        if is_seq:
            parts = ["" if x is None else str(x) for x in list(val)]
            vm = len(parts)
            raw = "\\".join(parts)  # IMPORTANT: backslash-separated for JS
            return {"raw": raw, "vm": vm, "blen": ""}

        # scalar
        return {"raw": str(val), "vm": 1, "blen": ""}

    # ------------------------ tree rendering ------------------------

    def _render_tree(
        self,
        root: "_Node",
        depth: int = 0,
        row_no: List[int] = None,
        export_rows: List[Dict[str, Any]] = None,
    ) -> str:
        if row_no is None:
            row_no = [0]
        if export_rows is None:
            export_rows = []

        out: List[str] = []
        # Render children of root only; root itself is artificial
        for child in root.children:
            out.append(self._render_node(child, depth, row_no, export_rows))
        return "".join(out)

    def _render_node(
        self, node: "_Node", depth: int, row_no: List[int], export_rows: List[Dict[str, Any]]
    ) -> str:
        html_rows: List[str] = []

        def next_bg() -> str:
            row_no[0] += 1
            return "var(--panel)" if (row_no[0] % 2 == 1) else "transparent"

        # 1) If node represents a Sequence or an Item N, render a collapsible header row.
        if node.is_sequence() or node.is_item_row():
            bg = next_bg()
            keyword = node.display_keyword()
            tag_hex = node.tag_hex_guess()
            vr = node.display_vr()

            # Value cells for sequences / item headers show item count
            left_disp, right_disp, cls_left, cls_right = node.sequence_value_cells()
            # derive header tint from subtree
            sc = node.severity_counts()
            if sc["diff"] > 0:
                header_cls = "cell-diff"  # red-ish background
            elif (sc["warn"] == 0) and (sc["diff"] == 0):
                header_cls = "cell-ok"  # green-ish background (everything matches)
            else:
                header_cls = ""  # neutral for warnings/info

            if sc["diff"] > 0:
                header_agg = "diff"
            elif sc["warn"] == 0:
                header_agg = "ok"
            else:
                header_agg = "warn"

            html_rows.append(
                f"""
    <tr class="row" data-depth="{depth}" data-seq="1" data-agg="{header_agg}" data-basecls="{header_cls}" style="background:{bg}">
    <td class="tdCell">
        <div class="kwcell" style="padding-left:{8 + depth*16}px">
        <span class="twistyWrap"><button class="twistyBtn" data-open="1" title="Collapse">–</button></span>
        <span class="kwtext">{self._e(keyword)}</span>
        </div>
    </td>
    <td class="tdCell mono">{self._e(tag_hex)}</td>
    <td class="tdCell mono">{self._e(vr)}</td>
    <td class="tdCell valCell headerVal {header_cls}" data-vr="SQ" data-raw="{self._e(left_disp)}" data-length="" data-missing="0">
        <span class="copypill">Copied</span>{self._e(left_disp)}
    </td>
    <td class="tdCell valCell headerVal {header_cls}" data-vr="SQ" data-raw="{self._e(right_disp)}" data-length="" data-missing="0">
        <span class="copypill">Copied</span>{self._e(right_disp)}
    </td>
    </tr>
    """
            )

            # Children
            for c in node.children:
                html_rows.append(self._render_node(c, depth + 1, row_no, export_rows=export_rows))
            return "".join(html_rows)

        # 2) Otherwise, it’s a leaf field row
        bg = next_bg()
        keyword = node.display_keyword()
        tag_hex = node.tag_hex_guess()
        vr = node.display_vr()

        # A node may aggregate multiple Diff rows (e.g., value + some metadata);
        # We’ll render only "value-like" rows (not the .VR row, which we already attached to node.vr_*).
        # If there are multiple value diffs, we render each as its own row under the same keyword.
        value_rows = [r for r in node.rows if not str(r.path).endswith(".VR")]
        if not value_rows and not node.children:
            # no visible rows; skip
            return ""

        if not value_rows:
            # show a single row with empty values (e.g., only had .VR diff)
            left_disp, right_disp, cls_left, cls_right = node.value_cells_for(None)
            sc = node.severity_counts()
            if sc["diff"] > 0:
                header_cls = "cell-diff"
                header_agg = "diff"
            elif sc["warn"] == 0:
                header_cls = "cell-ok"
                header_agg = "ok"
            else:
                header_cls = ""
                header_agg = "warn"

            html_rows.append(
                f"""
<tr class="row" data-depth="{depth}" data-seq="1" data-agg="{header_agg}" data-basecls="{header_cls}" style="background:{bg}">
  <td class="tdCell">
    <div class="kwcell" style="padding-left:{8 + depth*16}px">
      <span class="twistyWrap"><button class="twistyBtn" data-open="1" title="Collapse">–</button></span>
      <span class="kwtext">{self._e(keyword)}</span>
    </div>
  </td>
  <td class="tdCell mono">{self._e(tag_hex)}</td>
  <td class="tdCell mono">{self._e(vr)}</td>
  <td class="tdCell valCell headerVal {cls_left}"  data-vr="SQ" data-raw="{self._e(left_disp)}"  data-length="" data-missing="0">
    <span class="copypill">Copied</span>{self._e(left_disp)}
  </td>
  <td class="tdCell valCell headerVal {cls_right}" data-vr="SQ" data-raw="{self._e(right_disp)}" data-length="" data-missing="0">
    <span class="copypill">Copied</span>{self._e(right_disp)}
  </td>
</tr>
"""
            )
        else:
            for r in value_rows:
                vr_display = node.display_vr()
                vr_for_parse = (
                    node.vr_left
                    if node.vr_left and node.vr_left == node.vr_right
                    else node.vr_left or node.vr_right or ""
                )
                # left_missing = r.left is None or str(r.left) in ("", "None")
                # right_missing = r.right is None or str(r.right) in ("", "None")
                left_missing = r.left is None
                right_missing = r.right is None
                left_class = (
                    "cell-missing"
                    if left_missing
                    else (
                        "cell-diff"
                        if r.severity == "diff"
                        else "cell-warn" if r.severity == "warn" else "cell-ok"
                    )
                )
                right_class = (
                    "cell-missing"
                    if right_missing
                    else (
                        "cell-diff"
                        if r.severity == "diff"
                        else "cell-warn" if r.severity == "warn" else "cell-ok"
                    )
                )
                # prepare payloads for JS truncation/array preview
                lp = self._prepare_cell_payload(r.left)
                rp = self._prepare_cell_payload(r.right)
                blob = f"{r.path} {r.note} {r.severity} {r.left} {r.right}"
                export_rows.append(
                    {
                        "type": "leaf",
                        "path": str(r.path),
                        "keyword": keyword,
                        "tag": tag_hex,
                        "vr": vr_display,
                        "severity": r.severity,
                        "note": r.note,
                        "left_raw": lp["raw"],
                        "right_raw": rp["raw"],
                        "left_vm": lp["vm"],
                        "right_vm": rp["vm"],
                        "left_len": lp["blen"],
                        "right_len": rp["blen"],
                    }
                )
                html_rows.append(
                    f"""
<tr class="row rowsev-{self._e(r.severity)}" data-depth="{depth}" data-seq="0" data-sev="{self._e(r.severity)}" data-path="{self._e(str(r.path))}" style="background:{bg}" data-all="{self._e(blob)}">
  <td class="tdCell">
    <div class="kwcell" style="padding-left:{8 + depth*16}px">
      <span class="twistyWrap"></span>
      <span class="kwtext">{self._e(keyword)}</span>
    </div>
  </td>
  <td class="tdCell mono">{self._e(tag_hex)}</td>
  <td class="tdCell mono">{self._e(vr_display)}</td>

  <td class="tdCell valCell {left_class}"
      data-vr="{self._e(vr_for_parse)}"
      data-raw="{self._e(lp['raw'])}"
      data-length="{self._e(lp['blen'])}"
      data-vm="{self._e(lp['vm'])}"
      data-missing="{1 if left_missing else 0}">
    <span class="copypill">Copied</span>
  </td>

  <td class="tdCell valCell {right_class}"
      data-vr="{self._e(vr_for_parse)}"
      data-raw="{self._e(rp['raw'])}"
      data-length="{self._e(rp['blen'])}"
      data-vm="{self._e(rp['vm'])}"
      data-missing="{1 if right_missing else 0}">
    <span class="copypill">Copied</span>
  </td>
</tr>
"""
                )

        # children (shouldn't typically happen for non-seq leaves, but safe)
        for c in node.children:
            html_rows.append(self._render_node(c, depth + 1, row_no, export_rows=export_rows))
        return "".join(html_rows)

    # ------------------------ helpers ------------------------

    @staticmethod
    def _e(x: Any) -> str:
        return html.escape("" if x is None else str(x), quote=True)

    # _ITEM_RE = re.compile(r"^\[(\d+)\]$")  # “[0]” token in path

    @classmethod
    def _split_path(cls, path: str) -> List[str]:
        """
        Split a dotted path into tokens; preserve [N] as its own segment.
        Examples:
          "BeamSequence[0].GantryAngle" -> ["BeamSequence", "[0]", "GantryAngle"]
          "PatientName" -> ["PatientName"]
          "BeamSequence.VR" -> ["BeamSequence","VR"]
        """
        s = str(path)
        out: List[str] = []
        buf = ""
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == ".":
                if buf:
                    out.append(buf)
                    buf = ""
                i += 1
                continue
            buf += ch
            i += 1
        if buf:
            out.append(buf)
        # ensure “[N]” are separated if someone produced “...Sequence[0]”
        norm: List[str] = []
        for tok in out:
            m = re.match(r"^(.*?)(\[\d+\])$", tok)
            if m and m.group(1):
                norm.append(m.group(1))
                norm.append(m.group(2))
            else:
                norm.append(tok)
        return norm

    # -------- model → HTML table helpers --------
    def _render_groups_as_table(
        self, groups: List[Dict[str, Any]], depth: int = 0, row_no: List[int] = None
    ) -> str:
        if row_no is None:
            row_no = [0]
        out: List[str] = []
        for g in groups:
            # group header row
            row_no[0] += 1
            even = row_no[0] % 2 == 1
            row_bg = "var(--panel)" if even else "transparent"
            out.append(
                f"""
<tr class="row" data-depth="{depth}" data-seq="1" style="background:{row_bg}">
  <td class="tdCell">
    <div class="kwcell" style="padding-left:{8 + depth*16}px">
      <span class="twistyWrap"><button class="twistyBtn" data-open="1" title="Collapse">–</button></span>
      <span class="kwtext">{self._e(g['name'])}</span>
    </div>
  </td>
  <td class="tdCell mono">{self._e(g['name'])}</td>
  <td class="tdCell"></td>
  <td class="tdCell"></td>
  <td class="tdCell"></td>
</tr>"""
            )
            # rows in this group
            for r in g["rows"]:
                row_no[0] += 1
                even = row_no[0] % 2 == 1
                row_bg = "var(--panel)" if even else "transparent"
                search_blob = f"{r['path']} {r['note']} {r['severity']} {r['left']} {r['right']}"
                out.append(
                    f"""
<tr class="row rowsev-{self._e(r['severity'])}" data-depth="{depth+1}" data-seq="0" data-all="{self._e(search_blob)}" style="background:{row_bg}">
  <td class="tdCell">
    <div class="kwcell" style="padding-left:{8 + (depth+1)*16}px">
      <span class="twistyWrap"></span>
      <span class="kwtext">{self._e(r['keyword'])}</span>
    </div>
  </td>
  <td class="tdCell mono">{self._e(r['path'])}</td>
  <td class="tdCell valCell">{self._e(r['left'])}</td>
  <td class="tdCell valCell">{self._e(r['right'])}</td>
  <td class="tdCell"><span class="sev {self._e(r['severity'])}">{self._e(r['severity'])}</span> — <span class="mono">{self._e(r['note'])}</span></td>
</tr>"""
                )
            # children groups (if any)
            if g.get("children"):
                out.append(self._render_groups_as_table(g["children"], depth + 1, row_no))
        return "".join(out)

    # -------- utilities --------

    @staticmethod
    def _last_segment(path: str) -> str:
        parts = str(path).split(".")
        return parts[-1] if parts else str(path)

    @staticmethod
    def _first_segment(path: str) -> str:
        return str(path).split(".", 1)[0]

    @staticmethod
    def _maybe_tag(seg: str) -> Optional[str]:
        s = str(seg).strip()
        return s if re.match(r"^\([0-9A-Fa-f]{4},[0-9A-Fa-f]{4}\)$", s) else None

    @staticmethod
    def _preview_cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (bytes, bytearray)):
            return "[binary]"
        s = str(v)
        return s[:297] + "…" if len(s) > 300 else s

    def _group_hierarchy(self, flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # first segment grouping; order groups and rows for stable render
        groups: Dict[str, Dict[str, Any]] = {}
        for r in flat_rows:
            g = r["group"]
            bucket = groups.setdefault(g, {"name": g, "rows": [], "children": [], "counts": {}})
            bucket["rows"].append(r)
        # sort rows within group
        for g in groups.values():
            g["rows"].sort(key=lambda r: (r["keyword"], r["path"]))
            # severity counts per group (optional)
            counts = {"diff": 0, "warn": 0, "info": 0, "ok": 0}
            for r in g["rows"]:
                if r["severity"] in counts:
                    counts[r["severity"]] += 1
            g["counts"] = counts
        # sort groups
        ordered = [groups[k] for k in sorted(groups.keys(), key=str.lower)]
        return ordered


# ============================ Node model ============================


class _Node:
    __slots__ = ("name", "children", "rows", "vr_left", "vr_right", "has_vr_diff_flag")

    def __init__(self, name: str):
        self.name = name  # "BeamSequence" or "[0]" or "GantryAngle" or "root"
        self.children: List["_Node"] = []
        self.rows: List["Diff"] = []
        # presentation metadata
        self.vr_left: str = ""
        self.vr_right: str = ""
        self.has_vr_diff_flag: bool = False  # present when ".VR" row existed

    # building
    def ensure_path(self, parts: List[str]) -> "_Node":
        if not parts:
            return self
        head, tail = parts[0], parts[1:]
        child = self._child_named(head)
        if child is None:
            child = _Node(head)
            self.children.append(child)
        return child.ensure_path(tail)

    def _child_named(self, name: str) -> Optional["_Node"]:
        for c in self.children:
            if c.name == name:
                return c
        return None

    # traversal
    def walk(self, fn):
        fn(self)
        for c in self.children:
            c.walk(fn)

    # semantics
    def is_item_row(self) -> bool:
        return bool(re.match(r"^\[\d+\]$", self.name))

    def is_sequence(self) -> bool:
        # Heuristics: a non-root node is "sequence" if it has children and name isn't an “[N]” item.
        return (self.name != "root") and (not self.is_item_row()) and (len(self.children) > 0)

    # display helpers
    def display_keyword(self) -> str:
        if self.is_item_row():
            # Render as "Item N"
            n = re.findall(r"\d+", self.name)[0]
            return f"Item {n}"
        return self.name or ""

    def keyword_candidate(self) -> Optional[str]:
        # For VR/tag inference; items and 'VR' nodes excluded.
        if self.is_item_row() or self.name in ("root", "VR"):
            return None
        # If looks like a tag "(0010,0010)" skip keyword lookup
        if re.match(r"^\([0-9A-Fa-f]{4},[0-9A-Fa-f]{4}\)$", self.name):
            return None
        return self.name

    def tag_hex_guess(self) -> str:
        # If name is already a literal tag like "(0010,0010)", keep it.
        if re.match(r"^\([0-9A-Fa-f]{4},[0-9A-Fa-f]{4}\)$", self.name or ""):
            return self.name

        kw = self.keyword_candidate()
        if not kw:
            return ""

        try:
            tg = tag_for_keyword(kw)
            if tg is None:
                return ""
            # Use pydicom’s canonical formatting: "(GGGG,EEEE)"
            return str(Tag(tg))
        except Exception:
            return ""

    def display_vr(self) -> str:
        # Prefer an explicit .VR mismatch we captured; else fall back to an inferred common VR.
        if self.vr_left and self.vr_right and self.vr_left == self.vr_right:
            return self.vr_left
        if self.vr_left and self.vr_right and self.vr_left != self.vr_right:
            return f"{self.vr_left} ≠ {self.vr_right}"
        return self.vr_left or self.vr_right or ""

    # value cells + coloring
    def _cell_classes(self, r: Optional["Diff"], side: str) -> str:
        """
        Decide CSS class for a value TD on a given side:
          - missing element (left/right value looks empty / None) => black pill
          - diff severity => red-ish
          - warn severity => orange-ish
          - else ok
        """
        val = "" if r is None else (r.left if side == "left" else r.right)
        missing = (val is None) or (str(val) == "") or (str(val) == "None")
        if missing:
            return "cell-missing"
        if r is None:
            return "cell-ok"
        if r.severity == "diff":
            return "cell-diff"
        if r.severity == "warn":
            return "cell-warn"
        return "cell-ok"

    def _cell_value(self, val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, (bytes, bytearray)):
            return "[binary]"
        s = str(val)
        return s if len(s) <= 500 else (s[:497] + "…")

    def value_cells_for(self, r: Optional["Diff"]) -> Tuple[str, str, str, str]:
        left_val = "" if r is None else self._cell_value(r.left)
        right_val = "" if r is None else self._cell_value(r.right)
        return left_val, right_val, self._cell_classes(r, "left"), self._cell_classes(r, "right")

    def sequence_value_cells(self) -> Tuple[str, str, str, str]:
        """
        For sequence or item header rows, show 'Items: N' where applicable.
        If we have leaf rows attached directly (rare), fall back to first row values.
        """
        if self.is_item_row():
            # Item rows: don't show values; children carry the fields
            return "", "", "cell-ok", "cell-ok"
        # Sequence header: show item count (children that are items)
        item_children = [c for c in self.children if c.is_item_row()]
        if item_children:
            n = len(item_children)
            return f"Items: {n}", f"Items: {n}", "cell-ok", "cell-ok"
        # Fallback
        if self.rows:
            return self.value_cells_for(self.rows[0])
        return "", "", "cell-ok", "cell-ok"

    def iter_subtree_rows(self):
        # Yeild all Diff rows under this node (including this node's rows)
        for r in getattr(self, "rows", []):
            yield r
        for c in getattr(self, "children", []):
            yield from c.iter_subtree_rows()

    def severity_counts(self):
        # Count severities for all descendant rows (excluding .VR rows)
        counts = {"diff": 0, "warn": 0, "info": 0, "ok": 0}
        for r in self.iter_subtree_rows():
            if str(r.path).endswith(".VR"):
                continue
            if r.severity in counts:
                counts[r.severity] += 1
        return counts


# --------------------------- Helpers ---------------------------
SEQ_VRS = {"SQ"}


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


def tag_hex(tag: Union[Tag, int, tuple]) -> str:
    t = Tag(tag)
    return f"({int(t.group):04x},{int(t.element):04x})".lower()


def elem_display_value(e: Optional[DataElement]) -> str:
    if e is None:
        return None
    v = e.value
    # bytes/bulk keep short markers; else stringify
    if isinstance(v, (bytes, bytearray)):
        return ""  # value present but binary; JS will show "[binary]" based on VR
    return v if v is not None else ""  # present element with empty value


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
        if (e1 is None) or (e2 is None):
            if (e1 is None) and (e2 is None):
                # Shouldn't happen because tags is the union, but guard anyway.
                rep.add(path, None, None, note="both missing")
            elif e1 is None:
                # Left dataset lacks the element
                rep.add(path, None, elem_display_value(e2), note="left missing")
            else:  # e2 is None
                # Right dataset lacks the element
                rep.add(path, elem_display_value(e1), None, note="right missing")

            # rep.add(
            #     path,
            #     elem_display_value(e1),
            #     elem_display_value(e2),
            #     note="element presence differs",
            # )
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
        # left has item, right lacks it
        rep.add(f"{prefix}item[key={k}]", "present", None, note="unmatched item (right missing)")

        # rep.add(
        #     f"{prefix}item[key={k}]", "present", "absent", note="unmatched item (right missing)"
        # )
    for k in sorted(keys2 - keys1):
        # right has item, left lacks it
        rep.add(f"{prefix}item[key={k}]", None, "present", note="unmatched item (left missing)")
        # rep.add(
        #     f"{prefix}item[key={k}]", "absent", "present", note="unmatched item (left missing)"
        # )

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
    - Handles empty YAML (safe_load -> None) by treating it as {}.
    - Ensures base_opts is initialized before reading defaults from it.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config. Install with `pip install pyyaml`."
        ) from e

    # 1) Load file; treat empty file as {}
    cfg = yaml.safe_load(Path(path).read_text()) or {}

    # 2) Ensure base_opts is usable before reading defaults from it
    if base_opts is None:
        base_opts = DiffOptions()

    # 3) Scalars / simple lists
    ignore_private = bool(cfg.get("ignore_private", base_opts.ignore_private))
    ignore_bulk = bool(cfg.get("ignore_bulk", base_opts.ignore_bulk))
    ci = bool(cfg.get("case_insensitive_strings", base_opts.case_insensitive_strings))
    tol = float(cfg.get("numeric_tol", base_opts.numeric_tol.abs_tol))
    ignore_list = list(cfg.get("ignore", base_opts.ignore_tokens or []))

    # 4) Sequence keys + fallback
    seq_keys: Dict[str, List[str]] = dict(base_opts.sequence_keys or {})
    for k, v in (cfg.get("sequence_keys") or {}).items():
        seq_keys[str(k)] = [str(x) for x in v]

    seq_fallback = str(cfg.get("sequence_fallback", base_opts.sequence_fallback)).lower()
    if seq_fallback not in {"order", "report"}:
        seq_fallback = "order"

    # 5) OK rows / HTML controls
    collect_all = bool(cfg.get("collect_all_matches", base_opts.collect_all_matches))
    max_ok = int(cfg.get("max_ok_rows", base_opts.max_ok_rows))
    show_for = list(cfg.get("show_matches_for", base_opts.show_matches_for or []))

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
    p.add_argument(
        "--theme",
        type=str,
        default="dark",
        help="Built-in theme name (e.g., light, dark, nord, dracula, solarizedDark, etc.)",
    )
    p.add_argument(
        "--theme-json", type=str, help="Path to JSON file containing custom ThemeTokens."
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

    custom_tokens = None
    if args.theme_json:
        try:
            custom_tokens = json.loads(Path(args.theme_json).read_text())
            assert isinstance(custom_tokens, dict)
        except Exception as e:
            raise SystemExit(f"Failed to read --theme-json: {e}")

    if args.html:
        page = rep.write_html(
            args.html, title=args.title, theme=args.theme if not custom_tokens else args.theme
        )
        if custom_tokens:
            # rebuild CSS vars with custom tokens and inject
            css_vars = _theme_tokens_to_css_vars(_resolve_theme_tokens(None, custom_tokens))
            page = page.replace("<style>", f"<style>\n{css_vars}\n", 1)
        Path(args.html).write_text(page, encoding="utf-8")
        # still print JSON or text if requested, otherwise print a short confirmation
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2, default=str))
        else:
            print(f"HTML report written to: {page}")
        return

    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(rep.to_text())


if __name__ == "__main__":
    _cli()
