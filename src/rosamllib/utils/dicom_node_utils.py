from __future__ import annotations
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Set, Union

import pandas as pd

if TYPE_CHECKING:
    from rosamllib.nodes import DatasetNode, PatientNode, SeriesNode, InstanceNode

    NodeT = Union["SeriesNode", "InstanceNode"]


def get_referenced_nodes(
    node: Union[SeriesNode, InstanceNode],
    modality: Optional[Union[str, Iterable[str]]] = None,
    level: str = "INSTANCE",
    recursive: bool = True,
    include_start: bool = False,
) -> List[Union[SeriesNode, InstanceNode]]:
    """
    Return referenced nodes of a specified level (INSTANCE|SERIES), optionally filtered
    by modality.
    - Direct neighbors are always considered; if `recursive=False`, stop after depth=1.
    - If `recursive=True`, traverse further (no depth limit).
    - `include_start=False` by default (doesn't include the input `node` in results).
    """
    from rosamllib.nodes import SeriesNode, InstanceNode

    def norm_modalities(m) -> Optional[Set[str]]:
        if m is None:
            return None
        if isinstance(m, str):
            return {m.upper()}
        return {str(x).upper() for x in m}

    def modality_ok(obj) -> bool:
        if wanted is None:
            return True
        mod = getattr(obj, "Modality", None)
        return (mod or "").upper() in wanted

    def maybe_add(n):
        if level == "INSTANCE" and isinstance(n, InstanceNode) and modality_ok(n):
            out.append(n)
        elif level == "SERIES" and isinstance(n, SeriesNode) and modality_ok(n):
            out.append(n)

    level = level.upper()
    if level not in {"INSTANCE", "SERIES"}:
        raise ValueError("level must be 'INSTANCE' or 'SERIES'")

    wanted = norm_modalities(modality)
    out: List[Union[SeriesNode, InstanceNode]] = []
    seen: Set[int] = set()

    # BFS
    q = deque()
    # depth=0 is the start node; we still may include it if requested
    q.append((node, 0))
    if include_start:
        maybe_add(node)

    max_depth = None if recursive else 1

    while q:
        n, d = q.popleft()
        nid = id(n)
        if nid in seen:
            continue
        seen.add(nid)

        # collect (except the start if include_start=False)
        if d > 0:
            maybe_add(n)

        # stop expanding if we've hit the depth limit
        if max_depth is not None and d >= max_depth:
            continue

        # neighbors
        if isinstance(n, SeriesNode):
            # 1) direct series->series links (e.g., REG/SEG edges resolved at series level)
            for s in getattr(n, "referenced_series", []) or []:
                q.append((s, d + 1))
            # 2) go through instances to follow instance-level links
            for inst in getattr(n, "instances", {}).values():
                # instance->instance
                for ref in getattr(inst, "referenced_instances", []) or []:
                    q.append((ref, d + 1))
                # instance->series
                for s in getattr(inst, "referenced_series", []) or []:
                    q.append((s, d + 1))

        elif isinstance(n, InstanceNode):
            # instance->instance
            for ref in getattr(n, "referenced_instances", []) or []:
                q.append((ref, d + 1))
            # instance->series
            for s in getattr(n, "referenced_series", []) or []:
                q.append((s, d + 1))

    # de-dup while preserving order (by id)
    seen_ids = set()
    deduped = []
    for x in out:
        xid = id(x)
        if xid not in seen_ids:
            seen_ids.add(xid)
            deduped.append(x)

    return deduped


def get_referencing_nodes(
    node: Union[SeriesNode, InstanceNode],
    modality: Optional[Union[str, Iterable[str]]] = None,
    level: str = "INSTANCE",
    recursive: bool = True,
    include_start: bool = False,
) -> List[Union[SeriesNode, InstanceNode]]:
    """
    Return nodes that share the same FrameOfReferenceUID as the given node.

    Parameters
    ----------
    node : SeriesNode | InstanceNode
        Anchor node. If an InstanceNode is provided, its parent SeriesNode is used.
    level : {'SERIES', 'INSTANCE'}, default 'SERIES'
        - 'SERIES': return SeriesNode peers in the same Frame of Reference (FoR).
        - 'INSTANCE': return InstanceNode peers from all series in the same FoR.
        Note: with 'INSTANCE' and include_self=False, result can be empty if
        there are no peer series (i.e., anchor is the only series in its FoR).
    include_self : bool, default False
        Include the anchor in the results:
        - 'SERIES': include the anchor series.
        - 'INSTANCE': include all instances from the anchor series.
    modality : str or Iterable[str], optional
        Case-insensitive modality filter.
        - For level='SERIES', filters by `series.Modality` (e.g., 'CT', 'MR').
        - For level='INSTANCE', filters by `instance.Modality` (e.g., 'CT', 'RTDOSE').
        You may pass a single string (e.g., "CT") or an iterable (e.g., ["CT","MR"]).

    Returns
    -------
    list[SeriesNode] | list[InstanceNode]
        Peers in the same Frame of Reference, filtered by level/modality.

    Notes
    -----
    - This method prefers the precomputed `series.frame_of_reference_registered`
    filled during `_associate_dicoms`. If that list is empty, it falls back
    to scanning `self.dicom_files`.
    - Passing a single string for `modality` is supported and treated as a set
    with one element (e.g., "CT" -> {"CT"}).

    Examples
    --------
    >>> # Series peers (CT or MR) sharing the same FoR as a dose's CT
    >>> peers = loader.get_frame_registered_nodes(dose.parent_series,
    ...                                           level="SERIES",
    ...                                           modality=["CT","MR"])
    >>> # All RTDOSE instances within the same FoR (including the anchor series)
    >>> doses = loader.get_frame_registered_nodes(ct_series,
    ...                                           level="INSTANCE",
    ...                                           include_self=True,
    ...                                           modality="RTDOSE")
    """

    from rosamllib.nodes import SeriesNode, InstanceNode

    def norm_modalities(m) -> Optional[Set[str]]:
        if m is None:
            return None
        if isinstance(m, str):
            return {m.upper()}
        return {str(x).upper() for x in m}

    def modality_ok(obj) -> bool:
        if wanted is None:
            return True
        mod = getattr(obj, "Modality", None)
        return (mod or "").upper() in wanted

    def maybe_add(n):
        if level == "INSTANCE" and isinstance(n, InstanceNode) and modality_ok(n):
            out.append(n)
        elif level == "SERIES" and isinstance(n, SeriesNode) and modality_ok(n):
            out.append(n)

    def enqueue(nei, depth):
        if nei is None:
            return
        q.append((nei, depth))

    level = level.upper()
    if level not in {"INSTANCE", "SERIES"}:
        raise ValueError("level must be 'INSTANCE' or 'SERIES'")

    wanted = norm_modalities(modality)
    out: List[Union[SeriesNode, InstanceNode]] = []
    seen: Set[int] = set()
    q = deque()
    q.append((node, 0))
    if include_start:
        maybe_add(node)

    max_depth = None if recursive else 1

    while q:
        n, d = q.popleft()
        nid = id(n)
        if nid in seen:
            continue
        seen.add(nid)

        # collect (except depth 0 unless include_start)
        if d > 0:
            maybe_add(n)

        # stop expanding if depth cap
        if max_depth is not None and d >= max_depth:
            continue

        # ---- incoming neighbors ----
        if isinstance(n, InstanceNode):
            # instances that reference this instance
            for rin in getattr(n, "referencing_instances", []) or []:
                enqueue(rin, d + 1)
                # their parent series are also referrers at the series level
                enqueue(getattr(rin, "parent_series", None), d + 1)

            # series that reference this instance directly (if you maintain such a list)
            # Not standard in your model; typically we discover series via the instances above.

            # the instance's parent series might be referenced by other series;
            # climb to series and continue
            ps = getattr(n, "parent_series", None)
            if ps is not None:
                # series that reference this series (if populated)
                for rs in getattr(ps, "referencing_series", []) or []:
                    enqueue(rs, d + 1)

        elif isinstance(n, SeriesNode):
            # series that reference this series (if populated)
            for rs in getattr(n, "referencing_series", []) or []:
                enqueue(rs, d + 1)

            # instances that reference any instance within this series
            for inst in getattr(n, "instances", {}).values():
                for rin in getattr(inst, "referencing_instances", []) or []:
                    enqueue(rin, d + 1)
                    enqueue(getattr(rin, "parent_series", None), d + 1)

    # stable de-dup by object id
    uniq_ids = set()
    deduped = []
    for x in out:
        xid = id(x)
        if xid not in uniq_ids:
            uniq_ids.add(xid)
            deduped.append(x)

    return deduped


def get_frame_registered_nodes(
    node: Union[SeriesNode, InstanceNode],
    *,
    level: str = "SERIES",
    include_self: bool = False,
    modality: Optional[Union[str, Iterable[str]]] = None,
    dicom_files: Optional[Dict[str, Dict[str, SeriesNode]]] = None,
    derive_frame_from_references: bool = True,
) -> List[Union[SeriesNode, InstanceNode]]:
    """
    Return nodes that share at least one effective FrameOfReferenceUID with the anchor.

    Effective FoR of a series is the union of:
    - series.FrameOfReferenceUID
    - (if derive_frame_from_references) any inst.FrameOfReferenceUIDs
    - (if derive_frame_from_references) FoR of any series referenced by its instances
    """
    from rosamllib.nodes import SeriesNode, InstanceNode

    def _wanted_set(m):
        if m is None:
            return None
        return {m.upper()} if isinstance(m, str) else {str(x).upper() for x in m}

    def _series_mod_ok(s):
        return wanted is None or (getattr(s, "Modality", None) or "").upper() in wanted

    def _inst_mod_ok(i):
        return wanted is None or (getattr(i, "Modality", None) or "").upper() in wanted

    def _effective_fors(series: SeriesNode) -> set[str]:
        fors: set[str] = set()
        fo_direct = getattr(series, "FrameOfReferenceUID", None)
        if fo_direct:
            fors.add(str(fo_direct))
        if derive_frame_from_references:
            for inst in getattr(series, "instances", {}).values():
                for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                    if u:
                        fors.add(str(u))
                for rs in getattr(inst, "referenced_series", []) or []:
                    u = getattr(rs, "FrameOfReferenceUID", None)
                    if u:
                        fors.add(str(u))
        return fors

    lvl = str(level).upper()
    if lvl not in {"SERIES", "INSTANCE"}:
        raise ValueError("level must be 'SERIES' or 'INSTANCE'")

    wanted = _wanted_set(modality)

    # Anchor series + anchor FoR set
    anchor_series = node if isinstance(node, SeriesNode) else getattr(node, "parent_series", None)
    anchor_fors: set[str] = set()
    if isinstance(node, InstanceNode) and getattr(node, "FrameOfReferenceUIDs", None):
        anchor_fors |= {str(u) for u in (node.FrameOfReferenceUIDs or []) if u}
    if anchor_series and getattr(anchor_series, "FrameOfReferenceUID", None):
        anchor_fors.add(str(anchor_series.FrameOfReferenceUID))
    # If still empty and we’re allowed to derive, derive from anchor series’ instances
    if not anchor_fors and anchor_series and derive_frame_from_references:
        anchor_fors |= _effective_fors(anchor_series)

    if not anchor_fors:
        # no FoR context — return only self if requested
        if include_self:
            if lvl == "SERIES" and isinstance(node, SeriesNode) and _series_mod_ok(node):
                return [node]
            if lvl == "INSTANCE" and isinstance(node, InstanceNode) and _inst_mod_ok(node):
                return [node]
        return []

    # Collect peer series: intersection of effective FoRs with anchor_fors
    peer_series: list[SeriesNode] = []
    seen_sid: set[int] = set()

    # Prefer dicom_files when provided (covers RTSTRUCT/SEG/REG cases correctly)
    if dicom_files:
        for _pid, sdict in dicom_files.items():
            for s in sdict.values():
                if anchor_series is not None and s is anchor_series:
                    continue
                eff = _effective_fors(s)
                if eff and (eff & anchor_fors):
                    if id(s) not in seen_sid:
                        peer_series.append(s)
                        seen_sid.add(id(s))
    else:
        # Fallback to precomputed FoR neighbors (series-level only; may miss RTSTRUCT)
        if anchor_series:
            for s in list(getattr(anchor_series, "frame_of_reference_registered", []) or []):
                if id(s) in seen_sid:
                    continue
                # verify intersection using effective FoRs to avoid false negatives
                if _effective_fors(s) & anchor_fors:
                    peer_series.append(s)
                    seen_sid.add(id(s))

    if lvl == "SERIES":
        out = []
        if include_self and anchor_series and _series_mod_ok(anchor_series):
            out.append(anchor_series)
        out.extend([s for s in peer_series if _series_mod_ok(s)])
        # de-dup
        seen = set()
        dedup = []
        for x in out:
            if id(x) not in seen:
                seen.add(id(x))
                dedup.append(x)
        return dedup

    # INSTANCE level: return instances from anchor (optional) + peer series
    out_i: list[InstanceNode] = []

    def add_series(series: SeriesNode):
        for inst in getattr(series, "instances", {}).values():
            if _inst_mod_ok(inst):
                out_i.append(inst)

    if include_self and anchor_series:
        add_series(anchor_series)
    for s in peer_series:
        add_series(s)

    seen = set()
    dedup = []
    for x in out_i:
        if id(x) not in seen:
            seen.add(id(x))
            dedup.append(x)
    return dedup


def get_frame_registered_clusters(
    dicom_files,
    *,
    scope: str = "dataset",  # 'dataset' | 'patient' | 'study'
    patient_id: Optional[str] = None,
    study_uid: Optional[str] = None,
    modality: Optional[Union[str, Iterable[str]]] = None,
    include_missing_for: bool = False,
    derive_frame_from_references: bool = True,  # <— generic & True by default
    min_cluster_size: int = 1,
) -> Dict[str, List[SeriesNode]]:
    """
    Group series by FrameOfReferenceUID (FoR) within the chosen scope.

    Scope
    -----
    - 'dataset': consider all series in `self.dicom_files` (all patients/studies).
    - 'patient': consider only series under `patient_id`.
    - 'study'  : consider only series under (`patient_id`, `study_uid`).
                Cross-study series with the same FoR are excluded by design.

    Parameters
    ----------
    scope : {'dataset','patient','study'}, default 'dataset'
    patient_id : str, optional
        Required when scope is 'patient' or 'study'.
    study_uid : str, optional
        Required when scope is 'study'.
    modality : str or Iterable[str], optional
        Case-insensitive modality filter applied to `series.Modality`.
        Accepts a single string (e.g., "CT") or an iterable (e.g., ["CT","MR"]).
    include_missing_for : bool, default False
        If True, include series with no resolvable FoR under key '<MISSING>'.
    derive_frame_from_references : bool, default True
        When True, derive additional FoR memberships for a series by inspecting:
        (1) each instance's `InstanceNode.FrameOfReferenceUIDs`, and
        (2) each instance's `referenced_series.FrameOfReferenceUID`.
        The final FoR set is the union of the series' own FoR (if any) and the derived FoRs.
        A series may therefore appear in multiple clusters (e.g., RTSTRUCT that references
        multiple FoRs). When False, only the series' own FoR is used.
    min_cluster_size : int, default 1
        Drop clusters smaller than this size (by number of series).

    Returns
    -------
    dict[str, list[SeriesNode]]
        Map FoR UID -> list of SeriesNodes (sorted by Modality then SeriesInstanceUID).
        If `include_missing_for` is True, includes key '<MISSING>' when no FoR is found.

    Notes
    -----
    - With `derive_frame_from_references=True` (default), RTSTRUCT or any modality that carries
    or references multiple FoRs can appear in multiple FoR clusters, which matches DICOM
    allowances.
    - With `derive_frame_from_references=False`, clustering is a strict series-level FoR
    grouping.

    Examples
    --------
    >>> # dataset-wide clusters (CT only)
    >>> clusters = loader.get_frame_registered_clusters(modality="CT")
    >>> # patient-scope clusters (CT or MR)
    >>> clusters_p = loader.get_frame_registered_clusters(scope="patient",
    ...                                                  patient_id="P001",
    ...                                                  modality=["CT","MR"])
    >>> # study-scope clusters (no cross-study FoR pulling)
    >>> clusters_s = loader.get_frame_registered_clusters(scope="study",
    ...                                                  patient_id="P001",
    ...                                                  study_uid="1.2.3")
    >>> # derive FoR from references (default True) and keep only clusters with ≥2 series
    >>> clusters_rt = loader.get_frame_registered_clusters(scope="patient",
    ...     patient_id="P001", min_cluster_size=2)
    """

    def _wanted_set(m: Optional[Union[str, Iterable[str]]]) -> Optional[Set[str]]:
        if m is None:
            return None
        if isinstance(m, str):
            return {m.upper()}
        return {str(x).upper() for x in m}

    def mod_ok(s: SeriesNode) -> bool:
        return wanted is None or (getattr(s, "Modality", None) or "").upper() in wanted

    wanted = _wanted_set(modality)

    scope_l = scope.lower()
    if scope_l == "dataset":
        series_iter = (s for _pid, sdict in (dicom_files or {}).items() for s in sdict.values())
    elif scope_l == "patient":
        if not patient_id:
            raise ValueError("patient_id is required when scope='patient'")
        series_iter = iter((dicom_files.get(patient_id, {}) or {}).values())
    elif scope_l == "study":
        if not (patient_id and study_uid):
            raise ValueError("patient_id and study_uid are required when scope='study'")
        series_iter = (
            s
            for s in (dicom_files.get(patient_id, {}) or {}).values()
            if getattr(s, "StudyInstanceUID", None) == study_uid
        )
    else:
        raise ValueError("scope must be one of {'dataset','patient','study'}")

    groups: Dict[str, List[SeriesNode]] = defaultdict(list)

    for s in series_iter:
        if not mod_ok(s):
            continue

        # Start with the series' own FoR (if any)
        fors: Set[str] = set()
        fo_direct = getattr(s, "FrameOfReferenceUID", None)
        if fo_direct:
            fors.add(str(fo_direct))

        if derive_frame_from_references:
            # Add FoRs from instances
            for inst in getattr(s, "instances", {}).values():
                # Instance-declared FoRs (multi-FoR friendly, e.g., RTSTRUCT)
                for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                    if u:
                        fors.add(str(u))
                # FoRs from explicitly referenced series
                for rs in getattr(inst, "referenced_series", []) or []:
                    u = getattr(rs, "FrameOfReferenceUID", None)
                    if u:
                        fors.add(str(u))

        if fors:
            for u in fors:
                groups[u].append(s)
        else:
            if include_missing_for:
                groups["<MISSING>"].append(s)

    # Sort and filter by min_cluster_size
    for k in list(groups.keys()):
        groups[k] = sorted(
            groups[k], key=lambda x: ((x.Modality or ""), (x.SeriesInstanceUID or ""))
        )
    if min_cluster_size > 1:
        groups = {k: v for k, v in groups.items() if len(v) >= min_cluster_size}

    return dict(groups)


def get_nodes_for_patient(
    patient_node,
    level="SERIES",
    modality=None,
    uid=None,
):
    """
    Retrieves StudyNode, SeriesNode, or InstanceNode objects from a given PatientNode.

    Parameters
    ----------
    patient_node : PatientNode
        The patient node to search under.

    level : str, optional
        One of {"STUDY", "SERIES", "INSTANCE"} (case-insensitive).
        Determines which level of nodes to return. Default is "SERIES".

    modality : str, optional
        If specified, filters nodes by Modality (only applicable for SERIES/INSTANCE levels).

    uid : str, optional
        If specified, filters for a specific UID:
        - For level="STUDY": matches StudyInstanceUID
        - For level="SERIES": matches SeriesInstanceUID
        - For level="INSTANCE": matches SOPInstanceUID

    Returns
    -------
    List[StudyNode | SeriesNode | InstanceNode]
        A list of matching nodes at the requested level.
        If `uid` is specified, returns at most one element.

    Raises
    ------
    ValueError
        If `level` is not one of {"STUDY", "SERIES", "INSTANCE"}.
    """
    level = level.upper()
    if level not in {"STUDY", "SERIES", "INSTANCE"}:
        raise ValueError("level must be 'STUDY', 'SERIES', or 'INSTANCE'")

    results = []

    for study_node in patient_node:
        if level == "STUDY":
            if uid and study_node.StudyInstanceUID != uid:
                continue
            results.append(study_node)

        elif level == "SERIES":
            for series_node in study_node:
                if uid and series_node.SeriesInstanceUID != uid:
                    continue
                if modality and series_node.Modality != modality:
                    continue
                results.append(series_node)

        elif level == "INSTANCE":
            for series_node in study_node:
                for instance_node in series_node:
                    if uid and instance_node.SOPInstanceUID != uid:
                        continue
                    if modality and instance_node.Modality != modality:
                        continue
                    results.append(instance_node)

    return results


def _norm_modalities(modality: Optional[Union[str, Iterable[str]]]) -> Optional[Set[str]]:
    if modality is None:
        return None
    if isinstance(modality, str):
        return {modality.upper()}
    return {str(x).upper() for x in modality}


def _level_ok(node: NodeT, level: str) -> bool:
    from rosamllib.nodes import SeriesNode, InstanceNode

    if level == "SERIES":
        return isinstance(node, SeriesNode)
    if level == "INSTANCE":
        return isinstance(node, InstanceNode)
    raise ValueError("level must be 'SERIES' or 'INSTANCE'")


def _modality_ok(node: NodeT, wanted: Optional[Set[str]]) -> bool:
    if wanted is None:
        return True
    mod = getattr(node, "Modality", None)
    return (mod or "").upper() in wanted


def _iter_nodes_patient(patient: "PatientNode", level: str) -> Iterable[NodeT]:
    if level == "SERIES":
        for st in patient.studies.values():
            for se in st:
                yield se
    elif level == "INSTANCE":
        for st in patient.studies.values():
            for se in st:
                for inst in se:
                    yield inst
    else:
        raise ValueError("level must be 'SERIES' or 'INSTANCE'")


def _neighbors_via(
    dataset_or_patient, method: str, n: NodeT, *, level: str, recursive: bool, include_start: bool
) -> List[NodeT]:
    """
    Dispatch to your existing static helpers (already in your code):
      - get_referenced_nodes(...)
      - get_referencing_nodes(...)
      - get_frame_registered_nodes(...)
    `dataset_or_patient` can be a PatientNode (preferred) or DatasetNode where those statics live.
    """
    if method == "referenced":
        return dataset_or_patient.get_referenced_nodes(
            n, modality=None, level=level, recursive=recursive, include_start=include_start
        )
    if method == "referencing":
        return dataset_or_patient.get_referencing_nodes(
            n, modality=None, level=level, recursive=recursive, include_start=include_start
        )
    if method == "frame":
        return dataset_or_patient.get_frame_registered_nodes(
            n, level=level, include_self=include_start, modality=None
        )
    raise ValueError("method must be one of {'references','referencing','frame'}")


def _same_for_any_target(
    dataset_or_patient, n: NodeT, *, level: str, is_target: Callable[[NodeT], bool]
) -> bool:
    peers = dataset_or_patient.get_frame_registered_nodes(
        n, level=level, include_self=False, modality=None
    )
    return any(is_target(x) for x in peers)


def _node_ids(n: NodeT) -> Dict[str, Optional[str]]:
    """Collect UIDs/IDs in a uniform way for Series vs Instance."""
    from rosamllib.nodes import SeriesNode

    if isinstance(n, SeriesNode):
        st = getattr(n, "parent_study", None)
        pt = getattr(st, "parent_patient", None) if st else None
        return {
            "PatientID": getattr(pt, "PatientID", None),
            "StudyInstanceUID": getattr(st, "StudyInstanceUID", None) if st else None,
            "SeriesInstanceUID": getattr(n, "SeriesInstanceUID", None),
            "SOPInstanceUID": None,
        }
    else:  # InstanceNode
        se = getattr(n, "parent_series", None)
        st = getattr(se, "parent_study", None) if se else None
        pt = getattr(st, "parent_patient", None) if st else None
        return {
            "PatientID": getattr(pt, "PatientID", None),
            "StudyInstanceUID": getattr(st, "StudyInstanceUID", None) if st else None,
            "SeriesInstanceUID": getattr(se, "SeriesInstanceUID", None) if se else None,
            "SOPInstanceUID": getattr(n, "SOPInstanceUID", None),
        }


def build_reachability_summary(
    dataset: "DatasetNode",
    *,
    start_level: str = "SERIES",
    start_modality: Optional[Union[str, Iterable[str]]] = None,
    target_level: str = "SERIES",
    target_modality: Optional[Union[str, Iterable[str]]] = None,
    traversal: str = "references",  # 'references' | 'referencing' | 'frame'
    recursive: bool = True,
    include_start: bool = False,
    # Optional extra filters
    start_predicate: Optional[Callable[[NodeT], bool]] = None,
    target_predicate: Optional[Callable[[NodeT], bool]] = None,
    # Fallback acceptance if no referenced target found (for references/referencing modes)
    allow_same_for: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build three DataFrames:
      - details_df: one row per start node (with reachability result & metadata)
      - patient_agg_df: counts per patient
      - global_agg_df: overall totals (+ per start modality if available)

    Returns
    -------
    dict with keys {'details_df', 'patient_agg_df', 'global_agg_df'} mapping to DataFrames.
    """
    from rosamllib.nodes import SeriesNode

    s_level = start_level.upper()
    t_level = target_level.upper()
    if s_level not in {"SERIES", "INSTANCE"} or t_level not in {"SERIES", "INSTANCE"}:
        raise ValueError("start_level/target_level must be 'SERIES' or 'INSTANCE'")
    trav = traversal.lower()
    if trav not in {"references", "referencing", "frame"}:
        raise ValueError("traversal must be one of {'references','referencing','frame'}")
    s_wanted = _norm_modalities(start_modality)
    t_wanted = _norm_modalities(target_modality)

    def is_start(n: NodeT) -> bool:
        return (
            _level_ok(n, s_level)
            and _modality_ok(n, s_wanted)
            and (start_predicate(n) if start_predicate else True)
        )

    def is_target(n: NodeT) -> bool:
        return (
            _level_ok(n, t_level)
            and _modality_ok(n, t_wanted)
            and (target_predicate(n) if target_predicate else True)
        )

    rows = []

    for patient in dataset:
        # Gather starts for this patient
        starts: List[NodeT] = [n for n in _iter_nodes_patient(patient, s_level) if is_start(n)]
        if not starts:
            continue

        for s in starts:
            # Expand neighbors using helpers on PatientNode (which hosts the static methods)
            neighbors = _neighbors_via(
                patient, trav, s, level=t_level, recursive=recursive, include_start=include_start
            )
            # Filter to targets
            targets = [x for x in neighbors if is_target(x)]
            num_targets = len(targets)

            reached = num_targets > 0
            used_same_for_fallback = False
            if (not reached) and allow_same_for and trav in {"references", "referencing"}:
                used_same_for_fallback = _same_for_any_target(
                    patient, s, level=t_level, is_target=is_target
                )
                reached = used_same_for_fallback

            meta = _node_ids(s)
            start_mod = getattr(s, "Modality", None)
            start_for = getattr(
                s if isinstance(s, SeriesNode) else getattr(s, "parent_series", None),
                "FrameOfReferenceUID",
                None,
            )

            rows.append(
                {
                    **meta,
                    "StartLevel": s_level,
                    "StartModality": start_mod,
                    "StartFoR": start_for,
                    "Traversal": trav,
                    "Recursive": bool(recursive),
                    "IncludeStart": bool(include_start),
                    "AllowSameFoR": bool(allow_same_for),
                    "TargetLevel": t_level,
                    "TargetModalities": None if t_wanted is None else sorted(t_wanted),
                    "TargetsFound": num_targets,
                    "Reached": bool(reached),
                    "ReachedViaFoR": bool(used_same_for_fallback),
                }
            )

    if not rows:
        # Return empty frames with consistent columns
        cols = [
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "StartLevel",
            "StartModality",
            "StartFoR",
            "Traversal",
            "Recursive",
            "IncludeStart",
            "AllowSameFoR",
            "TargetLevel",
            "TargetModalities",
            "TargetsFound",
            "Reached",
            "ReachedViaFoR",
        ]
        details_df = pd.DataFrame(columns=cols)
    else:
        details_df = pd.DataFrame(rows)

    # Patient-level aggregation
    if len(details_df) > 0:
        grp = details_df.groupby("PatientID", dropna=False)
        patient_agg_df = grp.agg(
            TotalStarts=("Reached", "size"),
            Failures=("Reached", lambda x: (~x).sum()),
        ).reset_index()
        patient_agg_df["FailureRate"] = (
            patient_agg_df["Failures"] / patient_agg_df["TotalStarts"]
        ).round(3)
    else:
        patient_agg_df = pd.DataFrame(
            columns=["PatientID", "TotalStarts", "Failures", "FailureRate"]
        )

    # Global aggregation (+ by start modality if present)
    if len(details_df) > 0:
        total = {
            "TotalStarts": int(len(details_df)),
            "Failures": int((~details_df["Reached"]).sum()),
        }
        total["FailureRate"] = round(total["Failures"] / max(1, total["TotalStarts"]), 3)
        by_mod = (
            details_df.assign(StartModality=details_df["StartModality"].fillna("UNKNOWN"))
            .groupby("StartModality")
            .agg(TotalStarts=("Reached", "size"), Failures=("Reached", lambda x: (~x).sum()))
            .reset_index()
        )
        by_mod["FailureRate"] = (by_mod["Failures"] / by_mod["TotalStarts"]).round(3)
        global_agg_df = pd.concat(
            [pd.DataFrame([{"StartModality": "__ALL__", **total}]), by_mod], ignore_index=True
        )
    else:
        global_agg_df = pd.DataFrame(
            columns=["StartModality", "TotalStarts", "Failures", "FailureRate"]
        )

    return {
        "details_df": details_df,
        "patient_agg_df": patient_agg_df,
        "global_agg_df": global_agg_df,
    }


def associate_dicoms(
    dicom_src: Union[DatasetNode, PatientNode, Dict[str, Dict[str, SeriesNode]]],
):
    """
    Associates DICOM files based on referenced SOPInstanceUIDs and SeriesInstanceUIDs.

    Accepts either:
      - DatasetNode
      - PatientNode
      - Legacy mapping: dict[patient_id -> dict[series_uid -> SeriesNode]]

    What this does:
      - Builds SOP/Series/FrameOfReference maps
      - Wires instance<->instance, instance->series, and series<->series edges
      - Populates reverse edges for efficient "referencing" queries
      - Ensures FoR connectivity is symmetric
      - NEW: RTSTRUCT InstanceNode.FrameOfReferenceUIDs becomes the union of FoRs parsed
        from the RTSTRUCT file plus FoRs of the referenced image series present.

    Notes
    -----
    - This mutates SeriesNode/InstanceNode attributes in-place (adds lists like
      `referenced_series`, `referencing_series`, `referenced_instances`, etc.).
    - Assumes SeriesInstanceUID is unique within a patient. If duplicates exist across
      studies for the same patient, the last one encountered will win in the dict.
    """
    from rosamllib.nodes import DatasetNode, PatientNode, SeriesNode

    # Normalization: make dicom_files -> Dict[pid -> Dict[sid -> SeriesNode]]
    def _collect_series_by_patient(src) -> Dict[str, Dict[str, "SeriesNode"]]:
        # Back-compat: already in the expected dict shape
        if isinstance(src, dict):
            return src

        out: Dict[str, Dict[str, SeriesNode]] = {}

        def _add_patient(p: PatientNode):
            # Support both attribute styles
            pid = getattr(p, "PatientID", None)
            if not pid:
                raise ValueError("PatientNode missing PatientID")

            series_map: Dict[str, SeriesNode] = {}

            # Prefer p.studies
            studies = getattr(p, "studies", None)
            if isinstance(studies, dict):
                study_iter = studies.values()
            elif studies is not None:
                study_iter = studies
            else:
                study_iter = p

            for study in study_iter:
                series_dict = getattr(study, "series", {})
                # If it's adict, iterate items; otherwise, assume dict-like or empty
                if isinstance(series_dict, dict):
                    for sid, series in series_dict.items():
                        sid = getattr(series, "SeriesInstanceUID", None)
                        if sid:
                            series_map[sid] = series

            out[pid] = series_map

        # DatasetNode?
        if isinstance(src, DatasetNode):
            patients = getattr(src, "patients", None)
            if patients is not None:
                # if dict: patients.values(); if list-like: iterate; otherwise, try 'for p in src'
                if isinstance(patients, dict):
                    for p in patients.values():
                        _add_patient(p)
                else:
                    for p in patients:
                        _add_patient(p)
                return out

        elif isinstance(src, PatientNode):

            # PatientNode?
            if getattr(src, "PatientID", None) is not None:
                _add_patient(src)
                return out

        else:
            # If we get here, it's not a suported type

            raise TypeError(
                "Unsupported input to _associate_dicoms: expected DatasetNode, PatientNode, "
                "or Dict[patient_id -> Dict[series_uid -> SeriesNode]]"
            )

    dicom_files = _collect_series_by_patient(dicom_src)
    # sidecar state for fast identity de-dup per list
    _seen_map = {}

    def _append_unique(lst, obj):
        if obj is None:
            return
        lid = id(lst)
        s = _seen_map.get(lid)
        if s is None:
            s = set(id(x) for x in lst)
            _seen_map[lid] = s
        oid = id(obj)
        if oid not in s:
            lst.append(obj)
            s.add(oid)

    # Build lookup maps
    sop_instance_uid_map = {}
    series_uid_map = {}
    frame_of_reference_uid_map = {}

    for _pid, series_dict in dicom_files.items():
        for sid, series in series_dict.items():
            series_uid_map[sid] = series

            # ensure series containers exist
            series.referencing_series = getattr(series, "referencing_series", [])
            series.referenced_series = getattr(series, "referenced_series", [])
            series.frame_of_reference_registered = getattr(
                series, "frame_of_reference_registered", []
            )

            for sop_uid, inst in series.instances.items():
                sop_instance_uid_map[sop_uid] = inst

                # ensure instance containers exist
                inst.referenced_instances = getattr(inst, "referenced_instances", [])
                inst.referencing_instances = getattr(inst, "referencing_instances", [])
                inst.referenced_series = getattr(inst, "referenced_series", [])
                inst.other_referenced_series = getattr(inst, "other_referenced_series", [])
                inst.referenced_sop_instance_uids = getattr(
                    inst, "referenced_sop_instance_uids", []
                )
                inst.referenced_sids = getattr(inst, "referenced_sids", [])
                inst.other_referenced_sids = getattr(inst, "other_referenced_sids", [])

                # NEW: normalize multi-FoR field on instance (esp. RTSTRUCT)
                inst.FrameOfReferenceUIDs = list(getattr(inst, "FrameOfReferenceUIDs", []) or [])

            # map FoR -> series list
            if getattr(series, "FrameOfReferenceUID", None):
                frame_of_reference_uid_map.setdefault(series.FrameOfReferenceUID, []).append(
                    series
                )

    # Wire references
    for _pid, series_dict in dicom_files.items():
        for sid, series in series_dict.items():
            for sop_uid, inst in series.instances.items():
                modality = getattr(inst, "Modality", None)

                # A) SOPInstanceUID links (instance -> instance)
                for ref_sop_uid in getattr(inst, "referenced_sop_instance_uids", []):
                    ref_inst = sop_instance_uid_map.get(ref_sop_uid)
                    if not ref_inst:
                        continue

                    _append_unique(inst.referenced_instances, ref_inst)
                    _append_unique(ref_inst.referencing_instances, inst)

                    # promote to series-level edges
                    src_series = inst.parent_series
                    dst_series = getattr(ref_inst, "parent_series", None)
                    if src_series and dst_series and src_series is not dst_series:
                        _append_unique(src_series.referenced_series, dst_series)
                        _append_unique(dst_series.referencing_series, src_series)
                        _append_unique(inst.referenced_series, dst_series)

                # B) Modality-specific aggregation of series links
                if modality in {"RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"}:
                    for ref_inst in inst.referenced_instances:
                        rs = ref_inst.parent_series
                        if rs:
                            _append_unique(inst.referenced_series, rs)
                            _append_unique(inst.referenced_sids, rs.SeriesInstanceUID)

                if modality == "REG":
                    for ref_sid in getattr(inst, "referenced_sids", []):
                        rs = series_uid_map.get(ref_sid)
                        if rs:
                            _append_unique(inst.referenced_series, rs)
                    for other_sid in getattr(inst, "other_referenced_sids", []):
                        rs = series_uid_map.get(other_sid)
                        if rs:
                            _append_unique(inst.other_referenced_series, rs)
                            _append_unique(inst.referenced_series, rs)

                if modality == "SEG":
                    for ref_sid in getattr(inst, "referenced_sids", []):
                        rs = series_uid_map.get(ref_sid)
                        if rs:
                            _append_unique(inst.referenced_series, rs)

                # C) Promote instance->series to series<->series (reverse edges)
                for rs in list(getattr(inst, "referenced_series", [])):
                    src_series = inst.parent_series
                    if src_series and rs and src_series is not rs:
                        _append_unique(src_series.referenced_series, rs)
                        _append_unique(rs.referencing_series, src_series)

                for rs in list(getattr(inst, "other_referenced_series", [])):
                    src_series = inst.parent_series
                    if src_series and rs and src_series is not rs:
                        _append_unique(src_series.referenced_series, rs)
                        _append_unique(rs.referencing_series, src_series)

                # D) NEW: RTSTRUCT multi-FoR reconciliation on the instance
                if (modality or "").upper() == "RTSTRUCT":
                    # FoRs from referenced image series present in this dataset
                    fors_from_series = {
                        s.FrameOfReferenceUID
                        for s in getattr(inst, "referenced_series", [])
                        if getattr(s, "FrameOfReferenceUID", None)
                    }
                    # FoRs parsed from the RTSTRUCT dataset earlier (loader populated)
                    fors_from_ds = set(inst.FrameOfReferenceUIDs or [])
                    # Union and store back on the instance
                    inst.FrameOfReferenceUIDs = sorted(
                        {str(u) for u in (fors_from_series | fors_from_ds) if u}
                    )

    # --- EFFECTIVE FrameOfReference connectivity (symmetric, series-level) ---

    # Reset precomputed neighbors
    for _pid, series_dict in dicom_files.items():
        for _sid, s in series_dict.items():
            s.frame_of_reference_registered = getattr(s, "frame_of_reference_registered", [])
            s.frame_of_reference_registered[:] = []

    # Build FoR -> [series] buckets directly
    for_to_series = {}  # str -> list[SeriesNode]

    for _pid, series_dict in dicom_files.items():
        for _sid, s in series_dict.items():
            fors = set()

            # series-level FoR
            fo = getattr(s, "FrameOfReferenceUID", None)
            if fo:
                fors.add(str(fo))

            # derive from instances + their referenced series
            for inst in getattr(s, "instances", {}).values():
                for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                    if u:
                        fors.add(str(u))
                for rs in getattr(inst, "referenced_series", []) or []:
                    u = getattr(rs, "FrameOfReferenceUID", None)
                    if u:
                        fors.add(str(u))

            # (optional) only set if the field exists on a non-slotted class
            if hasattr(s, "EffectiveFrameOfReferenceUIDs"):
                try:
                    setattr(s, "EffectiveFrameOfReferenceUIDs", sorted(fors))
                except Exception:
                    pass  # ignore if slotted

            # bucket this series by each FoR
            for u in fors:
                for_to_series.setdefault(u, []).append(s)

    # Dedup helper (by identity)
    _seen_map = {}

    def _append_unique(lst, obj):
        if obj is None:
            return
        lid = id(lst)
        seen = _seen_map.get(lid)
        if seen is None:
            seen = set(id(x) for x in lst)
            _seen_map[lid] = seen
        oid = id(obj)
        if oid not in seen:
            lst.append(obj)
            seen.add(oid)

    # Symmetric neighbors within each effective FoR bucket
    for u, series_list in for_to_series.items():
        n = len(series_list)
        for i in range(n):
            si = series_list[i]
            for j in range(n):
                if i == j:
                    continue
                _append_unique(si.frame_of_reference_registered, series_list[j])

    # Link unresolved raw links
    for _pid, series_dict in dicom_files.items():
        # print(f"relinking raw for {_pid}...({len(series_dict)=})")
        for sid, series in series_dict.items():
            raw_uid = getattr(series, "raw_series_reference_uid", None)
            if raw_uid:
                parent = series_dict.get(raw_uid)
                if parent:
                    series.raw_series_reference = parent


def relink_raw_series_for_patient(patient):
    # Build local SeriesInstanceUID -> SeriesNode map
    sid_map = {s.SeriesInstanceUID: s for study in patient for s in study}

    # # Prepare reverse children list on each potential parent
    # for s in sid_map.values():
    #     if not hasattr(s, "embedded_children"):
    #         s.embedded_children = []

    # Resolve each series' parent pointer
    for s in sid_map.values():
        if getattr(s, "is_embedded_in_raw", None):
            raw_uid = getattr(s, "raw_series_reference_uid", None)

            if raw_uid:
                parent = sid_map.get(raw_uid)
                if parent:
                    s.raw_series_reference = parent
                    # parent.embedded_children.append(s)
                else:
                    pass
        # # Fallback inference: majority vote from instances (if available)
        # if not raw_uid and getattr(s, "is_embedded_in_raw", False):
        #     votes = []
        #     for inst in getattr(s, "instances", {}).values():
        #         uid = getattr(inst, "raw_series_reference_uid", None)
        #         if uid:
        #             votes.append(uid)

        #     if votes:
        #         # Most frequent UID among instances
        #         raw_uid = max(set(votes), key=votes.count)
        #         s.raw_series_reference_uid = raw_uid  # persist back on the object


def relink_raw_series_after_load(dataset):
    """
    After DatasetNode.from_dict(), rebuild RAW -> child links using
    `series.raw_series_reference_uid` that we persisted in JSON.
    """

    # Build a lookup of all series by UID
    sid_map = {s.SeriesInstanceUID: s for s in dataset.iter_series()}

    for s in sid_map.values():
        raw_uid = getattr(s, "raw_series_reference_uid", None)
        if raw_uid:
            parent = sid_map.get(raw_uid)
            if parent:
                s.raw_series_reference = parent


def link_raw_objects(node: Union[DatasetNode, PatientNode]):
    from rosamllib.nodes import DatasetNode, PatientNode

    if isinstance(node, DatasetNode):
        sid_map = {s.SeriesInstanceUID: s for s in node.iter_series()}
    elif isinstance(node, PatientNode):
        sid_map = {s.SeriesInstanceUID: s for study in node for s in study}

    for s in sid_map.values():
        raw_uid = getattr(s, "raw_series_refernce_uid", None)
        if raw_uid:
            parent = sid_map.get(raw_uid)
            if parent:
                s.raw_series_reference = parent
