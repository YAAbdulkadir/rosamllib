from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from rosamllib.readers.query_dicom import query_instances, QueryOptions
import weakref


# Base Visitor & Runner
class NodeVisitor:
    """Default traversal: Dataset -> Patients -> Studies -> Series -> Instances."""

    def visit_dataset(self, dataset):
        for patient in dataset:
            patient.accept(self) if hasattr(patient, "accept") else self.visit_patient(patient)
        return dataset

    def visit_patient(self, patient):
        for study in patient:
            study.accept(self) if hasattr(study, "accept") else self.visit_study(study)

    def visit_study(self, study):
        for series in study:
            series.accept(self) if hasattr(series, "accept") else self.visit_series(series)

    def visit_series(self, series):
        for inst in series:
            inst.accept(self) if hasattr(inst, "accept") else self.visit_instance(inst)

    def visit_instance(self, instance):
        # leaf by default
        return instance


def run_visitors(dataset, visitors: list[NodeVisitor]) -> None:
    """Run a list of visitors over a dataset in order."""
    for v in visitors:
        dataset.accept(v) if hasattr(dataset, "accept") else v.visit_dataset(dataset)


# Index Builder (fast lookups)
@dataclass
class BuildIndexVisitor(NodeVisitor):
    patients: Dict[str, Any] = field(default_factory=dict)
    studies: Dict[str, Any] = field(default_factory=dict)
    series: Dict[str, Any] = field(default_factory=dict)
    instances: Dict[str, Any] = field(default_factory=dict)

    def visit_dataset(self, dataset):
        # clear in case the same visitor is reused
        self.patients.clear()
        self.studies.clear()
        self.series.clear()
        self.instances.clear()
        return super().visit_dataset(dataset)

    def visit_patient(self, patient):
        self.patients[patient.PatientID] = patient
        return super().visit_patient(patient)

    def visit_study(self, study):
        self.studies[study.StudyInstanceUID] = study
        return super().visit_study(study)

    def visit_series(self, series):
        self.series[series.SeriesInstanceUID] = series
        for inst in series:
            self.instances[inst.SOPInstanceUID] = inst
        return super().visit_series(series)


# Cross-ref builder (Series <-> Series)
@dataclass
class SeriesCrossRefBuilder(NodeVisitor):
    """Populate SeriesNode.referenced_series and .referencing_sids using referenced_sids."""

    index: Optional[BuildIndexVisitor] = None

    def visit_dataset(self, dataset):
        if self.index is None:
            # Build a local index if one not supplied
            self.index = BuildIndexVisitor()
            self.index.visit_dataset(dataset)
        return super().visit_dataset(dataset)

    def visit_series(self, series):
        # ensure lists exist
        if not hasattr(series, "referenced_series"):
            series.referenced_series = []
        if not hasattr(series, "referencing_sids"):
            series.referencing_sids = []

        series.referenced_series.clear()
        for uid in getattr(series, "referenced_sids", []) or []:
            tgt = self.index.series.get(uid) if self.index else None
            if tgt:
                series.referenced_series.append(tgt)
                # add reverse sid if not already present
                if series.SeriesInstanceUID not in getattr(tgt, "referencing_sids", []):
                    tgt.referencing_sids.append(series.SeriesInstanceUID)
        return super().visit_series(series)


# Modality counter
@dataclass
class ModalityCounter(NodeVisitor):
    predicate: Callable[[Any], bool] = lambda s: True
    counts: Dict[str, int] = field(default_factory=dict)

    def visit_series(self, series):
        if self.predicate(series):
            key = (series.Modality or "UNKNOWN").upper()
            self.counts[key] = self.counts.get(key, 0) + 1
        return super().visit_series(series)


# Instance tag editor
@dataclass
class InstanceTagEditor(NodeVisitor):
    """
    edits: dict[str, Callable[[Any], Any]]
    e.g. {'SeriesDescription': lambda old: 'CT Simulation' if ... else old}

    Examples
    >>> edit = InstanceTagEditor(
    >>>     edits={
    >>>         "SeriesDescription":
    >>>         lambda old: "CT Simulation" if str(old).upper().startswith("SIM") else old
    >>>         },
    >>>     instance_predicate=lambda i: (i.Modality or "").upper()=="CT",
    >>>     dry_run=True
    >>> )
    >>> run_visitors(dataset, [edit])
    >>> print(f"Would change {edit.changed} instances; errors: {len(edit.errors)}")
    """

    edits: Dict[str, Callable[[Any], Any]]
    instance_predicate: Callable[[Any], bool] = lambda i: True
    dry_run: bool = True
    changed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def visit_instance(self, inst):
        if not self.instance_predicate(inst):
            return

        try:
            import pydicom
            from pydicom.datadict import tag_for_keyword

            ds = pydicom.dcmread(inst.FilePath)
            mutated = False

            for kw, fn in self.edits.items():
                tag = tag_for_keyword(kw)
                old = getattr(ds, kw, None)
                new = fn(old)
                if new is not None and new != old:
                    mutated = True
                    if not self.dry_run:
                        # create if missing; otherwise set value
                        if hasattr(ds, kw):
                            setattr(ds, kw, new)
                        else:
                            ds.add_new(tag, pydicom.datadict.dictionary_VR(tag), new)

            if mutated and not self.dry_run:
                ds.save_as(inst.FilePath)

            self.changed += int(mutated)

        except Exception as e:
            self.errors.append((inst.FilePath, str(e)))

        return inst


@dataclass
class QueryVisitor(NodeVisitor):
    """
    Run advanced DICOM tag queries across the graph (instances/files), using pydicom.

    - Filters are the same schema as query_df / query_instances:
      * exacts, wildcards (*, ? with escaping), RegEx/NotRegEx, ranges (gte/lte/gt/lt/eq/neq)
      * date/time aware comparisons
      * dot-paths through sequences with [*] or [index], e.g.:
        'ROIContourSequence[*].ContourSequence[*].ContourImageSequence[*].ReferencedSOPInstanceUID'
    - Batches by series to minimize file open calls.
    - You can prefilter series (e.g., by Modality) without I/O.
    - Collect either matching instances or the series that contain at least one match.

    Example:
        q = QueryVisitor(
            filters={
                "StudyDate": {"gte":"2025-03-01", "lte":"2025-03-31"},
                "Modality": "CT",
            },
            collect="instance"
        )
        dataset.accept(q)
        print(len(q.matches_instances))
    """

    filters: Dict[str, Union[str, List[Any], Dict[str, Any]]]  # deep DICOM filters
    options: QueryOptions = field(default_factory=QueryOptions)
    series_predicate: Optional[Callable[[Any], bool]] = None
    return_paths: bool = False

    # outputs
    hits_instances: List[Union[Any, str]] = field(default_factory=list)
    per_series_hitcount: Dict[str, int] = field(default_factory=dict)
    traces: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # SOP -> {key: matched_values}

    # Optional: inject a function that returns match traces from query_instances (see note below)
    _trace_hook: Optional[Callable[[List[Union[Any, str]]], Dict[str, Dict[str, Any]]]] = None

    def visit_series(self, series):
        if self.series_predicate and not self.series_predicate(series):
            return
        items = list(series)  # InstanceNode objects
        if not items:
            return
        hits = query_instances(items, options=self.options, **self.filters)
        if hits:
            sid = series.SeriesInstanceUID
            self.per_series_hitcount[sid] = len(hits)
            self.hits_instances.extend(
                hits if self.return_paths else [h for h in hits if not isinstance(h, str)]
            )

            # (Optional) if you extend query_instances to return or expose trace info:
            if self._trace_hook:
                self.traces.update(self._trace_hook(hits))


@dataclass
class LabelRule:
    target: str
    label: str
    when: Callable[[Any], bool]
    propagate: str = "none"  # 'none' | 'up' | 'down'
    condition_any: bool = True  # for aggregate levels
    namespace: Optional[str] = None  # optional 'ns:label'

    def full_label(self) -> str:
        return f"{self.namespace}:{self.label}" if self.namespace else self.label


# We avoid mutating nodes by storing labels in side maps keyed by object identity or UID strings.
@dataclass
class LabelStore:
    # object-id keyed sets; works across dataclasses with slots
    _obj_labels: "weakref.WeakKeyDictionary[Any, Set[str]]" = field(
        default_factory=weakref.WeakKeyDictionary
    )
    # UID keyed sets (useful for serialization/reloads)
    _uid_labels: Dict[Tuple[str, str], Set[str]] = field(
        default_factory=dict
    )  # (level, uid) -> labels

    def _ensure(self, obj) -> Set[str]:
        s = self._obj_labels.get(obj)
        if s is None:
            s = set()
            self._obj_labels[obj] = s
        return s

    # --- object-based ---
    def add(self, obj, *labels: str) -> None:
        self._ensure(obj).update(labels)

    def remove(self, obj, *labels: str) -> None:
        if obj in self._obj_labels:
            for lab in labels:
                self._obj_labels[obj].discard(lab)

    def has(self, obj, label: str) -> bool:
        return label in self._obj_labels.get(obj, set())

    def get(self, obj) -> Set[str]:
        return set(self._obj_labels.get(obj, set()))

    # --- UID-based (optional, for persistence) ---
    def add_uid(self, level: str, uid: str, *labels: str) -> None:
        k = (level.upper(), uid)
        self._uid_labels.setdefault(k, set()).update(labels)

    def get_uid(self, level: str, uid: str) -> Set[str]:
        return set(self._uid_labels.get((level.upper(), uid), set()))

    def export(self) -> Dict[str, Dict[str, Set[str]]]:
        # group by level in the UID map
        out: Dict[str, Dict[str, Set[str]]] = {}
        for (lvl, uid), labs in self._uid_labels.items():
            out.setdefault(lvl, {})[uid] = set(labs)
        return out


@dataclass
class LabelingVisitor(NodeVisitor):
    store: LabelStore
    rules: List[LabelRule]
    dry_run: bool = True
    # bookkeeping for undo/report
    applied: List[tuple[Any, str]] = field(default_factory=list)

    # Helpers
    def _apply(self, obj, lab: str):
        if not self.dry_run and not self.store.has(obj, lab):
            self.store.add(obj, lab)
            self.applied.append((obj, lab))

    # Aggregation utilities
    def _children(self, obj):
        # Return iterator over immediate children in the hierarchy
        if hasattr(obj, "series"):  # StudyNode
            return obj.series.values()
        if hasattr(obj, "studies"):  # PatientNode
            return obj.studies.values()
        if hasattr(obj, "instances"):  # SeriesNode
            return obj.instances.values()
        return []

    def _parent(self, obj):
        if hasattr(obj, "parent_series"):
            return obj.parent_series
        if hasattr(obj, "parent_study"):
            return obj.parent_study
        if hasattr(obj, "parent_patient"):
            return obj.parent_patient
        if hasattr(obj, "parent_dataset"):
            return obj.parent_dataset
        return None

    # Visit methods
    def visit_instance(self, inst):
        for rule in self.rules:
            if rule.target.upper() == "INSTANCE" and rule.when(inst):
                self._apply(inst, rule.full_label())
                if rule.propagate == "up":
                    s = self._parent(inst)
                    if s:
                        self._apply(s, rule.full_label())
                    st = self._parent(s) if s else None
                    if st:
                        self._apply(st, rule.full_label())
                    p = self._parent(st) if st else None
                    if p:
                        self._apply(p, rule.full_label())
        return inst

    def visit_series(self, series):
        # Evaluate series-level rules
        for rule in self.rules:
            if rule.target.upper() == "SERIES" and rule.when(series):
                self._apply(series, rule.full_label())
                if rule.propagate == "up":
                    st = self._parent(series)
                    p = self._parent(st) if st else None
                    if st:
                        self._apply(st, rule.full_label())
                    if p:
                        self._apply(p, rule.full_label())

        # Aggregate rules from children when target != INSTANCE and propagate='down'
        for rule in self.rules:
            if rule.target.upper() in {"STUDY", "PATIENT"} and rule.propagate == "down":
                # decide based on children
                kids = list(self._children(series))
                if not kids:
                    continue
                hits = [rule.when(k) for k in kids]
                cond = any(hits) if rule.condition_any else all(hits)
                if cond:
                    self._apply(series, rule.full_label())

        return super().visit_series(series)

    def visit_study(self, study):
        for rule in self.rules:
            if rule.target.upper() == "STUDY" and rule.when(study):
                self._apply(study, rule.full_label())
                if rule.propagate == "up":
                    p = self._parent(study)
                    if p:
                        self._apply(p, rule.full_label())

        # Downward aggregate to series if requested
        for rule in self.rules:
            if rule.target.upper() == "STUDY" and rule.propagate == "down":
                kids = list(self._children(study))
                if not kids:
                    continue
                hits = [rule.when(k) for k in kids]  # note: rule.when applied to series
                cond = any(hits) if rule.condition_any else all(hits)
                if cond:
                    for s in kids:
                        self._apply(s, rule.full_label())

        return super().visit_study(study)

    def visit_patient(self, patient):
        for rule in self.rules:
            if rule.target.upper() == "PATIENT" and rule.when(patient):
                self._apply(patient, rule.full_label())
        return super().visit_patient(patient)

    def undo(self):
        # revert applied labels (last-applied-first)
        for obj, lab in reversed(self.applied):
            self.store.remove(obj, lab)
        self.applied.clear()
