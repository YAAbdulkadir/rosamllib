from __future__ import annotations
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field


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

    def visit_instance(self, instance):
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
