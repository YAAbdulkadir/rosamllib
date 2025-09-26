from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Any
import re

_UID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _is_uid(s: str) -> bool:
    return isinstance(s, str) and bool(_UID_RE.match(s))


@dataclass(slots=True)
class DatasetNode:
    """
    Represents a dataset or collection of patients in the DICOM hierarchy.

    This class serves as a container for all patients within a particular dataset,
    grouping them under a single node. This can represent an institution, study group,
    or any higher-level categorization above individual patients.

    Parameters
    ----------
    dataset_id : str
        The unique identifier for the dataset or collection.
    dataset_name : str, optional
        The name or description of the dataset (e.g., institution name). Default is None.

    Attributes
    ----------
    dataset_id : str
        The unique identifier for the dataset.
    dataset_name : str or None
        The name or description of the dataset.
    patients : dict of str to PatientNode
        A dictionary containing `PatientNode` objects associated with this dataset.
        Keys are PatientIDs (str), and values are `PatientNode` instances.

    Methods
    -------
    add_patient(patient_node)
        Adds a `PatientNode` to the dataset.
    get_patient(patient_id)
        Retrieves a `PatientNode` from the dataset by PatientID.

    Examples
    --------
    >>> dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    >>> dataset.dataset_id
    'Institution_123'
    >>> dataset.dataset_name
    'XYZ Medical Center'
    >>> dataset.patients
    {}

    >>> dataset.add_patient(PatientNode(patient_id="12345", patient_name="John Doe"))
    >>> patient = dataset.get_patient("12345")
    >>> print(patient.PatientID, patient.PatientName)
    '12345', 'John Doe'
    """

    dataset_id: str
    dataset_name: Optional[str] = None
    patients: Dict[str, PatientNode] = field(default_factory=dict)

    def accept(self, visitor):
        return visitor.visit_dataset(self)

    # --- mutation ---
    def add_patient(self, patient_node: PatientNode, *, overwrite: bool = False) -> None:
        pid = patient_node.PatientID
        if not pid:
            raise ValueError("PatientNode must have a non-empty PatientID")
        if (pid in self.patients) and not overwrite:
            raise KeyError(f"Patient '{pid}' already exists (set overwrite=True to replace).")
        self.patients[pid] = patient_node
        patient_node.parent_dataset = self

    def get_or_create_patient(
        self, patient_id: str, patient_name: Optional[str] = None
    ) -> PatientNode:
        p = self.patients.get(patient_id)
        if p is None:
            p = PatientNode(patient_id=patient_id, patient_name=patient_name, parent_dataset=self)
            self.patients[patient_id] = p
        return p

    # --- access ---
    def get_patient(self, patient_id: str) -> Optional[PatientNode]:
        return self.patients.get(patient_id)

    def __getitem__(self, patient_id: str) -> PatientNode:
        return self.patients[patient_id]

    def __contains__(self, patient_id: str) -> bool:
        return patient_id in self.patients

    # --- iter / lens / repr ---
    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self) -> Iterator[PatientNode]:
        return iter(self.patients.values())

    def __repr__(self) -> str:
        return (
            f"DatasetNode(dataset_id={self.dataset_id!r}, "
            f"dataset_name={self.dataset_name!r}, NumPatients={len(self)})"
        )

    # --- traversal helpers ---
    def iter_studies(self) -> Iterator[StudyNode]:
        for patient in self:
            yield from patient

    def iter_series(self) -> Iterator[SeriesNode]:
        for study in self.iter_studies():
            yield from study

    def iter_instances(self) -> Iterator[InstanceNode]:
        for series in self.iter_series():
            yield from series

    # --- finders ---
    def find_series(self, series_uid: str) -> Optional[SeriesNode]:
        for s in self.iter_series():
            if s.SeriesInstanceUID == series_uid:
                return s
        return None

    def find_instance(self, sop_uid: str) -> Optional[InstanceNode]:
        for inst in self.iter_instances():
            if inst.SOPInstanceUID == sop_uid:
                return inst
        return None

    # --- (de)serialization ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "patients": {pid: p.to_dict() for pid, p in self.patients.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetNode:
        ds = cls(dataset_id=d["dataset_id"], dataset_name=d.get("dataset_name"))
        for p in d.get("patients", {}).values():
            ds.add_patient(PatientNode.from_dict(p, parent_dataset=ds))
        return ds


@dataclass(slots=True)
class PatientNode:
    """
    Represents a patient in the DICOM hierarchy.

    This class serves as a container for all studies associated with a patient.
    It stores the patient's unique identifier and name, along with a dictionary
    of `StudyNode` objects that represent the studies within the patient.

    Parameters
    ----------
    patient_id : str
        The unique identifier for the patient (PatientID).
    patient_name : str, optional
        The name of the patient (PatientName). Default is None.
    parent_dataset : DatasetNode, optional
        The `DatasetNode` this patient belongs to. Default is None.

    Attributes
    ----------
    PatientID : str
        The unique identifier for the patient.
    PatientName : str or None
        The name of the patient, if available.
    studies : dict of str to StudyNode
        A dictionary containing `StudyNode` objects associated with the patient.
        Keys are `StudyInstanceUID`s (str), and values are `StudyNode` instances.
    parent_dataset : DatasetNode or None
        Reference to the `DatasetNode` that this patient belongs to.

    Methods
    -------
    add_study(study_node)
        Adds a `StudyNode` to this patient's studies.
    get_study(study_uid)
        Retrieves a `StudyNode` by its `StudyInstanceUID`.

    Examples
    --------
    >>> patient = PatientNode(
                    patient_id="12345",
                    patient_name="John Doe",
                    parent_dataset=dataset
                    )
    >>> patient.PatientID
    '12345'
    >>> patient.PatientName
    'John Doe'
    >>> patient.studies
    {}
    >>> patient.add_study(StudyNode(study_uid="1.2.3.4.5", study_description="CT Chest"))
    >>> study = patient.get_study("1.2.3.4.5")
    >>> study.StudyInstanceUID
    '1.2.3.4.5'
    """

    patient_id: str
    patient_name: Optional[str] = None
    parent_dataset: Optional[DatasetNode] = None
    studies: Dict[str, StudyNode] = field(default_factory=dict)

    @property
    def PatientID(self) -> str:
        return self.patient_id

    @property
    def PatientName(self) -> Optional[str]:
        return self.patient_name

    def accept(self, visitor):
        return visitor.visit_patient(self)

    # --- mutation ---
    def add_study(self, study_node: StudyNode, *, overwrite: bool = False) -> None:
        uid = study_node.StudyInstanceUID
        if not _is_uid(uid):
            raise ValueError(f"Invalid StudyInstanceUID: {uid!r}")
        if (uid in self.studies) and not overwrite:
            raise KeyError(f"Study '{uid}' already exists (set overwrite=True to replace).")
        self.studies[uid] = study_node
        study_node.parent_patient = self

    def get_or_create_study(
        self, study_uid: str, study_description: Optional[str] = None
    ) -> StudyNode:
        s = self.studies.get(study_uid)
        if s is None:
            if not _is_uid(study_uid):
                raise ValueError(f"Invalid StudyInstanceUID: {study_uid!r}")
            s = StudyNode(
                study_uid=study_uid, study_description=study_description, parent_patient=self
            )
            self.studies[study_uid] = s
        return s

    # --- access / dunder ---
    def get_study(self, study_uid: str) -> Optional[StudyNode]:
        return self.studies.get(study_uid)

    def __getitem__(self, study_uid: str) -> StudyNode:
        return self.studies[study_uid]

    def __contains__(self, study_uid: str) -> bool:
        return study_uid in self.studies

    def __len__(self) -> int:
        return len(self.studies)

    def __iter__(self) -> Iterator[StudyNode]:
        return iter(self.studies.values())

    def __getattr__(self, name: str) -> Any:
        if self.parent_dataset is not None:
            return getattr(self.parent_dataset, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        return (
            f"PatientNode(PatientID={self.PatientID!r}, "
            f"PatientName={self.PatientName!r}, NumStudies={len(self)})"
        )

    # --- serialization ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "PatientID": self.PatientID,
            "PatientName": self.PatientName,
            "studies": {uid: s.to_dict() for uid, s in self.studies.items()},
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], parent_dataset: Optional[DatasetNode] = None
    ) -> PatientNode:
        p = cls(
            patient_id=d["PatientID"],
            patient_name=d.get("PatientName"),
            parent_dataset=parent_dataset,
        )
        for s in d.get("studies", {}).values():
            p.add_study(StudyNode.from_dict(s, parent_patient=p))
        return p


@dataclass(slots=True)
class StudyNode:
    """
    Represents a study in the DICOM hierarchy.

    This class serves as a container for all the series associated with a study.
    It stores the study's unique identifier and description, along with a dictionary
    of `SeriesNode` objects that represent the series within the study. Each study is
    also linked to a parent patient, which can be accessed through the `parent_patient`
    attribute.

    Parameters
    ----------
    study_uid : str
        The unique identifier for the study (StudyInstanceUID).
    study_description : str, optional
        A description of the study (StudyDescription). Default is None.
    parent_patient : PatientNode, optional
        The `PatientNode` instance associated with this study. Default is None.

    Attributes
    ----------
    StudyInstanceUID : str
        The unique identifier for the study.
    StudyDescription : str or None
        The description of the study, if available.
    series : dict of str to SeriesNode
        A dictionary containing `SeriesNode` objects associated with the study.
        Keys are `SeriesInstanceUID`s (str), and values are `SeriesNode` instances.
    parent_patient : PatientNode or None
        The `PatientNode` instance that this study is associated with, providing access
        to the parent patient's data.

    Methods
    -------
    add_series(series_node)
        Adds a `SeriesNode` to this study's series.
    get_series(series_uid)
        Retrieves a `SeriesNode` by its `SeriesInstanceUID`.

    Examples
    --------
    >>> study = StudyNode(
                    study_uid='1.2.840.113619.2.55.3.604688654.783.1590531004.467',
                    study_description='CT Chest'
                    )
    >>> study.StudyInstanceUID
    '1.2.840.113619.2.55.3.604688654.783.1590531004.467'
    >>> study.StudyDescription
    'CT Chest'
    >>> study.series
    {}
    >>> study.parent_patient
    None
    >>> study.add_series(SeriesNode(series_uid="1.2.840.113619.2.55.4"))
    >>> series = study.get_series("1.2.840.113619.2.55.4")
    >>> series.SeriesInstanceUID
    '1.2.840.113619.2.55.4'
    """

    study_uid: str
    study_description: Optional[str] = None
    parent_patient: Optional[PatientNode] = None
    series: Dict[str, SeriesNode] = field(default_factory=dict)

    @property
    def StudyInstanceUID(self) -> str:
        return self.study_uid

    @property
    def StudyDescription(self) -> Optional[str]:
        return self.study_description

    def accept(self, visitor):
        return visitor.visit_study(self)

    # --- mutation ---
    def add_series(self, series_node: SeriesNode, *, overwrite: bool = False) -> None:
        uid = series_node.SeriesInstanceUID
        if not _is_uid(uid):
            raise ValueError(f"Invalid SeriesInstanceUID: {uid!r}")
        if (uid in self.series) and not overwrite:
            raise KeyError(f"Series '{uid}' already exists (set overwrite=True to replace).")
        self.series[uid] = series_node
        series_node.parent_study = self

    def get_or_create_series(
        self, series_uid: str, modality: Optional[str] = None, desc: Optional[str] = None
    ) -> SeriesNode:
        s = self.series.get(series_uid)
        if s is None:
            if not _is_uid(series_uid):
                raise ValueError(f"Invalid SeriesInstanceUID: {series_uid!r}")
            s = SeriesNode(series_uid=series_uid, parent_study=self)
            s.Modality = modality
            s.SeriesDescription = desc
            self.series[series_uid] = s
        return s

    # --- access / dunder ---
    def get_series(self, series_uid: str) -> Optional[SeriesNode]:
        return self.series.get(series_uid)

    def __getitem__(self, series_uid: str) -> SeriesNode:
        return self.series[series_uid]

    def __contains__(self, series_uid: str) -> bool:
        return series_uid in self.series

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self) -> Iterator[SeriesNode]:
        return iter(self.series.values())

    def __getattr__(self, name: str) -> Any:
        if self.parent_patient is not None:
            return getattr(self.parent_patient, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        return (
            f"StudyNode(StudyInstanceUID={self.StudyInstanceUID!r}, "
            f"StudyDescription={self.StudyDescription!r}, NumSeries={len(self)})"
        )

    # --- serialization ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "StudyInstanceUID": self.StudyInstanceUID,
            "StudyDescription": self.StudyDescription,
            "series": {uid: s.to_dict() for uid, s in self.series.items()},
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], parent_patient: Optional[PatientNode] = None
    ) -> StudyNode:
        st = cls(
            study_uid=d["StudyInstanceUID"],
            study_description=d.get("StudyDescription"),
            parent_patient=parent_patient,
        )
        for s in d.get("series", {}).values():
            st.add_series(SeriesNode.from_dict(s, parent_study=st))
        return st


@dataclass(slots=True)
class SeriesNode:
    """
    Represents a series in the DICOM hierarchy.

    This class serves as a container for all the instances associated with a series.
    It stores metadata related to the series and provides methods to interact with the instances.
    Each series is also linked to a parent study, which can be accessed through the `parent_study`
    attribute.

    Parameters
    ----------
    series_uid : str
        The unique identifier for the DICOM series (SeriesInstanceUID).
    parent_study : StudyNode, optional
        The `StudyNode` instance associated with this series. Default is None.

    Attributes
    ----------
    SeriesInstanceUID : str
        The unique identifier for the series.
    Modality : str or None
        The modality of the series (e.g., 'CT', 'MR').
    SeriesDescription : str or None
        A description of the series, if available.
    FrameOfReferenceUID : str or None
        The Frame of Reference UID for the series.
    SOPInstances : list of str
        A list of SOP Instance UIDs associated with the series.
    instances : dict of str to InstanceNode
        A dictionary of `InstanceNode` objects in the series.
        Keys are SOPInstanceUIDs (str), values are `InstanceNode` instances.
    instance_paths : list of str
        A list of file paths to the DICOM instances.
    referencing_series : list of SeriesNode
        A list of `SeriesNode` objects that reference this series.
    referenced_series : list of SeriesNode
        A list of `SeriesNode` objects referenced by this series.
    referenced_sids : list of str
        SeriesInstanceUIDs referenced by this series.
    referencing_sids : list of str
        SeriesInstanceUIDs that reference this series.
    frame_of_reference_registered : list of SeriesNode
        A list of other `SeriesNode` objects registered to the same frame of reference.
    is_embedded_in_raw : bool
        Indicates whether the series is embedded within a RAW series.
    raw_series_reference : SeriesNode or None
        Reference to the RAW series in which this series is embedded.
    parent_study : StudyNode or None
        The `StudyNode` that this series is associated with, providing access to the parent study.

    Methods
    -------
    add_instance(instance)
        Adds an `InstanceNode` to the series.
    get_instance(sop_instance_uid)
        Retrieves an `InstanceNode` by its SOPInstanceUID.

    Examples
    --------
    >>> series = SeriesNode("1.2.840.113619.2.55.3")
    >>> series.SeriesInstanceUID
    '1.2.840.113619.2.55.3'
    >>> len(series)
    0
    >>> series.add_instance(InstanceNode("1.2.3.4.5.6.7", "/path/to/file.dcm"))
    >>> len(series)
    1
    >>> instance = series.get_instance("1.2.3.4.5.6.7")
    >>> instance.sop_instance_uid
    '1.2.3.4.5.6.7'
    """

    series_uid: str
    parent_study: Optional[StudyNode] = None

    # DICOM-ish exposed attributes
    Modality: Optional[str] = None
    SeriesDescription: Optional[str] = None
    FrameOfReferenceUID: Optional[str] = None

    # contents & relationships
    instances: Dict[str, InstanceNode] = field(default_factory=dict)
    instance_paths: list[str] = field(default_factory=list)

    # cross-reference fields
    referencing_series: list[SeriesNode] = field(default_factory=list)
    referenced_series: list[SeriesNode] = field(default_factory=list)
    referenced_sids: list[str] = field(default_factory=list)
    referencing_sids: list[str] = field(default_factory=list)
    frame_of_reference_registered: list[SeriesNode] = field(default_factory=list)

    is_embedded_in_raw: bool = False
    raw_series_reference: Optional[SeriesNode] = None

    @property
    def SeriesInstanceUID(self) -> str:
        return self.series_uid

    @property
    def SOPInstances(self) -> list[str]:
        # derived to avoid duplication with `instances`
        return list(self.instances.keys())

    def accept(self, visitor):
        return visitor.visit_series(self)

    # --- mutation ---
    def add_instance(self, instance: InstanceNode, *, overwrite: bool = False) -> None:
        uid = instance.SOPInstanceUID
        if not _is_uid(uid):
            raise ValueError(f"Invalid SOPInstanceUID: {uid!r}")
        if (uid in self.instances) and not overwrite:
            raise KeyError(f"Instance '{uid}' already exists (set overwrite=True to replace).")
        self.instances[uid] = instance
        instance.parent_series = self
        if instance.FilePath:
            self.instance_paths.append(instance.FilePath)

    # --- access / dunder ---
    def get_instance(self, sop_instance_uid: str) -> Optional[InstanceNode]:
        return self.instances.get(sop_instance_uid)

    def __getitem__(self, sop_instance_uid: str) -> InstanceNode:
        return self.instances[sop_instance_uid]

    def __contains__(self, sop_instance_uid: str) -> bool:
        return sop_instance_uid in self.instances

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[InstanceNode]:
        return iter(self.instances.values())

    def __getattr__(self, name: str) -> Any:
        if self.parent_study is not None:
            return getattr(self.parent_study, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        return (
            f"SeriesNode(SeriesInstanceUID={self.SeriesInstanceUID!r}, "
            f"Modality={self.Modality!r}, SeriesDescription={self.SeriesDescription!r}, "
            f"NumInstances={len(self)})"
        )

    # --- serialization ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "SeriesInstanceUID": self.SeriesInstanceUID,
            "Modality": self.Modality,
            "SeriesDescription": self.SeriesDescription,
            "FrameOfReferenceUID": self.FrameOfReferenceUID,
            "instances": {uid: inst.to_dict() for uid, inst in self.instances.items()},
            "instance_paths": list(self.instance_paths),
            "referenced_sids": list(self.referenced_sids),
            "referencing_sids": list(self.referencing_sids),
            "is_embedded_in_raw": self.is_embedded_in_raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], parent_study: Optional[StudyNode] = None) -> SeriesNode:
        se = cls(series_uid=d["SeriesInstanceUID"], parent_study=parent_study)
        se.Modality = d.get("Modality")
        se.SeriesDescription = d.get("SeriesDescription")
        se.FrameOfReferenceUID = d.get("FrameOfReferenceUID")
        se.instance_paths = list(d.get("instance_paths", []))
        se.referenced_sids = list(d.get("referenced_sids", []))
        se.referencing_sids = list(d.get("referencing_sids", []))
        se.is_embedded_in_raw = bool(d.get("is_embedded_in_raw", False))
        for inst in d.get("instances", {}).values():
            se.add_instance(InstanceNode.from_dict(inst, parent_series=se))
        return se


@dataclass(slots=True)
class InstanceNode:
    """
    Represents a DICOM instance (SOP instance) in the DICOM hierarchy.

    This class stores metadata and relationships associated with a DICOM instance,
    such as references to other instances or series.

    Parameters
    ----------
    SOPInstanceUID : str
        The unique identifier for the DICOM instance (SOPInstanceUID).
    FilePath : str
        The file path to the DICOM file.
    modality : str, optional
        The modality of the instance (e.g., 'CT', 'MR', 'RTSTRUCT'). Default is None.
    parent_series : SeriesNode, optional
        The `SeriesNode` this instance belongs to. Default is None.

    Attributes
    ----------
    SOPInstanceUID : str
    FilePath : str
    Modality : str or None
    FrameOfReferenceUIDs : list[str]
        Zero or more FrameOfReference UIDs associated with this instance.
        - For typical image instances (CT/MR/PT), this is usually empty; FoR is on the Series.
        - For RTSTRUCT, this may contain one or more UIDs collected from:
          * ReferencedFrameOfReferenceSequence[*].FrameOfReferenceUID
          * StructureSetROISequence[*].ReferencedFrameOfReferenceUID
          * Plus the FoRs of any referenced image Series
          (sanity union performed during association).
    references : list[Any]
    referenced_sop_instance_uids : list[str]
    referenced_sids : list[str]
    referenced_series : list[SeriesNode]
    other_referenced_sids : list[str]
    other_referenced_series : list[SeriesNode]
    referenced_instances : list[InstanceNode]
    referencing_instances : list[InstanceNode]
    parent_series : SeriesNode or None
    """

    SOPInstanceUID: str
    FilePath: str
    Modality: Optional[str] = None
    parent_series: Optional["SeriesNode"] = None  # quote if no __future__ annotations

    # RTSTRUCT (and any future multi-FoR cases)
    FrameOfReferenceUIDs: List[str] = field(default_factory=list)

    # references & relations
    references: List[Any] = field(default_factory=list)
    referenced_sop_instance_uids: List[str] = field(default_factory=list)
    referenced_sids: List[str] = field(default_factory=list)
    referenced_series: List["SeriesNode"] = field(default_factory=list)
    other_referenced_sids: List[str] = field(default_factory=list)
    other_referenced_series: List["SeriesNode"] = field(default_factory=list)
    referenced_instances: List["InstanceNode"] = field(default_factory=list)
    referencing_instances: List["InstanceNode"] = field(default_factory=list)

    # --- visitors ---
    def accept(self, visitor):
        return visitor.visit_instance(self)

    # --- convenience ---
    @property
    def primary_for_uid(self) -> Optional[str]:
        """Return a representative FrameOfReferenceUID if present (first in list)."""
        return self.FrameOfReferenceUIDs[0] if self.FrameOfReferenceUIDs else None

    def __getattr__(self, name: str) -> Any:
        if self.parent_series is not None:
            return getattr(self.parent_series, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        return (
            f"InstanceNode(SOPInstanceUID={self.SOPInstanceUID!r}, "
            f"Modality={self.Modality!r}, FilePath={self.FilePath!r})"
        )

    # --- serialization ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "SOPInstanceUID": self.SOPInstanceUID,
            "FilePath": self.FilePath,
            "Modality": self.Modality,
            "FrameOfReferenceUIDs": list(self.FrameOfReferenceUIDs),
            "referenced_sop_instance_uids": list(self.referenced_sop_instance_uids),
            "referenced_sids": list(self.referenced_sids),
            "other_referenced_sids": list(self.other_referenced_sids),
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], parent_series: Optional["SeriesNode"] = None
    ) -> "InstanceNode":
        inst = cls(
            SOPInstanceUID=d["SOPInstanceUID"],
            FilePath=d["FilePath"],
            Modality=d.get("Modality"),
            parent_series=parent_series,
        )
        inst.FrameOfReferenceUIDs = list(d.get("FrameOfReferenceUIDs", []))
        inst.referenced_sop_instance_uids = list(d.get("referenced_sop_instance_uids", []))
        inst.referenced_sids = list(d.get("referenced_sids", []))
        inst.other_referenced_sids = list(d.get("other_referenced_sids", []))
        return inst
