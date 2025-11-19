from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Any, Union, Callable, Iterable
import re
from rosamllib.utils import (
    get_referenced_nodes,
    get_referencing_nodes,
    get_frame_registered_nodes,
    _level_ok,
    _norm_modalities,
    _modality_ok,
)


_UID_RE = re.compile(r"^\d+(?:\.\d+)*$")

NodeT = Union["SeriesNode", "InstanceNode"]


def _is_uid(s: str) -> bool:
    return isinstance(s, str) and bool(_UID_RE.match(s))


class _ExtensibleAttrs:
    # NOTE: name-mangled to avoid collisions: _ExtensibleAttrs__extras
    __slots__ = ("_ExtensibleAttrs__extras",)

    # internal helpers
    def _ensure_extras(self) -> None:
        # Create the extras dict exactly once, without invoking __getattr__
        try:
            object.__getattribute__(self, "_ExtensibleAttrs__extras")
        except AttributeError:
            object.__setattr__(self, "_ExtensibleAttrs__extras", {})

    def _extras_dict(self) -> dict:
        self._ensure_extras()
        return object.__getattribute__(self, "_ExtensibleAttrs__extras")

    def _get_extra_maybe(self, name: str):
        ex = self._extras_dict()
        if name in ex:
            return ex[name]
        raise AttributeError

    # public helpers
    def set_attrs(self, **kwargs):
        self._extras_dict().update(kwargs)

    def get_attr(self, name: str, default=None):
        return self._extras_dict().get(name, default)

    def del_attr(self, name: str):
        self._extras_dict().pop(name, None)

    def iter_attrs(self):
        return self._extras_dict().items()

    def __setattr__(self, name, value):
        # If it's a declared dataclass field, write it normally
        fields = getattr(type(self), "__dataclass_fields__", None)
        if fields and name in fields:
            return object.__setattr__(self, name, value)
        # If it's an internal/private attr, set normally
        if name.startswith("_"):
            return object.__setattr__(self, name, value)
        # Otherwise, stash as an optional attribute
        self._extras_dict()[name] = value

    def __getattr__(self, name):
        # Called only if normal attribute lookup failed
        ex = self._extras_dict()
        if name in ex:
            return ex[name]
        # Let caller decide what to do next
        raise AttributeError


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

        if not p.PatientName and patient_name:
            p.PatientName = patient_name
        return p

    # --- access ---
    def get_patient(self, patient_id: str) -> Optional[PatientNode]:
        return self.patients.get(patient_id)

    def get_study(
        self, study_uid: str, *, patient_id: Optional[str] = None
    ) -> Optional["StudyNode"]:
        """
        Retrieve a StudyNode by StudyInstanceUID.

        Parameters
        ----------
        study_uid : str
            The StudyInstanceUID of the study to retrieve.
        patient_id : str, optional
            If provided, restrict the search to this PatientID for O(1-ish) lookup.

        Returns
        -------
        StudyNode or None
        """
        # Fast path: restrict to known patient
        if patient_id is not None:
            p = self.get_patient(patient_id)
            if p is None:
                return None
            return p.get_study(study_uid)

        # General path: search all patients
        for p in self:
            st = p.get_study(study_uid)
            if st is not None:
                return st
        return None

    def get_series(
        self,
        series_uid: str,
        *,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
    ) -> Optional["SeriesNode"]:
        """
        Retrieve a SeriesNode by SeriesInstanceUID.

        Parameters
        ----------
        series_uid : str
            The SeriesInstanceUID of the series to retrieve.
        patient_id : str, optional
            If provided, restrict the search to this PatientID.
        study_uid : str, optional
            If provided, restrict the search to this StudyInstanceUID.

        Returns
        -------
        SeriesNode or None
        """
        # Fastest path: known patient and study
        if patient_id is not None and study_uid is not None:
            p = self.get_patient(patient_id)
            if p is None:
                return None
            st = p.get_study(study_uid)
            return None if st is None else st.get_series(series_uid)

        # Next-fast path: known patient
        if patient_id is not None:
            p = self.get_patient(patient_id)
            if p is None:
                return None
            return p.get_series(series_uid)

        # Next-fast path: known study (but not patient)
        if study_uid is not None:
            st = self.get_study(study_uid)
            return None if st is None else st.get_series(series_uid)

        # Fallback: use traversal helper
        return self.find_series(series_uid)

    def get_instance(
        self,
        sop_uid: str,
        *,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
        series_uid: Optional[str] = None,
    ) -> Optional["InstanceNode"]:
        """
        Retrieve an InstanceNode by SOPInstanceUID.

        Parameters
        ----------
        sop_uid : str
            The SOPInstanceUID of the instance to retrieve.
        patient_id : str, optional
            If provided, restrict the search to this PatientID.
        study_uid : str, optional
            If provided, restrict the search to this StudyInstanceUID.
        series_uid : str, optional
            If provided, restrict the search to this SeriesInstanceUID.

        Returns
        -------
        InstanceNode or None
        """
        # Fastest path: known patient -> delegate to PatientNode
        if patient_id is not None:
            p = self.get_patient(patient_id)
            if p is None:
                return None
            return p.get_instance(sop_uid, series_uid=series_uid, study_uid=study_uid)

        # Next-fast path: known study (but not patient)
        if study_uid is not None:
            st = self.get_study(study_uid)
            if st is None:
                return None
            return st.get_instance(sop_uid, series_uid=series_uid)

        # Next-fast path: known series (but not study/patient)
        if series_uid is not None:
            se = self.find_series(series_uid)
            return None if se is None else se.get_instance(sop_uid)

        # Fallback: use traversal helper
        return self.find_instance(sop_uid)

    def get_referenced_nodes(
        self,
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

        return get_referenced_nodes(node, modality, level, recursive, include_start)

    def get_referencing_nodes(
        self,
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

        return get_referencing_nodes(node, modality, level, recursive, include_start)

    def get_frame_registered_nodes(
        self,
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

        return get_frame_registered_nodes(
            node,
            level=level,
            include_self=include_self,
            modality=modality,
            dicom_files=dicom_files,
            derive_frame_from_references=derive_frame_from_references,
        )

    def get_modality_distribution(self):
        modality_counts = {}

        for patient_node in self:
            for study_node in patient_node:
                for series_node in study_node:
                    modality = series_node.Modality or "Unknown"
                    if modality in ["RTPLAN", "RTDOSE", "RTSTRUCT", "RTRECORD"]:
                        for instance_node in series_node:
                            modality_counts[modality] = modality_counts.get(modality, 0) + 1
                    else:
                        modality_counts[modality] = modality_counts.get(modality, 0) + 1

        return modality_counts

    def report_sources_without_reach(
        self, **kwargs
    ) -> Dict[str, List[Union["SeriesNode", "InstanceNode"]]]:
        """
        Run PatientNode.find_sources_without_reach for every patient.

        Accepts the same keyword arguments as PatientNode.find_sources_without_reach.
        Returns a dict: {PatientID: [start_nodes_failing]} (only patients with failures included).
        """
        report: Dict[str, List[Union[SeriesNode, InstanceNode]]] = {}
        for p in self:
            bad = p.find_sources_without_reach(**kwargs)
            if bad:
                report[p.PatientID] = bad
        return report

    def dangling_references_report(self, *, return_df: bool = False):
        """
        Run PatientNode.find_dangling_references for all patients.
        Returns list of dict rows or a single concatenated DataFrame (if return_df=True).
        """
        all_rows = []
        for p in self:
            res = p.find_dangling_references(return_df=False)
            all_rows.extend(res)
        if return_df:
            try:
                import pandas as pd

                return pd.DataFrame(
                    all_rows,
                    columns=[
                        "PatientID",
                        "SourceLevel",
                        "SourceModality",
                        "SourceSeriesUID",
                        "SourceSOPInstanceUID",
                        "MissingKind",
                        "MissingUID",
                    ],
                )
            except Exception:
                return all_rows
        return all_rows

    def orphan_report(
        self,
        *,
        level: str = "SERIES",
        modality: Optional[Union[str, Iterable[str]]] = None,
        include_frame: bool = False,
        return_df: bool = False,
    ):
        rows = []
        nodes = []
        for p in self:
            res = p.find_orphans(
                level=level, modality=modality, include_frame=include_frame, return_df=False
            )
            nodes.extend((p.PatientID, n) for n in res)

        if not return_df:
            return [n for _, n in nodes]

        try:
            import pandas as pd

            for pid, n in nodes:
                if isinstance(n, SeriesNode):
                    st = getattr(n, "parent_study", None)
                    rows.append(
                        {
                            "PatientID": pid,
                            "Level": "SERIES",
                            "StudyInstanceUID": (
                                getattr(st, "StudyInstanceUID", None) if st else None
                            ),
                            "SeriesInstanceUID": n.SeriesInstanceUID,
                            "SOPInstanceUID": None,
                            "Modality": getattr(n, "Modality", None),
                            "FrameOfReferenceUID": getattr(n, "FrameOfReferenceUID", None),
                            "IsOrphan": True,
                        }
                    )
                else:
                    se = getattr(n, "parent_series", None)
                    st = getattr(se, "parent_study", None) if se else None
                    rows.append(
                        {
                            "PatientID": pid,
                            "Level": "INSTANCE",
                            "StudyInstanceUID": (
                                getattr(st, "StudyInstanceUID", None) if st else None
                            ),
                            "SeriesInstanceUID": (
                                getattr(se, "SeriesInstanceUID", None) if se else None
                            ),
                            "SOPInstanceUID": getattr(n, "SOPInstanceUID", None),
                            "Modality": getattr(se, "Modality", None) if se else None,
                            "FrameOfReferenceUID": (
                                getattr(se, "FrameOfReferenceUID", None) if se else None
                            ),
                            "IsOrphan": True,
                        }
                    )
            return pd.DataFrame(
                rows,
                columns=[
                    "PatientID",
                    "Level",
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "SOPInstanceUID",
                    "Modality",
                    "FrameOfReferenceUID",
                    "IsOrphan",
                ],
            )
        except Exception:
            return [n for _, n in nodes]

    def associate_dicoms(self):
        from rosamllib.utils import associate_dicoms

        return associate_dicoms(self)

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
        from rosamllib.utils import associate_dicoms

        associate_dicoms(ds)
        return ds


@dataclass(slots=True)
class PatientNode(_ExtensibleAttrs):
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
        if not s.StudyDescription and study_description:
            s.StudyDescription = study_description
        return s

    # --- access / dunder ---
    def get_study(self, study_uid: str) -> Optional[StudyNode]:
        return self.studies.get(study_uid)

    def get_series(
        self, series_uid: str, study_uid: Optional[str] = None
    ) -> Optional["SeriesNode"]:
        """
        Retrieve a SeriesNode (by SeriesInstanceUID) within this patient.

        Parameters
        ----------
        series_uid : str
            The SeriesInstanceUID of the series to retrieve.
        study_uid : str, optional
            If provided, restrict the search to this StudyInstanceUID for O(1) lookup.

        Returns
        -------
        SeriesNode or None
            The SeriesNode associated with the given series_uid, or None if not found.
        """
        # Fast path: restrict to known study
        if study_uid is not None:
            st = self.get_study(study_uid)
            if st is None:
                return None
            return st.get_series(series_uid)

        # General path: search all studies
        for st in self.studies.values():
            se = st.get_series(series_uid)
            if se is not None:
                return se
        return None

    def get_instance(
        self,
        sop_instance_uid: str,
        *,
        series_uid: Optional[str] = None,
        study_uid: Optional[str] = None,
    ) -> Optional["InstanceNode"]:
        """
        Retrieve an InstanceNode (by SOPInstanceUID) within this patient.

        Parameters
        ----------
        sop_instance_uid : str
            The SOPInstanceUID of the instance to retrieve.
        series_uid : str, optional
            If provided, restrict the search to this SeriesInstanceUID (faster).
        study_uid : str, optional
            If provided, restrict the search to this StudyInstanceUID (faster).

        Returns
        -------
        InstanceNode or None
            The InstanceNode associated with the given sop_instance_uid, or None if not found.

        Notes
        -----
        - If `study_uid` is provided, lookup is delegated to that StudyNode
        (and optionally to the given series).
        - Else if only `series_uid` is provided, we first locate the series across studies,
        then get the instance.
        - Otherwise, all studies/series are scanned and the first match is returned.
        """
        # Fastest path: known study (and optional series)
        if study_uid is not None:
            st = self.get_study(study_uid)
            if st is None:
                return None
            return st.get_instance(sop_instance_uid, series_uid=series_uid)

        # Next-fast path: known series but unknonw study
        if series_uid is not None:
            for st in self.studies.values():
                se = st.get_series(series_uid)
                if se is not None:
                    return se.get_instance(sop_instance_uid)
            return None

        # General path: search all studies
        for st in self.studies.values():
            inst = st.get_instance(sop_instance_uid)
            if inst is not None:
                return inst
        return None

    def get_referenced_nodes(
        self,
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

        return get_referenced_nodes(node, modality, level, recursive, include_start)

    def get_referencing_nodes(
        self,
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

        return get_referencing_nodes(node, modality, level, recursive, include_start)

    def get_frame_registered_nodes(
        self,
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

        return get_frame_registered_nodes(
            node,
            level=level,
            include_self=include_self,
            modality=modality,
            dicom_files=dicom_files,
            derive_frame_from_references=derive_frame_from_references,
        )

    def find_sources_with_reach(
        self,
        *,
        start_level: str = "SERIES",
        start_modality: Optional[Union[str, Iterable[str]]] = None,
        target_level: str = "SERIES",
        target_modality: Optional[Union[str, Iterable[str]]] = None,
        recursive: bool = True,
        traversal: str = "references",  # <- if you prefer your exact spelling, use "referenced"
        include_start: bool = False,
        # Optional power-users: custom predicates (applied in addition to modality/level)
        start_predicate: Optional[Callable[[NodeT], bool]] = None,
        target_predicate: Optional[Callable[[NodeT], bool]] = None,
        # Optional: accept same-FrameOfReferenceUID as a fallback "connection"
        allow_same_for: bool = False,
    ) -> List[NodeT]:
        """
        Generalized positive reachability: return 'start' nodes that CAN reach at least one
        'target'.

        Parameters mirror `find_sources_without_reach`, but this returns the positives.
        """
        s_level = start_level.upper()
        t_level = target_level.upper()
        if s_level not in {"SERIES", "INSTANCE"} or t_level not in {"SERIES", "INSTANCE"}:
            raise ValueError("start_level/target_level must be 'SERIES' or 'INSTANCE'")

        trav = traversal.lower()
        # Match your accepted spellings; keep both for convenience
        if trav == "references":
            trav = "referenced"
        if trav not in {"referenced", "referencing", "frame"}:
            raise ValueError("traversal must be one of {'referenced', 'referencing', 'frame'}")

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

        starts: List[NodeT] = [n for n in self._iter_nodes(s_level) if is_start(n)]
        if not starts:
            return []

        # Strategy functions
        def neighbors_references(n: NodeT) -> List[NodeT]:
            return get_referenced_nodes(
                n,
                modality=None,  # filter to targets after expansion
                level=t_level,
                recursive=recursive,
                include_start=include_start,
            )

        def neighbors_referencing(n: NodeT) -> List[NodeT]:
            return get_referencing_nodes(
                n,
                modality=None,
                level=t_level,
                recursive=recursive,
                include_start=include_start,
            )

        def neighbors_frame(n: NodeT) -> List[NodeT]:
            return get_frame_registered_nodes(
                n,
                level=t_level,
                include_self=include_start,
                modality=None,
            )

        if trav == "referenced":
            get_neighbors = neighbors_references
        elif trav == "referencing":
            get_neighbors = neighbors_referencing
        else:
            get_neighbors = neighbors_frame

        # Optional same-FoR fallback for ref-based traversals
        def same_for_any_target(n: NodeT) -> bool:
            if trav == "frame":
                return False  # already using FoR
            peers = get_frame_registered_nodes(n, level=t_level, include_self=False, modality=None)
            return any(is_target(x) for x in peers)

        hits: List[NodeT] = []
        for s in starts:
            reachable = get_neighbors(s)
            reached = any(is_target(x) for x in reachable)
            if not reached and allow_same_for and same_for_any_target(s):
                reached = True
            if reached:
                hits.append(s)

        return hits

    def find_sources_without_reach(
        self,
        *,
        start_level: str = "SERIES",
        start_modality: Optional[Union[str, Iterable[str]]] = None,
        target_level: str = "SERIES",
        target_modality: Optional[Union[str, Iterable[str]]] = None,
        recursive: bool = True,
        traversal: str = "references",
        include_start: bool = False,
        # Optional power-users: custom predicates (applied in addition to modality/level)
        start_predicate: Optional[Callable[[NodeT], bool]] = None,
        target_predicate: Optional[Callable[[NodeT], bool]] = None,
        # Optional: accept same-FrameOfReferenceUID as a fallback "connection"
        allow_same_for: bool = False,
    ) -> List[NodeT]:
        """
        Generalized consistency check: find 'start' nodes that cannot reach any 'target' nodes.

        Parameters
        ----------
        start_level, target_level : {'SERIES','INSTANCE'}
            Which node granularity to consider for starts/targets.
        start_modality, target_modality : str | Iterable[str] | None
            Filter by modality for starts/targets (case-insensitive). None = no filter.
        recursive : bool
            If True, multi-hop traversal; else only direct neighbors (depth=1).
        traversal : {'referenced','referencing','frame'}
            - 'referenced': follow outgoing references
            - 'referencing': follow incoming references
            - 'frame': use same FrameOfReference neighborhood
        include_start : bool
            If True, include the start node in traversal results (some queries may want this).
        start_predicate, target_predicate : Callable[[NodeT], bool] | None
            Extra filters; both must return True to keep a node as start/target.
        allow_same_for : bool
            If True and traversal is 'referenced'/'referencing', also accept a same-FoR CT/target
            as a fallback (only applied if no referenced target is found).

        Returns
        -------
        List[NodeT]
            Start nodes for which no target is reachable under the chosen traversal.
        """
        s_level = start_level.upper()
        t_level = target_level.upper()
        if s_level not in {"SERIES", "INSTANCE"} or t_level not in {"SERIES", "INSTANCE"}:
            raise ValueError("start_level/target_level must be 'SERIES' or 'INSTANCE'")
        trav = traversal.lower()
        if trav not in {"referenced", "referencing", "frame"}:
            raise ValueError("traversal must be one of {'referenced', 'referencing', 'frame'}")

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

        starts: List[NodeT] = [n for n in self._iter_nodes(s_level) if is_start(n)]
        if not starts:
            return []

        # Strategy functions
        def neighbors_references(n: NodeT) -> List[NodeT]:
            return get_referenced_nodes(
                n,
                modality=None,  # we'll filter by is_target after expansion
                level=t_level,
                recursive=recursive,
                include_start=include_start,
            )

        def neighbors_referencing(n: NodeT) -> List[NodeT]:
            return get_referencing_nodes(
                n,
                modality=None,
                level=t_level,
                recursive=recursive,
                include_start=include_start,
            )

        def neighbors_frame(n: NodeT) -> List[NodeT]:
            return get_frame_registered_nodes(
                n,
                level=t_level,
                include_self=include_start,
                modality=None,
            )

        if trav == "referenced":
            get_neighbors = neighbors_references
        elif trav == "referencing":
            get_neighbors = neighbors_referencing
        else:
            get_neighbors = neighbors_frame

        # Optinoal same-FoR fallback for references-based traversals
        def same_for_any_target(n: NodeT) -> bool:
            if trav == "frame":
                return False  # already using FoR
            peers = get_frame_registered_nodes(n, level=t_level, include_self=False, modality=None)
            return any(is_target(x) for x in peers)

        missing: List[NodeT] = []
        for s in starts:
            reachable = get_neighbors(s)
            if any(is_target(x) for x in reachable):
                continue
            if allow_same_for and same_for_any_target(s):
                continue
            missing.append(s)

        return missing

    def _iter_nodes(self, level: str) -> Iterable[NodeT]:
        """Yield all nodes for this patient at the requested level."""
        if level == "SERIES":
            for st in self.studies.values():
                for se in st:
                    yield se
        elif level == "INSTANCE":
            for st in self.studies.values():
                for se in st:
                    for inst in se:
                        yield inst
        else:
            raise ValueError("level must be 'SERIES' or 'INSTANCE'")

    def find_dangling_references(self, *, return_df: bool = False):
        """
        Scan this patient for reference UIDs that fail to resolve to any node in the same patient.

        Looks at:
        - SeriesNode.referenced_sids (series->series by UID)
        - InstanceNode.ReferencedSeriesInstanceUIDs (instance->series by UID)
        - InstanceNode.ReferencedSOPInstanceUIDs (instance->instance by UID)

        Returns
        -------
        list[dict] or pd.DataFrame (if return_df=True)
        Each row: {
            'PatientID', 'SourceLevel', 'SourceModality',
            'SourceSeriesUID', 'SourceSOPInstanceUID',
            'MissingKind' ('SERIES'|'INSTANCE'),
            'MissingUID'
        }
        """
        # Build local UID maps for O(1) lookups
        series_map: Dict[str, SeriesNode] = {}
        inst_map: Dict[str, InstanceNode] = {}

        for st in self.studies.values():
            for se in st:
                series_map[se.SeriesInstanceUID] = se
                for inst in se:
                    inst_map[inst.SOPInstanceUID] = inst

        rows: List[Dict[str, Optional[str]]] = []

        def add_row(
            src_node: Union[SeriesNode, InstanceNode], missing_uid: str, missing_kind: str
        ):
            if isinstance(src_node, SeriesNode):
                src_series_uid = src_node.SeriesInstanceUID
                src_sop_uid = None
                src_mod = getattr(src_node, "Modality", None)
                src_level = "SERIES"
            else:
                src_series = getattr(src_node, "parent_series", None)
                src_series_uid = getattr(src_series, "SeriesInstanceUID", None)
                src_sop_uid = getattr(src_node, "SOPInstanceUID", None)
                src_mod = getattr(src_series, "Modality", None) if src_series is not None else None
                src_level = "INSTANCE"

            rows.append(
                {
                    "PatientID": self.PatientID,
                    "SourceLevel": src_level,
                    "SourceModality": src_mod,
                    "SourceSeriesUID": src_series_uid,
                    "SourceSOPInstanceUID": src_sop_uid,
                    "MissingKind": missing_kind,
                    "MissingUID": missing_uid,
                }
            )

        # Inspect series-level raw UID references
        for st in self.studies.values():
            for se in st:
                # Series -> Series by UID
                for uid in getattr(se, "referenced_sids", []) or []:
                    if uid not in series_map:
                        add_row(se, uid, "SERIES")

                # Instance-level refereces inside this series
                for inst in se.instances.values():
                    # Instance -> Series by UID
                    for uid in getattr(inst, "ReferencedSeriesInstanceUIDs", []) or []:
                        if uid not in series_map:
                            add_row(inst, uid, "SERIES")
                    # Instance -> Instance by UID
                    for uid in getattr(inst, "ReferencedSOPInstanceUIDs", []) or []:
                        if uid not in inst_map:
                            add_row(inst, uid, "INSTANCE")

        if return_df:
            try:
                import pandas as pd

                return pd.DataFrame(
                    rows,
                    columns=[
                        "PatientID",
                        "SourceLevel",
                        "SourceModality",
                        "SourceSeriesUID",
                        "SourceSOPInstanceUID",
                        "MissingKind",
                        "MissingUID",
                    ],
                )
            except Exception:
                # If pandas isn't available, fall back to raw rows
                return rows

        return rows

    def find_orphans(
        self,
        *,
        level: str = "SERIES",
        modality: Optional[Union[str, Iterable[str]]] = None,
        include_frame: bool = False,
        return_df: bool = False,
    ):
        """
        Find nodes with NO incoming *and* NO outgoing connections within this patient.

        Connections are defined as:
        - Outgoing: nodes returned by get_referenced_nodes(node, level=..., recursive=False)
        - Incoming: nodes returned by get_referencing_nodes(node, level=..., recursive=False)
        - Optional: if include_frame=True, nodes in the same FoR via
                    get_frame_registered_nodes(node, level=..., include_self=False)

        Parameters
        ----------
        level : {'SERIES','INSTANCE'}
            Node type to evaluate.
        modality : str | Iterable[str] | None
            Filter candidate nodes by modality (case-insensitive). For INSTANCE level,
            the series' modality is used.
        include_frame : bool
            If True, being in the same FrameOfReferenceUID (FoR) *also* counts as connected.
            (Useful as a lenient fallback for imperfect exports.)
        return_df : bool
            If True, return a pandas DataFrame; else return a list of nodes.

        Returns
        -------
        list[SeriesNode|InstanceNode] or pd.DataFrame
        """
        lvl = str(level).upper()
        if lvl not in {"SERIES", "INSTANCE"}:
            raise ValueError("level must be 'SERIES' or 'INSTANCE'")

        def norm_mods(m):
            if m is None:
                return None
            if isinstance(m, str):
                return {m.upper()}
            return {str(x).upper() for x in m}

        wanted = norm_mods(modality)

        def is_mod_ok(n) -> bool:
            if wanted is None:
                return True
            if isinstance(n, SeriesNode):
                mod = getattr(n, "Modality", None)
            else:
                se = getattr(n, "parent_series", None)
                mod = getattr(se, "Modality", None) if se is not None else None
            return (mod or "").upper() in wanted

        # --- enumerate candidates within this patient ---
        candidates: list[Union[SeriesNode, InstanceNode]] = []
        if lvl == "SERIES":
            for st in self.studies.values():
                for se in st:
                    if is_mod_ok(se):
                        candidates.append(se)
        else:  # INSTANCE
            for st in self.studies.values():
                for se in st:
                    for inst in se:
                        if is_mod_ok(inst):
                            candidates.append(inst)

        def outgoing(n):
            # One-hop only (recursive=False), we just care about *some* edge existing
            return get_referenced_nodes(
                n, modality=None, level=lvl, recursive=False, include_start=False
            )

        def incoming(n):
            return get_referencing_nodes(
                n, modality=None, level=lvl, recursive=False, include_start=False
            )

        def frame_peers(n):
            if not include_frame:
                return []
            return get_frame_registered_nodes(n, level=lvl, include_self=False, modality=None)

        orphans = []
        for node in candidates:
            has_out = bool(outgoing(node))
            if not has_out:
                # fast fail? we still need to check incoming and (optionally) frame
                pass
            has_in = bool(incoming(node))
            if include_frame:
                # If neither in nor out via references, FoR peers can make it "non-orphan"
                has_frame = bool(frame_peers(node))
            else:
                has_frame = False

            if not has_out and not has_in and not has_frame:
                orphans.append(node)

        if not return_df:
            return orphans

        # Build a tidy DataFrame
        try:
            import pandas as pd

            rows = []
            for n in orphans:
                if isinstance(n, SeriesNode):
                    st = getattr(n, "parent_study", None)
                    pt = getattr(st, "parent_patient", None) if st else None
                    rows.append(
                        {
                            "PatientID": getattr(pt, "PatientID", None),
                            "Level": "SERIES",
                            "StudyInstanceUID": (
                                getattr(st, "StudyInstanceUID", None) if st else None
                            ),
                            "SeriesInstanceUID": n.SeriesInstanceUID,
                            "SOPInstanceUID": None,
                            "Modality": getattr(n, "Modality", None),
                            "FrameOfReferenceUID": getattr(n, "FrameOfReferenceUID", None),
                            "IsOrphan": True,
                        }
                    )
                else:
                    se = getattr(n, "parent_series", None)
                    st = getattr(se, "parent_study", None) if se else None
                    pt = getattr(st, "parent_patient", None) if st else None
                    rows.append(
                        {
                            "PatientID": getattr(pt, "PatientID", None),
                            "Level": "INSTANCE",
                            "StudyInstanceUID": (
                                getattr(st, "StudyInstanceUID", None) if st else None
                            ),
                            "SeriesInstanceUID": (
                                getattr(se, "SeriesInstanceUID", None) if se else None
                            ),
                            "SOPInstanceUID": getattr(n, "SOPInstanceUID", None),
                            "Modality": getattr(se, "Modality", None) if se else None,
                            "FrameOfReferenceUID": (
                                getattr(se, "FrameOfReferenceUID", None) if se else None
                            ),
                            "IsOrphan": True,
                        }
                    )
            return pd.DataFrame(
                rows,
                columns=[
                    "PatientID",
                    "Level",
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "SOPInstanceUID",
                    "Modality",
                    "FrameOfReferenceUID",
                    "IsOrphan",
                ],
            )
        except Exception:
            return orphans

    def associate_dicoms(self):
        from rosamllib.utils import associate_dicoms

        return associate_dicoms(self)

    def __getitem__(self, study_uid: str) -> StudyNode:
        return self.studies[study_uid]

    def __contains__(self, study_uid: str) -> bool:
        return study_uid in self.studies

    def __len__(self) -> int:
        return len(self.studies)

    def __iter__(self) -> Iterator[StudyNode]:
        return iter(self.studies.values())

    def __getattr__(self, name: str) -> Any:
        # 1) try extras (from mixin)
        try:
            return _ExtensibleAttrs._get_extra_maybe(self, name)
        except AttributeError:
            pass
        # 2) delegate to parent (existing behavior)
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
            "__extras__": dict(self.iter_attrs()),
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
        for k, v in (d.get("__extras__") or {}).items():
            p.set_attrs(**{k: v})
        for s in d.get("studies", {}).values():
            p.add_study(StudyNode.from_dict(s, parent_patient=p))

        return p


@dataclass(slots=True)
class StudyNode(_ExtensibleAttrs):
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
        if not s.SeriesDescription and desc:
            s.SeriesDescription = desc
        return s

    # --- access / dunder ---
    def get_series(self, series_uid: str) -> Optional[SeriesNode]:
        return self.series.get(series_uid)

    def get_intance(
        self, sop_instance_uid: str, series_uid: Optional[str] = None
    ) -> Optional["InstanceNode"]:
        """
        Retrieve an InstanceNode (by SOPInstanceUID) from this study.

        Parameters
        ----------
        sop_instance_uid : str
            The SOPInstanceUID of the instance to retrieve.
        series_uid : str, optional
            If provided, restrict the search to this SeriesInstanceUID for O(1) lookup.

        Returns
        -------
        InstanceNode or None
            The InstanceNode associated with the given SOPInstanceUID, or None if not found.

        Notes
        -----
        - If `series_uid` is provided, lookup is done only in that series (fast path).
        - Otherwise, the method scans all series in the study and returns the first match.
        """
        # Fast path: restrict to a known series
        if series_uid is not None:
            se = self.get_series(series_uid)
            if se is None:
                return None
            return se.get_instance(sop_instance_uid)

        # General path: search all series
        for se in self.series.values():
            inst = se.get_instance(sop_instance_uid)
            if inst is not None:
                return inst
        return None

    def __getitem__(self, series_uid: str) -> SeriesNode:
        return self.series[series_uid]

    def __contains__(self, series_uid: str) -> bool:
        return series_uid in self.series

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self) -> Iterator[SeriesNode]:
        return iter(self.series.values())

    def __getattr__(self, name: str) -> Any:
        # 1) extras
        try:
            return _ExtensibleAttrs._get_extra_maybe(self, name)
        except AttributeError:
            pass
        # 2) delegate to parent
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
            "__extras__": dict(self.iter_attrs()),
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
        for k, v in (d.get("__extras__") or {}).items():
            st.set_attrs(**{k: v})
        for s in d.get("series", {}).values():
            st.add_series(SeriesNode.from_dict(s, parent_study=st))

        return st


@dataclass(slots=True)
class SeriesNode(_ExtensibleAttrs):
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
    raw_series_reference_uid: Optional[str] = None

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

        # Track file path if present; avoid duplicates
        file_path = getattr(instance, "FilePath", None)
        if file_path:
            if file_path not in self.instance_paths:
                self.instance_paths.append(file_path)

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
        # 1) extras
        try:
            return _ExtensibleAttrs._get_extra_maybe(self, name)
        except AttributeError:
            pass
        # 2) delegate to parent
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
            "raw_series_reference_uid": (
                self.raw_series_reference.SeriesInstanceUID if self.raw_series_reference else None
            ),
            "__extras__": dict(self.iter_attrs()),
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
        se.raw_series_reference_uid = d.get("raw_series_reference_uid")
        for k, v in (d.get("__extras__") or {}).items():
            se.set_attrs(**{k: v})
        for inst in d.get("instances", {}).values():
            se.add_instance(InstanceNode.from_dict(inst, parent_series=se))
        return se


@dataclass(slots=True)
class InstanceNode(_ExtensibleAttrs):
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
    parent_series: Optional["SeriesNode"] = None

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

    # source of dicom
    sources: List[str] = field(default_factory=list)

    # --- visitors ---
    def accept(self, visitor):
        return visitor.visit_instance(self)

    # --- convenience ---
    @property
    def primary_for_uid(self) -> Optional[str]:
        """Return a representative FrameOfReferenceUID if present (first in list)."""
        return self.FrameOfReferenceUIDs[0] if self.FrameOfReferenceUIDs else None

    def __getattr__(self, name: str) -> Any:
        # 1) extras
        try:
            return _ExtensibleAttrs._get_extra_maybe(self, name)
        except AttributeError:
            pass
        # 2) delegate to parent
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
            "sources": list(self.sources),
            "Modality": self.Modality,
            "FrameOfReferenceUIDs": list(self.FrameOfReferenceUIDs),
            "referenced_sop_instance_uids": list(self.referenced_sop_instance_uids),
            "referenced_sids": list(self.referenced_sids),
            "other_referenced_sids": list(self.other_referenced_sids),
            "__extras__": dict(self.iter_attrs()),
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
        for k, v in (d.get("__extras__") or {}).items():
            inst.set_attrs(**{k: v})
        return inst
