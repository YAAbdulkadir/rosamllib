from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from pydicom.dataset import Dataset
from pydicom.uid import (
    RTBeamsTreatmentRecordStorage,
    RTBrachyTreatmentRecordStorage,
    RTTreatmentSummaryRecordStorage,
)


class RTRecord(Dataset):
    """
    Convenience wrapper around a DICOM RTRECORD (:class:`pydicom.dataset.Dataset`)
    with helpers for delivered beams, fractions, and dose references.

    This class is intended to be constructed from an existing RTRECORD dataset
    via :meth:`RTRecord.from_dataset`.

    Notes
    -----
    This class does **not** modify or reinterpret the underlying DICOM content.
    It simply adds Pythonic accessors for commonly used RT treatment record
    information, such as treatment session beams, delivered metersets/doses, and
    references back to the planning objects.
    """

    # ------------- Construction helpers -------------

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "RTRecord":
        """
        Create an :class:`RTRecord` instance from a pre-loaded DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            A DICOM dataset whose Modality is ``"RTRECORD"``.

        Returns
        -------
        RTRecord
            A shallow copy of the input dataset, upgraded to :class:`RTRecord`.

        Raises
        ------
        TypeError
            If ``ds`` is not a :class:`pydicom.dataset.Dataset`.
        ValueError
            If ``ds.Modality`` is not ``"RTRECORD"``.
        """
        if not isinstance(ds, Dataset):
            raise TypeError("RTRecord.from_dataset expects a pydicom.Dataset instance.")

        modality = getattr(ds, "Modality", None)
        if modality != "RTRECORD":
            raise ValueError(f"Dataset Modality must be 'RTRECORD', got {modality!r}.")

        # Optional SOP Class sanity check (any of the RT*TreatmentRecord storages)
        sop_class_uid = getattr(ds, "SOPClassUID", None)
        if sop_class_uid is not None:
            uid_str = str(sop_class_uid)
            valid_uids = {
                str(RTBeamsTreatmentRecordStorage),
                str(RTBrachyTreatmentRecordStorage),
                str(RTTreatmentSummaryRecordStorage),
            }
            if uid_str not in valid_uids:
                # You can log a warning here if you have a logger.
                pass

        new_ds = cls()
        new_ds.update(ds)
        return new_ds

    # ------------- Basic metadata / references -------------

    @property
    def referenced_rtplan_uid(self) -> Optional[str]:
        """
        Return the referenced RTPLAN SOPInstanceUID, if present.

        This uses the first item in ``ReferencedRTPlanSequence`` (if any).
        """
        seq = getattr(self, "ReferencedRTPlanSequence", None)
        if not seq:
            return None
        item = seq[0]
        return getattr(item, "ReferencedSOPInstanceUID", None)

    @property
    def treatment_date(self) -> Optional[str]:
        """Treatment date from ``TreatmentDate`` (DA), if present."""
        return getattr(self, "TreatmentDate", None)

    @property
    def treatment_time(self) -> Optional[str]:
        """Treatment time from ``TreatmentTime`` (TM), if present."""
        return getattr(self, "TreatmentTime", None)

    # ------------- Treatment session beams -------------

    @property
    def treatment_session_beam_sequence(self) -> List[Dataset]:
        """
        The DICOM ``TreatmentSessionBeamSequence`` as a list.

        Returns an empty list if ``TreatmentSessionBeamSequence`` is not present.
        """
        return list(getattr(self, "TreatmentSessionBeamSequence", []) or [])

    @property
    def num_session_beams(self) -> int:
        """Number of treatment session beams."""
        return len(self.treatment_session_beam_sequence)

    def iter_session_beams(self) -> Iterable[Dataset]:
        """Iterate over items in ``TreatmentSessionBeamSequence``."""
        yield from self.treatment_session_beam_sequence

    def get_session_beam_by_number(self, session_beam_number: int) -> Optional[Dataset]:
        """
        Return the treatment session beam whose ``TreatmentSessionBeamNumber`` matches.

        Parameters
        ----------
        session_beam_number : int
            The DICOM TreatmentSessionBeamNumber to search for.

        Returns
        -------
        pydicom.Dataset or None
            The matching session beam dataset, or ``None`` if not found.
        """
        for beam in self.treatment_session_beam_sequence:
            if getattr(beam, "TreatmentSessionBeamNumber", None) == session_beam_number:
                return beam
        return None

    def get_session_beam_by_referenced_beam(
        self, referenced_beam_number: int
    ) -> Optional[Dataset]:
        """
        Return the treatment session beam whose ``ReferencedBeamNumber`` matches.

        This is often the most convenient way to link delivered beams back to
        the planning RTPLAN beams.

        Parameters
        ----------
        referenced_beam_number : int
            The beam number from the RTPLAN.

        Returns
        -------
        pydicom.Dataset or None
            The matching session beam dataset, or ``None`` if not found.
        """
        for beam in self.treatment_session_beam_sequence:
            if getattr(beam, "ReferencedBeamNumber", None) == referenced_beam_number:
                return beam
        return None

    # --- Delivered meterset / dose per beam ---

    def get_beam_meterset(self, referenced_beam_number: int) -> Optional[float]:
        """
        Get the delivered meterset for a given RTPLAN beam number.

        Parameters
        ----------
        referenced_beam_number : int
            BeamNumber from the RTPLAN.

        Returns
        -------
        float or None
            Beam meterset (MU) if ``BeamMeterset`` is present, else ``None``.
        """
        beam = self.get_session_beam_by_referenced_beam(referenced_beam_number)
        if beam is None:
            return None
        meterset = getattr(beam, "BeamMeterset", None)
        return float(meterset) if meterset is not None else None

    def get_beam_dose(self, referenced_beam_number: int) -> Optional[float]:
        """
        Get the delivered dose for a given RTPLAN beam number.

        Parameters
        ----------
        referenced_beam_number : int
            BeamNumber from the RTPLAN.

        Returns
        -------
        float or None
            Delivered beam dose (as encoded in ``BeamDose``), or ``None`` if
            not present.
        """
        beam = self.get_session_beam_by_referenced_beam(referenced_beam_number)
        if beam is None:
            return None
        dose = getattr(beam, "BeamDose", None)
        return float(dose) if dose is not None else None

    @property
    def beam_metersets(self) -> Dict[int, float]:
        """
        Mapping of ReferencedBeamNumber -> delivered meterset (MU).

        Only beams that have both ``ReferencedBeamNumber`` and ``BeamMeterset``
        are included.
        """
        result: Dict[int, float] = {}
        for beam in self.treatment_session_beam_sequence:
            ref_bn = getattr(beam, "ReferencedBeamNumber", None)
            meterset = getattr(beam, "BeamMeterset", None)
            if ref_bn is not None and meterset is not None:
                result[int(ref_bn)] = float(meterset)
        return result

    @property
    def beam_doses(self) -> Dict[int, float]:
        """
        Mapping of ReferencedBeamNumber -> delivered beam dose.

        Only beams that have both ``ReferencedBeamNumber`` and ``BeamDose``
        are included.
        """
        result: Dict[int, float] = {}
        for beam in self.treatment_session_beam_sequence:
            ref_bn = getattr(beam, "ReferencedBeamNumber", None)
            dose = getattr(beam, "BeamDose", None)
            if ref_bn is not None and dose is not None:
                result[int(ref_bn)] = float(dose)
        return result

    # ------------- Fraction group summaries -------------

    @property
    def fraction_group_summary_sequence(self) -> List[Dataset]:
        """
        The DICOM ``FractionGroupSummarySequence`` as a list.

        Returns an empty list if not present.
        """
        return list(getattr(self, "FractionGroupSummarySequence", []) or [])

    @property
    def num_fraction_group_summaries(self) -> int:
        """Number of items in ``FractionGroupSummarySequence``."""
        return len(self.fraction_group_summary_sequence)

    @property
    def fraction_group_doses(self) -> Dict[int, float]:
        """
        Mapping of ReferencedFractionGroupNumber -> dose for that fraction group.

        This inspects ``FractionGroupSummarySequence`` and uses attributes
        such as ``ReferencedFractionGroupNumber`` and ``BeamDose`` if present.

        Notes
        -----
        The exact dose attributes present may vary by TPS; this helper is a
        best-effort summary and can be extended later for specific vendors.
        """
        result: Dict[int, float] = {}
        for item in self.fraction_group_summary_sequence:
            fg_num = getattr(item, "ReferencedFractionGroupNumber", None)
            fg_dose = getattr(item, "BeamDose", None)
            if fg_num is not None and fg_dose is not None:
                result[int(fg_num)] = float(fg_dose)
        return result

    # ------------- Delivered dose references -------------

    @property
    def delivered_dose_reference_sequence(self) -> List[Dataset]:
        """
        The DICOM ``DeliveredDoseReferenceSequence`` as a list.

        Returns an empty list if not present.
        """
        return list(getattr(self, "DeliveredDoseReferenceSequence", []) or [])

    @property
    def dose_reference_doses(self) -> Dict[int, float]:
        """
        Mapping of DoseReferenceNumber -> delivered dose to that reference.

        This inspects ``DeliveredDoseReferenceSequence`` and collects
        ``DoseReferenceNumber`` and ``CumulativeDoseToDoseReference``.

        Returns
        -------
        dict
            Keys: dose reference numbers (int).
            Values: delivered cumulative dose (float).
        """
        result: Dict[int, float] = {}
        for item in self.delivered_dose_reference_sequence:
            num = getattr(item, "DoseReferenceNumber", None)
            dose = getattr(item, "CumulativeDoseToDoseReference", None)
            if num is not None and dose is not None:
                result[int(num)] = float(dose)
        return result
