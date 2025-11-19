from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pydicom.dataset import Dataset
from pydicom.uid import RTPlanStorage


class RTPlan(Dataset):
    """
    Convenience wrapper around a DICOM RT Plan (:class:`pydicom.dataset.Dataset`)
    with helpers for beams, fraction groups, and referenced objects.

    This class is intended to be constructed from an existing RTPLAN dataset
    via :meth:`RTPLAN.from_dataset`.

    Notes
    -----
    This class does **not** modify or reinterpret the underlying DICOM content.
    It simply adds Pythonic accessors for commonly used plan information, such as
    beams, control points, and referenced structure sets.
    """

    # ------------- Construction helpers -------------

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "RTPlan":
        """
        Create an :class:`RTPLAN` instance from a pre-loaded DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            A DICOM dataset whose Modality is ``"RTPLAN"``.

        Returns
        -------
        RTPLAN
            A shallow copy of the input dataset, upgraded to :class:`RTPLAN`.

        Raises
        ------
        TypeError
            If ``ds`` is not a :class:`pydicom.dataset.Dataset`.
        ValueError
            If ``ds.Modality`` is not ``"RTPLAN"``.
        """
        if not isinstance(ds, Dataset):
            raise TypeError("RTPLAN.from_dataset expects a pydicom.Dataset instance.")

        modality = getattr(ds, "Modality", None)
        if modality != "RTPLAN":
            raise ValueError(f"Dataset Modality must be 'RTPLAN', got {modality!r}.")

        sop_class_uid = getattr(ds, "SOPClassUID", None)
        if sop_class_uid is not None and str(sop_class_uid) != RTPlanStorage:
            # You can log a warning here if you have a logger.
            pass

        new_ds = cls()
        new_ds.update(ds)
        return new_ds

    # ------------- Basic plan metadata -------------

    @property
    def plan_label(self) -> Optional[str]:
        """Return ``RTPlanLabel`` or ``None`` if not present."""
        return getattr(self, "RTPlanLabel", None)

    @property
    def plan_name(self) -> Optional[str]:
        """Return ``RTPlanName`` or ``None`` if not present."""
        return getattr(self, "RTPlanName", None)

    @property
    def plan_description(self) -> Optional[str]:
        """Return ``RTPlanDescription`` or ``None`` if not present."""
        return getattr(self, "RTPlanDescription", None)

    # ------------- Beam / fraction group helpers -------------

    @property
    def beam_sequence(self) -> List[Dataset]:
        """
        The DICOM ``BeamSequence`` as a list.

        Returns an empty list if ``BeamSequence`` is not present.
        """
        return list(getattr(self, "BeamSequence", []) or [])

    @property
    def num_beams(self) -> int:
        """Number of beams in the plan (length of :attr:`beam_sequence`)."""
        return len(self.beam_sequence)

    def iter_beams(self) -> Iterable[Dataset]:
        """Iterate over beams in ``BeamSequence``."""
        yield from self.beam_sequence

    def get_beam_by_number(self, beam_number: int) -> Optional[Dataset]:
        """
        Return the beam whose DICOM ``BeamNumber`` matches ``beam_number``.

        Parameters
        ----------
        beam_number : int
            The DICOM beam number to search for.

        Returns
        -------
        pydicom.Dataset or None
            The matching beam dataset, or ``None`` if not found.
        """
        for beam in self.beam_sequence:
            if getattr(beam, "BeamNumber", None) == beam_number:
                return beam
        return None

    @property
    def fraction_group_sequence(self) -> List[Dataset]:
        """
        The DICOM ``FractionGroupSequence`` as a list.

        Returns an empty list if ``FractionGroupSequence`` is not present.
        """
        return list(getattr(self, "FractionGroupSequence", []) or [])

    @property
    def num_fraction_groups(self) -> int:
        """Number of fraction groups in the plan."""
        return len(self.fraction_group_sequence)

    # ------------- Referenced objects -------------

    @property
    def referenced_structure_set_uid(self) -> Optional[str]:
        """
        Return the referenced Structure Set SOPInstanceUID, if present.

        This uses the first item in ``ReferencedStructureSetSequence`` (if any).
        """
        seq = getattr(self, "ReferencedStructureSetSequence", None)
        if not seq:
            return None
        item = seq[0]
        return getattr(item, "ReferencedSOPInstanceUID", None)

    @property
    def referenced_dose_uids(self) -> List[str]:
        """
        Return a list of referenced RTDOSE SOPInstanceUIDs, if present.

        This inspects ``ReferencedDoseSequence``.
        """
        uids: List[str] = []
        ref_dose_seq = getattr(self, "ReferencedDoseSequence", None)
        if ref_dose_seq:
            for item in ref_dose_seq:
                uid = getattr(item, "ReferencedSOPInstanceUID", None)
                if uid is not None:
                    uids.append(str(uid))
        return uids

    # ------------- Control point helpers -------------

    def iter_control_points(
        self,
        beam_number: Optional[int] = None,
    ) -> Iterable[Dataset]:
        """
        Iterate over control points in the plan.

        Parameters
        ----------
        beam_number : int, optional
            If provided, only iterate over control points for the specified beam.
            Otherwise, iterate over control points for all beams.

        Yields
        ------
        pydicom.Dataset
            Control point datasets from each beam's ``ControlPointSequence``.
        """
        beams = (
            [self.get_beam_by_number(beam_number)]
            if beam_number is not None
            else self.beam_sequence
        )

        for beam in beams:
            if beam is None:
                continue
            cps = getattr(beam, "ControlPointSequence", None)
            if not cps:
                continue
            for cp in cps:
                yield cp

    # ------------- Beam geometry helpers -------------

    def get_beam_isocenter(
        self,
        beam_number: int,
        cp_index: int = 0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Return the isocenter position for a given beam and control point.

        Parameters
        ----------
        beam_number : int
            DICOM ``BeamNumber`` of the desired beam.
        cp_index : int, optional
            Index into the beam's ``ControlPointSequence`` (default: 0).

        Returns
        -------
        tuple of float or None
            (x, y, z) isocenter in DICOM patient coordinates (mm), or ``None``
            if not available.
        """
        beam = self.get_beam_by_number(beam_number)
        if beam is None:
            return None

        cps = getattr(beam, "ControlPointSequence", None)
        if not cps or cp_index < 0 or cp_index >= len(cps):
            return None

        cp = cps[cp_index]
        iso = getattr(cp, "IsocenterPosition", None)
        if iso is None or len(iso) != 3:
            return None

        return tuple(float(v) for v in iso)

    @property
    def beam_isocenters(self) -> Dict[int, Tuple[float, float, float]]:
        """
        Convenience mapping of BeamNumber -> isocenter (from first control point).

        Returns
        -------
        dict
            Keys are ``BeamNumber`` (int), values are (x, y, z) tuples (mm).
            Beams without an isocenter are omitted.
        """
        iso_map: Dict[int, Tuple[float, float, float]] = {}
        for beam in self.beam_sequence:
            bn = getattr(beam, "BeamNumber", None)
            if bn is None:
                continue
            iso = self.get_beam_isocenter(bn, cp_index=0)
            if iso is not None:
                iso_map[int(bn)] = iso
        return iso_map

    def _get_angles_for_beams(
        self,
        tag_name: str,
        beam_number: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Internal helper to extract sequences of angles per beam.

        Parameters
        ----------
        tag_name : str
            Name of the control point attribute, e.g. ``"GantryAngle"``.
        beam_number : int, optional
            If provided, only this beam is inspected.

        Returns
        -------
        dict
            Mapping from BeamNumber -> list of angles (float).
        """
        result: Dict[int, List[float]] = {}

        beams = (
            [self.get_beam_by_number(beam_number)]
            if beam_number is not None
            else self.beam_sequence
        )

        for beam in beams:
            if beam is None:
                continue
            bn = getattr(beam, "BeamNumber", None)
            if bn is None:
                continue

            cps = getattr(beam, "ControlPointSequence", None) or []
            angles: List[float] = []
            for cp in cps:
                if hasattr(cp, tag_name):
                    angles.append(float(getattr(cp, tag_name)))
            result[int(bn)] = angles

        return result

    def get_gantry_angles(
        self,
        beam_number: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Get gantry angle trajectories per beam.

        Parameters
        ----------
        beam_number : int, optional
            If provided, only that beam is returned; otherwise, all beams.

        Returns
        -------
        dict
            Mapping from BeamNumber -> list of gantry angles (degrees).
        """
        return self._get_angles_for_beams("GantryAngle", beam_number)

    def get_collimator_angles(
        self,
        beam_number: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Get collimator angle trajectories per beam.

        Parameters
        ----------
        beam_number : int, optional
            If provided, only that beam is returned; otherwise, all beams.

        Returns
        -------
        dict
            Mapping from BeamNumber -> list of collimator angles (degrees).
        """
        return self._get_angles_for_beams("BeamLimitingDeviceAngle", beam_number)

    def get_couch_angles(
        self,
        beam_number: Optional[int] = None,
    ) -> Dict[int, List[float]]:
        """
        Get couch (patient support) angle trajectories per beam.

        Parameters
        ----------
        beam_number : int, optional
            If provided, only that beam is returned; otherwise, all beams.

        Returns
        -------
        dict
            Mapping from BeamNumber -> list of couch angles (degrees).
        """
        # Depending on TPS, this might be PatientSupportAngle or TableTopEccentricAngle.
        # We prefer PatientSupportAngle if present.
        angles = self._get_angles_for_beams("PatientSupportAngle", beam_number)
        # If nothing found, fall back to TableTopEccentricAngle
        if not any(angles.values()):
            angles = self._get_angles_for_beams("TableTopEccentricAngle", beam_number)
        return angles

    # --- Leaf/jaw helpers ---

    def _beam_device_type_index_map(self, beam: Dataset) -> Dict[str, int]:
        """
        Build a mapping RTBeamLimitingDeviceType -> index in BeamLimitingDeviceSequence.
        """
        mapping: Dict[str, int] = {}
        blds = getattr(beam, "BeamLimitingDeviceSequence", None) or []
        for idx, item in enumerate(blds):
            dev_type = getattr(item, "RTBeamLimitingDeviceType", None)
            if dev_type:
                mapping[str(dev_type)] = idx
        return mapping

    def get_leafjaw_positions(
        self,
        beam_number: int,
        device_type: str,
        cp_index: int = 0,
    ) -> Optional[List[float]]:
        """
        Get leaf/jaw positions for a given device type and control point.

        Parameters
        ----------
        beam_number : int
            DICOM ``BeamNumber`` of the desired beam.
        device_type : str
            DICOM device type from ``BeamLimitingDeviceSequence``, e.g.
            ``"MLCX"``, ``"MLCY"``, ``"ASYMX"``, ``"ASYMY"``, ``"X"``, ``"Y"``.
        cp_index : int, optional
            Index into the beam's ``ControlPointSequence`` (default: 0).

        Returns
        -------
        list of float or None
            The raw ``LeafJawPositions`` for that device and control point,
            or ``None`` if not present.

        Notes
        -----
        - For MLC devices, this typically returns a flat list of leaf edge
          positions (mm).
        - For jaw devices, this usually contains [neg, pos] positions (mm).
        """
        beam = self.get_beam_by_number(beam_number)
        if beam is None:
            return None

        cps = getattr(beam, "ControlPointSequence", None)
        if not cps or cp_index < 0 or cp_index >= len(cps):
            return None

        dev_index_map = self._beam_device_type_index_map(beam)
        if device_type not in dev_index_map:
            return None

        cp = cps[cp_index]
        pos_seq = getattr(cp, "BeamLimitingDevicePositionSequence", None) or []
        idx = dev_index_map[device_type]
        if idx < 0 or idx >= len(pos_seq):
            return None

        item = pos_seq[idx]
        positions = getattr(item, "LeafJawPositions", None)
        if positions is None:
            return None

        return [float(v) for v in positions]

    def get_jaw_positions(
        self,
        beam_number: int,
        axis: str = "X",
        cp_index: int = 0,
    ) -> Optional[Tuple[float, float]]:
        """
        Convenience helper for X/Y jaw positions for a given control point.

        Parameters
        ----------
        beam_number : int
            DICOM ``BeamNumber``.
        axis : {"X", "Y"}, optional
            Jaw axis to query.
        cp_index : int, optional
            Control point index (default: 0).

        Returns
        -------
        (float, float) or None
            (neg, pos) jaw positions in mm, or ``None`` if not available.
        """
        axis = axis.upper()
        if axis not in {"X", "Y"}:
            raise ValueError("axis must be 'X' or 'Y'")

        # Common device type names for jaws
        device_types = {
            "X": ["ASYMX", "X"],
            "Y": ["ASYMY", "Y"],
        }
        for dev_type in device_types[axis]:
            positions = self.get_leafjaw_positions(beam_number, dev_type, cp_index)
            if positions is not None and len(positions) >= 2:
                return float(positions[0]), float(positions[1])

        return None

    def get_mlc_leaf_positions(
        self,
        beam_number: int,
        cp_index: int = 0,
        device_type_candidates: Optional[List[str]] = None,
    ) -> Optional[List[float]]:
        """
        Convenience helper for MLC leaf positions for a given control point.

        Parameters
        ----------
        beam_number : int
            DICOM ``BeamNumber``.
        cp_index : int, optional
            Control point index (default: 0).
        device_type_candidates : list of str, optional
            Ordered list of device types to try, e.g. ``["MLCX", "MLCY"]``.
            If None, defaults to ``["MLCX", "MLCY"]``.

        Returns
        -------
        list of float or None
            Flat list of leaf positions (mm) for the selected MLC device,
            or ``None`` if no MLC device is found.
        """
        if device_type_candidates is None:
            device_type_candidates = ["MLCX", "MLCY"]

        for dev_type in device_type_candidates:
            positions = self.get_leafjaw_positions(beam_number, dev_type, cp_index)
            if positions is not None:
                return positions

        return None
