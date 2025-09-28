import os
from typing import List
import numpy as np
from pathlib import Path
from pydicom import dcmread
import SimpleITK as sitk


def sort_by_image_position_patient(file_names_or_datasets, reverse=False):
    """
    Sorts DICOM image files or datasets based on their position along the imaging axis.

    This function sorts a list of DICOM file paths or datasets based on the ImagePositionPatient
    tag and the orientation of the image (using ImageOrientationPatient).

    Parameters
    ----------
    file_names_or_datasets : list of str or list of pydicom.Dataset
        The list of DICOM file paths or datasets to sort.

    Returns
    -------
    list of str or list of pydicom.Dataset
        The sorted list of DICOM file paths or datasets.

    Notes
    -----
    This function computes the imaging axis using the ImageOrientationPatient tag
    and sorts the files based on the ImagePositionPatient tag along that axis.

    Examples
    --------
    >>> sorted_files = sort_by_image_position_patient(dicom_file_list)
    >>> print(sorted_files)
    ['file1.dcm', 'file2.dcm', 'file3.dcm']
    """

    def get_image_position_along_imaging_axis(ds):
        try:
            if isinstance(ds, (str, Path)):
                ds = dcmread(ds, stop_before_pixels=True)

            image_position_patient = np.array(ds.ImagePositionPatient, dtype=float)
            image_orientation_patient = np.array(ds.ImageOrientationPatient, dtype=float)
            row_cosines = image_orientation_patient[:3]
            col_cosines = image_orientation_patient[3:]
            imaging_axis = np.cross(row_cosines, col_cosines)
            return np.dot(image_position_patient, imaging_axis)
        except Exception as e:
            print(f"Could not read dataset: {e}")
            return float("inf")

    sorted_items = sorted(
        file_names_or_datasets, key=get_image_position_along_imaging_axis, reverse=reverse
    )
    return sorted_items


def get_referenced_sop_instance_uids(ds):
    """
    Extracts referenced SOPClassUIDs along with their associated SOPInstanceUIDs
    from RTSTRUCT, RTPLAN, and RTDOSE DICOM files, ensuring no duplicates.

    This method scans the DICOM dataset for references to other DICOM instances
    and returns a dictionary where keys are SOPClassUIDs and values are lists
     of unique associated SOPInstanceUIDs.

    Parameters
    ----------
    ds : pydicom.Dataset
        The DICOM dataset to extract references from.

    Returns
    -------
    dict
        A dictionary where keys are SOPClassUIDs and values are sets of
        unique associated SOPInstanceUIDs.

    Examples
    --------
    >>> sop_class_to_uids = DICOMLoader.get_referenced_sop_class_to_instance_uids(ds)
    >>> print(sop_class_to_uids)
    {'1.2.840.10008.5.1.4.1.1.2': {'1.2.3.4.5.6.7', '1.2.3.4.5.6.9'},
     '1.2.840.10008.5.1.4.1.1.4': {'1.2.3.4.5.6.8'}}
    """
    sop_class_to_instance_uids = {}

    def add_reference(item):
        if hasattr(item, "ReferencedSOPInstanceUID") and hasattr(item, "ReferencedSOPClassUID"):
            sop_instance_uid = item.ReferencedSOPInstanceUID
            sop_class_uid = item.ReferencedSOPClassUID
            if sop_class_uid not in sop_class_to_instance_uids:
                sop_class_to_instance_uids[sop_class_uid] = set()
            sop_class_to_instance_uids[sop_class_uid].add(sop_instance_uid)

    if ds.Modality == "RTSTRUCT":
        if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
            for item in ds.ReferencedFrameOfReferenceSequence:
                if hasattr(item, "RTReferencedStudySequence"):
                    for study_item in item.RTReferencedStudySequence:
                        if hasattr(study_item, "RTReferencedSeriesSequence"):
                            for series_item in study_item.RTReferencedSeriesSequence:
                                if hasattr(series_item, "ContourImageSequence"):
                                    for contour_item in series_item.ContourImageSequence:
                                        add_reference(contour_item)
        if hasattr(ds, "ROIContourSequence"):
            for roi_item in ds.ROIContourSequence:
                if hasattr(roi_item, "ContourSequence"):
                    for contour_seq in roi_item.ContourSequence:
                        if hasattr(contour_seq, "ContourImageSequence"):
                            for image_seq in contour_seq.ContourImageSequence:
                                add_reference(image_seq)

    else:
        if hasattr(ds, "ReferencedStructureSetSequence"):
            for item in ds.ReferencedStructureSetSequence:
                if hasattr(item, "ReferencedSOPInstanceUID"):
                    add_reference(item)

        if hasattr(ds, "ReferencedDoseSequence"):
            for item in ds.ReferencedDoseSequence:
                if hasattr(item, "ReferencedSOPInstanceUID"):
                    add_reference(item)

        if hasattr(ds, "ReferencedRTPlanSequence"):
            for item in ds.ReferencedRTPlanSequence:
                if hasattr(item, "ReferencedSOPInstanceUID"):
                    add_reference(item)

    return {k: list(v) for k, v in sop_class_to_instance_uids.items()}


def extract_rtstruct_for_uids(ds) -> List[str]:
    uids = set()
    if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
        for it in ds.ReferencedFrameOfReferenceSequence:
            if hasattr(it, "FrameOfReferenceUID"):
                uids.add(str(it.FrameOfReferenceUID))
    if hasattr(ds, "StructureSetROISequence"):
        for it in ds.StructureSetROISequence:
            if hasattr(it, "ReferencedFrameOfReferenceUID"):
                uids.add(str(it.ReferencedFrameOfReferenceUID))
    return sorted(uids)


def validate_dicom_path(path):
    """
    Validates whether the provided path is a valid file or directory.

    This function checks if the given path exists on the filesystem. It raises an error
    if the path does not exist or if it is neither a file nor a directory.

    Parameters
    ----------
    path : str
        The file system path to be validated.

    Raises
    ------
    IOError
        If the provided path does not exist.
    IOError
        If the provided path is neither a file nor a directory.
    """
    if not os.path.exists(path):
        raise IOError(f"Provided path does not exist: {path}")
    if not (os.path.isfile(path) or os.path.isdir(path)):
        raise IOError(f"Provided path is neither a file nor a directory: {path}")


def compute_dvh(
    dose_image,
    roi_masks,
    relative_volume=True,
    relative_dose=False,
    prescription_dose=None,
    bin_width=0.1,
):
    """
    Computes the Dose-Volume Histogram (DVH) for each ROI.

    Parameters:
    ----------
    dose_image : SimpleITK.Image
        A SimpleITK image containing the dose distribution.
    roi_masks : dict
        A dictionary where keys are ROI names (str) and values are binary SimpleITK images
        representing the masks for each ROI (0 for background, 1 for ROI).
    relative_volume : bool
        If True, volume is expressed as a percentage of the total ROI volume; otherwise,
        it is in absolute units (e.g., mm³).
    relative_dose : bool
        If True, dose bins are expressed as percentages of the prescription dose (if provided)
        or maximum dose (if prescription_dose is None).
    prescription_dose : float, optional
        If provided, the relative dose will be calculated as a percentage of the prescription dose.
    bin_width : float
        The width of each dose bin (Gy) when relative_dose=False,
        or in percentage when relative_dose=True.
    Returns:
    -------
    dict
        A dictionary where keys are ROI names and values are dictionaries containing:
        - 'dose_bins': Array of dose bin edges (Gy or % if relative_dose=True).
        - 'volume': Array of volume fractions (or absolute volumes if relative_volume=False).
    """
    dose_array = sitk.GetArrayFromImage(dose_image)
    if dose_image.HasMetaDataKey("3004|000e"):
        scaling_factor = float(dose_image.GetMetaData("3004|000e"))
    else:
        scaling_factor = 1
    dose_array = dose_array * scaling_factor
    max_dose = np.max(dose_array)

    def _compute_dvh(dose_array, mask_image, bin_width=bin_width):
        voxel_volume = np.prod(mask_image.GetSpacing())

        mask_array = sitk.GetArrayFromImage(mask_image).astype(bool)
        structure_dose = dose_array[mask_array]
        if len(structure_dose) > 0:
            roi_max_dose = np.max(structure_dose)
        else:
            roi_max_dose = 0
        bin_edges = np.arange(0, roi_max_dose + bin_width, bin_width)
        hist, _ = np.histogram(structure_dose, bins=bin_edges)

        cumulative_volume = np.cumsum(hist[::-1])[::-1] * voxel_volume
        return bin_edges[:-1], cumulative_volume

    structure_dvhs = {}
    for structure, mask_image in roi_masks.items():
        bin_edges, cumulative_volume = _compute_dvh(dose_array, mask_image)
        if relative_volume and len(cumulative_volume) > 0:
            cumulative_volume = cumulative_volume / cumulative_volume[0] * 100
        if relative_dose:
            if prescription_dose:
                bin_edges = bin_edges / prescription_dose * 100
            else:
                bin_edges = bin_edges / max_dose * 100
        structure_dvhs[structure] = (bin_edges, cumulative_volume)

    return structure_dvhs
