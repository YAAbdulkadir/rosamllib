import os
import re
import tempfile
from typing import Any, Dict, List, Union
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path
from datetime import datetime
from pydicom import dcmread
from pydicom.datadict import dictionary_VR
from pydicom.multival import MultiValue
from rosamllib.constants import VR_TO_DTYPE
from ipaddress import ip_address


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


def query_df(df: pd.DataFrame, **filters: Union[str, List[Any], Dict[str, Any]]) -> pd.DataFrame:
    """
    Filters a Pandas DataFrame based on a set of conditions, including wildcards (*, ?),
    ranges, lists, regular expressions, and inverse regular expressions. Supports escaping
    for literal wildcards.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to query.
    **filters : dict
        A set of filter conditions passed as keyword arguments. Each key is a column
        name, and its value is a condition. Supported conditions:
        - Exact match: {"column": "value"}
        - Wildcards: {"column": "value*"} or {"column": "val?e"}
          (* matches multiple characters, ? matches one character).
        - Ranges: {"column": {"gte": min_value, "lte": max_value}}
        - Regular expressions: {"column": {"RegEx": "pattern"}}
        - Inverse regular expressions: {"column": {"NotRegEx": "pattern"}}
        - Escaped wildcards: {"column": "val\\*e"} to match the literal `*` or `?`.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame based on the conditions provided.

    Notes
    -----
    - If the filter value contains `*`, it will be treated as a wildcard matching zero
      or more characters. Similarly, `?` will match exactly one character.
    - To match the literal characters `*` or `?`, escape them with a backslash (\\),
      e.g., `{"column": "value\\*"}`

    Examples
    --------
    # Sample DataFrame
    >>> data = {
    ...     "PatientID": ["123", "456", "789", "101", "121"],
    ...     "StudyDate": ["2023-01-01", "2023-02-15", "2023-03-01", "2023-04-20", "2023-05-10"],
    ...     "Age": [30, 45, 29, 60, 35],
    ... }
    >>> df = pd.DataFrame(data)

    # Example 1: Wildcard and exact match
    >>> filters = {"PatientID": ["1*", "456"]}
    >>> query_df(df, **filters)
      PatientID StudyDate  Age
    0       123 2023-01-01   30
    3       101 2023-04-20   60
    4       121 2023-05-10   35
    1       456 2023-02-15   45

    # Example 2: Date range
    >>> filters = {"StudyDate": {"gte": "2023-03-01"}}
    >>> query_df(df, **filters)
      PatientID StudyDate  Age
    2       789 2023-03-01   29
    3       101 2023-04-20   60
    4       121 2023-05-10   35
    """

    def _apply_condition(column: str, condition: Any) -> pd.Series:
        """
        Applies a single condition to a column of the DataFrame.

        Parameters
        ----------
        column : str
            The column to apply the condition on.
        condition : Any
            The condition to apply (exact match, wildcard, range, RegEx, etc.).

        Returns
        -------
        pd.Series
            A boolean mask indicating the rows that match the condition.
        """

        def process_literal(value: str) -> str:
            """
            Process escaped literals for wildcards.

            Parameters
            ----------
            value : str
                The input string potentially containing escaped literals.

            Returns
            -------
            str
                A regex-safe pattern with escaped wildcards handled.
            """
            return (
                value.replace(r"\*", r"\x1B")  # Temporarily replace \* with \x1B
                .replace(r"\?", r"\x1C")  # Temporarily replace \? with \x1C
                .replace("*", ".*")  # Convert * to regex wildcard
                .replace("?", ".")  # Convert ? to regex wildcard
                .replace(r"\x1B", r"\*")  # Restore literal *
                .replace(r"\x1C", r"\?")  # Restore literal ?
            )

        # Exact match or wildcard
        if isinstance(condition, str):
            if "*" in condition or "?" in condition:  # Wildcard filtering
                pattern = process_literal(condition)
                return df[column].astype(str).str.match(f"^{pattern}$", na=False)
            else:  # Exact match
                return df[column] == condition

        # Complex filtering
        elif isinstance(condition, dict):
            mask = pd.Series(True, index=df.index)
            for op, value in condition.items():
                if op == "RegEx":  # RegEx matching
                    if not isinstance(value, str):
                        raise ValueError("RegEx operator requires a string pattern.")
                    mask &= df[column].astype(str).str.contains(value, na=False)
                elif op == "NotRegEx":  # Inverse RegEx matching
                    if not isinstance(value, str):
                        raise ValueError("NotRegEx operator requires a string pattern.")
                    mask &= ~df[column].astype(str).str.contains(value, na=False)
                elif isinstance(value, str) and ("*" in value or "?" in value):
                    # Convert wildcard to regex pattern
                    pattern = process_literal(value)
                    if op == "eq":  # Equal with wildcard
                        mask &= df[column].astype(str).str.match(f"^{pattern}$", na=False)
                    elif op == "neq":  # Not equal with wildcard
                        mask &= ~df[column].astype(str).str.match(f"^{pattern}$", na=False)
                    else:
                        raise ValueError(
                            f"Operator '{op}' does not support wildcards in range filters."
                        )
                else:
                    if op == "gte":  # Greater than or equal to
                        mask &= df[column] >= value
                    elif op == "lte":  # Less than or equal to
                        mask &= df[column] <= value
                    elif op == "gt":  # Greater than
                        mask &= df[column] > value
                    elif op == "lt":  # Less than
                        mask &= df[column] < value
                    elif op == "eq":  # Equal
                        mask &= df[column] == value
                    elif op == "neq":  # Not equal
                        mask &= df[column] != value
                    else:
                        raise ValueError(f"Unsupported operator '{op}' in range filter.")
            return mask

        # List of values
        elif isinstance(condition, list):
            return df[column].isin(condition)

        raise ValueError(f"Unsupported condition type for column '{column}'.")

    filtered_df = df.copy()

    for column, condition in filters.items():
        if column not in filtered_df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        if isinstance(condition, list):  # Multiple conditions for the same column
            combined_mask = pd.Series(False, index=filtered_df.index)
            for sub_condition in condition:
                combined_mask |= _apply_condition(column, sub_condition)
            filtered_df = filtered_df.loc[
                combined_mask.reindex(filtered_df.index, fill_value=False)
            ]
        else:
            mask = _apply_condition(column, condition)
            filtered_df = filtered_df.loc[mask.reindex(filtered_df.index, fill_value=False)]

    return filtered_df


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


def get_referenced_sop_instance_uids(ds):
    """
    Extracts referenced SOPClassUIDs along with their associated SOPInstanceUIDs
    from RTSTRUCT, RTPLAN, and RTDOSE DICOM files, ensuring no duplicates.

    This method scans the DICOM dataset for references to other DICOM instances
    and returns a dictionary where keys are SOPClassUIDs and values are sets
    (or lists, if converted) of unique associated SOPInstanceUIDs.

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


def sitk_to_nifti(sitk_image):
    """
    Convert a SimpleITK Image to a nibabel NIfTI image by writing to a temporary file.

    Parameters:
    sitk_image (SimpleITK.Image): The SimpleITK Image object.

    Returns:
    nibabel.Nifti1Image: The converted NIfTI image object.
    """
    # Create a temporary file to hold the NIfTI image
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as temp_nifti_file:
        temp_nifti_file_path = temp_nifti_file.name

    try:
        # Write the SimpleITK image to the temporary NIfTI file
        sitk.WriteImage(sitk_image, temp_nifti_file_path)

        # Read the temporary NIfTI file using nibabel
        nifti_image = nib.load(temp_nifti_file_path)

        data = nifti_image.get_fdata()
        nifti_image_loaded = nib.Nifti1Image(data, nifti_image.affine, nifti_image.header)
        nifti_image_loaded.extra = {}

    finally:
        # Remove the temporary file after it has been read
        os.remove(temp_nifti_file_path)

    return nifti_image_loaded


def nifti_to_sitk(nifti_image):
    """
    Convert a nibabel NIfTI image to a SimpleITK Image by writing to a temporary file.

    Parameters:
    nifti_image (nibabel.Nifti1Image): The NIfTI image object.

    Returns:
    SimpleITK.Image: The converted SimpleITK Image object.
    """
    data = nifti_image.get_fdata()
    nifti_image_loaded = nib.Nifti1Image(data, nifti_image.affine, nifti_image.header)
    nifti_image_loaded.extra = {}

    # Create a temporary file to hold the NIfTI image
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as temp_nifti_file:
        temp_nifti_file_path = temp_nifti_file.name

    try:
        # Write the NIfTI image to the temporary file using nibabel
        nib.save(nifti_image, temp_nifti_file_path)

        # Read the temporary NIfTI file using SimpleITK
        sitk_image = sitk.ReadImage(temp_nifti_file_path)
    finally:
        if os.path.exists(temp_nifti_file_path):
            # Remove the temporary file
            os.remove(temp_nifti_file_path)

    return sitk_image


def parse_vr_value(vr, value):
    """
    Parses DICOM tag values based on VR.

    Parameters
    ----------
    vr : str
        The VR of the DICOM tag.
    value : str
        The raw value of the DICOM tag.

    Returns
    -------
    Parsed value in the appropriate type (e.g., date, time).
    """
    if value:
        if vr == "DA":
            try:
                return datetime.strptime(value, "%Y%m%d").date()
            except ValueError:
                return None
        elif vr == "TM":
            try:
                return datetime.strptime(value, "%H%M%S.%f").time()
            except ValueError:
                try:
                    return datetime.strptime(value, "%H%M%S").time()
                except ValueError:
                    return None
        elif vr == "DT":
            try:
                return datetime.strptime(value, "%Y%m%d%H%M%S.%f")
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y%m%d%H%M%S")
                except ValueError:
                    return None
        elif vr in ["IS", "SL", "SS", "UL", "US"]:
            try:
                if isinstance(value, MultiValue):
                    return [int(v) for v in value]
                else:
                    return int(value)
            except ValueError:
                return None
        elif vr in ["DS", "FL", "FD"]:
            try:
                if isinstance(value, MultiValue):
                    return [float(v) for v in value]
                else:
                    return float(value)
            except ValueError:
                return None
        elif vr == "LO":
            try:
                if isinstance(value, MultiValue):
                    return [str(v) for v in value]
                else:
                    return str(value)
            except ValueError:
                return None

    return value


def get_pandas_column_dtype(tag):
    """
    Determines the Pandas dtype for a given DICOM tag based on its VR.

    Parameters
    ----------
    tag : tuple
        The DICOM tag in (group, element) format.

    Returns
    -------
    type or str
        The corresponding Pandas dtype, or `object` if the VR is unknown.
    """
    try:
        vr = dictionary_VR(tag)
        return VR_TO_DTYPE.get(vr, object)
    except KeyError:
        return object


def validate_ae_title(ae_title: str) -> bool:
    """
    Validate a DICOM AE (Application Entity) Title.

    Parameters
    ----------
    ae_title : str
        The AE Title to validate.

    Returns
    -------
    bool
        True if the AE Title is valid, False otherwise.

    Notes
    -----
    - AE Titles must be between 1 and 16 characters long.
    - Allowed characters are uppercase letters (A-Z), digits (0-9),
      space, underscore (_), dash (-), and period (.).
    """
    if not (1 <= len(ae_title) <= 16):
        return False

    if not re.match(r"^[A-Z0-9 _\-.]+$", ae_title):
        return False

    return True


def validate_host(host: str) -> bool:
    """
    Validate a host address (IP or hostname).

    Parameters
    ----------
    host : str
        The host address to validate. This can be an IP address or hostname.

    Returns
    -------
    bool
        True if the host address is valid, False otherwise.

    Notes
    -----
    - For IP addresses, both IPv4 and IPv6 are supported.
    - Hostnames must be alphanumeric, may include hyphens, and
      must not exceed 253 characters.
    """
    try:
        # Try to parse as an IP address
        ip_address(host)
        return True
    except ValueError:
        # If not an IP address, validate as hostname
        if len(host) > 253:
            return False
        if re.match(r"^[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*$", host):
            return True
    return False


def validate_port(port: int) -> bool:
    """
    Validate a port number.

    Parameters
    ----------
    port : int
        The port number to validate.

    Returns
    -------
    bool
        True if the port number is valid, False otherwise.

    Notes
    -----
    - Valid port numbers are integers between 1 and 65535.
    """
    return 1 <= port <= 65535


def validate_entry(input_text: Union[str, int], entry_type: str) -> bool:
    """Checks whether a text input from the user contains invalid characters.

    Parameters
    ----------
    input_text : Union[str, int]
        The text input to a given field.
    entry_type : str
        The type of field where the text was input. The different
        types are:
        * AET
        * Port
        * IP

    Returns
    -------
    bool
        Whether the input was valid or not.
    """
    if entry_type == "AET":
        return validate_ae_title(input_text)
    elif entry_type == "IP":
        return validate_host(input_text)
    elif entry_type == "Port":
        return validate_port(input_text)

    else:
        return False


def get_running_env():
    try:
        from IPython import get_ipython
        import sys

        shell = get_ipython()
        if shell is None:
            return "script"  # Running in a regular script

        # Check if running in a Jupyter environment
        if "ipykernel" in sys.modules:
            # Check if JupyterLab or Jupyter Notebook
            from jupyter_server.serverapp import list_running_servers

            if any("lab" in server["url"] for server in list_running_servers()):
                return "jupyterlab"
            return "jupyter_notebook"
    except Exception:
        return "script"  # Fallback to script mode
