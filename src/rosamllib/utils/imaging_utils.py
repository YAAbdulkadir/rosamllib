import os
import tempfile
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path
from pydicom import dcmread


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


def transform_image(
    image: sitk.Image,
    angle_degrees: float = 0.0,
    axis: np.ndarray = np.array([0, 0, 1]),
    translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    interpolator=sitk.sitkLinear,
    default_value=None,
) -> sitk.Image:
    """
    Applies a rigid transformation (rotation about arbitrary axis + translation)
    to a 3D SimpleITK image.

    Parameters
    ----------
    image : sitk.Image
        The input 3D image.
    angle_degrees : float
        Rotation angle in degrees. Default is 0 (no rotation).
    axis : np.ndarray
        Axis of rotation as a 3-element array. Will be normalized.
    translation : np.ndarray
        Translation vector [dx, dy, dz] in mm.
    interpolator : sitk.InterpolatorEnum
        Interpolation method (e.g., sitk.sitkLinear or sitk.sitkNearestNeighbor).
    default_value : float
        Default value for pixels outside the image domain.

    Returns
    -------
    sitk.Image
        The transformed image.
    """

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Convert angle to radians
    angle_rad = np.deg2rad(angle_degrees)
    half_angle = angle_rad / 2.0

    # Compute versor (vector part of quaternion)
    versor = np.sin(half_angle) * axis

    # Get image center in physical coordinates
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    center = [origin[i] + 0.5 * spacing[i] * size[i] for i in range(3)]

    # Create the transform
    transform = sitk.VersorRigid3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(versor, angle_rad)
    transform.SetTranslation(translation.tolist())

    # Get default value
    if default_value is None:
        default_value = np.min(sitk.GetArrayViewFromImage(image)).astype(float)

    # Resample
    transformed_image = sitk.Resample(image, image, transform, interpolator, default_value)

    return transformed_image
