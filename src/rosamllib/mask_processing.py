import numpy as np
import xmltodict
import nibabel as nib
import SimpleITK as sitk
from skimage.measure import marching_cubes


def get_contour_points_from_mask(binary_mask, level=0.5, spacing=(1.0, 1.0, 1.0)):
    """
    Extracts contour points from a 3D binary mask using the marching cubes algorithm.

    This function computes the surface of the 3D mask using marching cubes and returns
    the vertices (contour points) that represent the surface.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        A 3D binary NumPy array where the value 1 represents the region of interest (ROI)
        and 0 represents the background.
    level : float, optional
        The contour value to search for isosurfaces in `binary_mask`. Default is 0.5,
        which is suitable for binary masks.
    spacing : tuple of float, optional
        The voxel spacing along each axis (z, y, x). This is used to scale the contour
        points to real-world dimensions. Default is (1.0, 1.0, 1.0).

    Returns
    -------
    vertices : numpy.ndarray
        An array of shape (N, 3) containing the coordinates of the contour points.
        Each row corresponds to a point (x, y, z).
    faces : numpy.ndarray
        An array of shape (M, 3) containing indices of the vertices that form triangular
        faces of the surface.

    Examples
    --------
    >>> binary_mask = np.zeros((100, 100, 100), dtype=np.uint8)
    >>> binary_mask[30:70, 30:70, 30:70] = 1  # Create a cube-like ROI
    >>> vertices, faces = get_contour_points(binary_mask, spacing=(1.0, 1.0, 1.0))
    >>> print("Number of contour points:", len(vertices))

    Notes
    -----
    - The marching cubes algorithm finds isosurfaces in volumetric data.
    - The `spacing` parameter adjusts the points to real-world dimensions, making it
      useful for medical images with non-isotropic voxel sizes.
    - For a smoother surface, you can increase the resolution of the binary mask.
    """
    # Perform marching cubes to obtain vertices and faces of the contour
    vertices, faces, _, _ = marching_cubes(binary_mask, level=level, spacing=spacing)

    return vertices, faces


def extract_largest_structure(image, spacing=None, min_volume=None):
    """
    Extracts the largest connected structure from a 3D binary mask using SimpleITK.
    Optionally, removes components smaller than a specified volume.

    Parameters
    ----------
    image : numpy.ndarray or sitk.Image
        The input 3D binary mask where foreground pixels are non-zero.
    spacing : tuple of float, optional
        The spacing of the image along each axis. If provided and the input is a sitk.Image,
        this spacing overrides the original spacing of the image.
    min_volume : float, optional
        The minimum volume (in mm³) for a component to be kept.
        Components with volumes smaller than this value are removed.
        If None, only the largest component is kept.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing only the largest or filtered connected structures.
        The output array will have the same dimensions as the input image.
    """
    # Convert numpy array to sitk.Image if necessary
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)
        if spacing is not None:
            image.SetSpacing(spacing)
    elif isinstance(image, sitk.Image):
        # Override spacing if provided
        if spacing is not None:
            image.SetSpacing(spacing)

    # Ensure the input image is binary (0 and 1)
    binary_image = sitk.Cast(image > 0, sitk.sitkUInt8)

    # Label connected components in the binary image
    labeled_image = sitk.ConnectedComponent(binary_image)

    # Relabel components by size (largest first)
    relabeled_image = sitk.RelabelComponent(labeled_image, sortByObjectSize=True)

    # Extract statistics about the labeled components
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabeled_image)

    # If a minimum volume is specified, filter components based on volume
    if min_volume is not None:
        # Create a new empty image for the filtered components
        filtered_image = sitk.Image(relabeled_image.GetSize(), sitk.sitkUInt8)
        filtered_image.CopyInformation(relabeled_image)

        # Iterate through all labeled components
        for label in stats.GetLabels():
            # Get the physical size (volume) of the component in mm³
            volume_mm3 = stats.GetPhysicalSize(label)

            # If the volume is greater than or equal to the specified minimum volume, keep it
            if volume_mm3 >= min_volume:
                # Add the component to the filtered image
                filtered_image = filtered_image | sitk.BinaryThreshold(
                    relabeled_image,
                    lowerThreshold=label,
                    upperThreshold=label,
                    insideValue=1,
                    outsideValue=0,
                )

        # Use the filtered image as the final output
        largest_structure_mask = filtered_image

    # If no minimum volume is specified, keep only the largest component
    else:
        # Get the label of the largest component (after relabeling, largest is always labeled as 1)
        largest_label = (
            stats.GetNumberOfLabels()
        )  # Largest component is labeled as 1 after relabeling

        # Create a binary mask for the largest component
        largest_structure_mask = sitk.BinaryThreshold(
            relabeled_image,
            lowerThreshold=largest_label,
            upperThreshold=largest_label,
            insideValue=1,
            outsideValue=0,
        )

    # Convert the final mask to a NumPy array and return it
    return sitk.GetArrayFromImage(largest_structure_mask)


def parse_totalseg_output(nifti_input):
    """
    Parse TotalSegmentator output and return binary masks for each segmented ROI.

    This function accepts either a file path to a NIfTI segmentation image or an already loaded
    NIfTI object. It parses the extended header to retrieve the label information, and then creates
    a dictionary with ROI names as keys and their corresponding binary masks as values. Each binary
    mask is a separate NIfTI image, with 0 representing the background and 1 representing the
    segmented ROI.

    Parameters
    ----------
    nifti_input : str or nibabel.Nifti1Image
        The input NIfTI data, either as a file path (string) to a NIfTI image or an already loaded
        NIfTI object (`nibabel.Nifti1Image`).

    Returns
    -------
    dict
        A dictionary where the keys are ROI (region of interest) names, and the values are the
        corresponding binary mask NIfTI images (`nibabel.Nifti1Image`).
        The mask values are 1 for the foreground (the ROI) and 0 for the background.

    Examples
    --------
    >>> roi_masks = parse_totalseg_output("path_to_segmentation.nii.gz")
    >>> liver_mask = roi_masks.get('Liver')
    >>> nib.save(liver_mask, 'liver_binary_mask.nii.gz')

    Notes
    -----
    This function expects the NIfTI file to contain segmentations generated by TotalSegmentator,
    where the extended header contains the ROI names and their corresponding label values. The
    segmentation image is assumed to have integer values where each unique value represents a
    different ROI.
    """
    # Check if the input is a string (file path) or a NIfTI object
    if isinstance(nifti_input, str):
        # nifti_input is a path, so load the NIfTI image
        img = nib.load(nifti_input)
    else:
        # nifti_input is already a NIfTI object
        img = nifti_input

    # Extract the extended header content
    ext_header = img.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)

    # Parse the label table from the extended header
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"]["Label"]

    # If only one label, convert the dict to a list
    if isinstance(ext_header, dict):
        ext_header = [ext_header]

    # Create a label map (key: label number, value: label name)
    label_map = {int(e["@Key"]): e["#text"] for e in ext_header}

    # Get the segmentation data array
    seg_data = img.get_fdata()

    # Create a dictionary to store ROI names and their corresponding binary masks
    roi_masks = {}

    # Iterate over each label and create a binary mask
    for label_value, roi_name in label_map.items():
        # Create a binary mask where 1 is the ROI and 0 is the background
        binary_mask = (seg_data == label_value).astype(np.uint8)  # Convert to binary (0 or 1)

        # Create a new NIfTI image for the binary mask
        binary_mask_nifti = nib.Nifti1Image(binary_mask, img.affine)

        # Store the binary mask in the dictionary with the ROI name as the key
        roi_masks[roi_name] = binary_mask_nifti

    return roi_masks
