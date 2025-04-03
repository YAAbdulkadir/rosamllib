from typing import List, Union
import cv2
import pydicom
import rt_utils
import numpy as np
import SimpleITK as sitk
import warnings
from rt_utils.utils import ROIData
from rt_utils.ds_helper import (
    create_structure_set_roi,
    create_rtroi_observation,
    create_roi_contour,
)


class RTStruct(rt_utils.RTStruct):
    """
    A class for handling DICOM RTSTRUCT (Radiotherapy Structure Set) data. It provides
    methods for manipulating and extracting structure contours, masks, and associated
    metadata, as well as adding new ROIs (Regions of Interest).

    Parameters
    ----------
    ds : pydicom.Dataset
        The DICOM dataset representing the RTSTRUCT file.
    series_data : SimpleITK.Image, optional
        The image series data associated with the RTSTRUCT file, used for generating ROI masks.

    Attributes
    ----------
    series_data : SimpleITK.Image or None
        The image series data associated with the RTSTRUCT file.
    ds : pydicom.Dataset
        The DICOM dataset of the RTSTRUCT.
    roi_contours : list of pydicom.Dataset
        List of ROI contour sequences from the DICOM dataset.
    frame_of_reference_uid : str
        Frame of Reference UID extracted from the DICOM dataset.
    referenced_image : SimpleITK.Image or None
        The referenced image used to generate structure masks and transformations.
    structure_names : list of str
        Cached list of structure names from the DICOM dataset.
    structure_masks : dict
        Cache for storing generated structure masks.
    structure_contours : dict
        Cache for storing structure contours in both pixel and physical space.

    Methods
    -------
    set_series_data(series_data)
        Sets the image series data associated with the RTSTRUCT.
    set_referenced_image(referenced_image)
        Sets the referenced image for generating structure masks.
    get_structure_names()
        Returns the names of all structures defined in the RTSTRUCT.
    get_structure_color(structure_name)
        Returns the display color of a specific structure.
    get_structure_index(structure_name)
        Returns the index of a specific structure based on its name.
    get_structure_mask(structure_name)
        Generates and returns a binary mask for a specific structure.
    get_contour_points_in_pixel_space(structure_name)
        Returns the contour points for a specific structure in pixel space.
    get_contour_points_in_physical_space(structure_name)
        Returns the contour points for a specific structure in physical space.
    get_physical_to_pixel_transformation_matrix()
        Returns the transformation matrix from physical to pixel space.
    get_pixel_to_physical_transformation_matrix()
        Returns the transformation matrix from pixel to physical space.
    add_roi(mask, color, name, description, use_pin_hole, approximate_contours,
        roi_generation_algorithm)
        Adds a new ROI to the RTSTRUCT with specified properties.

    Examples
    --------
    >>> rtstruct = RTStruct(ds)
    >>> rtstruct.set_referenced_image(image)
    >>> mask = rtstruct.get_structure_mask("PTV")
    >>> rtstruct.add_roi(mask=mask, name="New ROI", color=[255, 0, 0])

    See Also
    --------
    rt_utils.RTStruct : The base class for RTSTRUCT manipulation.
    """

    def __init__(self, ds, series_data=None):
        self.series_data = series_data
        self.ds = ds
        self.roi_contours = self.ds.ROIContourSequence
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID
        self.referenced_image = None
        self.structure_names = []

        # cache for storing generated masks and contours
        self.structure_masks = {}
        self.structure_contours = {}

    def __getattr__(self, attr):
        """
        Allows attribute access for DICOM metadata via dot notation.

        Parameters
        ----------
        attr : str
            The attribute being accessed (e.g., 'PatientID').

        Returns
        -------
        str
            The value of the requested DICOM metadata tag from the self.ds attribute.

        Raises
        ------
        AttributeError
            If the attribute is not a valid DICOM keyword or metadata is not found.
        """
        if attr in self.dir():
            return getattr(self.ds, attr)

        # Fallback to the parent class method if not a valid DICOM attribute
        try:
            return super().__getattr__(attr)
        except Exception:
            raise AttributeError(f"'RTStruct' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        """
        Allows setting DICOM metadata attributes via dot notation.

        Parameters
        ----------
        attr : str
            The attribute being set (e.g., 'PatientID').
        value : str
            The value to be assigned to the attribute.
        """
        try:
            tag = pydicom.datadict.tag_for_keyword(attr)
            if tag is not None:
                # Set the value in the DICOM dataset (self.ds)
                self.ds[tag].value = value
            else:
                super().__setattr__(attr, value)
        except Exception:
            super().__setattr__(attr, value)

    def __dir__(self):
        """
        Returns a list of attributes and DICOM metadata keywords.

        This method dynamically combines standard attributes and the available DICOM metadata
        keywords from the `self.ds` attribute.

        Returns
        -------
        list
            A combined list of attributes and DICOM metadata keywords.
        """
        default_dir = super().__dir__()
        dicom_keywords = [pydicom.datadict.keyword_for_tag(tag) for tag in self.ds.keys()]
        return default_dir + [kw for kw in dicom_keywords if kw]

    def dir(self):
        """
        Custom `dir` method to return a list of available attributes and DICOM metadata keywords.

        This method provides a dynamic list of attributes and metadata for easier access.

        Returns
        -------
        list of str
            List of all attributes, including DICOM metadata keywords.
        """
        return self.__dir__()

    def set_series_data(self, series_data):
        """
        Sets the image series data associated with the RTSTRUCT.

        Parameters
        ----------
        series_data : SimpleITK.Image
            The image series data to be associated with the RTSTRUCT.
        """
        self.series_data = series_data

    def set_referenced_image(self, referenced_image):
        """
        Sets the referenced image for generating structure masks.

        Parameters
        ----------
        referenced_image : SimpleITK.Image
            The referenced image used for mask generation and transformations.
        """
        self.referenced_image = referenced_image

    def get_structure_names(self):
        """
        Returns the names of all structures defined in the RTSTRUCT.

        Returns
        -------
        list of str
            A list of structure names.

        Examples
        --------
        >>> rtstruct.get_structure_names()
        ['GTV', 'PTV', 'CTV']
        """
        if not self.structure_names:
            self.structure_names = [
                structure.ROIName for structure in self.ds.StructureSetROISequence
            ]
        return self.structure_names

    def get_structure_color(self, structure_name):
        """
        Returns the display color of a specific structure.

        Parameters
        ----------
        structure_name : str
            The name of the structure to get the color for.

        Returns
        -------
        list of int
            The RGB color values of the structure.

        Examples
        --------
        >>> rtstruct.get_structure_color("GTV")
        [255, 0, 0]
        """
        structure_index = self.get_structure_index(structure_name)
        color = self.ds.ROIContourSequence[structure_index].ROIDisplayColor
        return color

    def get_structure_index(self, structure_name):
        """
        Returns the index of a specific structure based on its name.

        Parameters
        ----------
        structure_name : str
            The name of the structure to find.

        Returns
        -------
        int
            The index of the structure in the ROIContourSequence.

        Raises
        ------
        ValueError
            If the structure name is not found in the RTSTRUCT.

        Examples
        --------
        >>> rtstruct.get_structure_index("PTV")
        1
        """
        return self.get_structure_names().index(structure_name)

    def get_structure_mask(self, structure_name):
        """
        Generates and returns a binary mask for a specific structure.

        Parameters
        ----------
        structure_name : str
            The name of the structure to generate the mask for.

        Returns
        -------
        numpy.ndarray
            A 3D numpy array representing the binary mask of the structure.

        Raises
        ------
        RuntimeError
            If the referenced image has not been set.
        ValueError
            If the structure name is not found in the RTSTRUCT.

        Examples
        --------
        >>> mask = rtstruct.get_structure_mask("PTV")
        >>> mask.shape
        (512, 512, 100)
        """

        if self.referenced_image is None:
            raise RuntimeError(
                "referenced image has not been set. Call `set_referenced_image` first."
            )

        if structure_name in self.structure_masks:
            return self.structure_masks[structure_name]

        structure_index = self.get_structure_index(structure_name)

        if structure_index is None:
            raise ValueError(f"Structure {structure_name} not found in RTSTRUCT.")

        contour_data = self.roi_contours[structure_index]
        try:
            mask = self._create_mask_from_contours(self.referenced_image, contour_data)
        except Exception as e:
            mask = np.zeros(sitk.GetArrayFromImage(self.referenced_image).shape, dtype=np.uint8)
            warnings.warn(
                f"Couldn't process contours for {structure_name}: {e}",
                UserWarning,
            )
        mask = sitk.GetImageFromArray(mask)
        mask.SetDirection(self.referenced_image.GetDirection())
        mask.SetOrigin(self.referenced_image.GetOrigin())
        mask.SetSpacing(self.referenced_image.GetSpacing())
        self.structure_masks[structure_name] = mask  # Cache the generated mask

        return mask

    def _create_mask_from_contours(self, image, contour_data):
        """
        Creates a binary mask from the contour data for a given structure.

        Parameters
        ----------
        image : SimpleITK.Image
            The referenced image.
        contour_data : pydicom.Sequence
            The contour data sequence for the structure.

        Returns
        -------
        numpy.ndarray
            A 3D binary mask of the structure.
        """
        image_array_shape = sitk.GetArrayFromImage(image).shape
        mask = np.zeros(image_array_shape, dtype=np.uint8)

        for contour in contour_data.ContourSequence:
            self._process_contour(contour, mask)

        return mask

    def _process_contour(self, contour, mask):
        """
        Processes a single contour and updates the mask with its binary representation.

        Parameters
        ----------
        contour : pydicom.dataset.Dataset
            The contour dataset containing ContourData.
        mask : numpy.ndarray
            The binary mask array to update with the contour.
        """
        try:
            contour_points = np.array(contour.ContourData).reshape(-1, 3)

            # Transform physical contour points into pixel coordinates
            indices = self.referenced_image.transform_to_pixel_coordinates(
                contour_points, indexing="xy"
            )

            # Ensure all points are on a single slice
            # TODO
            # handle cases where points do not lie on a single slices becuase
            # sitk might have re-oriented to one of the standard orientations
            # and therefore, slice_index might not be unique
            slice_index = np.unique(indices[:, 2])
            if len(slice_index) != 1:
                raise ValueError("Contour points do not lie on a single slice")
            slice_index = int(slice_index[0])

            # Extract 2D point for the slice
            points_2d = indices[:, :2].astype(np.int32)

            # Create an empty mask for the slice
            slice_shape = mask[slice_index].shape
            slice_mask = np.zeros(slice_shape, dtype=np.uint8)

            # Fill the polygon
            cv2.fillPoly(slice_mask, [points_2d], 1)

            # Update the mask array with this filled polygon mask
            mask[slice_index] = np.maximum(mask[slice_index], slice_mask)

        except Exception as e:
            raise RuntimeError(f"Error processing contour: {e}")

    def get_contour_points_in_pixel_space(self, structure_name):
        """
        Returns the contour points for a specific structure in pixel space.

        Parameters
        ----------
        structure_name : str
            The name of the structure to retrieve the contour points for.

        Returns
        -------
        dict of int : list of numpy.ndarray
            A dictionary where keys are slice indices and values are numpy arrays representing
            the pixel coordinates of the contours on each slice.

        Examples
        --------
        >>> contours = rtstruct.get_contour_points_in_pixel_space("GTV")
        >>> contours[5].shape
        (10, 2)
        """
        if self.referenced_image is None:
            raise RuntimeError(
                "The referenced image has not been set. Call `set_referenced_image` first."
            )

        # Initialize the structure cache if it's not already present
        if structure_name not in self.structure_contours:
            self.structure_contours[structure_name] = {}

        # Check if the contours for this structure are already cached
        if "pixel_space" in self.structure_contours[structure_name]:
            return self.structure_contours[structure_name]["pixel_space"]

        # Get the structure index
        structure_index = self.get_structure_index(structure_name)
        contour_data = self.roi_contours[structure_index]

        contours_in_pixel_space = {}

        # Define a function to process each contour
        def process_contour(contour):
            contour_points = np.array(contour.ContourData).reshape(-1, 3)
            try:
                # Transform to pixel coordinates
                pixel_coords = self.referenced_image.transform_to_pixel_coordinates(contour_points)
            except Exception as e:
                raise RuntimeError(f"Error transforming contour points to pixel space: {e}")

            # Get the slice index directly (no need for unique check)
            slice_index = self.referenced_image.TransformPhysicalPointToIndex(contour_points[0])[2]
            return slice_index, pixel_coords

        # Process the contours and concatenate if multiple contours exist for the same slice
        for contour in contour_data.ContourSequence:
            slice_index, pixel_coords = process_contour(contour)

            # If slice_index already exists, concatenate the new coordinates to the existing array
            if slice_index in contours_in_pixel_space:
                contours_in_pixel_space[slice_index] = np.concatenate(
                    (contours_in_pixel_space[slice_index], pixel_coords), axis=0
                )
            else:
                # Initialize the NumPy array with the first set of coordinates
                contours_in_pixel_space[slice_index] = pixel_coords

        # Cache the converted contour points for this structure
        self.structure_contours[structure_name]["pixel_space"] = contours_in_pixel_space

        return contours_in_pixel_space

    def get_contour_points_in_physical_space(self, structure_name):
        """
        Returns the contour points for a specific structure in physical space.

        Parameters
        ----------
        structure_name : str
            The name of the structure to retrieve the contour points for.

        Returns
        -------
        dict of int : list of numpy.ndarray
            A dictionary where keys are slice indices and values are numpy arrays representing
            the physical coordinates of the contours on each slice.
        """
        if self.referenced_image is None:
            raise RuntimeError(
                "The referenced image has not been set. Call `set_referenced_image` first."
            )

        # Initialize the structure cache if it's not already present
        if structure_name not in self.structure_contours:
            self.structure_contours[structure_name] = {}

        # Check if the contours for this structure are already cached
        if "physical_space" in self.structure_contours[structure_name]:
            return self.structure_contours[structure_name]["physical_space"]

        # Get the structure index
        structure_index = self.get_structure_index(structure_name)

        contour_data = self.roi_contours[structure_index]

        contours_in_physical_space = {}
        for contour in contour_data.ContourSequence:
            contour_points = np.array(contour.ContourData).reshape(-1, 3)

            # Add the contour points to the appropriate slice index
            slice_index = self.referenced_image.TransformPhysicalPointToIndex(contour_points[0])[2]
            if slice_index not in contours_in_physical_space:
                contours_in_physical_space[slice_index] = []
            contours_in_physical_space[slice_index].append(contour_points)

        # Cache the converted contour points for this structure
        self.structure_contours[structure_name]["physical_space"] = contours_in_physical_space

        return contours_in_physical_space

    def get_physical_to_pixel_transformation_matrix(self):
        """
        Returns the transformation matrix from physical coordinates to pixel coordinates.

        Returns
        -------
        numpy.ndarray
            A 4x4 transformation matrix from physical to pixel space.

        Raises
        ------
        RuntimeError
            If the referenced image has not been set.
        """
        if self.referenced_image is None:
            raise RuntimeError(
                "The referenced image has not been set. Call `set_referenced_image` first."
            )

        return self.referenced_image.get_physical_to_pixel_transformation_matrix()

    def get_pixel_to_physical_transformation_matrix(self):
        """
        Returns the transformation matrix from pixel coordinates to physical coordinates.

        Returns
        -------
        numpy.ndarray
            A 4x4 transformation matrix from pixel to physical space.

        Raises
        ------
        RuntimeError
            If the referenced image has not been set.
        """
        if self.referenced_image is None:
            raise RuntimeError(
                "The referenced image has not been set. Call `set_referenced_image` first."
            )

        return self.referenced_image.get_pixel_to_physical_transformation_matrix()

    def add_roi(
        self,
        mask: np.ndarray,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = "",
        use_pin_hole: bool = False,
        approximate_contours: bool = True,
        roi_generation_algorithm: Union[str, int] = 0,
        imaging_axis: int = 0,
    ):
        """
        Adds a new Region of Interest (ROI) to the RTSTRUCT.

        Parameters
        ----------
        mask : numpy.ndarray
            A binary mask representing the new ROI.
        color : list of int, optional
            A list of RGB values representing the display color of the ROI.
        name : str, optional
            The name of the new ROI.
        description : str, optional
            A description of the ROI.
        use_pin_hole : bool, optional
            Whether to use pinhole technique for contouring.
        approximate_contours : bool, optional
            Whether to approximate the contours.
        roi_generation_algorithm : int, optional
            Algorithm for generating the ROI.
        imaging_axis : int, optional
            Integer representing the imaging axis (0, 1, or 2)

        Raises
        ------
        ValueError
            If the series data has not been set.

        Notes
        -----
        This method is the same as the `add_roi` method from the `rt_utils.RTStruct` class
        with imaging_axis as an additional parameter.

        See Also
        --------
        rt_utils.RTStruct.add_roi : Base method for adding ROIs.
        """
        if self.series_data is None:
            raise ValueError(
                "series_data is not set. Please use the `set_series_data` method first."
            )
        # get the roi number of the last contour
        last_roi_number = self.get_last_roi_number()
        roi_number = last_roi_number + 1

        if imaging_axis not in (0, 1, 2):
            raise ValueError("imaging_axis must be 0, 1, or 2.")

        # move axis if it's not already the last axis
        if imaging_axis != mask.ndim - 1:
            mask = np.moveaxis(mask, imaging_axis, -1)

        mask = mask > 0

        roi_data = ROIData(
            mask,
            color,
            roi_number,
            name,
            self.frame_of_reference_uid,
            description,
            use_pin_hole,
            approximate_contours,
            roi_generation_algorithm,
        )
        self.ds.StructureSetROISequence.append(create_structure_set_roi(roi_data))
        self.ds.RTROIObservationsSequence.append(create_rtroi_observation(roi_data))
        self.ds.ROIContourSequence.append(create_roi_contour(roi_data, self.series_data))

    def get_last_roi_number(self):
        structure_roi_numbers = [
            structure.ROINumber for structure in self.ds.StructureSetROISequence
        ]
        if structure_roi_numbers:
            last_roi_number = structure_roi_numbers[-1]
        else:
            last_roi_number = 0
        return last_roi_number
