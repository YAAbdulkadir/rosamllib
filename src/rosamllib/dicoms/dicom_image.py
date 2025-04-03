import pydicom
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


class DICOMImage(sitk.Image):
    """
    A class that represents a DICOM image and provides methods for retrieving image properties,
    transforming points between physical and pixel space, resampling, and visualization.

    This class subclasses `SimpleITK.Image`, meaning it behaves like a standard `SimpleITK.Image`
    with additional methods for handling DICOM data, including transformations, resampling, and
    image visualization.

    Parameters
    ----------
    image : SimpleITK.Image
        A SimpleITK Image object representing the DICOM image.

    Examples
    --------
    >>> reader = sitk.ImageSeriesReader()
    >>> dicom_image = DICOMImage(reader.Execute())
    >>> dicom_image.GetSize()
    >>> dicom_image.visualize(axial_index=10)

    See Also
    --------
    SimpleITK.Image : The parent class that provides core image processing methods.
    """

    def __init__(self, image):
        """
        Initializes the DICOMImage class by wrapping a SimpleITK Image object.

        Parameters
        ----------
        image : SimpleITK.Image
            A SimpleITK Image object representing the DICOM image.

        Notes
        -----
        This method caches transformation matrices for pixel-to-physical and physical-to-pixel
        coordinate systems, which are calculated and stored upon request.
        """
        super().__init__(image)
        self._pixel_to_physical_matrix = None
        self._physical_to_pixel_matrix = None

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
            The value of the requested DICOM metadata tag.

        Raises
        ------
        AttributeError
            If the attribute is not a valid DICOM keyword or metadata is not found.
        """
        # Try to convert the attribute (keyword) to a DICOM tag
        try:
            tag = pydicom.datadict.tag_for_keyword(attr)
            if tag is None:
                raise ValueError("Not a DICOM metadata.")

            # Convert the integer tag into a pydicom Tag object (this separates group and element)
            tag_obj = pydicom.tag.Tag(tag)

            # Format the tag into the "group|element" format that SimpleITK expects
            tag_str = f"{tag_obj.group:04X}|{tag_obj.element:04X}"
            tag_str = tag_str.lower()
            if self.HasMetaDataKey(tag_str):
                # Fetch the metadata using the SimpleITK GetMetaData method
                return self.GetMetaData(tag_str)
            else:
                raise AttributeError(f"'DICOMImage' object has no attribute '{attr}'")

        except ValueError:
            # not a valid DICOM metadata, check if parent has the method
            try:
                return super().__getattr__(attr)
            except Exception:
                raise AttributeError(f"'DICOMImage' object has no attribute '{attr}'")

        except Exception:
            raise AttributeError(f"'DICOMImage' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        """
        Allows setting of DICOM metadata attributes via dot notation.

        Parameters
        ----------
        attr : str
            The attribute being set (e.g., 'PatientID').
        value : str
            The value to set for the given attribute.

        Raises
        ------
        AttributeError
            If the attribute is not a valid DICOM keyword or cannot be set.
        """
        # Try to convert the attribute (keyword) to a DICOM tag
        tag = pydicom.datadict.tag_for_keyword(attr)
        if tag is not None:
            try:
                # Convert the integer tag into a pydicom Tag object
                tag_obj = pydicom.tag.Tag(tag)

                # Format the tag into the "group|element" format that SimpleITK expects
                tag_str = f"{tag_obj.group:04X}|{tag_obj.element:04X}"
                tag_str = tag_str.lower()
                # Set the DICOM metadata using the SimpleITK SetMetaData method
                self.SetMetaData(tag_str, str(value))
                return
            except Exception as e:
                raise AttributeError(f"Error setting DICOM metadata for '{attr}': {e}")

        # If it's not DICOM metadata, set it as a normal attribute
        super().__setattr__(attr, value)

    def __dir__(self):
        """
        Extends the built-in dir() method to include available DICOM metadata keywords.

        Returns
        -------
        list of str
            A list of DICOM keywords (attributes) that can be accessed via dot notation, in
            addition to the default attributes of the SimpleITK.Image class.
        """
        # Get the default attributes from the parent class
        default_dir = super().__dir__()

        # Get DICOM metadata keys and convert them to their respective keywords
        metadata_keys = self.GetMetaDataKeys()
        dicom_keywords = []
        for key in metadata_keys:
            try:
                group, element = key.split("|")
                tag = pydicom.tag.Tag(int(group, 16), int(element, 16))
                keyword = pydicom.datadict.keyword_for_tag(tag)
                if keyword:
                    dicom_keywords.append(keyword)
            except ValueError:
                # not a dicom tag
                pass

        # Combine the default attributes with the DICOM keywords
        return default_dir + dicom_keywords

    def dir(self):
        """
        Custom dir method to return a list of available attributes and DICOM metadata keywords.

        Returns
        -------
        list of str
            List of all attributes, including DICOM metadata keywords.
        """
        return self.__dir__()

    def get_image_array(self):
        """
        Converts the current DICOM image to a NumPy array.

        Returns
        -------
        numpy.ndarray
            A NumPy array representing the image data.

        See Also
        --------
        SimpleITK.GetArrayFromImage : Converts a SimpleITK Image object to a NumPy array.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> image_array = dicom_image.get_image_array()
        """
        return sitk.GetArrayFromImage(self)

    def resample_image(self, new_spacing):
        """
        Resamples the image to the specified spacing.

        Parameters
        ----------
        new_spacing : tuple of float
            The desired spacing between pixels in each dimension (X, Y, Z).

        Returns
        -------
        DICOMImage
            A resampled DICOMImage object.

        Notes
        -----
        This method uses linear interpolation for resampling and maintains the original
        orientation and origin of the image.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> resampled_image = dicom_image.resample_image((1.0, 1.0, 1.0))
        """
        original_spacing = self.GetSpacing()
        original_size = self.GetSize()

        # Compute the new size based on the new spacing
        new_size = [
            int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(self.GetDirection())
        resample.SetOutputOrigin(self.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(self.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkLinear)

        return DICOMImage(resample.Execute(self))

    def visualize(
        self,
        axial_index=None,
        sagittal_index=None,
        coronal_index=None,
        window_width=400,
        window_level=40,
    ):
        """
        Visualizes the DICOM image in axial, sagittal, and coronal planes.

        Parameters
        ----------
        axial_index : int, optional
            The index of the axial slice to display. Defaults to the middle of the volume.
        sagittal_index : int, optional
            The index of the sagittal slice to display. Defaults to the middle of the volume.
        coronal_index : int, optional
            The index of the coronal slice to display. Defaults to the middle of the volume.
        window_width : float, optional
            The window width for image intensity display. Default is 400.
        window_level : float, optional
            The window level for image intensity display. Default is 40.

        Notes
        -----
        The method applies windowing to the image data and displays three orthogonal slices (axial,
        sagittal, coronal) using matplotlib.

        See Also
        --------
        matplotlib.pyplot : Used to generate the image visualizations.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> dicom_image.visualize(axial_index=30)
        """

        # img = sitk.GetArrayFromImage(self)
        img = sitk.GetArrayFromImage(self.resample_image([1.0, 1.0, 1.0]))
        min_val = window_level - (window_width / 2)
        max_val = window_level + (window_width / 2)

        windowed_img = np.clip(img, min_val, max_val)

        if not axial_index:
            axial_index = img.shape[0] // 2
        if not coronal_index:
            coronal_index = img.shape[1] // 2
        if not sagittal_index:
            sagittal_index = img.shape[2] // 2

        fig = plt.figure(figsize=(10, 8))

        # axial
        axial_ax = fig.add_subplot(1, 2, 1)
        axial_ax.imshow(windowed_img[axial_index], cmap="gray")
        axial_ax.set_title("Axial")
        axial_ax.axis("off")

        # sagittal
        sagittal_ax = fig.add_subplot(2, 2, 2)
        sagittal_ax.imshow(windowed_img[:, :, sagittal_index], cmap="gray", origin="lower")
        sagittal_ax.set_title("Sagittal")
        sagittal_ax.axis("off")

        # coronal
        coronal_ax = fig.add_subplot(2, 2, 4)
        coronal_ax.imshow(windowed_img[:, coronal_index, :], cmap="gray", origin="lower")
        coronal_ax.set_title("Coronal")
        coronal_ax.axis("off")

        plt.tight_layout()
        plt.show()

    def get_pixel_to_physical_transformation_matrix(self):
        """
        Returns the transformation matrix from pixel coordinates to physical coordinates.

        Returns
        -------
        numpy.ndarray
            A 4x4 matrix representing the transformation from pixel coordinates to physical space.

        Notes
        -----
        The matrix is cached after the first computation to improve performance for repeated
        transformations.

        See Also
        --------
        SimpleITK.Image.GetDirection : Gets the image orientation.
        SimpleITK.Image.GetSpacing : Gets the image spacing.
        SimpleITK.Image.GetOrigin : Gets the image origin.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> pixel_to_physical_matrix = dicom_image.get_pixel_to_physical_transformation_matrix()
        """
        if self._pixel_to_physical_matrix is None:
            origin = np.array(self.GetOrigin())  # (x0, y0, z0)
            spacing = np.array(self.GetSpacing())  # (sx, sy, sz)
            direction = np.array(self.GetDirection()).reshape(3, 3)

            # Construct the 3x3 scaling and rotation matrix
            affine_matrix = np.diag(spacing) @ direction

            # Create the 4x4 transformation matrix (homogeneous coordinates)
            T = np.eye(4)
            T[:3, :3] = affine_matrix
            T[:3, 3] = origin

            # Cache the matrix
            self._pixel_to_physical_matrix = T

        return self._pixel_to_physical_matrix

    def get_physical_to_pixel_transformation_matrix(self):
        """
        Returns the transformation matrix from physical coordinates to pixel coordinates.

        Returns
        -------
        numpy.ndarray
            A 4x4 matrix representing the transformation from physical coordinates to pixel space.

        Notes
        -----
        The matrix is computed as the inverse of the pixel-to-physical transformation matrix and
        cached for future use.

        See Also
        --------
        DICOMImage.get_pixel_to_physical_transformation_matrix : Computes the pixel-to-physical
        matrix.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> physical_to_pixel_matrix = dicom_image.get_physical_to_pixel_transformation_matrix()
        """
        if self._physical_to_pixel_matrix is None:
            # Compute and cache the inverse of the pixel-to-physical matrix
            self._physical_to_pixel_matrix = np.linalg.inv(
                self.get_pixel_to_physical_transformation_matrix()
            )

        return self._physical_to_pixel_matrix

    def transform_to_physical_coordinates(self, pixel_points, indexing="ij"):
        """
        Transforms pixel coordinates into physical coordinates.

        Parameters
        ----------
        pixel_points : numpy.ndarray
            An array of pixel coordinates, with shape (N, 3), where N is the number of points.
        indexing : str, optional
            Indexing convention to use. Options are 'ij' (row, column) or 'xy'. Default is 'ij'.

        Returns
        -------
        numpy.ndarray
            The physical coordinates corresponding to the input pixel coordinates.

        Raises
        ------
        ValueError
            If an invalid indexing parameter is provided.

        Notes
        -----
        The method converts pixel coordinates into physical space using the transformation matrix
        cached in the object.

        See Also
        --------
        DICOMImage.get_pixel_to_physical_transformation_matrix : Returns the pixel-to-physical
        matrix.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> pixel_points = np.array([[10, 15, 20], [25, 30, 35]])
        >>> physical_points = dicom_image.transform_to_physical_coordinates(pixel_points, "xy")
        """
        if indexing not in ["ij", "xy"]:
            raise ValueError("Invalid indexing parameter. Use 'ij' or 'xy'.")

        # Swap the first and second columns (to x,y) if indexing is 'ij' before transforming
        if indexing == "ij":
            pixel_points = pixel_points.copy()
            pixel_points[:, [0, 1]] = pixel_points[:, [1, 0]]

        # Add a fourth 1 to each point to convert it to homogeneous coordinates
        num_points = pixel_points.shape[0]
        pixel_points_homogeneous = np.hstack([pixel_points, np.ones((num_points, 1))])

        # Get the transformation matrix to physical coordinates
        T = self.get_pixel_to_physical_transformation_matrix()

        # Apply the transformation matrix to all points
        physical_points_homogeneous = np.dot(T, pixel_points_homogeneous.T).T

        # Drop the homogeneous coordinate (fourth dimension) to get back to 3D
        physical_points = physical_points_homogeneous[:, :3]

        return physical_points

    def transform_to_pixel_coordinates(self, physical_points, indexing="ij"):
        """
        Transforms physical coordinates into pixel coordinates.

        Parameters
        ----------
        physical_points : numpy.ndarray
            An array of physical coordinates, with shape (N, 3), where N is the number of points.
        indexing : str, optional
            Indexing convention to use. Options are 'ij' (row, column) or 'xy'. Default is 'ij'.

        Returns
        -------
        numpy.ndarray
            The pixel coordinates corresponding to the input physical coordinates.

        Notes
        -----
        The method converts physical coordinates into pixel space using the inverse of the
        pixel-to-physical transformation matrix.

        See Also
        --------
        DICOMImage.get_physical_to_pixel_transformation_matrix : Returns the physical-to-pixel
        matrix.

        Examples
        --------
        >>> dicom_image = DICOMImage(sitk_image)
        >>> physical_points = np.array([[10.5, 15.2, 20.7], [25.1, 30.3, 35.5]])
        >>> pixel_points = dicom_image.transform_to_pixel_coordinates(physical_points, "xy")
        """
        # Add a fourth 1 to each point to convert it to homogeneous coordinates
        num_points = physical_points.shape[0]
        physical_points_homogeneous = np.hstack([physical_points, np.ones((num_points, 1))])

        # Get the transformation matrix to pixel coordinates
        T_inv = self.get_physical_to_pixel_transformation_matrix()

        # Apply the transformation matrix to all points
        pixel_points_homogeneous = np.dot(T_inv, physical_points_homogeneous.T).T

        # Drop the homogeneous coordinate (fourth dimension) to get back to 3D
        pixel_points = pixel_points_homogeneous[:, :3]

        if indexing == "ij":
            # Swap the x, y (col, row) to y, x (row, col)
            pixel_points[:, [0, 1]] = pixel_points[:, [1, 0]]

        return pixel_points
