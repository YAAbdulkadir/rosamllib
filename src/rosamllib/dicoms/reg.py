import pydicom
import struct
import numpy as np
import matplotlib.pyplot as plt


class REG:
    """
    A class to represent and process DICOM Registration (REG) files, extracting transformation
    matrices, metadata, and deformation grid information for both rigid and deformable image
    registrations.

    This class handles the extraction of transformation matrices, image information, and
    referenced series details for rigid and deformable image registration from a DICOM REG file.
    It provides access to the fixed and moving image details as well as the ability to plot
    deformation grids.

    Parameters
    ----------
    reg_dataset : pydicom.Dataset
        The DICOM dataset representing the REG file.

    Attributes
    ----------
    reg_dataset : pydicom.Dataset
        The raw DICOM dataset for the REG file.
    fixed_image_info : dict
        Metadata and transformation matrix for the fixed image, including grid-based information
        for deformable registrations.
    moving_image_info : dict
        Metadata and transformation matrix for the moving image, including grid-based information
        for deformable registrations.
    registration_type : str or None
        Indicates whether the registration is 'rigid' or 'deformable'. None if the type is not
        identified.

    Methods
    -------
    extract_transformation_matrices_and_metadata():
        Extracts transformation matrices and image metadata for both rigid and deformable
        registrations.

    extract_rigid_transformation(reg_sequence):
        Extracts transformation matrices and metadata from the RegistrationSequence for rigid
        registrations.

    extract_deformable_transformation(deformable_reg_sequence):
        Extracts grid-based transformations for deformable registrations.

    extract_image_info(reg_item):
        Extracts transformation matrix and metadata from an individual RegistrationSequence item.

    extract_matrix_transformation(matrix_registration_sequence, index):
        Extracts matrix-based transformations from MatrixRegistrationSequence.

    extract_grid_transformation(grid_sequence, index):
        Extracts grid-based transformations from DeformableRegistrationGridSequence.

    extract_referenced_series_info():
        Extracts referenced series information from the REG file.

    check_other_references():
        Checks for additional references in StudiesContainingOtherReferencedInstancesSequence.

    match_series_with_image(series_instances, image_instances):
        Matches a series with an image based on SOPInstanceUIDs.

    extract_matrix_transformation_direct(matrix_registration_sequence, index, matrix_type=""):
        Extracts matrix transformations from sequences without a nested MatrixSequence.

    get_fixed_image_info():
        Returns the metadata and transformation matrix for the fixed image.

    get_moving_image_info():
        Returns the metadata and transformation matrix for the moving image.

    dir():
        Custom dir method to return a list of available attributes and DICOM metadata keywords.
    """

    def __init__(self, reg_dataset):
        self.reg_dataset = reg_dataset
        self.fixed_image_info = {}
        self.moving_image_info = {}
        self.registration_type = None

        # Automatically extract transformation matrices and metadata upon instantiation
        self.extract_transformation_matrices_and_metadata()

        # Extract referenced series information
        self.extract_referenced_series_info()

        # Check for other references
        self.check_other_references()

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
        if attr in self.dir():
            return getattr(self.reg_dataset, attr)

        raise AttributeError(f"'REG' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        """
        Allows setting DICOM metadata attributes via dot notation.

        Parameters
        ----------
        attr : str
            The attribute being set (e.g., 'PatientID').
        value : str
            The value to be assigned to the attribute.

        Raises
        ------
        AttributeError
            If the attribute is not a valid DICOM keyword or if setting fails.
        """
        if attr == "reg_dataset":
            super().__setattr__(attr, value)
            return

        try:
            # Try to find the corresponding DICOM tag for the keyword
            tag = pydicom.datadict.tag_for_keyword(attr)
            if tag is not None:
                tag_obj = pydicom.tag.Tag(tag)
                tag_str = f"{tag_obj.group:04X}|{tag_obj.element:04X}"
                self.reg_dataset[tag_str].value = value
            else:
                # If it's not a valid DICOM tag, set it as a regular attribute
                super().__setattr__(attr, value)
        except Exception:
            # If anything goes wrong, fall back to setting the attribute normally
            super().__setattr__(attr, value)

    def __dir__(self):
        """
        Returns a list of attributes and DICOM metadata keywords.

        Returns
        -------
        list
            A combined list of attributes and DICOM metadata keywords.
        """
        default_dir = super().__dir__()
        dicom_keywords = [pydicom.datadict.keyword_for_tag(tag) for tag in self.reg_dataset.keys()]
        dicom_keywords = [keyword for keyword in dicom_keywords if keyword]  # Filter out None

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

    def extract_transformation_matrices_and_metadata(self):
        """
        Extracts transformation matrices and metadata for both rigid and deformable registrations.

        Raises
        ------
        ValueError
            If neither a RegistrationSequence nor DeformableRegistrationSequence is found.
        """
        ds = self.reg_dataset
        if "RegistrationSequence" in ds:
            self.registration_type = "rigid"
            self.extract_rigid_transformation(ds.RegistrationSequence)

        elif "DeformableRegistrationSequence" in ds:
            self.registration_type = "deformable"
            self.extract_deformable_transformation(ds.DeformableRegistrationSequence)
        else:
            raise ValueError("No RegistrationSequence or DeformableRegistrationSequence found.")

    def extract_rigid_transformation(self, reg_sequence):
        """
        Extracts transformation matrices and metadata for rigid registration.

        Parameters
        ----------
        reg_sequence : pydicom.Sequence
            The DICOM sequence for rigid registration.

        Raises
        ------
        ValueError
            If there are not exactly two items in the RegistrationSequence
            (fixed and moving images).
        """
        if len(reg_sequence) != 2:
            raise ValueError(
                "Expected two items in RegistrationSequence for fixed and moving images."
            )

        img_info_1 = self.extract_image_info(reg_sequence[0])
        img_info_2 = self.extract_image_info(reg_sequence[1])

        # Identify fixed and moving images using Frame of Reference UID
        if "SourceFrameOfReferenceUID" in img_info_1 and "SourceFrameOfReferenceUID" in img_info_2:
            if img_info_1["SourceFrameOfReferenceUID"] == self.reg_dataset.FrameOfReferenceUID:
                self.fixed_image_info = img_info_1
                self.moving_image_info = img_info_2
            else:
                self.fixed_image_info = img_info_2
                self.moving_image_info = img_info_1

        else:
            raise ValueError(
                "Frame of Reference UID missing in one or both RegistrationSequence items."
            )

    def extract_deformable_transformation(self, deformable_reg_sequence):
        """
        Extracts transformation matrices and grid-based information for deformable registration.

        Parameters
        ----------
        deformable_reg_sequence : pydicom.Sequence
            The DICOM sequence for deformable registration.
        """
        for i, sequence_item in enumerate(deformable_reg_sequence):
            image_info = {}
            if "MatrixRegistrationSequence" in sequence_item:
                # Handle matrix-based transformation
                self.extract_matrix_transformation(sequence_item.MatrixRegistrationSequence)

            if "DeformableRegistrationGridSequence" in sequence_item:
                # Handle grid-based transformation
                self.extract_grid_transformation(
                    sequence_item.DeformableRegistrationGridSequence, i
                )

            image_info = {
                "referenced_images": [
                    inst_item.ReferencedSOPInstanceUID
                    for inst_item in sequence_item.ReferencedImageSequence
                ],
                "SOPClassUID": sequence_item.ReferencedImageSequence[0].ReferencedSOPClassUID,
                "SourceFrameOfReferenceUID": sequence_item.get("SourceFrameOfReferenceUID", None),
            }
            if i == 0:
                self.fixed_image_info.update(image_info)
            else:
                self.moving_image_info.update(image_info)

            # Handle pre- and post-deformation matrices
            if "PreDeformationMatrixRegistrationSequence" in sequence_item:
                self.extract_matrix_transformation_direct(
                    sequence_item.PreDeformationMatrixRegistrationSequence, i, "pre"
                )

            if "PostDeformationMatrixRegistrationSequence" in sequence_item:
                self.extract_matrix_transformation_direct(
                    sequence_item.PostDeformationMatrixRegistrationSequence, i, "post"
                )

    def extract_image_info(self, reg_item):
        """
        Extracts transformation matrix and metadata from a single item in the RegistrationSequence.

        Parameters
        ----------
        reg_item : pydicom.Dataset
            An item from the RegistrationSequence.

        Returns
        -------
        dict
            A dictionary containing the transformation matrix, transformation type,
            SOPClassUID, and referenced SOP Instance UIDs.
        """
        image_info = {}
        if "MatrixRegistrationSequence" in reg_item:
            # Extract the transformation matrix from the MatrixRegistrationSequence
            matrix_registration_seq = reg_item.MatrixRegistrationSequence
            if len(matrix_registration_seq) > 0:
                image_info.update(self.extract_matrix_transformation(matrix_registration_seq))

        if "ReferencedImageSequence" in reg_item:
            ref_image_seq = reg_item.ReferencedImageSequence
            image_info["referenced_images"] = [
                ref_item.ReferencedSOPInstanceUID for ref_item in ref_image_seq
            ]
            image_info["SOPClassUID"] = ref_image_seq[0].ReferencedSOPClassUID

        # TODO
        # FrameOfReferenceUID here is not a required field here if ReferencedImageSequence
        # is present. We'll need the referenced images in that case to get their FOR_UID
        image_info["SourceFrameOfReferenceUID"] = reg_item.FrameOfReferenceUID

        return image_info

    def extract_matrix_transformation(self, matrix_registration_sequence):
        """
        Extracts matrix-based transformations from MatrixRegistrationSequence.

        Parameters
        ----------
        matrix_registration_sequence : pydicom.Sequence
            The sequence containing transformation matrices.

        Raises
        ------
        ValueError
            If MatrixRegistrationSequence is not found.
        """
        matrix_seq = matrix_registration_sequence[0].MatrixSequence

        if len(matrix_seq) == 0:
            raise ValueError("MatrixSequence is empty.")

        # Multiply matrices in the correct order
        transformation_matrix = np.eye(4)
        for matrix_item in reversed(matrix_seq):
            matrix = np.array(matrix_item.FrameOfReferenceTransformationMatrix).reshape(4, 4)
            transformation_matrix = transformation_matrix @ matrix

        matrix_info = {
            "transformation_matrix": transformation_matrix,
            "transformation_type": matrix_seq[0].FrameOfReferenceTransformationMatrixType,
        }

        return matrix_info

    def extract_grid_transformation(self, grid_sequence, index):
        """
        Extracts grid-based transformations for deformable registration.

        Parameters
        ----------
        grid_sequence : pydicom.Sequence
            The sequence containing grid-based transformation data.
        index : int
            Index representing whether the data is for the fixed (0) or moving (1) image.

        Raises
        ------
        ValueError
            If there is a mismatch in the expected and actual grid data size.
        """
        grid_seq = grid_sequence[0]
        grid_data_bytes = grid_seq.VectorGridData

        # Determine the expected shape and number of elements (3 for vector components x, y, z)
        grid_dimensions = grid_seq.GridDimensions
        expected_elements = np.prod(grid_dimensions) * 3

        # Assuming 32-bit (4-byte) floating-point numbers (float32)
        element_size = 4  # bytes
        expected_byte_size = expected_elements * element_size

        # Check if the actual size matches the expected byte size
        if len(grid_data_bytes) != expected_byte_size:
            raise ValueError(
                f"Grid data size mismatch. Expected {expected_byte_size} bytes, "
                f"but got {len(grid_data_bytes)} bytes."
            )

        # Unpack the binary data (assuming it's in float32 format)
        unpacked_grid_data = struct.unpack(f"{expected_elements}f", grid_data_bytes)

        # Reshape the unpacked data into (dimX, dimY, dimZ, 3) where 3 represents
        # x, y, z components
        grid_data = np.array(unpacked_grid_data).reshape(grid_dimensions + [3])

        image_info = {
            "grid_data": grid_data,
            "grid_dimensions": grid_seq.GridDimensions,
            "grid_resolution": grid_seq.GridResolution,
            "image_orientation": grid_seq.ImageOrientationPatient,
            "image_position": grid_seq.ImagePositionPatient,
        }

        # Store grid information
        if index == 0:
            self.fixed_image_info.update(image_info)

        else:
            self.moving_image_info.update(image_info)

    def extract_referenced_series_info(self):
        """
        Extracts referenced series information from the DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            The DICOM dataset object representing the REG file.
        """
        ds = self.reg_dataset
        if "ReferencedSeriesSequence" in ds:
            for series_item in ds.ReferencedSeriesSequence:
                series_info = {
                    "SeriesInstanceUID": series_item.SeriesInstanceUID,
                    "ReferencedInstances": [
                        instance.ReferencedSOPInstanceUID
                        for instance in series_item.ReferencedInstanceSequence
                    ],
                }

                if self.match_series_with_image(
                    series_info["ReferencedInstances"],
                    self.fixed_image_info.get("referenced_images", []),
                ):
                    self.fixed_image_info["SeriesInstanceUID"] = series_info["SeriesInstanceUID"]
                elif self.match_series_with_image(
                    series_info["ReferencedInstances"],
                    self.moving_image_info.get("referenced_images", []),
                ):
                    self.moving_image_info["SeriesInstanceUID"] = series_info["SeriesInstanceUID"]

    def check_other_references(self):
        """
        Checks for additional references in StudiesContainingOtherReferencedInstancesSequence.

        Parameters
        ----------
        ds : pydicom.Dataset
            The DICOM dataset object representing the REG file.
        """
        ds = self.reg_dataset
        if "StudiesContainingOtherReferencedInstancesSequence" in ds:
            for study in ds.StudiesContainingOtherReferencedInstancesSequence:
                if "ReferencedSeriesSequence" in study:
                    for series_item in study.ReferencedSeriesSequence:
                        other_referenced_instances = [
                            instance.ReferencedSOPInstanceUID
                            for instance in series_item.ReferencedInstanceSequence
                        ]

                        if self.match_series_with_image(
                            other_referenced_instances,
                            self.fixed_image_info.get("referenced_images", []),
                        ):
                            self.fixed_image_info["SeriesInstanceUID"] = (
                                series_item.SeriesInstanceUID
                            )
                            if "StudyInstanceUID" in study.dir():
                                self.fixed_image_info["StudyInstanceUID"] = study.StudyInstanceUID
                        elif self.match_series_with_image(
                            other_referenced_instances,
                            self.moving_image_info.get("referenced_images", []),
                        ):
                            self.moving_image_info["SeriesInstanceUID"] = (
                                series_item.SeriesInstanceUID
                            )
                            if "StudyInstanceUID" in study.dir():
                                self.moving_image_info["StudyInstanceUID"] = study.StudyInstanceUID

    def match_series_with_image(self, series_instances, image_instances):
        """
        Matches a series with an image based on SOPInstanceUIDs.

        Parameters
        ----------
        series_instances : list of str
            SOPInstanceUIDs from the series.
        image_instances : list of str
            SOPInstanceUIDs from the image.

        Returns
        -------
        bool
            True if there is a match, False otherwise.
        """
        # Check if any SOPInstanceUID in the series matches the SOPInstanceUIDs in the image info
        return any(instance_uid in image_instances for instance_uid in series_instances)

    def extract_matrix_transformation_direct(
        self, matrix_registration_sequence, index, matrix_type=""
    ):
        """
        Extracts matrix transformations from sequences without a nested MatrixSequence.

        Parameters
        ----------
        matrix_registration_sequence : pydicom.Sequence
            The sequence containing the transformation matrices.
        index : int
            Index representing whether the data is for the fixed (0) or moving (1) image.
        matrix_type : str
            Type of matrix (e.g., "pre", "post").

        Raises
        ------
        ValueError
            If MatrixRegistrationSequence is not found.
        """
        if len(matrix_registration_sequence) > 0:
            # Directly extract FrameOfReferenceTransformationMatrix and Type
            transformation_matrix = np.array(
                matrix_registration_sequence[0].FrameOfReferenceTransformationMatrix
            ).reshape(4, 4)

            image_info = {
                "transformation_matrix": transformation_matrix,
                "transformation_type": matrix_registration_sequence[
                    0
                ].FrameOfReferenceTransformationMatrixType,
            }

            # Store the matrix in the appropriate place
            self._store_matrix_info(image_info, index, matrix_type)
        else:
            raise ValueError("MatrixRegistrationSequence not found.")

    def _store_matrix_info(self, image_info, index, matrix_type):
        """
        Stores transformation matrix information in the correct place for fixed or moving image.

        Parameters
        ----------
        image_info : dict
            A dictionary containing the transformation matrix and related metadata.
        index : int
            Index representing whether the data is for the fixed (0) or moving (1) image.
        matrix_type : str
            Type of matrix (e.g., "pre", "post").
        """
        if index == 0:
            # Fixed image
            if matrix_type == "pre":
                self.fixed_image_info["pre_deformation_matrix"] = image_info
            elif matrix_type == "post":
                self.fixed_image_info["post_deformation_matrix"] = image_info
            else:
                self.fixed_image_info["matrix"] = image_info
        else:
            # Moving image
            if matrix_type == "pre":
                self.moving_image_info["pre_deformation_matrix"] = image_info
            elif matrix_type == "post":
                self.moving_image_info["post_deformation_matrix"] = image_info
            else:
                self.moving_image_info["matrix"] = image_info

    def get_fixed_image_info(self):
        """
        Returns the transformation matrix and metadata for the fixed image.

        Returns
        -------
        dict
            Dictionary containing the transformation matrix and metadata for the fixed image.

        Raises
        ------
        ValueError
            If the fixed image information has not been loaded.
        """
        if not self.fixed_image_info:
            raise ValueError("Fixed image information not loaded. Call `read` method first.")
        return self.fixed_image_info

    def get_moving_image_info(self):
        """
        Returns the transformation matrix and metadata for the moving image.

        Returns
        -------
        dict
            Dictionary containing the transformation matrix and metadata for the moving image.

        Raises
        ------
        ValueError
            If the moving image information has not been loaded.
        """
        if not self.moving_image_info:
            raise ValueError("Moving image information not loaded. Call `read` method first.")
        return self.moving_image_info

    def plot_deformation_grid(self, slice_index=0):
        """
        Plots the deformation grid using matplotlib's quiver plot for a 2D view of the grid data.

        Parameters
        ----------
        slice_index : int
            The index of the slice in the 3D grid to visualize.

        Raises
        ------
        ValueError
            If the grid data is not available.
        """
        if "grid_data" in self.moving_image_info:
            grid = self.moving_image_info["grid_data"]
            dimensions = self.moving_image_info["grid_dimensions"]

            if slice_index >= dimensions[2]:
                print(f"Slice index {slice_index} is out of bounds.")
                return

            # Let's assume we are visualizing the X and Y deformation for the selected slice
            x, y = np.meshgrid(np.arange(dimensions[1]), np.arange(dimensions[0]))

            # Selecting the specific slice for visualization
            u = grid[:, :, slice_index, 0]  # X component of the deformation vectors
            v = grid[:, :, slice_index, 1]  # Y component of the deformation vectors

            plt.figure()
            plt.quiver(x, y, u, v)
            plt.title(f"Deformation Grid (Slice {slice_index})")
            plt.show()
        else:
            print("No deformation grid data available for visualization.")
