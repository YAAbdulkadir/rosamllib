import struct
import numpy as np
import matplotlib.pyplot as plt
from pydicom.dataset import Dataset


class REG(Dataset):
    """
    DICOM Registration (REG) dataset with convenience methods for extracting
    rigid and deformable registration information.

    This class subclasses :class:`pydicom.dataset.Dataset` and adds
    registration-specific utilities to parse:

    - Rigid transformations from ``RegistrationSequence`` /
      ``MatrixRegistrationSequence``.
    - Deformable vector fields from
      ``DeformableRegistrationSequence`` /
      ``DeformableRegistrationGridSequence``.
    - Referenced image, series, and study identifiers for fixed and moving
      images.

    Instances are typically constructed via :meth:`REG.from_dataset`, which
    copies all DICOM elements from a base :class:`pydicom.Dataset` and then
    populates the registration metadata.

    Attributes
    ----------
    fixed_image_info : dict
        Metadata and transformation information for the fixed image. May
        include keys such as:

        - ``"referenced_images"``: list of referenced SOPInstanceUIDs
        - ``"SOPClassUID"``: SOP class UID of the referenced images
        - ``"SourceFrameOfReferenceUID"``: frame of reference UID
        - ``"SeriesInstanceUID"`` / ``"StudyInstanceUID"``
        - ``"transformation_matrix"`` / ``"matrix"`` and related entries
        - Grid-related fields for deformable registration (e.g. ``"grid_data"``)
    moving_image_info : dict
        Same structure as ``fixed_image_info``, but for the moving image.
    registration_type : str or None
        ``"rigid"`` if a ``RegistrationSequence`` is present,
        ``"deformable"`` if a ``DeformableRegistrationSequence`` is present,
        otherwise ``None`` (before extraction is performed).

    Methods
    -------
    from_dataset(ds)
        Construct a :class:`REG` from an existing :class:`pydicom.Dataset`.
    extract_transformation_matrices_and_metadata()
        Detect and extract rigid or deformable transformation data.
    extract_rigid_transformation(reg_sequence)
        Parse rigid registration matrices and assign fixed/moving image info.
    extract_deformable_transformation(deformable_reg_sequence)
        Parse deformable grid and optional pre/post matrices.
    extract_image_info(reg_item)
        Extract transformation and reference info from a single registration item.
    extract_matrix_transformation(matrix_registration_sequence)
        Compose a 4x4 transformation matrix from ``MatrixSequence``.
    extract_grid_transformation(grid_sequence, index)
        Decode the deformation vector field into a 4D NumPy array.
    extract_referenced_series_info()
        Attach series-level UIDs to fixed/moving image info.
    check_other_references()
        Attach study/series UIDs from
        ``StudiesContainingOtherReferencedInstancesSequence`` when present.
    match_series_with_image(series_instances, image_instances)
        Helper to test SOPInstanceUID overlap between series and image refs.
    extract_matrix_transformation_direct(...)
        Handle matrix sequences without a nested ``MatrixSequence``.
    get_fixed_image_info()
        Return the fixed-image registration metadata.
    get_moving_image_info()
        Return the moving-image registration metadata.
    plot_deformation_grid(slice_index=0)
        Visualize a 2D slice of the deformable grid using a quiver plot.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fixed_image_info = {}
        self.moving_image_info = {}
        self.registration_type = None

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "REG":
        """
        Create a :class:`REG` instance from an existing DICOM dataset.

        This copies all data elements from ``ds`` into a new :class:`REG`
        instance and then runs the extraction pipeline to populate
        ``fixed_image_info``, ``moving_image_info``, and ``registration_type``.

        Parameters
        ----------
        ds : pydicom.Dataset
            A DICOM dataset representing a REG object (Modality ``"REG"``).

        Returns
        -------
        REG
            A fully initialized :class:`REG` instance with parsed registration
            metadata.
        """
        reg = cls()
        reg.update(ds)

        if hasattr(ds, "file_meta"):
            reg.file_meta = ds.file_meta
        reg.fixed_image_info = {}
        reg.moving_image_info = {}
        reg.registration_type = None
        reg.extract_transformation_matrices_and_metadata()
        reg.extract_referenced_series_info()
        reg.check_other_references()
        return reg

    def extract_transformation_matrices_and_metadata(self):
        """
        Extract transformation matrices and registration metadata.

        Determines whether the dataset represents a rigid or deformable
        registration, sets :attr:`registration_type`, and dispatches to
        the appropriate extraction method.

        Raises
        ------
        ValueError
            If neither ``RegistrationSequence`` nor
            ``DeformableRegistrationSequence`` is present in the dataset.
        """
        if "RegistrationSequence" in self:
            self.registration_type = "rigid"
            self.extract_rigid_transformation(self.RegistrationSequence)

        elif "DeformableRegistrationSequence" in self:
            self.registration_type = "deformable"
            self.extract_deformable_transformation(self.DeformableRegistrationSequence)
        else:
            raise ValueError("No RegistrationSequence or DeformableRegistrationSequence found.")

    def extract_rigid_transformation(self, reg_sequence):
        """
        Extract transformation matrices and metadata for rigid registration.

        Parameters
        ----------
        reg_sequence : pydicom.Sequence
            The ``RegistrationSequence`` from the REG dataset. Must contain
            exactly two items corresponding to the fixed and moving images.

        Raises
        ------
        ValueError
            If there are not exactly two items in the sequence, or if
            ``FrameOfReferenceUID`` is missing from one or both items.
        """
        if len(reg_sequence) != 2:
            raise ValueError(
                "Expected two items in RegistrationSequence for fixed and moving images."
            )

        img_info_1 = self.extract_image_info(reg_sequence[0])
        img_info_2 = self.extract_image_info(reg_sequence[1])

        # Identify fixed and moving images using Frame of Reference UID
        if "SourceFrameOfReferenceUID" in img_info_1 and "SourceFrameOfReferenceUID" in img_info_2:
            if img_info_1["SourceFrameOfReferenceUID"] == self.FrameOfReferenceUID:
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
        Extract transformation matrices and grid-based information for deformable registration.

        Parameters
        ----------
        deformable_reg_sequence : pydicom.Sequence
            The ``DeformableRegistrationSequence`` from the REG dataset.
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
        Extract transformation matrix and metadata from a single registration item.

        Parameters
        ----------
        reg_item : pydicom.Dataset
            An item from ``RegistrationSequence`` or a similar registration
            sequence.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"transformation_matrix"`` and ``"transformation_type"`` (if present)
            - ``"referenced_images"``: list of referenced SOPInstanceUIDs
            - ``"SOPClassUID"`` of the referenced images
            - ``"SourceFrameOfReferenceUID"`` derived from ``FrameOfReferenceUID``
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
        Extract and compose matrix-based transformations from a MatrixRegistrationSequence.

        Parameters
        ----------
        matrix_registration_sequence : pydicom.Sequence
            A ``MatrixRegistrationSequence`` containing one or more items, each
            with a nested ``MatrixSequence``.

        Returns
        -------
        dict
            Dictionary with:

            - ``"transformation_matrix"``: 4x4 NumPy array
            - ``"transformation_type"``: value of
              ``FrameOfReferenceTransformationMatrixType``

        Raises
        ------
        ValueError
            If the inner ``MatrixSequence`` is empty.
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
        Extract grid-based transformations for deformable registration.

        Parameters
        ----------
        grid_sequence : pydicom.Sequence
            A ``DeformableRegistrationGridSequence`` containing the vector grid.
        index : int
            Index representing whether the data corresponds to the fixed (0) or
            moving (1) image.

        Raises
        ------
        ValueError
            If there is a mismatch between the expected and actual grid data size.
        """
        grid_seq = grid_sequence[0]
        grid_data_bytes = grid_seq.VectorGridData

        # Normalize GridDimensions
        grid_dimensions = tuple(int(d) for d in grid_seq.GridDimensions)

        # Determine the expected shape and number of elements (3 for vector components x, y, z)
        expected_elements = int(np.prod(grid_dimensions)) * 3

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
        grid_data = np.array(unpacked_grid_data, dtype=np.float32).reshape(
            grid_dimensions[0],
            grid_dimensions[1],
            grid_dimensions[2],
            3,
        )

        image_info = {
            "grid_data": grid_data,
            "grid_dimensions": grid_dimensions,
            "grid_resolution": tuple(grid_seq.GridResolution),
            "image_orientation": list(grid_seq.ImageOrientationPatient),
            "image_position": list(grid_seq.ImagePositionPatient),
        }

        # Store grid information
        if index == 0:
            self.fixed_image_info.update(image_info)

        else:
            self.moving_image_info.update(image_info)

    def extract_referenced_series_info(self):
        """
        Extract referenced series information from the REG dataset.

        When ``ReferencedSeriesSequence`` is present at the top level, this
        method scans the referenced instances and, based on overlapping
        SOPInstanceUIDs, assigns:

        - ``"SeriesInstanceUID"`` to :attr:`fixed_image_info` and/or
          :attr:`moving_image_info`.

        This helps link the registration to specific image series.
        """
        if "ReferencedSeriesSequence" in self:
            for series_item in self.ReferencedSeriesSequence:
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
        Populate additional study and series references, if available.

        When ``StudiesContainingOtherReferencedInstancesSequence`` is present,
        this method traverses those structures and, based on SOPInstanceUID
        overlap, attaches:

        - ``"SeriesInstanceUID"`` and, when present,
        - ``"StudyInstanceUID"``

        to :attr:`fixed_image_info` and/or :attr:`moving_image_info`.
        """
        if "StudiesContainingOtherReferencedInstancesSequence" in self:
            for study in self.StudiesContainingOtherReferencedInstancesSequence:
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
        Check whether a referenced series matches a set of image SOPInstanceUIDs.

        Parameters
        ----------
        series_instances : list of str
            SOPInstanceUIDs referenced by a series.
        image_instances : list of str
            SOPInstanceUIDs associated with a fixed or moving image.

        Returns
        -------
        bool
            ``True`` if any SOPInstanceUID is shared between the two lists,
            otherwise ``False``.
        """
        # Check if any SOPInstanceUID in the series matches the SOPInstanceUIDs in the image info
        return any(instance_uid in image_instances for instance_uid in series_instances)

    def extract_matrix_transformation_direct(
        self, matrix_registration_sequence, index, matrix_type=""
    ):
        """
        Extract matrix transformations from a sequence without a nested MatrixSequence.

        This is used for sequences such as
        ``PreDeformationMatrixRegistrationSequence`` and
        ``PostDeformationMatrixRegistrationSequence`` where the
        transformation matrix is stored directly on the sequence items.

        Parameters
        ----------
        matrix_registration_sequence : pydicom.Sequence
            The sequence containing a single item with
            ``FrameOfReferenceTransformationMatrix`` and
            ``FrameOfReferenceTransformationMatrixType``.
        index : int
            Index representing whether the data is for the fixed (0) or moving (1) image.
        matrix_type : str, optional
            Type of matrix (e.g., ``"pre"``, ``"post"``). Used to decide
            where in the info dict the matrix is stored.

        Raises
        ------
        ValueError
            If the sequence is empty.
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
        Store transformation matrix information for the fixed or moving image.

        Parameters
        ----------
        image_info : dict
            Dictionary containing the transformation matrix and related metadata.
        index : int
            Index representing whether the data is for the fixed (0) or moving (1) image.
        matrix_type : str
            Type of matrix (e.g., ``"pre"``, ``"post"``, or ``""`` for main matrix).
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
        Return the registration metadata for the fixed image.

        Returns
        -------
        dict
            Dictionary containing the transformation matrix, referenced image
            identifiers, and any series/study/grid metadata associated with
            the fixed image.

        Raises
        ------
        ValueError
            If the fixed image information has not been populated. Ensure that
            the instance was created via :meth:`REG.from_dataset` or
            :class:`REGReader` before calling this method.
        """
        if not self.fixed_image_info:
            raise ValueError("Fixed image information not loaded. Call `read` method first.")
        return self.fixed_image_info

    def get_moving_image_info(self):
        """
        Return the registration metadata for the moving image.

        Returns
        -------
        dict
            Dictionary containing the transformation matrix, referenced image
            identifiers, and any series/study/grid metadata associated with
            the moving image.

        Raises
        ------
        ValueError
            If the moving image information has not been populated. Ensure that
            the instance was created via :meth:`REG.from_dataset` or
            :class:`REGReader` before calling this method.
        """
        if not self.moving_image_info:
            raise ValueError("Moving image information not loaded. Call `read` method first.")
        return self.moving_image_info

    def plot_deformation_grid(self, slice_index=0):
        """
        Visualize a 2D slice of the deformable registration grid.

        This uses :mod:`matplotlib`'s quiver plot to display the in-plane
        deformation vectors (x and y components) for a given slice of the
        3D deformation grid stored in :attr:`moving_image_info`.

        Parameters
        ----------
        slice_index : int, optional
            Index of the slice along the third grid dimension (z-axis) to
            visualize.

        Raises
        ------
        ValueError
            If grid data is expected but not available.
        """
        if "grid_data" in self.moving_image_info:
            grid = self.moving_image_info["grid_data"]
            dimensions = tuple(int(d) for d in self.moving_image_info["grid_dimensions"])

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

    def __repr__(self) -> str:
        """
        Return a concise string representation of the REG object.

        The representation includes the registration type, series instance UIDs
        (if available) for fixed and moving images, and the number of referenced
        images attached to each.

        Examples
        --------
        >>> reg
        REG(registration_type='rigid', fixed_series='1.2.3', moving_series='4.5.6',
            fixed_refs=3, moving_refs=3)
        """
        reg_type = self.registration_type or "unknown"
        fixed_series = self.fixed_image_info.get("SeriesInstanceUID", "N/A")
        moving_series = self.moving_image_info.get("SeriesInstanceUID", "N/A")
        fixed_refs = len(self.fixed_image_info.get("referenced_images", []))
        moving_refs = len(self.moving_image_info.get("referenced_images", []))

        return (
            "REG("
            f"registration_type={reg_type!r}, "
            f"fixed_series={fixed_series!r}, "
            f"moving_series={moving_series!r}, "
            f"fixed_refs={fixed_refs}, "
            f"moving_refs={moving_refs}"
            ")"
        )
