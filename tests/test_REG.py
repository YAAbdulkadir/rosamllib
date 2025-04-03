# import pytest
# import numpy as np
# import pydicom
# from pydicom.dataset import Dataset
# from pydicom.dataset import FileMetaDataset
# from pydicom.uid import ImplicitVRLittleEndian
# from rosamllib.dicoms import REG


# @pytest.fixture
# def sample_reg_dataset():
#     """
#     Creates a sample pydicom.Dataset for REG testing.

#     Returns
#     -------
#     pydicom.Dataset
#         A mock REG DICOM dataset.
#     """
#     dataset = Dataset()
#     dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.1"
#     dataset.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.10"
#     dataset.Modality = "REG"

#     # Add fake registration sequences for testing
#     dataset.RegistrationSequence = pydicom.Sequence([Dataset(), Dataset()])
#     dataset.RegistrationSequence[0].MatrixRegistrationSequence = pydicom.Sequence([Dataset()])
#     dataset.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence = (
#         pydicom.Sequence([Dataset()])
#     )
#     dataset.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence[
#         0
#     ].FrameOfReferenceTransformationMatrix = list(np.eye(4).flatten())
#     dataset.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence[
#         0
#     ].FrameOfReferenceTransformationMatrixType = "RIGID"

#     dataset.RegistrationSequence[1].MatrixRegistrationSequence = pydicom.Sequence([Dataset()])
#     dataset.RegistrationSequence[1].MatrixRegistrationSequence[0].MatrixSequence = (
#         pydicom.Sequence([Dataset()])
#     )
#     dataset.RegistrationSequence[1].MatrixRegistrationSequence[0].MatrixSequence[
#         0
#     ].FrameOfReferenceTransformationMatrix = list(np.eye(4).flatten())
#     dataset.RegistrationSequence[1].MatrixRegistrationSequence[0].MatrixSequence[
#         0
#     ].FrameOfReferenceTransformationMatrixType = "RIGID"

#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
#     file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     dataset.file_meta = file_meta

#     return dataset


# @pytest.fixture
# def sample_deformable_reg_dataset():
#     """
#     Creates a sample pydicom.Dataset for deformable registration testing.

#     Returns
#     -------
#     pydicom.Dataset
#         A mock deformable registration DICOM dataset.
#     """
#     dataset = Dataset()
#     dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.1"
#     dataset.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.10"
#     dataset.Modality = "REG"

#     # Add fake deformable registration sequences for testing
#     dataset.DeformableRegistrationSequence = pydicom.Sequence([Dataset(), Dataset()])
#     dataset.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence = (
#         pydicom.Sequence([Dataset()])
#     )
#     dataset.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[
#         0
#     ].GridDimensions = [10, 10, 10]
#     dataset.DeformableRegistrationSequence[0].DeformableRegistrationGridSequence[
#         0
#     ].VectorGridData = (np.random.rand(10 * 10 * 10 * 3).astype(np.float32).tobytes())

#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
#     file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     dataset.file_meta = file_meta

#     return dataset


# def test_rigid_registration_extraction(sample_reg_dataset):
#     # Test extraction of transformation matrices for rigid registration
#     reg = REG(sample_reg_dataset)

#     # Check that fixed and moving image information is extracted
#     assert "transformation_matrix" in reg.fixed_image_info
#     assert "transformation_matrix" in reg.moving_image_info

#     fixed_matrix = reg.fixed_image_info["transformation_matrix"]
#     moving_matrix = reg.moving_image_info["transformation_matrix"]

#     # Check that the transformation matrices are 4x4
#     assert fixed_matrix.shape == (4, 4)
#     assert moving_matrix.shape == (4, 4)

#     # Check that the matrices are identity matrices (from the mock dataset)
#     assert np.allclose(fixed_matrix, np.eye(4))
#     assert np.allclose(moving_matrix, np.eye(4))


# def test_deformable_registration_extraction(sample_deformable_reg_dataset):
#     # Test extraction of deformation grid for deformable registration
#     reg = REG(sample_deformable_reg_dataset)

#     # Check that grid data is extracted
#     assert "grid_data" in reg.fixed_image_info
#     grid_data = reg.fixed_image_info["grid_data"]

#     # Check the shape of the extracted grid data (should be 10x10x10x3)
#     assert grid_data.shape == (10, 10, 10, 3)

#     # Check that the grid data contains floating point numbers
#     assert np.issubdtype(grid_data.dtype, np.float32)


# def test_plot_deformation_grid(sample_deformable_reg_dataset):
#     # Test if the deformation grid plotting works without raising exceptions
#     reg = REG(sample_deformable_reg_dataset)

#     try:
#         reg.plot_deformation_grid(slice_index=5)
#     except Exception as e:
#         pytest.fail(f"plot_deformation_grid raised an exception: {e}")


# def test_dicom_metadata_access(sample_reg_dataset):
#     # Test DICOM metadata access via dot notation
#     reg = REG(sample_reg_dataset)

#     # Access a non-existent metadata field (should raise an AttributeError)
#     with pytest.raises(AttributeError):
#         _ = reg.PatientID  # Metadata field not present


# def test_dicom_metadata_assignment(sample_reg_dataset):
#     # Test setting DICOM metadata via dot notation
#     reg = REG(sample_reg_dataset)

#     # Set a new metadata field
#     reg.PatientID = "123456"

#     # Check that the metadata is correctly set in the DICOM dataset
#     assert reg.reg_dataset.PatientID == "123456"


# def test_dir_method(sample_reg_dataset):
#     # Test the dir method to ensure it includes DICOM metadata fields
#     reg = REG(sample_reg_dataset)
#     attributes = reg.dir()

#     # Check that the DICOM metadata keyword (e.g., 'PatientID') is in the list of attributes
#     assert "PatientID" in attributes
