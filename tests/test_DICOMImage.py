# import pytest
# import numpy as np
# import SimpleITK as sitk
# from rosamllib.dicoms import DICOMImage


# @pytest.fixture
# def sample_dicom_image():
#     # Create a sample SimpleITK Image for testing
#     image_array = np.random.rand(64, 64, 64)  # A 3D random image (64x64x64)
#     sitk_image = sitk.GetImageFromArray(image_array)
#     sitk_image.SetSpacing((1.0, 1.0, 1.0))
#     sitk_image.SetOrigin((0.0, 0.0, 0.0))
#     sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

#     return DICOMImage(sitk_image)


# def test_get_image_array(sample_dicom_image):
#     # Test that the image is correctly converted to a NumPy array
#     image_array = sample_dicom_image.get_image_array()
#     assert isinstance(image_array, np.ndarray)
#     assert image_array.shape == (64, 64, 64)


# def test_resample_image(sample_dicom_image):
#     # Test the resampling functionality
#     new_spacing = (2.0, 2.0, 2.0)
#     resampled_image = sample_dicom_image.resample_image(new_spacing)

#     assert isinstance(resampled_image, DICOMImage)
#     assert resampled_image.GetSpacing() == new_spacing
#     assert resampled_image.GetSize() != sample_dicom_image.GetSize()


# def test_get_pixel_to_physical_transformation_matrix(sample_dicom_image):
#     # Test the pixel-to-physical transformation matrix
#     matrix = sample_dicom_image.get_pixel_to_physical_transformation_matrix()

#     assert isinstance(matrix, np.ndarray)
#     assert matrix.shape == (4, 4)
#     assert np.allclose(matrix[:3, :3], np.eye(3))  # Identity for the default direction


# def test_get_physical_to_pixel_transformation_matrix(sample_dicom_image):
#     # Test the physical-to-pixel transformation matrix
#     matrix = sample_dicom_image.get_physical_to_pixel_transformation_matrix()

#     assert isinstance(matrix, np.ndarray)
#     assert matrix.shape == (4, 4)


# def test_transform_to_physical_coordinates(sample_dicom_image):
#     # Test transforming pixel coordinates to physical coordinates
#     pixel_points = np.array([[32, 32, 32]])
#     physical_points = sample_dicom_image.transform_to_physical_coordinates(pixel_points)

#     assert physical_points.shape == pixel_points.shape
#     assert np.allclose(physical_points, pixel_points)  # Since spacing and origin are identity


# def test_transform_to_pixel_coordinates(sample_dicom_image):
#     # Test transforming physical coordinates to pixel coordinates
#     physical_points = np.array([[32.0, 32.0, 32.0]])
#     pixel_points = sample_dicom_image.transform_to_pixel_coordinates(physical_points)

#     assert pixel_points.shape == physical_points.shape
#     assert np.allclose(pixel_points, physical_points)  # Since spacing and origin are identity


# def test_getattr_dicom_metadata(sample_dicom_image):
#     # Test accessing DICOM metadata via __getattr__
#     sample_dicom_image.SetMetaData("0010|0010", "Test Patient")

#     assert sample_dicom_image.PatientName == "Test Patient"


# def test_setattr_dicom_metadata(sample_dicom_image):
#     # Test setting DICOM metadata via __setattr__
#     sample_dicom_image.PatientName = "John Doe"

#     assert sample_dicom_image.GetMetaData("0010|0010") == "John Doe"


# def test_dir_method_includes_dicom_metadata(sample_dicom_image):
#     # Test that DICOM metadata is included in the dir() output
#     sample_dicom_image.SetMetaData("0010|0010", "John Doe")
#     attributes = sample_dicom_image.dir()

#     assert "PatientName" in attributes
