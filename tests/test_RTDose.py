# import pytest
# import numpy as np
# import SimpleITK as sitk
# from rosamllib.dicoms import RTDose
# from rosamllib.dicoms import DICOMImage


# @pytest.fixture
# def sample_rtdose():
#     """
#     Creates a sample RTDose object for testing.

#     Returns
#     -------
#     RTDose
#         A sample RTDose object with a blank 3D image and mock metadata.
#     """
#     # Create a dummy dose grid as a 3D image with random values
#     dose_array = np.random.rand(64, 64, 64)
#     sitk_dose = sitk.GetImageFromArray(dose_array)

#     # Add sample metadata, including DoseGridScaling
#     sitk_dose.SetSpacing((1.0, 1.0, 1.0))
#     sitk_dose.SetOrigin((0.0, 0.0, 0.0))
#     sitk_dose.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
#     sitk_dose.SetMetaData("3004|000e", "0.02")  # DoseGridScaling

#     return RTDose(sitk_dose)


# @pytest.fixture
# def sample_dicom_image():
#     """
#     Creates a sample DICOMImage object for testing.

#     Returns
#     -------
#     DICOMImage
#         A sample DICOMImage object with a blank 3D image.
#     """
#     # Create a dummy CT image
#     image_array = np.zeros((64, 64, 64))
#     sitk_image = sitk.GetImageFromArray(image_array)
#     sitk_image.SetSpacing((1.0, 1.0, 1.0))
#     sitk_image.SetOrigin((0.0, 0.0, 0.0))
#     sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

#     return DICOMImage(sitk_image)


# def test_get_dose_array(sample_rtdose):
#     # Test if the dose array is retrieved and properly scaled
#     dose_array = sample_rtdose.get_dose_array()
#     assert isinstance(dose_array, np.ndarray)
#     assert dose_array.shape == (64, 64, 64)
#     assert np.all(dose_array <= 1.0)  # Dose values are scaled by the factor in metadata


# def test_dose_grid_scaling(sample_rtdose):
#     # Test if the dose grid scaling is retrieved properly
#     scaling = sample_rtdose.dose_grid_scaling
#     assert scaling == 0.02


# def test_resample_dose_to_image_grid(sample_rtdose, sample_dicom_image):
#     # Test resampling the dose to match a referenced DICOM image grid
#     resampled_dose = sample_rtdose.resample_dose_to_image_grid(sample_dicom_image)
#     assert isinstance(resampled_dose, RTDose)

#     # Ensure that the resampled dose grid matches the size of the referenced image grid
#     assert resampled_dose.GetSize() == sample_dicom_image.GetSize()


# def test_dicom_metadata_access(sample_rtdose):
#     # Test accessing DICOM metadata via dot notation
#     assert sample_rtdose.dose_grid_scaling == 0.02

#     # Test accessing a metadata field
#     with pytest.raises(AttributeError):
#         _ = sample_rtdose.PatientID  # Metadata field not present


# def test_dicom_metadata_assignment(sample_rtdose):
#     # Test setting DICOM metadata via dot notation
#     sample_rtdose.PatientID = "123456"
#     assert sample_rtdose.GetMetaData("0010|0020") == "123456"


# def test_dicom_dir_method(sample_rtdose):
#     # Test the dir method to ensure it includes DICOM metadata fields
#     attributes = sample_rtdose.dir()
#     assert "dose_grid_scaling" in attributes
#     assert "3004|000e" not in attributes  # Should display the keyword, not the tag
