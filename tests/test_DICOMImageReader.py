# import os
# import tempfile
# import pydicom
# import pytest
# from pydicom.dataset import Dataset
# from pydicom.dataset import FileMetaDataset
# from pydicom.uid import ImplicitVRLittleEndian
# from rosamllib.dicoms import DICOMImage
# from rosamllib.readers import DICOMImageReader
# from unittest.mock import patch


# @pytest.fixture
# def dicom_file():
#     # Create a temporary DICOM file for testing
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
#     ds = Dataset()
#     ds.PatientID = "123456"
#     ds.SeriesInstanceUID = "1.2.840.113619"
#     ds.ImagePositionPatient = [0, 0, 0]
#     ds.PixelData = b"\0" * 512 * 512
#     ds.Rows = 512
#     ds.Columns = 512
#     ds.PhotometricInterpretation = "MONOCHROME2"
#     ds.SOPInstanceUID = "1.2.840.113619.2.55.3"
#     ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
#     ds.Modality = "CT"
#     ds.ImageOrientationPatient = [0, 0, 1, 0, 1, 0]
#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
#     file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     ds.file_meta = file_meta
#     ds.save_as(temp_file.name, write_like_original=False)

#     temp_file.close()
#     yield temp_file.name
#     os.remove(temp_file.name)


# def test_single_file_input(dicom_file):
#     reader = DICOMImageReader(dicom_file)
#     dicom_image = reader.read()
#     assert isinstance(dicom_image, DICOMImage)
#     assert dicom_image.PatientID == "123456"


# def test_directory_input(dicom_file):
#     # Create a temporary directory with DICOM files
#     temp_dir = tempfile.TemporaryDirectory()
#     file_path = os.path.join(temp_dir.name, "test_file.dcm")
#     ds = pydicom.dcmread(dicom_file)
#     ds.save_as(file_path)

#     reader = DICOMImageReader(temp_dir.name)
#     dicom_image = reader.read()
#     assert isinstance(dicom_image, DICOMImage)

#     temp_dir.cleanup()


# def test_single_dataset_input(dicom_file):
#     ds = pydicom.dcmread(dicom_file)
#     reader = DICOMImageReader(ds)
#     dicom_image = reader.read()
#     assert isinstance(dicom_image, DICOMImage)
#     assert dicom_image.PatientID == "123456"


# def test_invalid_input():
#     with pytest.raises(ValueError):
#         DICOMImageReader(123)  # Invalid input, should raise ValueError


# @patch("rosamllib.dicoms.DICOMImageReader._process_dicom_directory")
# def test_process_dicom_directory(mock_process_dir, dicom_file):
#     mock_process_dir.return_value = [dicom_file]
#     reader = DICOMImageReader(dicom_file)
#     dicom_image = reader.read()
#     assert isinstance(dicom_image, DICOMImage)


# def test_read_single_dataset(dicom_file):
#     ds = pydicom.dcmread(dicom_file)
#     reader = DICOMImageReader(ds)
#     dicom_image = reader._read_single_dataset(ds)
#     assert isinstance(dicom_image, DICOMImage)
