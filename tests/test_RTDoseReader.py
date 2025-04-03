# import os
# import tempfile
# import pytest
# import pydicom
# from rosamllib.readers import RTDoseReader
# from pydicom.dataset import Dataset
# from pydicom.dataset import FileMetaDataset
# from pydicom.uid import ImplicitVRLittleEndian
# from pydicom.errors import InvalidDicomError
# from rosamllib.dicoms import RTDose


# @pytest.fixture
# def rtdose_file():
#     # Create a temporary RTDOSE DICOM file for testing
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
#     ds = Dataset()
#     ds.Modality = "RTDOSE"
#     ds.PatientID = "12345"
#     ds.SOPInstanceUID = "1.2.840.113619.2.55.3"
#     ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
#     ds.PixelData = b"\x00" * 512 * 512 * 10  # Simulate pixel data
#     ds.Rows = 512
#     ds.Columns = 512
#     ds.NumberOfFrames = 10
#     ds.PixelSpacing = [1.0, 1.0]
#     ds.GridFrameOffsetVector = [i for i in range(10)]
#     ds.ImagePositionPatient = [0.0, 0.0, 0.0]
#     ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
#     file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     ds.file_meta = file_meta

#     ds.save_as(temp_file.name, write_like_original=False)

#     temp_file.close()

#     yield temp_file.name
#     os.remove(temp_file.name)


# def test_read_rtdose_file(rtdose_file):
#     reader = RTDoseReader(rtdose_file)
#     rtdose = reader.read()
#     assert isinstance(rtdose, RTDose)
#     assert rtdose.GetSpacing() == (1.0, 1.0, 1.0)


# def test_read_rtdose_from_dataset(rtdose_file):
#     ds = pydicom.dcmread(rtdose_file)
#     reader = RTDoseReader(ds)
#     rtdose = reader.read()
#     assert isinstance(rtdose, RTDose)
#     assert rtdose.GetSpacing() == (1.0, 1.0, 1.0)


# def test_invalid_modality_raises_error():
#     ds = Dataset()
#     ds.Modality = "CT"  # Not an RTDOSE
#     reader = RTDoseReader(ds)
#     with pytest.raises(InvalidDicomError, match="Provided dataset is not an RTDOSE."):
#         reader.read_from_dataset(ds)


# def test_no_rtdose_in_directory():
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create a temporary DICOM file with incorrect modality
#         temp_file = os.path.join(temp_dir, "invalid_rtdose.dcm")
#         ds = Dataset()
#         ds.Modality = "CT"  # Invalid modality
#         ds.save_as(temp_file)

#         reader = RTDoseReader(temp_dir)
#         with pytest.raises(IOError, match="No RTDOSE file found in directory"):
#             reader.read()


# def test_find_rtdose_in_directory(rtdose_file):
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Move the test RTDOSE file to the temporary directory
#         rtdose_path = os.path.join(temp_dir, "test_rtdose.dcm")
#         os.rename(rtdose_file, rtdose_path)

#         # Check if the RTDoseReader finds the RTDOSE in the directory
#         file_path = RTDoseReader.find_rtdose_in_directory(temp_dir)
#         assert file_path == rtdose_path


# def test_invalid_input_type():
#     with pytest.raises(ValueError, match="Input must be a file path or a pydicom.Dataset."):
#         RTDoseReader(123)  # Invalid input type


# def test_read_invalid_rtdose_file():
#     # Create a temporary file with no valid DICOM content
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
#         temp_file.write(b"Invalid data")

#     reader = RTDoseReader(temp_file.name)
#     with pytest.raises(IOError, match="Error reading RTDOSE file"):
#         reader.read()

#     os.remove(temp_file.name)
