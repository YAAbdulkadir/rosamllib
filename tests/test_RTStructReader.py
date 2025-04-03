# import os
# import tempfile
# import pytest
# import pydicom
# from pydicom.dataset import Dataset
# from pydicom.dataset import FileMetaDataset
# from pydicom.uid import ImplicitVRLittleEndian
# from pydicom.errors import InvalidDicomError
# from rosamllib.readers import RTStructReader
# from rosamllib.dicoms import RTStruct


# @pytest.fixture
# def rtstruct_file():
#     # Create a temporary RTSTRUCT DICOM file for testing
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
#     ds = Dataset()
#     ds.Modality = "RTSTRUCT"
#     ds.PatientID = "12345"
#     ds.SOPInstanceUID = "1.2.840.113619.2.55.3"
#     ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
#     ds.ROIContourSequence = []
#     ds.StructureSetROISequence = []

#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
#     file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     ds.file_meta = file_meta

#     ds.save_as(temp_file.name, write_like_original=False)

#     temp_file.close()

#     yield temp_file.name
#     os.remove(temp_file.name)


# def test_read_rtstruct_file(rtstruct_file):
#     reader = RTStructReader(rtstruct_file)
#     rtstruct = reader.read()
#     assert isinstance(rtstruct, RTStruct)
#     assert rtstruct.ds.PatientID == "12345"


# def test_read_rtstruct_from_dataset(rtstruct_file):
#     ds = pydicom.dcmread(rtstruct_file)
#     reader = RTStructReader(ds)
#     rtstruct = reader.read()
#     assert isinstance(rtstruct, RTStruct)
#     assert rtstruct.ds.PatientID == "12345"


# def test_invalid_modality_raises_error():
#     ds = Dataset()
#     ds.Modality = "CT"  # Not an RTSTRUCT
#     reader = RTStructReader(ds)
#     with pytest.raises(InvalidDicomError, match="Provided dataset is not an RTSTRUCT."):
#         reader.read_from_dataset(ds)


# def test_no_rtstruct_in_directory():
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create a temporary DICOM file with incorrect modality
#         temp_file = os.path.join(temp_dir, "invalid_rtstruct.dcm")
#         ds = Dataset()
#         ds.Modality = "CT"  # Invalid modality
#         ds.save_as(temp_file)

#         reader = RTStructReader(temp_dir)
#         with pytest.raises(IOError, match="No RTSTRUCT file found in directory"):
#             reader.read()


# def test_find_rtstruct_in_directory(rtstruct_file):
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Move the test RTSTRUCT file to the temporary directory
#         rtstruct_path = os.path.join(temp_dir, "test_rtstruct.dcm")
#         os.rename(rtstruct_file, rtstruct_path)

#         # Check if the RTStructReader finds the RTSTRUCT in the directory
#         file_path = RTStructReader.find_rtstruct_in_directory(temp_dir)
#         assert file_path == rtstruct_path


# def test_invalid_input_type():
#     with pytest.raises(TypeError, match="Input must be a file path"):
#         RTStructReader(123)  # Invalid input type


# def test_read_invalid_rtstruct_file():
#     # Create a temporary file with no valid DICOM content
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
#         temp_file.write(b"Invalid data")

#     reader = RTStructReader(temp_file.name)
#     with pytest.raises(IOError, match="Error reading RTSTRUCT file"):
#         reader.read()

#     os.remove(temp_file.name)
