# import os
# import pytest
# import pydicom
# import tempfile
# from rosamllib.readers import REGReader
# from pydicom.dataset import Dataset
# from pydicom.dataset import FileMetaDataset
# from pydicom.uid import ImplicitVRLittleEndian
# from pydicom.errors import InvalidDicomError
# from rosamllib.dicoms import REG


# @pytest.fixture
# def reg_file():
#     # Create a temporary REG DICOM file for testing
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
#     ds = Dataset()
#     ds.Modality = "REG"
#     ds.PatientID = "12345"
#     ds.SOPInstanceUID = "1.2.840.113619.2.55.3"
#     ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.1"

#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
#     file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     ds.file_meta = file_meta

#     ds.save_as(temp_file.name, write_like_original=False)

#     temp_file.close()

#     yield temp_file.name
#     os.remove(temp_file.name)


# def test_read_reg_file(reg_file):
#     reader = REGReader(reg_file)
#     reg = reader.read()
#     assert isinstance(reg, REG)
#     assert reg.reg_dataset.Modality == "REG"


# def test_read_reg_from_dataset(reg_file):
#     ds = pydicom.dcmread(reg_file)
#     reader = REGReader(ds)
#     reg = reader.read()
#     assert isinstance(reg, REG)
#     assert reg.reg_dataset.Modality == "REG"


# def test_invalid_modality_raises_error():
#     ds = Dataset()
#     ds.Modality = "CT"  # Not a REG modality
#     reader = REGReader(ds)
#     with pytest.raises(InvalidDicomError):
#         reader.read()


# def test_no_reg_in_directory():
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create a temporary DICOM file with incorrect modality
#         temp_file = os.path.join(temp_dir, "invalid_reg.dcm")
#         ds = Dataset()
#         ds.Modality = "CT"  # Invalid modality
#         ds.save_as(temp_file)

#         reader = REGReader(temp_dir)
#         with pytest.raises(IOError, match="No REG file found in directory"):
#             reader.read()


# def test_find_reg_in_directory(reg_file):
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Move the test REG file to the temporary directory
#         reg_path = os.path.join(temp_dir, "test_reg.dcm")
#         os.rename(reg_file, reg_path)

#         # Check if the REGReader finds the REG in the directory
#         file_path = REGReader._find_reg_in_directory(temp_dir)
#         assert file_path == reg_path


# def test_invalid_input_type():
#     with pytest.raises(ValueError, match="reg_input must be either a file path"):
#         REGReader(123)  # Invalid input type


# def test_read_invalid_reg_file():
#     # Create a temporary file with no valid DICOM content
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
#         temp_file.write(b"Invalid data")

#     reader = REGReader(temp_file.name)
#     with pytest.raises(InvalidDicomError):
#         reader.read()

#     os.remove(temp_file.name)
