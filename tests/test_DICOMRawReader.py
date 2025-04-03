# import os
# import pytest
# import pydicom
# from pydicom.dataset import Dataset
# from tempfile import NamedTemporaryFile
# from rosamllib.readers import DICOMRawReader
# from pydicom.dataset import FileMetaDataset
# from pydicom.uid import ImplicitVRLittleEndian


# @pytest.fixture
# def dicom_raw_file():
#     # Create a temporary DICOM RAW file for testing
#     temp_file = NamedTemporaryFile(delete=False, suffix=".dcm")
#     ds = Dataset()
#     ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
#     ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.10"
#     ds.Modality = "RAW"

#     # Simulate an embedded dataset in the MIMSoftwareSessionMetaSeq (0013, 2050) tag
#     embedded_ds = Dataset()
#     embedded_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
#     embedded_ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.11"
#     ds.MIMSoftwareSessionMetaSeq = [embedded_ds]

#     file_meta = FileMetaDataset()
#     file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
#     file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
#     file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
#     ds.file_meta = file_meta

#     temp_file.close()

#     # Save the dataset to a temporary file
#     ds.save_as(temp_file.name, write_like_original=False)
#     yield temp_file.name
#     os.remove(temp_file.name)


# def test_read_dicom_raw_file(dicom_raw_file):
#     # Test reading a DICOM RAW file
#     reader = DICOMRawReader(dicom_raw_file)
#     reader.read()

#     assert reader.dataset is not None
#     assert reader.dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.1"
#     assert len(reader.get_embedded_datasets()) == 1
#     assert reader.get_embedded_datasets()[0].SOPInstanceUID == "1.2.3.4.5.6.7.8.9.11"


# def test_read_dicom_raw_from_dataset(dicom_raw_file):
#     # Test reading a pre-loaded pydicom.Dataset
#     dataset = pydicom.dcmread(dicom_raw_file)
#     reader = DICOMRawReader(dataset)
#     reader.read()

#     assert reader.dataset is not None
#     assert reader.dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.1"
#     assert len(reader.get_embedded_datasets()) == 1
#     assert reader.get_embedded_datasets()[0].SOPInstanceUID == "1.2.3.4.5.6.7.8.9.11"


# def test_extract_embedded_datasets(dicom_raw_file):
#     # Test extraction of embedded datasets
#     reader = DICOMRawReader(dicom_raw_file)
#     reader.read()

#     embedded_datasets = reader.get_embedded_datasets()
#     assert len(embedded_datasets) == 1
#     assert embedded_datasets[0].SOPInstanceUID == "1.2.3.4.5.6.7.8.9.11"


# def test_no_mimsoftware_tag():
#     # Test error when MIMSoftwareSessionMetaSeq tag is missing
#     ds = Dataset()
#     ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"

#     reader = DICOMRawReader(ds)
#     with pytest.raises(ValueError, match="MIMSoftwareSessionMetaSeq.*not found"):
#         reader.extract_embedded_datasets()


# def test_referenced_series_uid_extraction(dicom_raw_file):
#     # Test extraction of ReferencedSeriesUID from ReferencedSeriesSequence
#     dataset = pydicom.dcmread(dicom_raw_file)

#     # Add a ReferencedSeriesSequence to the dataset
#     ref_series = Dataset()
#     ref_series.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.20"
#     dataset.ReferencedSeriesSequence = [ref_series]

#     reader = DICOMRawReader(dataset)
#     reader.read()
#     assert reader.referenced_series_uid == "1.2.3.4.5.6.7.8.9.20"


# def test_invalid_input_type():
#     # Test invalid input type
#     with pytest.raises(ValueError, match="raw_input must be either a file path"):
#         DICOMRawReader(123)  # Invalid input type


# def test_read_invalid_file():
#     # Test reading an invalid file
#     with NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
#         temp_file.write(b"Invalid data")

#     reader = DICOMRawReader(temp_file.name)
#     with pytest.raises(IOError):
#         reader.read()

#     os.remove(temp_file.name)
