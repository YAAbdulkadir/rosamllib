import pytest
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from rosamllib.readers import RTRecordReader
from rosamllib.dicoms import RTRecord


def _make_minimal_rtrecord_dataset() -> Dataset:
    """
    Create a minimal synthetic RTRECORD dataset.
    """
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.4"  # RT Beams Treatment Record Storage
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.Modality = "RTRECORD"
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.StudyInstanceUID = "9.8.7.6.5.4.3.2.1"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8"
    ds.TreatmentDate = "20250101"
    return ds


def _attach_basic_file_meta(ds: Dataset) -> Dataset:
    """
    Attach minimal file meta & transfer syntax so save_as() writes a valid DICOM.
    """
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.file_meta = file_meta
    return ds


def test_rtrecordreader_from_dataset():
    """
    RTRecordReader should accept an in-memory Dataset and return an RTRecord.
    """
    base_ds = _make_minimal_rtrecord_dataset()
    reader = RTRecordReader(base_ds)

    record = reader.read()

    assert isinstance(record, RTRecord)
    assert isinstance(record, Dataset)
    assert record.Modality == "RTRECORD"
    assert record.PatientID == "12345"


def test_rtrecordreader_from_file_path(tmp_path):
    """
    RTRecordReader should read an RTRECORD file from a file path.
    """
    base_ds = _make_minimal_rtrecord_dataset()
    base_ds = _attach_basic_file_meta(base_ds)

    dcm_path = tmp_path / "rtrecord.dcm"
    base_ds.save_as(dcm_path, enforce_file_format=True)

    reader = RTRecordReader(dcm_path)
    record = reader.read()

    assert isinstance(record, RTRecord)
    assert record.Modality == "RTRECORD"
    assert record.PatientName.family_name == "Test"
    assert record.PatientID == "12345"


def test_rtrecordreader_from_directory(tmp_path):
    """
    RTRecordReader should find an RTRECORD file when given a directory path.
    """
    base_ds = _make_minimal_rtrecord_dataset()
    base_ds = _attach_basic_file_meta(base_ds)

    subdir = tmp_path / "nested"
    subdir.mkdir()

    # Some junk file that is not DICOM
    (subdir / "not_dicom.txt").write_text("not a dicom")

    rtrecord_path = subdir / "my_rtrecord.dcm"
    base_ds.save_as(rtrecord_path, enforce_file_format=True)

    reader = RTRecordReader(tmp_path)
    record = reader.read()

    assert isinstance(record, RTRecord)
    assert record.Modality == "RTRECORD"
    assert record.PatientID == "12345"


def test_rtrecordreader_directory_no_rtrecord_raises(tmp_path):
    """
    If a directory contains no RTRECORD files, read() should raise IOError.
    """
    # Non-RTRECORD DICOM
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds = _attach_basic_file_meta(ds)
    ct_path = tmp_path / "ct.dcm"
    ds.save_as(ct_path, enforce_file_format=True)

    # And a non-DICOM file
    (tmp_path / "foo.txt").write_text("hello")

    reader = RTRecordReader(tmp_path)

    with pytest.raises(IOError):
        reader.read()


def test_rtrecordreader_invalid_input_type_raises():
    """
    RTRecordReader.__init__ should reject unsupported input types.
    """
    with pytest.raises(ValueError):
        RTRecordReader(123)  # not str/Path/Dataset
