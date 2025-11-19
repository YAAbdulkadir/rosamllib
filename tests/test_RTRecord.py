from pydicom.dataset import Dataset
from rosamllib.dicoms import RTRecord  # adjust import if in a different module


def _make_minimal_rtrecord_dataset() -> Dataset:
    """
    Create a minimal synthetic RTRECORD DICOM dataset.

    We don't need a fully standard-compliant object here, just enough
    tags to exercise RTRecord.from_dataset and attribute access.
    """
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.4"  # RT Beams Treatment Record Storage
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.Modality = "RTRECORD"
    ds.StudyInstanceUID = "9.8.7.6.5.4.3.2.1"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8"
    ds.TreatmentDate = "20250101"
    return ds


def test_rtrecord_from_dataset_basic():
    """
    RTRecord.from_dataset should return an RTRecord instance with the same
    DICOM attributes as the source Dataset.
    """
    base_ds = _make_minimal_rtrecord_dataset()

    record = RTRecord.from_dataset(base_ds)

    assert isinstance(record, RTRecord)
    # Also still a pydicom Dataset
    assert isinstance(record, Dataset)

    # Check a few key attributes are copied correctly
    assert record.PatientID == base_ds.PatientID
    assert record.PatientName == base_ds.PatientName
    assert record.Modality == "RTRECORD"
    assert record.SOPInstanceUID == base_ds.SOPInstanceUID
    assert record.StudyInstanceUID == base_ds.StudyInstanceUID
    assert record.SeriesInstanceUID == base_ds.SeriesInstanceUID


def test_rtrecord_dir_contains_dicom_keywords():
    """
    RTRecord should expose DICOM keywords via dir(), just like Dataset.
    """
    base_ds = _make_minimal_rtrecord_dataset()
    record = RTRecord.from_dataset(base_ds)

    attrs = dir(record)

    # A few expected DICOM keywords should appear
    assert "PatientID" in attrs
    assert "PatientName" in attrs
    assert "StudyInstanceUID" in attrs
    assert "SeriesInstanceUID" in attrs


def test_rtrecord_attribute_access_and_assignment():
    """
    RTRecord should behave like a normal pydicom Dataset for attribute access
    and assignment.
    """
    base_ds = _make_minimal_rtrecord_dataset()
    record = RTRecord.from_dataset(base_ds)

    # Read existing attribute
    assert record.PatientID == "12345"

    # Modify an attribute on the RTRecord
    record.PatientID = "99999"
    assert record.PatientID == "99999"

    # Ensure we're not accidentally breaking dot access for new attributes
    record.CustomComment = "test-comment"
    assert record.CustomComment == "test-comment"


def test_rtrecord_from_dataset_accepts_plain_dataset():
    """
    RTRecord.from_dataset should accept any pydicom.Dataset that at least
    looks like an RTRECORD (we don't enforce Modality in the class).
    """
    ds = Dataset()
    ds.PatientID = "ABC"
    ds.Modality = "RTRECORD"

    record = RTRecord.from_dataset(ds)

    assert isinstance(record, RTRecord)
    assert record.PatientID == "ABC"
    assert record.Modality == "RTRECORD"
