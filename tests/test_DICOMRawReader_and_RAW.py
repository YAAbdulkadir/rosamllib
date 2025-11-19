import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rosamllib.dicoms.raw import RAW
from rosamllib.readers.dicom_raw_reader import DICOMRawReader  # adjust module name/path


def _make_base_raw_dataset(with_embedded: bool = True) -> Dataset:
    """
    Create a synthetic RAW DICOM dataset, optionally with embedded datasets
    in the private tag (0013,2050).
    """
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage (good enough)
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "OT"  # or something generic
    ds.PatientID = "RAWTEST"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    if with_embedded:
        # Build two tiny embedded datasets
        embedded1 = Dataset()
        embedded1.SOPClassUID = "1.2.3.4.5.6.7.1"
        embedded1.SOPInstanceUID = "1.2.43.798798"
        embedded1.Modality = "REG"

        embedded2 = Dataset()
        embedded2.SOPClassUID = "1.2.3.4.5.6.7.2"
        embedded2.SOPInstanceUID = "1.2.43.552"
        embedded2.Modality = "RTSTRUCT"

        mim_seq = Sequence([embedded1, embedded2])
        # Private tag (0013,2050) VR = SQ
        ds.add_new(0x00132050, "SQ", mim_seq)

    # Add a simple ReferencedSeriesSequence so RAW can pick up SeriesInstanceUID
    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = "1.2.3.43"
    ref_series_item.ReferencedInstanceSequence = Sequence()
    ds.ReferencedSeriesSequence = Sequence([ref_series_item])

    return ds


def _attach_basic_file_meta(ds: Dataset) -> Dataset:
    """
    Attach minimal file meta so save_as(enforce_file_format=True) writes a valid DICOM.
    """
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta
    return ds


# ---------- RAW tests ----------


def test_raw_from_dataset_extracts_embedded_and_series_uid():
    base_ds = _make_base_raw_dataset(with_embedded=True)

    raw = RAW.from_dataset(base_ds)

    assert isinstance(raw, RAW)
    assert len(raw.embedded_datasets) == 2

    embedded = raw.get_embedded_datasets()
    assert embedded[0].SOPInstanceUID == "1.2.43.798798"
    assert embedded[1].SOPInstanceUID == "1.2.43.552"

    # File meta should be attached to each embedded dataset
    for ed in embedded:
        assert hasattr(ed, "file_meta")
        assert ed.file_meta is not None
        assert ed.file_meta.TransferSyntaxUID is not None

    # Referenced series UID should be populated
    assert raw.referenced_series_uid == "1.2.3.43"

    # __repr__ sanity
    rep = repr(raw)
    assert "num_embedded=2" in rep
    assert "1.2.3.43" in rep


def test_raw_extract_embedded_datasets_missing_tag_raises():
    base_ds = _make_base_raw_dataset(with_embedded=False)
    raw = RAW.from_dataset  # we don't want from_dataset to auto-raise here

    # Directly constructing RAW and calling extract_embedded_datasets()
    raw = RAW()
    raw.update(base_ds)

    with pytest.raises(ValueError):
        raw.extract_embedded_datasets()


def test_raw_get_embedded_datasets_without_extract_raises():
    base_ds = _make_base_raw_dataset(with_embedded=True)
    raw = RAW()
    raw.update(base_ds)

    # We never called extract_embedded_datasets()
    with pytest.raises(ValueError):
        raw.get_embedded_datasets()


# ---------- DICOMRawReader tests ----------


def test_dicomrawreader_from_dataset():
    base_ds = _make_base_raw_dataset(with_embedded=True)

    reader = DICOMRawReader(base_ds)
    raw = reader.read()

    assert isinstance(raw, RAW)
    assert len(raw.embedded_datasets) == 2
    assert raw.referenced_series_uid == "1.2.3.43"


def test_dicomrawreader_from_file_path(tmp_path):
    base_ds = _make_base_raw_dataset(with_embedded=True)
    base_ds = _attach_basic_file_meta(base_ds)

    dcm_path = tmp_path / "raw.dcm"
    base_ds.save_as(dcm_path, enforce_file_format=True)

    reader = DICOMRawReader(dcm_path)
    raw = reader.read()

    assert isinstance(raw, RAW)
    assert len(raw.embedded_datasets) == 2
    assert raw.embedded_datasets[0].Modality == "REG"
    assert raw.embedded_datasets[1].Modality == "RTSTRUCT"


def test_dicomrawreader_invalid_input_type_raises():
    with pytest.raises(ValueError):
        DICOMRawReader(123)  # not str/Path/Dataset
