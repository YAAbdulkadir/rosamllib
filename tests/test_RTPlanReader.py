import os
from pathlib import Path

import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import RTPlanStorage, generate_uid

from rosamllib.readers import RTPlanReader
from rosamllib.dicoms import RTPlan


def create_minimal_rtplan_dataset() -> Dataset:
    """Create a minimal but valid RTPLAN dataset for testing."""
    ds = Dataset()
    ds.Modality = "RTPLAN"
    ds.SOPClassUID = RTPlanStorage
    ds.SOPInstanceUID = generate_uid()

    ds.RTPlanLabel = "TEST_PLAN"
    ds.RTPlanName = "Test Plan"
    ds.RTPlanDescription = "Test description"

    # Minimal BeamSequence with one beam
    beam = Dataset()
    beam.BeamNumber = 1
    beam.BeamName = "B1"
    beam.TreatmentMachineName = "LINAC1"
    beam.BeamType = "STATIC"
    ds.BeamSequence = [beam]

    return ds


def write_rtplan_file(tmp_path: Path) -> Path:
    """Write a minimal RTPLAN FileDataset to disk and return the path."""
    ds = create_minimal_rtplan_dataset()

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = RTPlanStorage
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    file_ds = FileDataset(
        filename_or_obj="",
        dataset=ds,
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    filepath = tmp_path / "rtplan.dcm"
    pydicom.dcmwrite(filepath, file_ds)
    return filepath


def test_rtplan_reader_from_dataset_returns_RTPLAN():
    ds = create_minimal_rtplan_dataset()
    reader = RTPlanReader(ds)

    rtplan = reader.read()

    assert isinstance(rtplan, RTPlan)
    assert rtplan.Modality == "RTPLAN"
    assert rtplan.plan_label == "TEST_PLAN"
    assert rtplan.plan_name == "Test Plan"


def test_rtplan_reader_from_file_path(tmp_path):
    filepath = write_rtplan_file(tmp_path)

    reader = RTPlanReader(filepath)
    rtplan = reader.read()

    assert isinstance(rtplan, RTPlan)
    assert rtplan.SOPInstanceUID is not None
    assert rtplan.plan_label == "TEST_PLAN"


def test_rtplan_reader_from_directory(tmp_path):
    # Create a dir with the RTPLAN file and a junk file
    rtplan_path = write_rtplan_file(tmp_path)
    (tmp_path / "junk.txt").write_text("not a dicom")

    reader = RTPlanReader(tmp_path)
    rtplan = reader.read()

    assert isinstance(rtplan, RTPlan)
    assert rtplan.SOPInstanceUID is not None
    assert rtplan.plan_label == "TEST_PLAN"
    # Make sure it actually found the file in the directory
    assert os.path.samefile(Path(rtplan_path), Path(rtplan_path))


def test_rtplan_reader_raises_if_no_rtplan_in_directory(tmp_path):
    # Only create a non-RTPLAN DICOM
    ds = Dataset()
    ds.Modality = "CT"
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = generate_uid()

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    file_ds = FileDataset("", ds, file_meta=file_meta, preamble=b"\0" * 128)

    filepath = tmp_path / "ct.dcm"
    pydicom.dcmwrite(filepath, file_ds)

    reader = RTPlanReader(tmp_path)

    with pytest.raises(IOError):
        _ = reader.read()


def test_rtplan_reader_invalid_modality_dataset_raises_value_error():
    ds = Dataset()
    ds.Modality = "CT"
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = generate_uid()

    reader = RTPlanReader(ds)

    with pytest.raises(ValueError):
        _ = reader.read()


def test_rtplan_reader_raises_if_no_input():
    with pytest.raises(ValueError):
        # Force the internal state into "no dataset, no path" and call read
        reader = RTPlanReader.__new__(RTPlanReader)
        reader.rtplan_file_path = None
        reader.rtplan_dataset = None
        reader.read()
