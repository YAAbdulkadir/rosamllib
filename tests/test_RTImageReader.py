# tests/test_RTImageReader.py

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from rosamllib.dicoms import RTIMAGE
from rosamllib.readers import RTImageReader


def _make_minimal_rtimage_dataset(rows=8, cols=8):
    """
    Small helper to build a synthetic RTIMAGE dataset with valid pixel data
    and file_meta so that it can be written to disk.
    """
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.1"  # RT Image Storage
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.Modality = "RTIMAGE"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9.10"

    ds.file_meta = file_meta

    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [1.0, 1.0]
    ds.RTImageSID = 1500.0
    ds.RadiationMachineSAD = 1000.0

    ds.RTImageLabel = "TEST_RTIMAGE"

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    arr = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    ds.PixelData = arr.tobytes()

    return ds


def test_rtimage_reader_from_file_path(tmp_path):
    """
    RTImageReader should read a DICOM RTIMAGE file from a file path and
    return an RTIMAGE instance.
    """
    base_ds = _make_minimal_rtimage_dataset()
    dcm_path = tmp_path / "rtimage.dcm"
    base_ds.save_as(dcm_path, enforce_file_format=True)

    reader = RTImageReader(dcm_path)
    rtimg = reader.read()

    assert isinstance(rtimg, RTIMAGE)
    assert rtimg.Modality == "RTIMAGE"
    geom = rtimg.get_geometry()
    assert geom["size"] == (base_ds.Columns, base_ds.Rows)


def test_rtimage_reader_from_directory(tmp_path):
    """
    RTImageReader should find an RTIMAGE file when given a directory path.
    It should use _find_rtimage_in_directory under the hood.
    """
    base_ds = _make_minimal_rtimage_dataset()

    # nested structure with junk + one RTIMAGE
    subdir = tmp_path / "nested"
    subdir.mkdir()

    junk_file = subdir / "not_dicom.txt"
    junk_file.write_text("not a dicom")

    rtimage_path = subdir / "my_rtimage.dcm"
    base_ds.save_as(rtimage_path, enforce_file_format=True)

    reader = RTImageReader(tmp_path)
    rtimg = reader.read()

    assert isinstance(rtimg, RTIMAGE)
    assert rtimg.Modality == "RTIMAGE"
    assert rtimg.RTImageLabel == base_ds.RTImageLabel


def test_rtimage_reader_directory_no_rtimage_raises(tmp_path):
    """
    RTImageReader should raise an IOError when no RTIMAGE is found in a directory.
    """
    subdir = tmp_path / "empty"
    subdir.mkdir()

    # Add some non-RTIMAGE DICOM or non-DICOM
    junk = subdir / "junk.txt"
    junk.write_text("hello")

    reader = RTImageReader(tmp_path)
    try:
        _ = reader.read()
        raised = False
    except IOError:
        raised = True

    assert raised, "Expected IOError when no RTIMAGE file present in directory"


def test_rtimage_reader_from_dataset():
    """
    RTImageReader should accept a pre-loaded Dataset and wrap it in RTIMAGE.
    """
    base_ds = _make_minimal_rtimage_dataset()
    reader = RTImageReader(base_ds)
    rtimg = reader.read()

    assert isinstance(rtimg, RTIMAGE)
    assert rtimg.Modality == "RTIMAGE"
    assert rtimg.Rows == base_ds.Rows
    assert rtimg.Columns == base_ds.Columns


def test_rtimage_reader_invalid_input_raises():
    """
    RTImageReader should raise ValueError for unsupported input types.
    """
    import pytest

    with pytest.raises(ValueError):
        RTImageReader(12345)  # neither str/Path nor Dataset
