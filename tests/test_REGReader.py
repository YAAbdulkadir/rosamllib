import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rosamllib.readers.reg_reader import REGReader
from rosamllib.dicoms import REG


def _make_minimal_rigid_reg_dataset() -> Dataset:
    """
    Create a minimal synthetic REG dataset sufficient for REG.from_dataset
    and REGReader.read() to succeed.

    This intentionally does NOT include matrix sequences; it only sets:
    - Modality = "REG"
    - FrameOfReferenceUID on the root
    - RegistrationSequence with two items
      * each has FrameOfReferenceUID and ReferencedImageSequence
    """
    ds = Dataset()
    ds.Modality = "REG"
    ds.FrameOfReferenceUID = "1.2.32.34.39"

    reg_seq = Sequence()

    # Fixed item (FoR matches root FrameOfReferenceUID)
    fixed_item = Dataset()
    fixed_item.FrameOfReferenceUID = "1.2.32.34.39"
    fixed_ref_seq = Sequence()
    fixed_ref_item = Dataset()
    fixed_ref_item.ReferencedSOPInstanceUID = "1.2.32.34.3"
    fixed_ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    fixed_ref_seq.append(fixed_ref_item)
    fixed_item.ReferencedImageSequence = fixed_ref_seq
    reg_seq.append(fixed_item)

    # Moving item (FoR is different)
    moving_item = Dataset()
    moving_item.FrameOfReferenceUID = "1.2.32.24.43"
    moving_ref_seq = Sequence()
    moving_ref_item = Dataset()
    moving_ref_item.ReferencedSOPInstanceUID = "1.2.3243.4343"
    moving_ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    moving_ref_seq.append(moving_ref_item)
    moving_item.ReferencedImageSequence = moving_ref_seq
    reg_seq.append(moving_item)

    ds.RegistrationSequence = reg_seq

    return ds


def _attach_basic_file_meta(ds: Dataset) -> Dataset:
    """
    Attach minimal file meta and transfer syntax so ds.save_as() produces
    a proper DICOM file with File Meta Information and DICM preamble.
    """
    file_meta = Dataset()
    # Spatial Registration Storage SOP Class UID (good enough for tests)
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.file_meta = file_meta
    return ds


def test_regreader_from_dataset():
    """
    REGReader should accept an in-memory Dataset and return a REG instance.
    """
    base_ds = _make_minimal_rigid_reg_dataset()
    reader = REGReader(base_ds)

    reg = reader.read()

    assert isinstance(reg, REG)
    assert reg.registration_type == "rigid"
    # Fixed/moving refs should be set
    assert reg.get_fixed_image_info()["referenced_images"] == ["1.2.32.34.3"]
    assert reg.get_moving_image_info()["referenced_images"] == ["1.2.3243.4343"]


def test_regreader_from_file_path(tmp_path):
    """
    REGReader should read a DICOM REG file from a file path.
    """
    base_ds = _make_minimal_rigid_reg_dataset()
    base_ds = _attach_basic_file_meta(base_ds)

    dcm_path = tmp_path / "reg.dcm"
    # Important: ensure a proper DICOM file is written
    base_ds.save_as(dcm_path, enforce_file_format=True)

    reader = REGReader(dcm_path)
    reg = reader.read()

    assert isinstance(reg, REG)
    assert reg.registration_type == "rigid"
    assert reg.get_fixed_image_info()["referenced_images"] == ["1.2.32.34.3"]


def test_regreader_from_directory(tmp_path):
    """
    REGReader should find a REG file when given a directory path.
    It should use _find_reg_in_directory under the hood.
    """
    base_ds = _make_minimal_rigid_reg_dataset()
    base_ds = _attach_basic_file_meta(base_ds)

    # Create a nested directory structure with some junk + one REG.dcm
    subdir = tmp_path / "nested"
    subdir.mkdir()

    junk_file = subdir / "not_dicom.txt"
    junk_file.write_text("this is not a dicom")

    reg_path = subdir / "my_reg_file.dcm"
    base_ds.save_as(reg_path, enforce_file_format=True)

    reader = REGReader(tmp_path)
    reg = reader.read()

    assert isinstance(reg, REG)
    assert reg.registration_type == "rigid"
    assert reg.get_moving_image_info()["referenced_images"] == ["1.2.3243.4343"]


def test_regreader_directory_no_reg_raises(tmp_path):
    """
    If a directory contains no REG files, REGReader.read() should raise IOError.
    """
    # Create some non-REG garbage files
    (tmp_path / "foo.txt").write_text("hello")

    ds = Dataset()
    ds.Modality = "CT"
    ds = _attach_basic_file_meta(ds)
    ct_path = tmp_path / "ct.dcm"
    ds.save_as(ct_path, enforce_file_format=True)

    reader = REGReader(tmp_path)

    with pytest.raises(IOError):
        reader.read()


def test_regreader_invalid_input_type_raises():
    """
    REGReader.__init__ should reject unsupported input types.
    """
    with pytest.raises(ValueError):
        REGReader(123)  # not str/Path/Dataset
