import struct
import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rosamllib.dicoms import REG


def _make_rigid_reg_dataset() -> Dataset:
    """Create a minimal synthetic rigid REG DICOM dataset."""
    ds = Dataset()
    ds.Modality = "REG"
    ds.FrameOfReferenceUID = "1.2.89.43"

    # --- RegistrationSequence with 2 items: fixed + moving ---
    reg_seq = Sequence()

    # Fixed item (FoR matches ds.FrameOfReferenceUID)
    fixed_item = Dataset()
    fixed_item.FrameOfReferenceUID = "1.2.89.43"

    # MatrixRegistrationSequence for fixed
    fixed_matrix_seq = Sequence()
    fixed_matrix_item = Dataset()
    # Simple identity matrix (flattened)
    fixed_matrix_item.FrameOfReferenceTransformationMatrix = [1.0] * 16
    fixed_matrix_item.FrameOfReferenceTransformationMatrixType = "RIGID"
    fixed_matrix_item.MatrixSequence = Sequence([fixed_matrix_item])
    fixed_matrix_seq.append(fixed_matrix_item)
    fixed_item.MatrixRegistrationSequence = fixed_matrix_seq

    # ReferencedImageSequence for fixed
    fixed_ref_seq = Sequence()
    fixed_ref_item = Dataset()
    fixed_ref_item.ReferencedSOPInstanceUID = "1.2.89.43"
    fixed_ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
    fixed_ref_seq.append(fixed_ref_item)
    fixed_item.ReferencedImageSequence = fixed_ref_seq

    reg_seq.append(fixed_item)

    # Moving item (different FoR)
    moving_item = Dataset()
    moving_item.FrameOfReferenceUID = "1.2.7.87.7343"

    # MatrixRegistrationSequence for moving: simple translation in X (just to be different)
    moving_matrix_seq = Sequence()
    moving_matrix_item = Dataset()
    moving_matrix = np.eye(4)
    moving_matrix[0, 3] = 5.0  # translate +5 in x
    moving_matrix_item.FrameOfReferenceTransformationMatrix = moving_matrix.flatten().tolist()
    moving_matrix_item.FrameOfReferenceTransformationMatrixType = "RIGID"
    moving_matrix_item.MatrixSequence = Sequence([moving_matrix_item])
    moving_matrix_seq.append(moving_matrix_item)
    moving_item.MatrixRegistrationSequence = moving_matrix_seq

    # ReferencedImageSequence for moving
    moving_ref_seq = Sequence()
    moving_ref_item = Dataset()
    moving_ref_item.ReferencedSOPInstanceUID = "1.2.3.78"
    moving_ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    moving_ref_seq.append(moving_ref_item)
    moving_item.ReferencedImageSequence = moving_ref_seq

    reg_seq.append(moving_item)

    ds.RegistrationSequence = reg_seq

    # --- ReferencedSeriesSequence to match fixed/moving ---
    ref_series_seq = Sequence()

    fixed_series = Dataset()
    fixed_series.SeriesInstanceUID = "1.2.3.8.6.10"
    fixed_series.ReferencedInstanceSequence = Sequence()
    fixed_series_inst = Dataset()
    fixed_series_inst.ReferencedSOPInstanceUID = "1.2.89.43"
    fixed_series.ReferencedInstanceSequence.append(fixed_series_inst)

    moving_series = Dataset()
    moving_series.SeriesInstanceUID = "1.2.32.34.39"
    moving_series.ReferencedInstanceSequence = Sequence()
    moving_series_inst = Dataset()
    moving_series_inst.ReferencedSOPInstanceUID = "1.2.3.78"
    moving_series.ReferencedInstanceSequence.append(moving_series_inst)

    ref_series_seq.append(fixed_series)
    ref_series_seq.append(moving_series)

    ds.ReferencedSeriesSequence = ref_series_seq

    return ds


def _make_deformable_reg_dataset() -> Dataset:
    """Create a minimal synthetic deformable REG DICOM dataset."""
    ds = Dataset()
    ds.Modality = "REG"
    ds.FrameOfReferenceUID = "1.2.89.43"

    deform_seq = Sequence()

    # Common grid parameters
    grid_dims = [2, 2, 2]  # very small grid
    num_voxels = np.prod(grid_dims)
    num_components = 3  # x, y, z
    total_elements = int(num_voxels * num_components)

    # Simple vector field: all zeros
    grid_values = [0.0] * total_elements
    grid_bytes = struct.pack(f"{total_elements}f", *grid_values)

    for i, (ref_uid, series_uid) in enumerate(
        [("1.2.3.6.8.9", "1.2.3.8.67.73"), ("1.2.3.879", "1.2.32.34.35")]
    ):
        item = Dataset()

        # ReferencedImageSequence
        ref_seq = Sequence()
        ref_item = Dataset()
        ref_item.ReferencedSOPInstanceUID = ref_uid
        ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ref_seq.append(ref_item)
        item.ReferencedImageSequence = ref_seq

        item.SourceFrameOfReferenceUID = "1.2.89.43" if i == 0 else "1.2.7.87.7343"

        # DeformableRegistrationGridSequence
        grid_ds = Dataset()
        grid_ds.VectorGridData = grid_bytes
        grid_ds.GridDimensions = grid_dims
        grid_ds.GridResolution = [1.0, 1.0, 1.0]
        grid_ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        grid_ds.ImagePositionPatient = [0.0, 0.0, 0.0]

        item.DeformableRegistrationGridSequence = Sequence([grid_ds])

        deform_seq.append(item)

    ds.DeformableRegistrationSequence = deform_seq

    # Optional: add a simple other references structure
    # to exercise check_other_references
    ds.StudiesContainingOtherReferencedInstancesSequence = Sequence()
    study = Dataset()
    study.StudyInstanceUID = "1.2.3.6.8.10"
    study.ReferencedSeriesSequence = Sequence()
    for series_uid, ref_uid in [
        ("1.2.3.8.67.73", "1.2.3.6.8.9"),
        ("1.2.32.34.35", "1.2.3.879"),
    ]:
        series_item = Dataset()
        series_item.SeriesInstanceUID = series_uid
        series_item.ReferencedInstanceSequence = Sequence()
        inst = Dataset()
        inst.ReferencedSOPInstanceUID = ref_uid
        series_item.ReferencedInstanceSequence.append(inst)
        study.ReferencedSeriesSequence.append(series_item)
    ds.StudiesContainingOtherReferencedInstancesSequence.append(study)

    return ds


def test_reg_from_dataset_rigid():
    ds = _make_rigid_reg_dataset()
    reg = REG.from_dataset(ds)

    # Basic type and registration type
    assert isinstance(reg, REG)
    assert reg.registration_type == "rigid"

    # Fixed/moving image info
    fixed_info = reg.get_fixed_image_info()
    moving_info = reg.get_moving_image_info()

    # Fixed should match FrameOfReferenceUID on root dataset
    assert fixed_info["SourceFrameOfReferenceUID"] == ds.FrameOfReferenceUID
    assert "transformation_matrix" in fixed_info
    assert fixed_info["transformation_matrix"].shape == (4, 4)

    assert "transformation_matrix" in moving_info
    assert moving_info["transformation_matrix"].shape == (4, 4)

    # Referenced images
    assert fixed_info["referenced_images"] == ["1.2.89.43"]
    assert moving_info["referenced_images"] == ["1.2.3.78"]

    # SeriesInstanceUIDs from ReferencedSeriesSequence
    assert fixed_info["SeriesInstanceUID"] == "1.2.3.8.6.10"
    assert moving_info["SeriesInstanceUID"] == "1.2.32.34.39"

    # __repr__ sanity check
    rep = repr(reg)
    assert "registration_type='rigid'" in rep
    assert "1.2.3.8.6.10" in rep
    assert "1.2.32.34.39" in rep


def test_reg_from_dataset_deformable():
    ds = _make_deformable_reg_dataset()
    reg = REG.from_dataset(ds)

    assert isinstance(reg, REG)
    assert reg.registration_type == "deformable"

    fixed_info = reg.get_fixed_image_info()
    moving_info = reg.get_moving_image_info()

    # Grid info should be present
    assert "grid_data" in fixed_info
    assert "grid_data" in moving_info

    grid_fixed = fixed_info["grid_data"]
    grid_moving = moving_info["grid_data"]

    # Shape should be (dimX, dimY, dimZ, 3)
    assert grid_fixed.shape == (2, 2, 2, 3)
    assert grid_moving.shape == (2, 2, 2, 3)

    # All zeros as constructed
    assert np.allclose(grid_fixed, 0.0)
    assert np.allclose(grid_moving, 0.0)

    # Study/series mapping from check_other_references
    assert fixed_info["SeriesInstanceUID"] == "1.2.3.8.67.73"
    assert moving_info["SeriesInstanceUID"] == "1.2.32.34.35"
    assert fixed_info["StudyInstanceUID"] == "1.2.3.6.8.10"
    assert moving_info["StudyInstanceUID"] == "1.2.3.6.8.10"

    rep = repr(reg)
    assert "registration_type='deformable'" in rep


def test_get_fixed_moving_image_info_uninitialized():
    # Create an empty REG (no dataset, not from from_dataset)
    reg = REG()

    with pytest.raises(ValueError):
        reg.get_fixed_image_info()

    with pytest.raises(ValueError):
        reg.get_moving_image_info()
