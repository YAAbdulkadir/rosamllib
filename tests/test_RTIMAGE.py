import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from rosamllib.dicoms import RTIMAGE


def _make_minimal_rtimage_dataset(
    rows=8,
    cols=8,
    pixel_spacing=(1.0, 1.0),
    imager_spacing=None,
    imageplane_spacing=None,
    sid=1500.0,
    sad=1000.0,
    add_window=False,
):
    """
    Create a small synthetic RTIMAGE-like Dataset suitable for RTIMAGE.from_dataset tests.
    """

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.1"  # RT Image Storage
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.Modality = "RTIMAGE"

    ds.Rows = rows
    ds.Columns = cols

    # Pixel spacing tags (priority: ImagePlanePixelSpacing -> ImagerPixelSpacing -> PixelSpacing)
    if imageplane_spacing is not None:
        ds.ImagePlanePixelSpacing = list(imageplane_spacing)
    if imager_spacing is not None:
        ds.ImagerPixelSpacing = list(imager_spacing)
    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)

    ds.RTImageSID = float(sid)
    ds.RadiationMachineSAD = float(sad)

    ds.RTImageLabel = "TEST_RTIMAGE"
    ds.RTImageDescription = "Synthetic RTIMAGE for unit tests"
    ds.RTTreatmentMachineName = "TESTLINAC"

    # Angles
    ds.GantryAngle = 10.0
    ds.BeamLimitingDeviceAngle = 5.0
    ds.PatientSupportAngle = 0.0

    # Optional window/level tags
    if add_window:
        ds.WindowCenter = 100.0
        ds.WindowWidth = 200.0

    # Minimal pixel data: 16-bit unsigned, ramp
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    # File meta for when the dataset is written to disk (reader tests can reuse this helper)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9.10"

    ds.file_meta = file_meta

    arr = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    ds.PixelData = arr.tobytes()

    return ds


def test_rtimage_from_dataset_basic_geometry_and_repr():
    """
    RTIMAGE.from_dataset should populate _meta and __repr__ should be informative
    without raising, even when some tags are synthetic.
    """
    base_ds = _make_minimal_rtimage_dataset()
    rti = RTIMAGE.from_dataset(base_ds)

    # Basic type check
    assert isinstance(rti, RTIMAGE)
    assert rti.Modality == "RTIMAGE"

    # Geometry metadata
    geom = rti.get_geometry()
    assert geom["size"] == (base_ds.Columns, base_ds.Rows)
    # spacing is stored as (dx, dy) = (col_spacing, row_spacing)
    assert geom["spacing"] == (1.0, 1.0)
    assert geom["rtimage"]["SID"] == 1500.0
    assert geom["rtimage"]["SAD"] == 1000.0

    # Angles present and as floats
    assert geom["angles"]["Gantry"] == 10.0
    assert geom["angles"]["Collimator"] == 5.0
    assert geom["angles"]["Table"] == 0.0

    # __repr__ should be a non-empty string and contain some key fields
    rep = repr(rti)
    assert "RTIMAGE(" in rep
    assert "size=" in rep
    assert "SID=" in rep
    assert "SAD=" in rep


def test_rtimage_spacing_priority_imageplane_over_imager_over_pixelspacing():
    """
    Spacing priority should be:
      ImagePlanePixelSpacing -> ImagerPixelSpacing -> PixelSpacing
    """
    # Set all three with distinct values to test precedence
    base_ds = _make_minimal_rtimage_dataset(
        pixel_spacing=(1.0, 1.0),
        imager_spacing=(0.8, 0.9),
        imageplane_spacing=(0.5, 0.6),
    )
    rti = RTIMAGE.from_dataset(base_ds)
    geom = rti.get_geometry()

    # ImagePlanePixelSpacing is [row, col]; we store spacing as (dx, dy) = (col, row)
    # i.e., (0.6, 0.5) for (row=0.5, col=0.6)
    assert geom["spacing"] == (0.6, 0.5)


def test_rtimage_get_pixel_array_and_windowing_with_tags():
    """
    get_pixel_array_float and window_image should cooperate with DICOM
    WindowCenter/WindowWidth when provided.
    """
    ds = _make_minimal_rtimage_dataset(add_window=True)
    # Slightly tweak pixel data so it's not trivial
    arr = np.linspace(0, 400, ds.Rows * ds.Columns, dtype=np.float32).astype(np.uint16)
    ds.PixelData = arr.tobytes()

    rti = RTIMAGE.from_dataset(ds)

    float_img = rti.get_pixel_array_float()
    assert float_img.shape == (ds.Rows, ds.Columns)
    assert float_img.dtype == np.float32

    win_img = rti.window_image()
    assert win_img.shape == float_img.shape
    assert np.all(win_img >= 0.0) and np.all(win_img <= 1.0)

    img8 = rti.to_uint8()
    assert img8.dtype == np.uint8
    assert img8.shape == float_img.shape


def test_rtimage_pixel_iso_inverse_consistency():
    """
    pixel_to_isocenter_bev and iso_to_pixels should be approximately inverse
    mappings under the same geometry and principal point.
    """
    ds = _make_minimal_rtimage_dataset(
        rows=16,
        cols=16,
        pixel_spacing=(1.0, 1.0),
        sid=1500.0,
        sad=1000.0,
    )
    rti = RTIMAGE.from_dataset(ds)

    # pick a few arbitrary pixel coordinates
    xs = np.array([0, 5, 10, 15], dtype=float)
    ys = np.array([0, 4, 8, 15], dtype=float)

    u_iso, v_iso = rti.pixel_to_isocenter_bev(xs, ys)
    x2, y2 = rti.iso_to_pixels(u_iso, v_iso)

    # They should match (within floating error)
    assert np.allclose(xs, x2, atol=1e-6)
    assert np.allclose(ys, y2, atol=1e-6)


def test_rtimage_summary_keys_present():
    """
    summary() should return a dict with a stable schema and no crashes.
    """
    ds = _make_minimal_rtimage_dataset()
    rti = RTIMAGE.from_dataset(ds)

    summary = rti.summary()
    assert "size" in summary
    assert "spacing_mm" in summary
    assert "SID_mm" in summary
    assert "SAD_mm" in summary
    assert "Angles_deg" in summary
    assert "Machine" in summary
    assert "Link" in summary
