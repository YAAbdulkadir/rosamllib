"""
RTIMAGE utilities — EPID/portal imaging helpers for QA, overlays, and geometry.

OVERVIEW
--------
This module provides a convenience wrapper around DICOM RTIMAGE (EPID/portal) objects:
  • Robust, vendor-tolerant metadata extraction (spacing, SID/SAD, angles, plan linkage)
  • Pixel pipeline (Rescale → WL/WW → 8-bit) for quick visualization
  • IEC-style orientation helpers: Beam's-Eye-View (BEV) ↔ patient coordinates
  • RTPLAN overlay extraction and rendering (Jaws + MLC) onto the RTIMAGE
  • DICOM OverlayData (group 60xx) parsing and alpha-compositing

The goal is to make portal-image QA plots and geometry readouts “one-liners” while
stating all current assumptions explicitly so we can evolve safely.

CURRENT CONVENTIONS (READ ME)
-----------------------------
Patient axes (target frame for mapping):
    x = Left(+),  y = Posterior(+),  z = Superior(+)

Default BEV basis at angles = 0 (Gantry=Collimator=Couch=0):
    ŵ (beam axis) points toward -y (A→P beam),
    û aligns +x (patient left),
    v̂ aligns +z (superior).

Rotations:
    • Collimator: about current beam axis (ŵ)
    • Gantry:     about +z
    • Couch:      about +z (patient rotates under the beam)

Projection model (pinhole scaling; iso-plane only):
    Let (u_det, v_det) be detector-plane displacements in mm from principal point,
    with u right, v down. Let SID = source-to-imager distance, SAD = source-to-axis distance.

    BEV (iso-plane) coordinates:
        u_iso = (SAD/SID) * u_det
        v_iso = (SAD/SID) * (-v_det)      # flipped so +v is “up” in BEV

    Inverse (iso → pixels; using detector spacing dx, dy in mm/px and center (cx,cy) in px):
        u_det = (SID/SAD) * u_iso
        v_det = (SID/SAD) * (-v_iso)
        x = u_det/dx + cx
        y = v_det/dy + cy

Spacing priority (detector pixel size, mm/px), first present wins:
    ImagePlanePixelSpacing (3002,0011) → ImagerPixelSpacing (0018,1164) → PixelSpacing

ANGLES & PLAN LINKAGE
---------------------
Angles are read when present and default to 0.0 otherwise:
    • GantryAngle
    • BeamLimitingDeviceAngle (collimator)
    • PatientSupportAngle (couch)

Plan linkage (optional but recommended):
    ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
    ReferencedRTPlanSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber  (when provided)

RTPLAN OVERLAYS (Jaws + MLC)
----------------------------
We extract field geometry at the **isocenter plane** from the referenced (or supplied) RTPLAN:
    • Jaws: 'X'/'ASYMX' → (u_min, u_max), 'Y'/'ASYMY' → (v_min, v_max)
    • MLC:
        - MLCX: leaves move along u; boundaries define v bands
        - MLCY: leaves move along v; boundaries define u bands
    • We use the **first Control Point** today. If LeafPositionBoundaries are missing,
      we approximate equal band spacing across the corresponding jaw limits.

DICOM OVERLAYS (group 60xx)
---------------------------
For each 60xx group:
    Rows (60xx,0010), Columns (60xx,0011), OverlayOrigin (60xx,0050; 1-based), OverlayData
    (60xx,3000)
We unpack bit-packed OverlayData into a binary mask, place it at (origin) in a full-frame canvas,
and alpha-composite over the grayscale RTIMAGE.

MINIMAL TAGS BY FEATURE
-----------------------
Display (float & windowing):
    PixelData, (optional) RescaleSlope, RescaleIntercept, (optional) WindowCenter, WindowWidth

Iso-plane projection (pixels ↔ iso):
    PixelData, Rows, Columns, one of {ImagePlanePixelSpacing, ImagerPixelSpacing, PixelSpacing},
    RTImageSID (3002,0026), RadiationMachineSAD (3002,0022)

IEC BEV→patient mapping:
    (Above) + GantryAngle, BeamLimitingDeviceAngle, PatientSupportAngle
    (defaults to 0.0 if missing)

RTPLAN overlays:
    RTPLAN with BeamSequence.ControlPointSequence
    • Jaws: BeamLimitingDevicePositionSequence item with RTBeamLimitingDeviceType in
        {'X','ASYMX','Y','ASYMY'}
    • MLC:  BeamLimitingDeviceSequence with {RTBeamLimitingDeviceType in {'MLCX','MLCY'}}
            and corresponding BeamLimitingDevicePositionSequence in ControlPointSequence
    • Optional: LeafPositionBoundaries

DICOM 60xx overlays:
    (60xx,0010) Rows, (60xx,0011) Columns, (60xx,0050) Origin, (60xx,3000) OverlayData

LIMITATIONS (BY DESIGN, FOR NOW)
--------------------------------
    • Angles/axes match a common IEC 61217 interpretation. Sites/vendors may differ.
      We intend to add a `conventions=` preset (Varian/Elekta/ViewRay, etc.) with tests.
    • Only the **first control point** is rendered for MLC/Jaws (no animation yet).
    • Iso-plane pinhole model only; no full 3D source→detector ray tracing.
    • If LeafPositionBoundaries are absent, band spacing is approximated linearly across jaws.

FUTURE EXTENSIONS (ROADMAP)
---------------------------
    • conventions=: axis/angle presets with validation + unit tests
    • cp_index / animation: draw any control point or animate leaf motion across CPs
    • project_world_to_detector(): physically-accurate 3D ray model (source, panel normal/offset)
    • Export helpers: SVG/JSON of jaws/MLC/overlays for HTML reports

QUICK START
-----------
    >>> import pydicom
    >>> from rtimage import RTIMAGE
    >>> ds = pydicom.dcmread("PORTAL_0001.dcm")
    >>> rti = RTIMAGE(ds)
    >>> img8 = rti.to_uint8()                # WL/WW (uses tags or 5-95% auto)
    >>> u, v = rti.pixel_to_isocenter_bev(512, 512)
    >>> R = rti.bev_basis_patient()          # 3x3 BEV→patient rotation
    >>> plan = pydicom.dcmread("RTPLAN.dcm")
    >>> ax = rti.draw_field_from_rtplan(plan)  # jaws + MLC overlaid on image
    >>> ax = rti.render_overlays(ax=ax)        # draw any 60xx overlays
    >>> import matplotlib.pyplot as plt; plt.show()

VERSIONING & TESTING NOTES
--------------------------
    • When changing angle conventions or the projection model, bump a visible module version
      and update this docstring.
    • Add unit tests for: spacing priority, WL/WW fallbacks, pixel↔iso inverse consistency,
      MLC boundary use vs fallback, and overlay origin placement.
"""

import math
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pydicom
from pydicom.dataset import Dataset
import matplotlib.pyplot as plt


# ------------------------------ Small linear algebra helpers ------------------------------


def _deg2rad(a: float) -> float:
    return float(a) * math.pi / 180.0


def _Rz(theta_deg: float) -> np.ndarray:
    """Rotation about +Z by theta (deg)."""
    th = _deg2rad(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _axis_angle(axis: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rodrigues' rotation about arbitrary axis (unit) by theta (deg)."""
    th = _deg2rad(theta_deg)
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = ax
    c, s = math.cos(th), math.sin(th)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


class RTIMAGE(Dataset):
    """
    RTIMAGE convenience wrapper focused on **EPID/portal images** with:
      1) Robust, vendor-tolerant metadata extraction (spacing, SID/SAD, angles, linkage)
      2) Pixel pipeline (rescale → WL/WW → uint8)
      3) **IEC-style orientation** utilities to map Beam's-Eye-View (BEV) to patient coords
      4) **RTPLAN overlays** (jaws + MLC) rendered on the RTIMAGE via simple pinhole geometry
      5) **DICOM OverlayData** (group 60xx) unpacking and quick visualization

    WHY THIS EXISTS
    ---------------
    Portal images vary a lot by vendor and workflow. This wrapper hides the common
    “gotchas” (spacing priority, missing tags, WL/WW fallbacks, MLC boundary logic)
    and gives stable accessors + plotting helpers for QA/report pipelines.

    CURRENT CONVENTIONS (IMPORTANT)
    -------------------------------
    • **Patient axes (DICOM/IEC orientation used here)**:
        x = Left(+),  y = Posterior(+),  z = Superior(+)
      This is the target patient frame for mapping from BEV.

    • **Default beam/BEV frame at angles=0**:
        Gantry=0, Collimator=0, Couch=0:
          ŵ (beam axis) points toward -y (A→P beam),
          û aligns +x (patient left),
          v̂ aligns +z (superior).
      Gantry & Couch rotate about +z. Collimator rotates about the instantaneous beam axis (ŵ).
      This is a common IEC 61217 interpretation; if your site differs, we'll introduce a
      `conventions=` switch to flip/permute axes or angle senses.

    • **Projection model**:
        Simple **pinhole** projection using **SAD/SID** ratios between the detector plane
        and the isocenter plane. This is appropriate for BEV drawings on RTIMAGE
        (field box & leaf edges). It is **not** a full 3D ray-trace from source to detector;
        we can add that as an option later.

    • **Spacing priority** (detector pixel size, mm/px):
        ImagePlanePixelSpacing (3002,0011) → ImagerPixelSpacing (0018,1164) → PixelSpacing.
      Many RTIMAGEs have at least one; code gracefully degrades when missing.

    • **Angles**:
        Uses GantryAngle, BeamLimitingDeviceAngle (collimator), PatientSupportAngle (couch)
        when present; defaults to 0.0 if absent. Values assumed degrees.

    • **Plan linkage**:
        If ReferencedRTPlanSequence exists, we read `ReferencedSOPInstanceUID` and (when present)
        pick beam via `ReferencedBeamSequence[0].ReferencedBeamNumber`. If absent, you can pass
        any RTPLAN to overlay utilities and choose the beam manually.

    GUARANTEES (WHAT CALLERS CAN RELY ON)
    -------------------------------------
    • `get_geometry()` / `summary()` return dicts with stable keys and `None` where
        data is missing.
    • `get_pixel_array_float()` always returns float32 with RescaleSlope/Intercept applied.
    • `window_image()` always returns an image in [0,1]; if WL/WW tags are missing, robust 5-95%
      percentile window is used.
    • `pixel_to_isocenter_bev()` and `iso_to_pixels()` are consistent inverses under the pinhole
        model, given stable (SID,SAD,spacing) and the same principal point.
    • Jaws/MLC outlines returned are **on the isocenter plane** in BEV (u,v) mm.

    LIMITATIONS (BY DESIGN, FOR NOW)
    --------------------------------
    • **IEC variants**: Some vendors/sites adopt different zero-angle references or axis senses.
      Today we implement a widely used convention. We will add `conventions=`
      presets (Varian/Elekta/ViewRay) with unit tests.
    • **Animated MLC/Jaws**: We read only the **first control point** for the overlay.
      Extension: select CP index, time/sample, or animate.
    • **Missing `LeafPositionBoundaries`**: If absent, we approximate equal spacing across the
      corresponding jaws; good enough for quick QA drawings, but we'll add vendor-specific
      fallbacks.
    • **Full 3D ray model**: Not implemented (only pinhole scaling to/from isocenter plane).
      We'll add an optional physically-accurate projection if needed for dosimetric
      back-projection.

    EXTENSION HOOKS (FUTURE TODO)
    ------------------------------
    • `bev_basis_patient(conventions=...)` → axis/angle presets + validation.
    • `field_outline_at_iso(..., cp_index=0)` → pick any CP; optional animation helper.
    • `draw_field_from_rtplan(..., style=...)` → SVG/JSON export for HTML reports.
    • `project_world_to_detector(...)` → full 3D source-to-detector model (with SID, SDD).

    EXAMPLE
    -------
    >>> ds = pydicom.dcmread("PORTAL_0001.dcm")
    >>> rti = RTIMAGE(ds)
    >>> img8 = rti.to_uint8()                      # auto window (percentiles if no WL/WW tags)
    >>> geom = rti.get_geometry()                  # dict with SID/SAD/etc.
    >>> u, v = rti.pixel_to_isocenter_bev(512,512) # center pixel -> (u,v) at iso (mm)
    >>> R = rti.bev_basis_patient()                # BEV→patient 3x3 rotation matrix
    >>> plan = pydicom.dcmread("RTPLAN.dcm")
    >>> ax = rti.draw_field_from_rtplan(plan)      # jaws+MLC on top of RTIMAGE
    >>> ax = rti.render_overlays(ax=ax)            # 60xx overlays semi-transparent
    """

    # ------------------------------ Initialization ------------------------------

    def __init__(self, *args, **kwargs):
        """
        Initialize an RTIMAGE dataset.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to :class:`pydicom.dataset.Dataset`.

        Notes
        -----
        If initialized from an existing Dataset (e.g. RTIMAGE(ds)), all elements
        are copied and the core metadata cache is populated.
        """
        super().__init__(*args, **kwargs)
        self._meta: Dict[str, Any] = {}

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "RTIMAGE":
        """
        Create an :class:`RTIMAGE` from an existing DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            A DICOM RTIMAGE (EPID/portal) dataset.

        Returns
        -------
        RTIMAGE
            A new RTIMAGE instance with metadata extracted.
        """
        rti = cls()
        rti.update(ds)

        if hasattr(ds, "file_meta"):
            rti.file_meta = ds.file_meta

        rti._meta = {}
        rti._extract_core_metadata()
        return rti

    def __dir__(self):
        """
        Combine Dataset attributes with DICOM keyword names for convenience.

        This preserves normal Dataset behavior and adds keyword-based tab completion.
        """
        base = super().__dir__()
        kws = [pydicom.datadict.keyword_for_tag(tag) for tag in self.keys()]
        return base + [k for k in kws if k]

    def dir(self):
        """Alias for __dir__()."""
        return self.__dir__()

    # ------------------------------ Metadata ------------------------------------
    def _extract(self, keyword: str, default=None):
        """Safe DICOM keyword getter; returns `default` if missing or unreadable."""
        try:
            return getattr(self, keyword)
        except Exception:
            return default

    def _extract_core_metadata(self):
        """
        Populate `_meta` with stable, vendor-tolerant fields:
          size: (cols, rows)
          spacing: (dx, dy) mm/px
            (priority: ImagePlanePixelSpacing → ImagerPixelSpacing → PixelSpacing)
          rtimage: {SID, SAD, Plane, Orientation, Label, Description}
          angles:  {Gantry, Collimator, Table}  (deg; defaults to 0.0 if absent)
          machine: basic identifiers
          link:    Referenced RTPLAN UID and BeamNumber (if present)
        """
        ds = self

        rows = self._extract("Rows")
        cols = self._extract("Columns")
        imps = self._extract("ImagePlanePixelSpacing")  # (3002,0011)
        ips = self._extract("ImagerPixelSpacing")  # (0018,1164)
        ps = self._extract("PixelSpacing")  # fallback

        spacing = imps or ips or ps or [None, None]
        try:
            spacing = [float(spacing[0]), float(spacing[1])]
        except Exception:
            spacing = [None, None]

        sid = self._extract("RTImageSID")  # (3002,0026)
        sad = self._extract("RadiationMachineSAD")  # (3002,0022)
        gantry = self._extract("GantryAngle")
        collimator = self._extract("BeamLimitingDeviceAngle")
        table = self._extract("PatientSupportAngle")

        ref_plan_uid = None
        ref_beam_number = None
        if "ReferencedRTPlanSequence" in ds:
            try:
                rps = ds.ReferencedRTPlanSequence[0]
                ref_plan_uid = getattr(rps, "ReferencedSOPInstanceUID", None)
                if "ReferencedBeamSequence" in rps:
                    ref_beam_number = rps.ReferencedBeamSequence[0].ReferencedBeamNumber
            except Exception:
                pass

        self._meta = {
            "size": (
                int(cols) if cols is not None else None,
                int(rows) if rows is not None else None,
            ),  # (cols, rows)
            "spacing": (
                (spacing[1], spacing[0]) if spacing[0] and spacing[1] else (None, None)
            ),  # (dx, dy) mm/px
            "rtimage": {
                "SID": float(sid) if sid is not None else None,
                "SAD": float(sad) if sad is not None else None,
                "Plane": self._extract("RTImagePlane"),
                "Orientation": self._extract("RTImageOrientation"),
                "Label": self._extract("RTImageLabel"),
                "Description": self._extract("RTImageDescription"),
            },
            "angles": {
                "Gantry": float(gantry) if gantry is not None else 0.0,
                "Collimator": float(collimator) if collimator is not None else 0.0,
                "Table": float(table) if table is not None else 0.0,
            },
            "machine": {
                "TreatmentMachineName": self._extract("RTTreatmentMachineName"),
                "Manufacturer": self._extract("Manufacturer"),
                "ManufacturerModelName": self._extract("ManufacturerModelName"),
            },
            "link": {
                "ReferencedRTPlanUID": ref_plan_uid,
                "ReferencedBeamNumber": ref_beam_number,
            },
        }

    def get_geometry(self) -> Dict[str, Any]:
        """
        Returns the structured geometry dict assembled in `_extract_core_metadata()`.

        Keys are stable; missing elements are `None`.
        """
        return self._meta

    # ------------------------------ Pixel / Display ------------------------------

    def get_pixel_array_float(self) -> np.ndarray:
        """
        Return image as float32 with **RescaleSlope/Intercept** applied if present.

        Notes
        -----
        • Many EPID images are 16-bit with vendor-specific linear rescaling.
        • Downstream display should call `window_image()` to clamp to [0,1].
        """
        arr = self.pixel_array.astype(np.float32)
        slope = float(getattr(self, "RescaleSlope", 1.0))
        intercept = float(getattr(self, "RescaleIntercept", 0.0))
        return arr * slope + intercept

    def _compute_window(
        self, img: np.ndarray, window: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Decide (center, width) for display:
          1) If `window` provided → use it (center,width)
          2) Else if DICOM WindowCenter/WindowWidth present → use them
          3) Else → robust auto window via 5-95 percentiles
        """
        if window is not None and len(window) == 2:
            return float(window[0]), float(window[1])

        wc = getattr(self, "WindowCenter", None)
        ww = getattr(self, "WindowWidth", None)

        def _take(x):
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return float(x[0])
            try:
                return float(x)
            except Exception:
                return None

        c, w = _take(wc), _take(ww)
        if c is not None and w is not None and w > 0:
            return c, w

        lo, hi = np.percentile(img, [5, 95])
        return float((hi + lo) / 2.0), float(max(hi - lo, 1.0))

    def window_image(self, window: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Apply window/level and return a float image in [0,1].

        Guarantees
        ----------
        Output is always clipped to [0,1] and ready for display or conversion to uint8.
        """
        img = self.get_pixel_array_float()
        c, w = self._compute_window(img, window)
        lo, hi = c - w / 2.0, c + w / 2.0
        out = (img - lo) / max(hi - lo, 1e-6)
        return np.clip(out, 0.0, 1.0)

    def to_uint8(self, window: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Window and convert to uint8 [0..255] for plotting or saving (PNG, etc.).
        """
        return (self.window_image(window) * 255.0 + 0.5).astype(np.uint8)

    def quicklook(self, window: Optional[Tuple[float, float]] = None, title: Optional[str] = None):
        """
        Convenience plot: shows the windowed image (grayscale), hides axes,
        and uses RTImageLabel/SID in title when not provided.
        """
        img8 = self.to_uint8(window)
        plt.figure()
        plt.imshow(img8, cmap="gray", origin="upper")
        plt.axis("off")
        if title:
            plt.title(title)
        else:
            t = self._meta["rtimage"].get("Label") or "RTIMAGE"
            sid = self._meta["rtimage"].get("SID")
            plt.title(t + (f" [SID={sid:.0f}]" if sid else ""))
        plt.show()

    # ------------------------------ IEC rotations & BEV mapping ------------------------------

    def bev_basis_patient(self) -> np.ndarray:
        """
        Build a 3x3 rotation matrix **R** that maps BEV vectors to patient space:

            patient_vec = R @ bev_vec

        Where `bev_vec = [u, v, w]^T` lives in the BEV frame (u right, v up, w along the beam).

        Conventions implemented **today** (see header “CURRENT CONVENTIONS”):
        • Patient: x=Left(+), y=Posterior(+), z=Superior(+)
        • At Gantry=Collimator=Couch=0: w = -y, u = +x, v = +z
        • Rotations: Collimator about w; Gantry about +z; Couch about +z (patient rotates)

        Returns
        -------
        R : (3,3) ndarray
            Columns are the patient-space unit vectors of (û, v̂, ŵ).

        Future extension
        ----------------
        Add `conventions=` argument to select vendor/site presets; add regression tests.
        """
        ang = self._meta["angles"]
        g = float(ang.get("Gantry", 0.0))
        c = float(ang.get("Collimator", 0.0))
        t = float(ang.get("Table", 0.0))

        # Initial basis at 0/0/0
        u0 = np.array([1.0, 0.0, 0.0])  # left
        v0 = np.array([0.0, 0.0, 1.0])  # superior
        w0 = np.array([0.0, -1.0, 0.0])  # beam towards -y

        # Collimator (around w), then Gantry (+z), then Couch (+z)
        Rc = _axis_angle(w0, c)
        u1 = Rc @ u0
        v1 = Rc @ v0
        w1 = Rc @ w0

        Rg = _Rz(g)
        u2 = Rg @ u1
        v2 = Rg @ v1
        w2 = Rg @ w1

        Rt = _Rz(t)
        u3 = Rt @ u2
        v3 = Rt @ v2
        w3 = Rt @ w2

        R = np.column_stack([u3, v3, w3])
        return R

    def pixel_to_isocenter_bev(
        self,
        x: np.ndarray,
        y: np.ndarray,
        principal_point: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map detector pixel indices (x=col, y=row) → (u,v) coordinates **at isocenter** (mm).

        Model
        -----
        Simple pinhole scaling:  (u_iso, v_iso) = ( SAD/SID ) * (u_det, -v_det ),
        where (u_det, v_det) are detector-plane mm displacements from the principal point,
        with u to the right and v **down** on the detector. We flip sign on v to make
        +v point **up** in BEV coordinates.

        Requirements
        ------------
        Needs SID, SAD, detector spacing, and image size. Raises ValueError if missing.

        principal_point
        ----------------
        (cx, cy) in **pixels**. If None, uses image center ((W-1)/2, (H-1)/2).
        """
        (dx, dy) = self._meta.get("spacing", (None, None))
        sid = self._meta["rtimage"].get("SID")
        sad = self._meta["rtimage"].get("SAD")
        size = self._meta.get("size", (None, None))
        if None in (dx, dy, sid, sad, size[0], size[1]):
            raise ValueError("Insufficient geometry/spacing to project to isocenter plane.")

        cols, rows = size
        if principal_point is None:
            cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
        else:
            cx, cy = principal_point

        u_det = (np.asarray(x) - cx) * dx
        v_det = (np.asarray(y) - cy) * dy
        scale = float(sad) / float(sid)
        u_iso = u_det * scale
        v_iso = -v_det * scale
        return u_iso, v_iso

    def bev_to_patient(
        self, u: np.ndarray, v: np.ndarray, w: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Map BEV isocenter-plane points to patient coordinates.

        Inputs
        ------
        u, v : array-like
            BEV coordinates at the isocenter plane (mm).
        w : array-like or None
            Beam-axis component (mm). If None, assumes w=0 (isocenter plane).

        Returns
        -------
        xyz : ndarray[..., 3]
            Patient-space coordinates after applying `bev_basis_patient()`.

        Notes
        -----
        This does **not** perform source-to-detector ray intersection. It's a pure
        frame rotation (with optional w component) useful for reporting and analysis.
        """
        R = self.bev_basis_patient()
        if w is None:
            w = np.zeros_like(u, dtype=float)
        uvw = np.stack([u, v, w], axis=-1)  # (...,3)
        return uvw @ R.T

    # ------------------------------ RTPLAN overlay extraction ------------------------------

    @staticmethod
    def _find_beam(rtplan: Dataset, beam_number: Optional[int]) -> Optional[Dataset]:
        """Return the RTPLAN beam (by BeamNumber if provided, else first)."""
        if "BeamSequence" not in rtplan:
            return None
        if beam_number is None:
            return rtplan.BeamSequence[0]
        for b in rtplan.BeamSequence:
            if getattr(b, "BeamNumber", None) == beam_number:
                return b
        return None

    @staticmethod
    def _first_cp(beam: Dataset) -> Optional[Dataset]:
        """Return the first ControlPointSequence item or None."""
        if "ControlPointSequence" in beam and len(beam.ControlPointSequence) > 0:
            return beam.ControlPointSequence[0]
        return None

    @staticmethod
    def _device_by_type(cp: Dataset, dev_type: str) -> Optional[Dataset]:
        """
        Find a BeamLimitingDevicePositionSequence item by
        RTBeamLimitingDeviceType (e.g., 'X','Y','MLCX','MLCY').
        """
        if "BeamLimitingDevicePositionSequence" not in cp:
            return None
        for d in cp.BeamLimitingDevicePositionSequence:
            if getattr(d, "RTBeamLimitingDeviceType", "") == dev_type:
                return d
        return None

    def field_outline_at_iso(
        self, rtplan: Dataset, beam_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract the jaws box and MLC edges **on the isocenter plane** in BEV (u,v) mm.

        Returns
        -------
        {
          'jaws': {'X': (x1,x2), 'Y': (y1,y2)},   # mm in BEV (u along X, v along Y)
          'mlc':  [((u1,v1),(u2,v2)), ...],      # list of leaf edge segments at iso
          'mlc_info': {'type': 'MLCX'|'MLCY', 'leaf_boundaries': [...]}  # when present
        }

        Behavior
        --------
        • Uses **first control point** only (today).
        • For **MLCX**: leaves move along u; boundaries define v band edges.
          We return 2 vertical segments per leaf pair (bank A and B).
        • For **MLCY**: leaves move along v; boundaries define u band edges.
          We return 2 horizontal segments per leaf pair.

        Fallbacks
        ---------
        • If `LeafPositionBoundaries` missing, we evenly space boundaries between the jaw limits
          on the orthogonal axis. This is a pragmatic approximation for QA drawings.

        Future extension
        ----------------
        Add `cp_index` selection or an animation helper; vendor-specific boundary lookups.
        """
        beam = self._find_beam(rtplan, beam_number)
        if beam is None:
            raise ValueError("Beam not found in RTPLAN.")

        cp0 = self._first_cp(beam)
        if cp0 is None:
            raise ValueError("No ControlPointSequence in beam.")

        out: Dict[str, Any] = {"jaws": {}, "mlc": [], "mlc_info": {}}

        # --- Jaws
        for typ in ("ASYMX", "X", "ASYMY", "Y"):
            d = self._device_by_type(cp0, typ)
            if d is None:
                continue
            pos = list(map(float, getattr(d, "LeafJawPositions", [])))
            if len(pos) == 2:
                if "X" in typ:
                    out["jaws"]["X"] = (pos[0], pos[1])  # along u
                else:
                    out["jaws"]["Y"] = (pos[0], pos[1])  # along v

        # --- MLC
        mlc_dev = None
        for dev in getattr(beam, "BeamLimitingDeviceSequence", []):
            if "MLC" in getattr(dev, "RTBeamLimitingDeviceType", ""):
                mlc_dev = dev
                break

        mlc_type = getattr(mlc_dev, "RTBeamLimitingDeviceType", None) if mlc_dev else None
        out["mlc_info"]["type"] = mlc_type

        mlc_pos_seq = self._device_by_type(cp0, "MLCX") or self._device_by_type(cp0, "MLCY")
        if mlc_pos_seq is not None:
            leaf_pos = np.array(list(map(float, mlc_pos_seq.LeafJawPositions)), dtype=float)
            boundaries = None
            if mlc_dev is not None and hasattr(mlc_dev, "LeafPositionBoundaries"):
                boundaries = np.array(
                    list(map(float, mlc_dev.LeafPositionBoundaries)), dtype=float
                )
                out["mlc_info"]["leaf_boundaries"] = boundaries.tolist()

            if mlc_type == "MLCX":
                n = leaf_pos.size // 2
                a = leaf_pos[:n]  # bank A
                b = leaf_pos[n:]  # bank B
                if boundaries is None:
                    y1, y2 = out["jaws"].get("Y", (-100.0, 100.0))
                    boundaries = np.linspace(y1, y2, n + 1)
                for i in range(n):
                    v_lo, v_hi = float(boundaries[i]), float(boundaries[i + 1])
                    out["mlc"].append(((a[i], v_lo), (a[i], v_hi)))
                    out["mlc"].append(((b[i], v_lo), (b[i], v_hi)))
            elif mlc_type == "MLCY":
                n = leaf_pos.size // 2
                a = leaf_pos[:n]
                b = leaf_pos[n:]
                if boundaries is None:
                    x1, x2 = out["jaws"].get("X", (-100.0, 100.0))
                    boundaries = np.linspace(x1, x2, n + 1)
                for i in range(n):
                    u_lo, u_hi = float(boundaries[i]), float(boundaries[i + 1])
                    out["mlc"].append(((u_lo, a[i]), (u_hi, a[i])))
                    out["mlc"].append(((u_lo, b[i]), (u_hi, b[i])))

        return out

    # ------------------------------ Projection: isocenter plane <-> pixels ----------------

    def iso_to_pixels(
        self,
        u_iso: np.ndarray,
        v_iso: np.ndarray,
        principal_point: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project BEV isocenter-plane coordinates (u,v) mm → detector pixel indices (x,y).

        Inverse of `pixel_to_isocenter_bev()` under the same pinhole model:

            (u_det, v_det) = ( SID/SAD ) * (u_iso, -v_iso)
            x = u_det/dx + cx
            y = v_det/dy + cy

        Requirements
        ------------
        Needs SID, SAD, spacing, and image size. Raises ValueError if missing.

        principal_point
        ----------------
        (cx, cy) in **pixels**; defaults to image center if None.
        """
        (dx, dy) = self._meta.get("spacing", (None, None))
        sid = self._meta["rtimage"].get("SID")
        sad = self._meta["rtimage"].get("SAD")
        size = self._meta.get("size", (None, None))
        if None in (dx, dy, sid, sad, size[0], size[1]):
            raise ValueError("Insufficient geometry/spacing to project to pixels.")

        cols, rows = size
        if principal_point is None:
            cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
        else:
            cx, cy = principal_point

        scale = float(sid) / float(sad)
        u_det = np.asarray(u_iso) * scale
        v_det = -np.asarray(v_iso) * scale
        x = u_det / dx + cx
        y = v_det / dy + cy
        return x, y

    # ------------------------------ DICOM Overlays (60xx) ------------------------------

    def get_overlays(self) -> List[Dict[str, Any]]:
        """
        Parse DICOM Overlay (group 60xx) bitmaps.

        Returns
        -------
        A list of dicts:
        [
          {
            'group': 0x6000 | 0x6002 | ...,
            'rows': R,
            'cols': C,
            'origin': (row0, col0),      # 0-based placement into the full image
            'bits_allocated': B,         # usually 1
            'mask': np.uint8[R, C]       # 1 where overlay is set, 0 elsewhere
          },
          ...
        ]

        Behavior & Notes
        ----------------
        • Unpacks bit-packed OverlayData (60xx,3000) into a binary mask.
        • Truncates any padding beyond R*C bits.
        • Converts the 1-based OverlayOrigin (60xx,0050) to 0-based indices for placing into
          the full image canvas.
        """
        ds = self
        overlays: List[Dict[str, Any]] = []
        for gg in range(0x6000, 0x601F + 1, 2):
            group = pydicom.tag.Tag(gg, 0x0010)  # Rows tag probe
            if group not in ds:
                continue
            rows = int(ds[pydicom.tag.Tag(gg, 0x0010)].value)
            cols = int(ds[pydicom.tag.Tag(gg, 0x0011)].value)
            origin = ds.get(pydicom.tag.Tag(gg, 0x0050), [1, 1])  # 1-based (row, col)
            ov_bits = ds.get(pydicom.tag.Tag(gg, 0x0100), 1)
            data_elem = ds.get(pydicom.tag.Tag(gg, 0x3000))
            if data_elem is None:
                continue
            data = np.frombuffer(data_elem.value, dtype=np.uint8)
            bits = np.unpackbits(data)
            mask = (bits[: rows * cols].reshape(rows, cols) & 1).astype(np.uint8)
            overlays.append(
                {
                    "group": gg,
                    "rows": rows,
                    "cols": cols,
                    "origin": (int(origin[0]) - 1, int(origin[1]) - 1),
                    "bits_allocated": ov_bits,
                    "mask": mask,
                }
            )
        return overlays

    def render_overlays(
        self, ax: Optional[plt.Axes] = None, color: str = "r", alpha: float = 0.35
    ):
        """
        Alpha-composite all 60xx overlays on top of the current RTIMAGE view.

        Parameters
        ----------
        ax : matplotlib Axes or None
            If None, creates a new figure and draws the grayscale RTIMAGE first.
        color : str
            Currently unused for per-pixel masks (kept for parity with future vector overlays).
        alpha : float
            Overlay transparency.

        Notes
        -----
        Each overlay is placed at its (origin) in the full image canvas (rowsxcols).
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.imshow(self.to_uint8(), cmap="gray", origin="upper")
            ax.axis("off")
        for o in self.get_overlays():
            r0, c0 = o["origin"]
            mask = o["mask"]
            H, W = int(self._meta["size"][1]), int(self._meta["size"][0])
            canvas = np.zeros((H, W), dtype=float)
            rr, cc = mask.shape
            r1, c1 = min(r0 + rr, H), min(c0 + cc, W)
            canvas[r0:r1, c0:c1] = mask[: r1 - r0, : c1 - c0]
            ax.imshow(np.ma.masked_where(canvas == 0, canvas), origin="upper", alpha=alpha)
        return ax

    # ------------------------------ Drawing helpers for jaws/MLC ------------------------------

    def draw_field_from_rtplan(
        self,
        rtplan: Dataset,
        beam_number: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        window: Optional[Tuple[float, float]] = None,
        show_mlc: bool = True,
        show_jaws: bool = True,
        color: str = "y",
        lw: float = 1.5,
    ):
        """
        Draw Jaws rectangle and MLC edges (from RTPLAN) **on the RTIMAGE**.

        Steps
        -----
        1) Renders the windowed RTIMAGE (if `ax` not provided).
        2) Extracts field geometry at the isocenter plane via `field_outline_at_iso(...)`.
        3) Projects BEV (u,v) to detector pixels using `iso_to_pixels(...)`.
        4) Plots jaws polygon and MLC leaf edge segments.

        Parameters
        ----------
        rtplan : Dataset
            The RTPLAN DICOM object.
        beam_number : int or None
            Beam number to use (None → first beam).
        ax : matplotlib Axes or None
            Existing axes to draw on; if None, a new figure is created.
        window : (center,width) or None
            WL/WW for the background image; None → auto/DICOM tags.
        show_mlc, show_jaws : bool
            Toggle visibility of MLC and jaws.
        color : str
            Line color for field outlines.
        lw : float
            Line width.

        Future extension
        ----------------
        • `cp_index` to draw any control point.
        • Styling/export helpers (SVG/JSON) for HTML reports.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.imshow(self.to_uint8(window), cmap="gray", origin="upper")
            ax.axis("off")

        geom = self.field_outline_at_iso(rtplan, beam_number)

        if show_jaws and "X" in geom["jaws"] and "Y" in geom["jaws"]:
            x1, x2 = geom["jaws"]["X"]
            y1, y2 = geom["jaws"]["Y"]
            poly_u = np.array([x1, x2, x2, x1, x1])
            poly_v = np.array([y1, y1, y2, y2, y1])
            px, py = self.iso_to_pixels(poly_u, poly_v)
            ax.plot(px, py, color=color, lw=lw)

        if show_mlc and len(geom["mlc"]) > 0:
            for p0, p1 in geom["mlc"]:
                u = np.array([p0[0], p1[0]])
                v = np.array([p0[1], p1[1]])
                px, py = self.iso_to_pixels(u, v)
                ax.plot(px, py, color=color, lw=lw * 0.8, alpha=0.9)

        return ax

    # ------------------------------ Summaries & misc ------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Compact summary for logging/serialization. Schema is stable.

        Contains: size, spacing_mm, SID_mm, SAD_mm, Angles_deg, Machine, Link
        """
        return {
            "size": self._meta["size"],
            "spacing_mm": self._meta["spacing"],
            "SID_mm": self._meta["rtimage"]["SID"],
            "SAD_mm": self._meta["rtimage"]["SAD"],
            "Angles_deg": self._meta["angles"],
            "Machine": self._meta["machine"],
            "Link": self._meta["link"],
        }

    def __repr__(self) -> str:
        """
        Compact human-readable summary of the RTIMAGE.

        Example:
            RTIMAGE(size=1024x1024, spacing=0.40x0.40 mm,
                    SID=1000 mm, SAD=1000 mm,
                    angles={'Gantry':0.0,'Collimator':0.0,'Table':0.0},
                    plan_uid=1.2.3.4)
        """
        size = self._meta.get("size", (None, None))
        spacing = self._meta.get("spacing", (None, None))
        sid = self._meta.get("rtimage", {}).get("SID")
        sad = self._meta.get("rtimage", {}).get("SAD")
        ang = self._meta.get("angles", {})
        link = self._meta.get("link", {})

        cols, rows = size if size else (None, None)
        dx, dy = spacing if spacing else (None, None)

        gantry = ang.get("Gantry", None)
        collimator = ang.get("Collimator", None)
        table = ang.get("Table", None)

        plan_uid = link.get("ReferencedRTPlanUID", None)

        return (
            "RTIMAGE("
            f"size={cols}x{rows}, "
            f"spacing={dx}x{dy} mm, "
            f"SID={sid}, SAD={sad}, "
            f"angles={{Gantry:{gantry}, Collimator:{collimator}, Table:{table}}}, "
            f"plan_uid={plan_uid}"
            ")"
        )
