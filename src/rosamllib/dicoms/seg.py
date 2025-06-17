from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import SimpleITK as sitk
from highdicom.seg import Segmentation, segread
from pydicom.dataset import Dataset


class SEG(Segmentation):
    """Light-weight wrapper that streamlines common SEG tasks."""

    @classmethod
    def from_segmentation(cls, seg: Segmentation, *, copy: bool = False) -> "SEG":
        """Wrap an *existing* Segmentation in the SEG subclass.

        Parameters
        ----------
        seg   : highdicom.seg.Segmentation
            The object you got back from `segread()` or similar.
        copy  : bool, default False
            If ``True`` deep-copies the underlying pydicom Dataset
            first; otherwise the original object is mutated in place.
        """
        return cls.from_dataset(seg, copy=copy)

    @classmethod
    def from_file(cls, fp: Union[str, Path]) -> "SEG":
        """Load from disk and up-cast in one line."""
        return cls.from_segmentation(segread(fp), copy=False)

    @classmethod
    def from_dataset_obj(cls, ds: Dataset, *, copy: bool = False) -> "SEG":
        """Up-cast a raw `Dataset` that you already have in memory."""
        return cls.from_dataset(ds, copy=copy)

    def get_segment_labels(self, *, as_dict: bool = False) -> Union[List[str], Dict[int, str]]:
        """Return the DICOM *Segment Label* for every segment."""
        descriptions = self.SegmentSequence
        if as_dict:
            return {d.segment_number: d.segment_label for d in descriptions}
        return [d.segment_label for d in descriptions]

    def get_segment_mask(
        self,
        segment_number: int,
        *,
        ignore_spatial_locations: bool = True,
    ) -> sitk.Image:
        """
        Return one segment as a `SimpleITK.Image`.

        Parameters
        ----------
        segment_number : int
            DICOM *Segment Number* (not zero-based index!).
        ignore_spatial_locations : bool, default True
            Only relevant if the fast `get_volume()` path fails and we
            have to fall back to frame-based extraction.  Passed through
            to `get_pixels_by_source_instance()`.

        Returns
        -------
        SimpleITK.Image
            Binary mask aligned to the source image geometry.
        """
        # ---------------- fast path: regular 3‑D volume --------------
        vol = self.get_volume(
            segment_numbers=[segment_number],
            combine_segments=False,  # 4‑D: z,y,x,segments
            allow_missing_positions=False,  # require complete grid
        )
        if vol is not None:
            # Only one segment, so vol.array shape → (z, y, x, 1)
            mask_arr = np.squeeze(vol.array, axis=3)  # -> (z, y, x)

            img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))

            # volume spacing is (Δz, Δy, Δx); ITK expects (Δx, Δy, Δz)
            dz, dy, dx = vol.spacing
            img.SetSpacing((dx, dy, dz))
            img.SetOrigin(vol.position)
            dir_itk = np.column_stack(
                [
                    vol.direction[:, 2],  # X ← original column axis
                    vol.direction[:, 1],  # Y ← original row axis
                    vol.direction[:, 0],  # Z ← original slice axis
                ]
            ).ravel(order="C")
            img.SetDirection(tuple(dir_itk))
            return img

        # --------------- fallback: irregular / tiled SEG -------------
        src_uids = [u for _, _, u in self.get_source_image_uids()]
        mask4d = self.get_pixels_by_source_instance(
            source_sop_instance_uids=src_uids,
            segment_numbers=[segment_number],
            combine_segments=False,
            skip_overlap_checks=True,
            ignore_spatial_locations=ignore_spatial_locations,
        )
        mask_arr = np.squeeze(mask4d, axis=3).astype(np.uint8)

        img = sitk.GetImageFromArray(mask_arr)

        # Geometry: copy from first source slice (approximation)
        first_ds = self._source_images[0]
        dz = abs(float(first_ds.SliceThickness))
        dy, dx = map(float, first_ds.PixelSpacing)
        img.SetSpacing((dx, dy, dz))
        img.SetOrigin(tuple(map(float, first_ds.ImagePositionPatient)))
        orient = tuple(map(float, first_ds.ImageOrientationPatient))
        dir_itk = orient[:3] + orient[3:] + (0.0,)
        img.SetDirection(dir_itk)
        return img

    @staticmethod
    def resample_to_reference(
        moving: sitk.Image,
        reference: sitk.Image,
        *,
        is_label: bool = True,
        default_value: int | float = 0,
    ) -> sitk.Image:
        """
        Resample `moving` onto the grid of `reference` (size, spacing,
        origin, direction).

        Parameters
        ----------
        moving : SimpleITK.Image
            Mask or probability map you want to overlay.
        reference : SimpleITK.Image
            The MR / CT (or any image) that defines the desired grid.
        is_label : bool, default True
            • True   → use *nearest-neighbour* (keeps binary/label integrity).
            • False  → use *linear* (for probability / fractional masks).
        default_value : int|float, default 0
            Value for voxels outside the `moving` field of view.

        Returns
        -------
        SimpleITK.Image
            `moving`, resampled so that
            `resampled.GetSize() == reference.GetSize()`, etc.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(sitk.Transform())  # identity
        return resampler.Execute(moving)
