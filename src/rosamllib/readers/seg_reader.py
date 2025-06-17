import os
import pydicom
from highdicom.seg import Segmentation, segread
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms import SEG


class SEGReader:
    """
    A class for reading DICOM SEG files from a file path, directory, or pydicom.Dataset.
    The SEGReader class will return an instance of the SEG class, which contains methods
    for extracting beam sequences, fraction details, and treatment parameters.

    Parameters
    ----------
    seg_input : str or pydicom.Dataset
        Path to the SEG file, directory containing a SEG file, or a pydicom.Dataset.

    Methods
    -------
    read()
        Reads the SEG file or dataset and returns an instance of the SEG class.

    Examples
    --------
    >>> reader = SEGReader("path/to/dicom/SEG")
    >>> seg = reader.read()

    >>> dataset = pydicom.dcmread("path/to/dicom/SEG.dcm")
    >>> reader = SEGReader(dataset)
    >>> seg = reader.read()
    """

    def __init__(self, seg_input):
        self.seg_file_path = None
        self.seg_dataset = None

        if isinstance(seg_input, (str, Path)):
            # If seg_input is a file path or directory
            self.seg_file_path = seg_input
        elif isinstance(seg_input, Dataset):
            # If seg_input is a pre-loaded pydicom.Dataset
            self.seg_dataset = seg_input
        else:
            raise ValueError(
                "seg_input must be either a file path (str), a directory, or a pydicom.Dataset."
            )

    def read(self):

        if self.seg_file_path:
            if os.path.isdir(self.seg_file_path):
                seg_file = self._find_seg_in_directory(self.seg_file_path)
                if not seg_file:
                    raise IOError(f"No SEG file found in directory: {self.seg_file_path}")
                seg = segread(seg_file)
            else:
                seg = segread(self.seg_file_path)
        elif self.seg_dataset:
            seg = Segmentation.from_dataset(self.seg_dataset, copy=True)
        else:
            raise ValueError("No SEG file path or dataset provided.")

        return SEG.from_segmentation(seg, copy=True)

    @staticmethod
    def _assert_is_seg(ds: Dataset) -> None:
        if getattr(ds, "Modality", None) != "SEG":
            raise ValueError("Dataset is not a DICOM Segmentation object.")

    @staticmethod
    def _find_seg_in_directory(folder: Path) -> Path | None:
        """Return the first SEG file found (depth-first)."""
        for root, _, files in os.walk(folder):
            for name in files:
                fp = Path(root, name)
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True)
                    if getattr(ds, "Modality", None) == "SEG":
                        return fp
                except InvalidDicomError:
                    pass
        return None
