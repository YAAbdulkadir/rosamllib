import os
import pydicom
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
        """
        Reads the SEG file or dataset and returns an instance of the SEG class.

        If a file path is provided, it reads the file or searches for a SEG file
        in the directory. If a dataset is provided, it directly instantiates the SEG class.

        Returns
        -------
        SEG
            An instance of the `SEG` class, initialized with the DICOM SEG dataset.

        Raises
        ------
        IOError
            If no SEG file is found in the directory or if the file cannot be read.
        """
        if self.seg_file_path:
            if os.path.isdir(self.seg_file_path):
                seg_file = self._find_seg_in_directory(self.seg_file_path)
                if not seg_file:
                    raise IOError(f"No SEG file found in directory: {self.seg_file_path}")
                self.seg_dataset = pydicom.dcmread(seg_file)
            else:
                self.seg_dataset = pydicom.dcmread(self.seg_file_path)
        elif not self.seg_dataset:
            raise ValueError("No RTPLAN file path or dataset provided.")

        return SEG(self.seg_dataset)

    def _find_seg_in_directory(self, directory_path):
        """
        Searches a directory for a SEG file.

        Parameters
        ----------
        directory_path : str
            Path to the directory to search.

        Returns
        -------
        str
            The path to the SEG file if found, otherwise None.

        Raises
        ------
        InvalidDicomError
            If no valid SEG file is found.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "SEG":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None
