import os
import pydicom
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms import REG


class REGReader:
    """
    A class for reading DICOM REG files from a file path, directory, or pydicom.Dataset.
    The REGReader class will return an instance of the REG class, which contains methods
    for extracting transformation matrices, metadata, and referenced series information.

    Parameters
    ----------
    reg_input : str or pydicom.Dataset
        Path to the REG file, directory containing a REG file, or a pydicom.Dataset.

    Methods
    -------
    read()
        Reads the REG file or dataset and returns an instance of the REG class.

    Examples
    --------
    >>> reader = REGReader("path/to/dicom/REG")
    >>> reg = reader.read()

    >>> dataset = pydicom.dcmread("path/to/dicom/REG.dcm")
    >>> reader = REGReader(dataset)
    >>> reg = reader.read()
    """

    def __init__(self, reg_input):
        self.reg_file_path = None
        self.reg_dataset = None

        if isinstance(reg_input, (str, Path)):
            # If reg_input is a file path or directory
            self.reg_file_path = reg_input
        elif isinstance(reg_input, Dataset):
            # If reg_input is a pre-loaded pydicom.Dataset
            self.reg_dataset = reg_input
        else:
            raise ValueError(
                "reg_input must be either a file path (str), a directory, or a pydicom.Dataset."
            )

    def read(self):
        """
        Reads the REG file or dataset and returns an instance of the REG class.

        If a file path is provided, it reads the file or searches for a REG file
        in the directory. If a dataset is provided, it directly instantiates the REG class.

        Returns
        -------
        REG
            An instance of the REG class, initialized with the DICOM REG dataset.

        Raises
        ------
        IOError
            If no REG file is found in the directory or if the file cannot be read.
        """
        if self.reg_file_path:
            if os.path.isdir(self.reg_file_path):
                reg_file = self._find_reg_in_directory(self.reg_file_path)
                if not reg_file:
                    raise IOError(f"No REG file found in directory: {self.reg_file_path}")
                self.reg_dataset = pydicom.dcmread(reg_file)
            else:
                self.reg_dataset = pydicom.dcmread(self.reg_file_path)
        elif not self.reg_dataset:
            raise ValueError("No REG file path or dataset provided.")

        return REG(self.reg_dataset)

    def _find_reg_in_directory(self, directory_path):
        """
        Searches a directory for a REG file.

        Parameters
        ----------
        directory_path : str
            Path to the directory to search.

        Returns
        -------
        str
            The path to the REG file if found, otherwise None.

        Raises
        ------
        InvalidDicomError
            If no valid REG file is found.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "REG":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None
