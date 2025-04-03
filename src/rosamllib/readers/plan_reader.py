import os
import pydicom
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms import RTPlan


class RTPlanReader:
    """
    A class for reading DICOM RTPLAN files from a file path, directory, or pydicom.Dataset.
    The RTPlanReader class will return an instance of the RTPlan class, which contains methods
    for extracting beam sequences, fraction details, and treatment parameters.

    Parameters
    ----------
    rtplan_input : str or pydicom.Dataset
        Path to the RTPLAN file, directory containing an RTPLAN file, or a pydicom.Dataset.

    Methods
    -------
    read()
        Reads the RTPLAN file or dataset and returns an instance of the RTPlan class.

    Examples
    --------
    >>> reader = RTPlanReader("path/to/dicom/RTPLAN")
    >>> rtplan = reader.read()

    >>> dataset = pydicom.dcmread("path/to/dicom/RTPLAN.dcm")
    >>> reader = RTPlanReader(dataset)
    >>> rtplan = reader.read()
    """

    def __init__(self, rtplan_input):
        self.rtplan_file_path = None
        self.rtplan_dataset = None

        if isinstance(rtplan_input, (str, Path)):
            # If rtplan_input is a file path or directory
            self.rtplan_file_path = rtplan_input
        elif isinstance(rtplan_input, Dataset):
            # If rtplan_input is a pre-loaded pydicom.Dataset
            self.rtplan_dataset = rtplan_input
        else:
            raise ValueError(
                "rtplan_input must be either a file path (str), a directory, or a pydicom.Dataset."
            )

    def read(self):
        """
        Reads the RTPLAN file or dataset and returns an instance of the RTPlan class.

        If a file path is provided, it reads the file or searches for an RTPLAN file
        in the directory. If a dataset is provided, it directly instantiates the RTPlan class.

        Returns
        -------
        RTPlan
            An instance of the RTPlan class, initialized with the DICOM RTPLAN dataset.

        Raises
        ------
        IOError
            If no RTPLAN file is found in the directory or if the file cannot be read.
        """
        if self.rtplan_file_path:
            if os.path.isdir(self.rtplan_file_path):
                rtplan_file = self._find_rtplan_in_directory(self.rtplan_file_path)
                if not rtplan_file:
                    raise IOError(f"No RTPLAN file found in directory: {self.rtplan_file_path}")
                self.rtplan_dataset = pydicom.dcmread(rtplan_file)
            else:
                self.rtplan_dataset = pydicom.dcmread(self.rtplan_file_path)
        elif not self.rtplan_dataset:
            raise ValueError("No RTPLAN file path or dataset provided.")

        return RTPlan(self.rtplan_dataset)

    def _find_rtplan_in_directory(self, directory_path):
        """
        Searches a directory for an RTPLAN file.

        Parameters
        ----------
        directory_path : str
            Path to the directory to search.

        Returns
        -------
        str
            The path to the RTPLAN file if found, otherwise None.

        Raises
        ------
        InvalidDicomError
            If no valid RTPLAN file is found.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "RTPLAN":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None
