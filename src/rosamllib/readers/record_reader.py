import os
import pydicom
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms import RTRecord


class RTRecordReader:
    """
    A class for reading DICOM RTRECORD files from a file path, directory, or pydicom.Dataset.
    The RTRecordReader class will return an instance of the RTRecord class, which contains methods
    for extracting treatment records, dose information, and other relevant RTRecord data.

    Parameters
    ----------
    rtrecord_input : str or pydicom.Dataset
        Path to the RTRECORD file, directory containing an RTRECORD file, or a pydicom.Dataset.

    Methods
    -------
    read()
        Reads the RTRECORD file or dataset and returns an instance of the RTRecord class.

    Examples
    --------
    >>> reader = RTRecordReader("path/to/dicom/RTRECORD")
    >>> rtrecord = reader.read()

    >>> dataset = pydicom.dcmread("path/to/dicom/RTRECORD.dcm")
    >>> reader = RTRecordReader(dataset)
    >>> rtrecord = reader.read()
    """

    def __init__(self, rtrecord_input):
        self.rtrecord_file_path = None
        self.rtrecord_dataset = None

        if isinstance(rtrecord_input, (str, Path)):
            # If rtrecord_input is a file path or directory
            self.rtrecord_file_path = rtrecord_input
        elif isinstance(rtrecord_input, Dataset):
            # If rtrecord_input is a pre-loaded pydicom.Dataset
            self.rtrecord_dataset = rtrecord_input
        else:
            raise ValueError(
                "input must be either a file path (str), a directory, or a pydicom.Dataset."
            )

    def read(self):
        """
        Reads the RTRECORD file or dataset and returns an instance of the RTRecord class.

        If a file path is provided, it reads the file or searches for an RTRECORD file
        in the directory. If a dataset is provided, it directly instantiates the RTRecord class.

        Returns
        -------
        RTRecord
            An instance of the RTRecord class, initialized with the DICOM RTRECORD dataset.

        Raises
        ------
        IOError
            If no RTRECORD file is found in the directory or if the file cannot be read.
        """
        if self.rtrecord_file_path:
            if os.path.isdir(self.rtrecord_file_path):
                rtrecord_file = self._find_rtrecord_in_directory(self.rtrecord_file_path)
                if not rtrecord_file:
                    raise IOError(
                        f"No RTRECORD file found in directory: {self.rtrecord_file_path}"
                    )
                self.rtrecord_dataset = pydicom.dcmread(rtrecord_file)
            else:
                self.rtrecord_dataset = pydicom.dcmread(self.rtrecord_file_path)
        elif not self.rtrecord_dataset:
            raise ValueError("No RTRECORD file path or dataset provided.")

        return RTRecord.from_dataset(self.rtrecord_dataset)

    def _find_rtrecord_in_directory(self, directory_path):
        """
        Searches a directory for an RTRECORD file.

        Parameters
        ----------
        directory_path : str
            Path to the directory to search.

        Returns
        -------
        str
            The path to the RTRECORD file if found, otherwise None.

        Raises
        ------
        InvalidDicomError
            If no valid RTRECORD file is found.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if ds.Modality == "RTRECORD":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None
