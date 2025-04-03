import os
from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms import RTStruct
from rosamllib.utils import validate_dicom_path


class RTStructReader:
    """
    A class to read DICOM RTSTRUCT files from various input sources such as a file path,
    directory, or a `pydicom.Dataset` object.

    The RTStructReader provides methods to read RTSTRUCT files from disk, process DICOM datasets,
    and extract the RTSTRUCT data for further usage.

    Parameters
    ----------
    dicom_path_or_dataset : str or pydicom.Dataset
        A file path (str) to the RTSTRUCT DICOM file or a directory containing it, or
        a `pydicom.Dataset` representing the RTSTRUCT.

    Attributes
    ----------
    dicom_path : str or None
        The file path to the RTSTRUCT DICOM file or directory.
    dataset : pydicom.Dataset or None
        The DICOM dataset provided, if any.
    rtstruct : RTStruct or None
        The parsed RTSTRUCT object, once successfully read.

    Methods
    -------
    read()
        Reads the RTSTRUCT file from the dataset, file, or directory.
    read_from_file(file_path)
        Reads the RTSTRUCT file from the given file path and returns an `RTStruct` object.
    read_from_dataset(dataset)
        Reads the RTSTRUCT from a `pydicom.Dataset` and returns an `RTStruct` object.
    find_rtstruct_in_directory(directory_path)
        Searches for and returns the file path of the RTSTRUCT file within a directory.
    """

    def __init__(self, dicom_path_or_dataset):
        """
        Initializes the RTStructReader with either a file path or a DICOM dataset.

        Parameters
        ----------
        dicom_path_or_dataset : str or pydicom.Dataset
            Path to the RTSTRUCT DICOM file, a directory containing RTSTRUCT files, or
            a `pydicom.Dataset` representing an RTSTRUCT.

        Raises
        ------
        TypeError
            If the input is neither a file path (str) nor a `pydicom.Dataset`.
        """
        self.dicom_path = None
        self.dataset = None
        self.rtstruct = None
        if isinstance(dicom_path_or_dataset, Dataset):
            self.dataset = dicom_path_or_dataset
        elif isinstance(dicom_path_or_dataset, (str, Path)):
            self.dicom_path = dicom_path_or_dataset
        else:
            raise TypeError("Input must be a file path (str) or a pydicom.Dataset.")

    def read(self):
        """
        Reads the RTSTRUCT data from the provided dataset or file path.

        Depending on the input provided during initialization, this method attempts to
        load the RTSTRUCT data either from the dataset directly or by searching the file
        path or directory.

        Returns
        -------
        RTStruct
            An instance of `RTStruct` containing the RTSTRUCT data.

        Raises
        ------
        IOError
            If no RTSTRUCT file is found in the directory or if there is an error reading
            the RTSTRUCT file.
        """
        if self.dataset:
            self.rtstruct = self.read_from_dataset(self.dataset)
        else:
            validate_dicom_path(self.dicom_path)
            if os.path.isdir(self.dicom_path):
                rtstruct_file = self.find_rtstruct_in_directory(self.dicom_path)
                if not rtstruct_file:
                    raise IOError(f"No RTSTRUCT file found in directory: {self.dicom_path}")
                self.rtstruct = self.read_from_file(rtstruct_file)
            elif os.path.isfile(self.dicom_path):
                self.rtstruct = self.read_from_file(self.dicom_path)

        return self.rtstruct

    @staticmethod
    def read_from_file(file_path):
        """
        Reads the RTSTRUCT DICOM file from the specified file path.

        Parameters
        ----------
        file_path : str
            The file path to the RTSTRUCT DICOM file.

        Returns
        -------
        RTStruct
            An instance of `RTStruct` containing the parsed RTSTRUCT data.

        Raises
        ------
        IOError
            If the file cannot be read or if it is not a valid RTSTRUCT DICOM file.
        """
        validate_dicom_path(file_path)
        try:
            dataset = dcmread(file_path)
            modality = getattr(dataset, "Modality", None)
            if modality != "RTSTRUCT":
                raise InvalidDicomError(f"File at {file_path} is not an RTSTRUCT DICOM file.")
        except Exception as e:
            raise IOError(f"Error reading RTSTRUCT file: {e}")

        return RTStruct(dataset)

    @staticmethod
    def read_from_dataset(dataset):
        """
        Reads the RTSTRUCT data from a provided DICOM dataset.

        Parameters
        ----------
        dataset : pydicom.Dataset
            The DICOM dataset to read the RTSTRUCT data from.

        Returns
        -------
        RTStruct
            An instance of `RTStruct` containing the parsed RTSTRUCT data.

        Raises
        ------
        InvalidDicomError
            If the provided dataset is not an RTSTRUCT.
        """
        modality = getattr(dataset, "Modality", None)
        if modality != "RTSTRUCT":
            raise InvalidDicomError("Provided dataset is not an RTSTRUCT.")

        return RTStruct(dataset)

    @staticmethod
    def find_rtstruct_in_directory(directory_path):
        """
        Searches the specified directory for an RTSTRUCT DICOM file and returns its file path.

        Parameters
        ----------
        directory_path : str
            The path to the directory to search for the RTSTRUCT file.

        Returns
        -------
        str or None
            The file path of the RTSTRUCT DICOM file if found, otherwise None.

        Raises
        ------
        IOError
            If no RTSTRUCT file is found or there is an issue reading files in the directory.
        """
        validate_dicom_path(directory_path)
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = dcmread(file_path, stop_before_pixels=True)
                    modality = getattr(ds, "Modality", None)
                    if modality == "RTSTRUCT":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None
