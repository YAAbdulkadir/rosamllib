import os
import tempfile
import SimpleITK as sitk
from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms.rtdose import RTDose
from rosamllib.utils import validate_dicom_path


class RTDoseReader:
    """
    A class for reading and processing DICOM RTDOSE files.

    This class provides functionality to read RTDOSE files from either a file path or a
    pydicom.Dataset. It leverages SimpleITK for handling the DICOM image data and pydicom for
    working with DICOM metadata. The class also includes functionality to locate an RTDOSE
    file in a directory and validate DICOM file paths.

    Parameters
    ----------
    dose_path_or_dataset : str or pydicom.Dataset
        A string representing the path to an RTDOSE file or a pydicom.Dataset object containing
        RTDOSE data.

    Attributes
    ----------
    dose_file_path : str or None
        The path to the RTDOSE file, if provided.
    dataset : pydicom.Dataset or None
        A pydicom.Dataset object, if provided.

    Methods
    -------
    read()
        Reads and processes the RTDOSE file or dataset based on the input provided.
    read_from_file(file_path)
        Reads an RTDOSE file from a given file path and returns an RTDose object.
    read_from_dataset(dataset)
        Reads RTDOSE data from a pydicom.Dataset object and returns an RTDose object.
    find_rtdose_in_directory(directory_path)
        Searches a directory for an RTDOSE file and returns the first one found.

    Examples
    --------
    >>> reader = RTDoseReader("path/to/rtdose/file")
    >>> rtdose = reader.read()
    >>> print(rtdose.GetSpacing())

    >>> dataset = pydicom.dcmread("path/to/rtdose/file")
    >>> reader = RTDoseReader(dataset)
    >>> rtdose = reader.read()
    >>> print(rtdose.GetOrigin())

    See Also
    --------
    pydicom.Dataset : Provides additional functionality for handling DICOM datasets.
    SimpleITK.Image : Used for handling image data within the RTDose object.
    """

    def __init__(self, dose_path_or_dataset):
        """
        Initializes the RTDoseReader with either a file path or a pydicom.Dataset.

        Parameters
        ----------
        dose_path_or_dataset : str or pydicom.Dataset
            The path to an RTDOSE file or a pydicom.Dataset object containing RTDOSE data.

        Raises
        ------
        ValueError
            If the provided input is neither a file path nor a pydicom.Dataset object.
        """
        self.dose_file_path = None
        self.dataset = None
        if isinstance(dose_path_or_dataset, (str, Path)):
            self.dose_file_path = dose_path_or_dataset
        elif isinstance(dose_path_or_dataset, Dataset):
            self.dataset = dose_path_or_dataset
        else:
            raise ValueError("Input must be a file path or a pydicom.Dataset.")

    def read(self):
        """
        Reads and processes the RTDOSE file or dataset based on the input.

        If a dataset is provided during initialization, this method reads and processes the
        dataset. If a file path is provided, it either reads the file directly or searches
        for an RTDOSE file in the directory.

        Returns
        -------
        RTDose
            An RTDose object representing the RTDOSE image and metadata.

        Raises
        ------
        IOError
            If no RTDOSE file is found in the provided directory or the file cannot be read.
        InvalidDicomError
            If the file is not a valid RTDOSE DICOM file.
        """
        if self.dataset:
            return self.read_from_dataset(self.dataset)
        else:
            validate_dicom_path(self.dose_file_path)
            if os.path.isdir(self.dose_file_path):
                dose_file = self.find_rtdose_in_directory(self.dose_file_path)
                if not dose_file:
                    raise IOError(f"No RTDOSE file found in directory: {self.dose_file_path}")
                return self.read_from_file(dose_file)
            else:
                return self.read_from_file(self.dose_file_path)

    @staticmethod
    def read_from_file(file_path):
        """
        Reads an RTDOSE file from the given file path and returns an RTDose object.

        Parameters
        ----------
        file_path : str
            The path to the RTDOSE DICOM file.

        Returns
        -------
        RTDose
            An RTDose object representing the RTDOSE image and metadata.

        Raises
        ------
        InvalidDicomError
            If the file is not a valid RTDOSE DICOM file.
        IOError
            If there is an error reading the RTDOSE file.
        """
        validate_dicom_path(file_path)
        try:
            dataset = dcmread(file_path)
            modality = getattr(dataset, "Modality", None)
            if modality != "RTDOSE":
                raise InvalidDicomError(f"File at {file_path} is not an RTDOSE DICOM file.")
            return RTDose(sitk.ReadImage(file_path))
        except Exception as e:
            raise IOError(f"Error reading RTDOSE file: {e}")

    @staticmethod
    def read_from_dataset(dataset):
        """
        Reads RTDOSE data from a pydicom.Dataset object and returns an RTDose object.

        This method saves the dataset temporarily to disk in order to read it into SimpleITK.
        It also performs checks to ensure that the dataset is of type 'RTDOSE'.

        Parameters
        ----------
        dataset : pydicom.Dataset
            A pydicom.Dataset object representing an RTDOSE.

        Returns
        -------
        RTDose
            An RTDose object representing the RTDOSE image and metadata.

        Raises
        ------
        InvalidDicomError
            If the provided dataset is not an RTDOSE dataset.
        """
        modality = getattr(dataset, "Modality", None)
        if modality != "RTDOSE":
            raise InvalidDicomError("Provided dataset is not an RTDOSE.")

        # create temporary file (inefficient but it's easier this way to get DICOM metadata)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        dataset.save_as(temp_file.name)

        rtdose = RTDose(sitk.ReadImage(temp_file))

        # clean up temporary file
        os.remove(temp_file)

        return rtdose

    @staticmethod
    def find_rtdose_in_directory(directory_path):
        """
        Searches a directory for an RTDOSE DICOM file and returns the path to the first one found.

        Parameters
        ----------
        directory_path : str
            The path to the directory where RTDOSE files are searched for.

        Returns
        -------
        str or None
            The path to the first RTDOSE file found, or None if no file is found.

        Raises
        ------
        IOError
            If the provided path is not a valid directory.
        """
        validate_dicom_path(directory_path)
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = dcmread(file_path, stop_before_pixels=True)
                    modality = getattr(ds, "Modality", None)
                    if modality == "RTDOSE":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        return None
