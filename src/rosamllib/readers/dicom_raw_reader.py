from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from rosamllib.dicoms import RAW


class DICOMRawReader:
    """
    Reader class for loading RAW DICOM objects from disk or an in-memory Dataset.

    The reader accepts either:

    - A file path to a RAW DICOM file.
    - A pre-loaded :class:`pydicom.Dataset`.

    The :meth:`read` method returns an instance of :class:`rosamllib.dicoms.RAW`,
    which subclasses :class:`pydicom.dataset.Dataset` and provides utilities for
    extracting embedded datasets from the MIMSoftwareSessionMetaSeq (0013,2050)
    tag and referenced series information.

    Parameters
    ----------
    raw_input : str, pathlib.Path, or pydicom.Dataset
        Path to a RAW DICOM file or an in-memory DICOM dataset.
    """

    def __init__(self, raw_input):
        self.raw_file_path = None
        self.raw_dataset = None

        if isinstance(raw_input, (str, Path)):
            self.raw_file_path = raw_input  # RAW file path provided
        elif isinstance(raw_input, Dataset):
            self.raw_dataset = raw_input  # Dataset object provided
        else:
            raise ValueError(
                "raw_input must be either a file path (str) or a pydicom.Dataset object."
            )

    def read(self):
        """
        Load the RAW DICOM and return an initialized :class:`RAW` object.

        Returns
        -------
        RAW
            A :class:`RAW` instance created via :meth:`RAW.from_dataset`.

        Raises
        ------
        ValueError
            If neither a file path nor dataset was supplied.
        pydicom.errors.InvalidDicomError
            If the file at ``raw_file_path`` is not a valid DICOM.
        ValueError
            If the RAW dataset does not contain the expected (0013,2050) tag.
        """
        if self.raw_file_path:
            ds = dcmread(self.raw_file_path)  # Read RAW DICOM file
        elif self.raw_dataset is not None:
            ds = self.raw_dataset
        else:
            raise ValueError("No RAW file path or dataset provided.")

        return RAW.from_dataset(ds)
