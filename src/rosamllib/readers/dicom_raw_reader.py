from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian


class DICOMRawReader:
    """
    A class for reading DICOM RAW files and extracting embedded datasets from
    the MIMSoftwareSessionMetaSeq (0013, 2050) tag.

    This class reads DICOM RAW files or accepts a `pydicom.Dataset` directly and
    extracts all embedded datasets contained within the MIMSoftwareSessionMetaSeq tag,
    which could include datasets from other modalities such as REG or RTSTRUCT.

    Parameters
    ----------
    raw_input : str or pydicom.Dataset
        The path to the RAW DICOM file or a `pydicom.Dataset` representing the RAW DICOM.

    Attributes
    ----------
    raw_file_path : str or None
        The path to the RAW DICOM file, if provided.
    dataset : pydicom.Dataset or None
        The DICOM dataset object read from the RAW file or directly provided.
    embedded_datasets : list of pydicom.Dataset
        A list of extracted embedded datasets from the MIMSoftwareSessionMetaSeq tag.
    referenced_series_uid : str or None
        The SeriesInstanceUID extracted from the ReferencedSeriesSequence if available.

    Methods
    -------
    read()
        Reads the RAW DICOM file or dataset and extracts embedded datasets from the
        MIMSoftwareSessionMetaSeq tag.
    extract_embedded_datasets()
        Extracts all embedded datasets from the MIMSoftwareSessionMetaSeq tag in the dataset.
    get_embedded_datasets()
        Returns the list of extracted embedded datasets.
    _get_referenced_series_uid()
        Attempts to extract the ReferencedSeriesUID from the dataset.

    Raises
    ------
    ValueError
        If `raw_input` is neither a file path (str) nor a `pydicom.Dataset` object.
    IOError
        If the RAW DICOM file cannot be read or the MIMSoftwareSessionMetaSeq tag is not found.

    Examples
    --------
    Reading a DICOM RAW file and extracting embedded datasets:

    >>> reader = DICOMRawReader("path/to/dicom_raw.dcm")
    >>> reader.read()
    >>> embedded_datasets = reader.get_embedded_datasets()
    >>> for ds in embedded_datasets:
    ...     print(ds.Modality)

    Passing a pre-loaded `pydicom.Dataset` directly:

    >>> raw_dataset = pydicom.dcmread("path/to/dicom_raw.dcm")
    >>> reader = DICOMRawReader(raw_dataset)
    >>> reader.read()
    >>> embedded_datasets = reader.get_embedded_datasets()
    >>> for ds in embedded_datasets:
    ...     print(ds.Modality)
    """

    def __init__(self, raw_input):
        """
        Initializes the DICOMRawReader with either a path to the RAW DICOM file or a
        `pydicom.Dataset` object.

        Parameters
        ----------
        raw_input : str or pydicom.Dataset
            The path to the RAW DICOM file or a `pydicom.Dataset` object.

        Raises
        ------
        ValueError
            If `raw_input` is neither a file path (str) nor a `pydicom.Dataset` object.
        """
        self.raw_file_path = None
        self.dataset = None
        self.embedded_datasets = []
        self.referenced_series_uid = None
        if isinstance(raw_input, (str, Path)):
            self.raw_file_path = raw_input  # RAW file path provided
        elif isinstance(raw_input, Dataset):
            self.dataset = raw_input  # Dataset object provided
        else:
            raise ValueError(
                "raw_input must be either a file path (str) or a pydicom.Dataset object."
            )

    def read(self):
        """
        Reads the RAW DICOM file or provided dataset and extracts embedded datasets
        from the MIMSoftwareSessionMetaSeq tag.

        If the input is a file path, it reads the RAW DICOM file using `pydicom.dcmread`.
        Then, it calls `extract_embedded_datasets` to extract datasets within the
        MIMSoftwareSessionMetaSeq tag.

        Raises
        ------
        IOError
            If the RAW DICOM file cannot be read or the dataset is not properly loaded.
        ValueError
            If the MIMSoftwareSessionMetaSeq tag is not found.
        """
        try:
            if self.raw_file_path:
                self.dataset = dcmread(self.raw_file_path)  # Read RAW DICOM file
            elif self.dataset is not None:
                pass  # Use provided dataset
            else:
                raise ValueError("No RAW file path or dataset provided.")

            # Extract embedded datasets from the dataset
            self.extract_embedded_datasets()
            self._get_referenced_series_uid()

        except Exception as e:
            raise IOError(f"Failed to read RAW DICOM file or dataset: {e}")

    def extract_embedded_datasets(self):
        """
        Extracts all embedded datasets from the MIMSoftwareSessionMetaSeq tag.

        The method looks for the MIMSoftwareSessionMetaSeq (0013, 2050) tag and iterates
        through its items, extracting each one as an individual dataset. These datasets are
        stored in the `embedded_datasets` attribute.

        Raises
        ------
        ValueError
            If the MIMSoftwareSessionMetaSeq tag is not found in the RAW DICOM dataset.

        Examples
        --------
        >>> reader = DICOMRawReader("path/to/dicom_raw.dcm")
        >>> reader.read()
        >>> reader.extract_embedded_datasets()
        """
        if self.dataset is None:
            raise ValueError("RAW DICOM file not loaded. Call `read` method first.")

        # Check for MIMSoftwareSessionMetaSeq tag
        if (0x0013, 0x2050) in self.dataset:
            mim_seq = self.dataset[(0x0013, 0x2050)]

            # Iterate over the items in MIMSoftwareSessionMetaSeq
            for item in mim_seq:
                if isinstance(item, Dataset):
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = item.SOPClassUID
                    file_meta.MediaStorageSOPInstanceUID = item.SOPInstanceUID
                    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                    item.file_meta = file_meta
                    self.embedded_datasets.append(item)
        else:
            raise ValueError(
                "MIMSoftwareSessionMetaSeq (0013, 2050) tag not found in the RAW DICOM file."
            )

    def get_embedded_datasets(self):
        """
        Returns the list of embedded datasets extracted from the MIMSoftwareSessionMetaSeq tag.

        If no datasets have been extracted, this method will raise an exception.

        Returns
        -------
        list of pydicom.Dataset
            A list of embedded datasets extracted from the RAW file.

        Raises
        ------
        ValueError
            If no embedded datasets have been extracted or the dataset has not been loaded.

        Examples
        --------
        >>> reader = DICOMRawReader("path/to/dicom_raw.dcm")
        >>> reader.read()
        >>> embedded_datasets = reader.get_embedded_datasets()
        >>> for ds in embedded_datasets:
        ...     print(ds.Modality)
        """
        if not self.embedded_datasets:
            if not self.dataset:
                raise ValueError("No embedded datasets extracted. Call `read` method first.")
        return self.embedded_datasets

    def _get_referenced_series_uid(self):
        """
        Attempts to extract the ReferencedSeriesUID from the ReferencedSeriesSequence tag.

        This method looks for the ReferencedSeriesSequence in the dataset and retrieves the
        SeriesInstanceUID from it, storing it in the `referenced_series_uid` attribute.

        Raises
        ------
        AttributeError
            If the ReferencedSeriesSequence or SeriesInstanceUID is not found.

        Notes
        -----
        This is an internal helper method for extracting the referenced series information.
        """
        try:
            self.referenced_series_uid = getattr(
                getattr(self.dataset, "ReferencedSeriesSequence")[0], "SeriesInstanceUID"
            )
        except Exception as e:
            print(f"Couldn't extract ReferencedSeriesUID: {e}")
