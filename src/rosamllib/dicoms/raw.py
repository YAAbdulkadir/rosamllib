from typing import List, Optional

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian


class RAW(Dataset):
    """
    DICOM RAW dataset with helpers for extracting embedded datasets and
    referenced series information.

    This class subclasses :class:`pydicom.dataset.Dataset` and adds:

    - Extraction of embedded datasets from the private
      MIMSoftwareSessionMetaSeq (0013,2050) tag.
    - Attachment of minimal file meta to each embedded dataset.
    - Extraction of a referenced series UID from ``ReferencedSeriesSequence``
      when present.

    Typical usage
    -------------
    >>> from pydicom import dcmread
    >>> from rosamllib.dicoms import RAW
    >>> ds = dcmread("path/to/dicom_raw.dcm")
    >>> raw = RAW.from_dataset(ds)
    >>> embedded = raw.get_embedded_datasets()
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a RAW instance.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to :class:`pydicom.dataset.Dataset`.
        """
        super().__init__(*args, **kwargs)
        self.embedded_datasets: List[Dataset] = []
        self.referenced_series_uid: Optional[str] = None

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "RAW":
        """
        Create a :class:`RAW` instance from an existing DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            A DICOM dataset representing a RAW object.

        Returns
        -------
        RAW
            A new :class:`RAW` instance with embedded datasets extracted and
            referenced series UID populated when available.

        Raises
        ------
        ValueError
            If the MIMSoftwareSessionMetaSeq (0013,2050) tag is not present.
        """
        raw = cls()
        raw.update(ds)
        if hasattr(ds, "file_meta"):
            raw.file_meta = ds.file_meta

        raw.extract_embedded_datasets()
        raw.extract_referenced_series_uid()
        return raw

    def extract_embedded_datasets(self):
        """
        Extract all embedded datasets from the MIMSoftwareSessionMetaSeq tag.

        Looks for the private tag (0013,2050) and iterates through its items,
        treating each as an embedded :class:`pydicom.Dataset`. For each item,
        a minimal :class:`FileMetaDataset` is attached so it can behave like
        a top-level dataset if needed.

        Raises
        ------
        ValueError
            If the RAW dataset is empty or does not contain the
            MIMSoftwareSessionMetaSeq (0013,2050) tag.
        """
        if not len(self):
            raise ValueError("RAW dataset is empty.")

        # Check for MIMSoftwareSessionMetaSeq tag
        if (0x0013, 0x2050) in self:
            mim_seq = self[(0x0013, 0x2050)]

            # Iterate over the items in MIMSoftwareSessionMetaSeq
            for item in mim_seq:
                if isinstance(item, Dataset):
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = getattr(item, "SOPClassUID", None)
                    file_meta.MediaStorageSOPInstanceUID = getattr(item, "SOPInstanceUID", None)
                    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                    item.file_meta = file_meta
                    self.embedded_datasets.append(item)
        else:
            raise ValueError(
                "MIMSoftwareSessionMetaSeq (0013, 2050) tag not found in RAW dataset."
            )

    def get_embedded_datasets(self) -> List[Dataset]:
        """
        Return the list of embedded datasets extracted from (0013,2050).

        Returns
        -------
        list of pydicom.Dataset
            List of embedded datasets.

        Raises
        ------
        ValueError
            If embedded datasets have not been extracted yet.
        """
        if not self.embedded_datasets:
            raise ValueError("No embedded datasets found. Call extract_embedded_datasets() first.")
        return self.embedded_datasets

    def extract_referenced_series_uid(self) -> None:
        """
        Extract the referenced SeriesInstanceUID from ReferencedSeriesSequence.

        This method looks for ``ReferencedSeriesSequence`` in the RAW dataset and,
        if present, stores the first ``SeriesInstanceUID`` found in
        :attr:`referenced_series_uid`. If the sequence is missing or malformed,
        the attribute is left as ``None``.
        """
        try:
            seq = getattr(self, "ReferencedSeriesSequence", None)
            if not seq:
                return
            first = seq[0]
            self.referenced_series_uid = getattr(first, "SeriesInstanceUID", None)
        except Exception:
            # Fail silently; referenced_series_uid remains None
            self.referenced_series_uid = None

    def __repr__(self) -> str:
        """
        Return a concise string representation of the RAW object.

        Includes number of embedded datasets and referenced series UID if available.
        """
        num_embedded = len(self.embedded_datasets)
        series_uid = self.referenced_series_uid or "N/A"
        return f"RAW(num_embedded={num_embedded}, referenced_series_uid={series_uid!r})"
