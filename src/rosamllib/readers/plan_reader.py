import os
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from rosamllib.dicoms import RTPlan


class RTPlanReader:
    """
    Reader class for loading DICOM RTPLAN (treatment plan) objects from disk or
    an in-memory :class:`pydicom.Dataset`.

    The reader accepts either:

    - A file path to an RTPLAN DICOM file.
    - A directory containing at least one RTPLAN file (scanned recursively).
    - A pre-loaded :class:`pydicom.Dataset` whose Modality is ``"RTPLAN"`` (recommended).

    The :meth:`read` method returns an instance of :class:`rosamllib.dicoms.RTPLAN`,
    which subclasses :class:`pydicom.dataset.Dataset` and is intended to provide
    convenience helpers for beam geometry, control points, dose prescriptions,
    and referenced images/structures.

    Parameters
    ----------
    rtplan_input : str, pathlib.Path, or pydicom.Dataset
        A path to an RTPLAN file, a directory containing an RTPLAN file,
        or an in-memory DICOM dataset.

    Examples
    --------
    Reading from a file path:

    >>> reader = RTPlanReader("path/to/RTPLAN.dcm")
    >>> rtplan = reader.read()

    Reading from a directory:

    >>> reader = RTPlanReader("/path/to/folder/")
    >>> rtplan = reader.read()

    Reading from an existing dataset:

    >>> ds = pydicom.dcmread("path/to/RTPLAN.dcm")
    >>> reader = RTPlanReader(ds)
    >>> rtplan = reader.read()
    """

    def __init__(self, rtplan_input):
        self.rtplan_file_path = None
        self.rtplan_dataset = None

        if isinstance(rtplan_input, (str, Path)):
            # File path or directory
            self.rtplan_file_path = rtplan_input
        elif isinstance(rtplan_input, Dataset):
            # Pre-loaded DICOM dataset
            self.rtplan_dataset = rtplan_input
        else:
            raise ValueError(
                "rtplan_input must be either a file path (str/Path), "
                "a directory, or a pydicom.Dataset."
            )

    def read(self) -> RTPlan:
        """
        Load the RTPLAN DICOM and return an initialized :class:`RTPlan` object.

        Behavior
        --------
        • If initialized with a directory, scans recursively for the first file
          whose Modality is ``'RTPLAN'`` (using ``stop_before_pixels=True``).
        • If initialized with a file path, reads it via :func:`pydicom.dcmread`.
        • If initialized with a Dataset, uses it directly.

        Returns
        -------
        RTPlan
            A :class:`RTPlan` instance created via :meth:`RTPlan.from_dataset`.

        Raises
        ------
        IOError
            If no RTPLAN file can be located when a directory is provided.
        ValueError
            If neither a file path nor dataset was supplied.
        pydicom.errors.InvalidDicomError
            If the target file is not a valid DICOM file.
        """
        if self.rtplan_file_path:
            if os.path.isdir(self.rtplan_file_path):
                rtplan_file = self._find_rtplan_in_directory(self.rtplan_file_path)
                if not rtplan_file:
                    raise IOError(f"No RTPLAN file found in directory: {self.rtplan_file_path}")
                ds = pydicom.dcmread(rtplan_file)
            else:
                ds = pydicom.dcmread(self.rtplan_file_path)
        elif self.rtplan_dataset is not None:
            ds = self.rtplan_dataset
        else:
            raise ValueError("No RTPLAN file path or dataset provided.")

        return RTPlan.from_dataset(ds)

    def _find_rtplan_in_directory(self, directory_path):
        """
        Recursively search a directory for the first valid DICOM RTPLAN file.

        Parameters
        ----------
        directory_path : str or Path
            The directory in which to search for RTPLAN DICOM files.

        Returns
        -------
        str or None
            The full path to the first discovered RTPLAN file, or ``None`` if
            no valid RTPLAN dataset was found.

        Notes
        -----
        • Uses ``stop_before_pixels=True`` to keep directory scans fast.
        • Any unreadable or non-DICOM files are silently skipped.
        """
        directory_path = str(directory_path)
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if getattr(ds, "Modality", None) == "RTPLAN":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception:
                    # Best-effort scan; unexpected errors are ignored
                    pass
        return None
