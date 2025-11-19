import os
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from rosamllib.dicoms import RTIMAGE


class RTImageReader:
    """
    Reader class for loading DICOM RTIMAGE (EPID/portal) objects from disk or
    an in-memory :class:`pydicom.Dataset`.

    The reader accepts either:

    - A file path to a RTIMAGE DICOM file.
    - A directory containing at least one RTIMAGE file (scanned recursively).
    - A pre-loaded :class:`pydicom.Dataset` whose Modality is ``"RTIMAGE"`` (recommended).

    The :meth:`read` method returns an instance of :class:`rosamllib.dicoms.RTIMAGE`,
    which subclasses :class:`pydicom.dataset.Dataset` and provides convenience
    helpers for geometry, BEV mapping, overlays, and RTPLAN field drawing.

    Parameters
    ----------
    rtimage_input : str, pathlib.Path, or pydicom.Dataset
        A path to an RTIMAGE file, a directory containing an RTIMAGE file,
        or an in-memory DICOM dataset.

    Examples
    --------
    Reading from a file path:

    >>> reader = RTImageReader("path/to/RTIMAGE.dcm")
    >>> rtimg = reader.read()

    Reading from a directory:

    >>> reader = RTImageReader("/path/to/folder/")
    >>> rtimg = reader.read()

    Reading from an existing dataset:

    >>> ds = pydicom.dcmread("path/to/RTIMAGE.dcm")
    >>> reader = RTImageReader(ds)
    >>> rtimg = reader.read()
    """

    def __init__(self, rtimage_input):
        self.rtimage_file_path = None
        self.rtimage_dataset = None

        if isinstance(rtimage_input, (str, Path)):
            # File path or directory
            self.rtimage_file_path = rtimage_input
        elif isinstance(rtimage_input, Dataset):
            # Pre-loaded DICOM dataset
            self.rtimage_dataset = rtimage_input
        else:
            raise ValueError(
                "rtimage_input must be either a file path (str/Path), "
                "a directory, or a pydicom.Dataset."
            )

    def read(self):
        """
        Load the RTIMAGE DICOM and return an initialized :class:`RTIMAGE` object.

        Behavior
        --------
        • If initialized with a directory, scans recursively for the first file
          whose Modality is ``'RTIMAGE'`` (using ``stop_before_pixels=True``).
        • If initialized with a file path, reads it via :func:`pydicom.dcmread`.
        • If initialized with a Dataset, uses it directly.

        Returns
        -------
        RTIMAGE
            A :class:`RTIMAGE` instance created via :meth:`RTIMAGE.from_dataset`.

        Raises
        ------
        IOError
            If no RTIMAGE file can be located when a directory is provided.
        ValueError
            If neither a file path nor dataset was supplied.
        pydicom.errors.InvalidDicomError
            If the target file is not a valid DICOM file.
        """
        if self.rtimage_file_path:
            if os.path.isdir(self.rtimage_file_path):
                rtimage_file = self._find_rtimage_in_directory(self.rtimage_file_path)
                if not rtimage_file:
                    raise IOError(f"No RTIMAGE file found in directory: {self.rtimage_file_path}")
                ds = pydicom.dcmread(rtimage_file)
            else:
                ds = pydicom.dcmread(self.rtimage_file_path)
        elif self.rtimage_dataset is not None:
            ds = self.rtimage_dataset
        else:
            raise ValueError("No RTIMAGE file path or dataset provided.")

        return RTIMAGE.from_dataset(ds)

    def _find_rtimage_in_directory(self, directory_path):
        """
        Recursively search a directory for the first valid DICOM RTIMAGE file.

        Parameters
        ----------
        directory_path : str or Path
            The directory in which to search for RTIMAGE DICOM files.

        Returns
        -------
        str or None
            The full path to the first discovered RTIMAGE file, or ``None`` if
            no valid RTIMAGE dataset was found.

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
                    if getattr(ds, "Modality", None) == "RTIMAGE":
                        return file_path
                except InvalidDicomError:
                    continue
                except Exception:
                    # Best-effort scan; unexpected errors are logged and ignored
                    pass
        return None
