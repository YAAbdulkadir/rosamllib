import os
import pydicom
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from rosamllib.dicoms import REG


class REGReader:
    """
    Reader class for loading DICOM Registration (REG) objects from disk or an
    already-loaded :class:`pydicom.Dataset`.

    The reader accepts either:

    - A file path to a REG DICOM file.
    - A directory containing at least one REG file.
    - A pre-loaded :class:`pydicom.Dataset` whose modality is ``"REG"``.

    The :meth:`read` method returns an instance of :class:`rosamllib.dicoms.REG`,
    which is a subclass of :class:`pydicom.Dataset` with additional
    registration-specific utilities for parsing rigid/deformable registration
    information, transformation matrices, and referenced image metadata.

    Parameters
    ----------
    reg_input : str, pathlib.Path, or pydicom.Dataset
        A path to a REG file, a directory containing a REG file, or an
        in-memory DICOM dataset.

    Examples
    --------
    Reading from a file path:

    >>> reader = REGReader("path/to/REG.dcm")
    >>> reg = reader.read()

    Reading from a directory:

    >>> reader = REGReader("/path/to/folder/")
    >>> reg = reader.read()

    Reading from an existing dataset:

    >>> ds = pydicom.dcmread("path/to/REG.dcm")
    >>> reader = REGReader(ds)
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
        Load the REG file or dataset and return an initialized :class:`REG` object.

        If ``reg_input`` was a directory, the reader searches recursively for the
        first valid file with DICOM Modality ``REG``.
        If ``reg_input`` was already a :class:`pydicom.Dataset`, it is used as is.

        Returns
        -------
        REG
            A :class:`REG` instance created via :meth:`REG.from_dataset`,
            containing all DICOM elements along with parsed registration
            metadata and transformation structures.

        Raises
        ------
        IOError
            If no REG file can be located when a directory is provided.
        ValueError
            If neither a file path nor dataset was supplied.
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

        return REG.from_dataset(self.reg_dataset)

    def _find_reg_in_directory(self, directory_path):
        """
        Recursively search a directory for the first valid DICOM REG file.

        Parameters
        ----------
        directory_path : str or Path
            The directory in which to search for REG DICOM files.

        Returns
        -------
        str or None
            The full path to the first discovered REG file, or ``None`` if
            no valid REG dataset was found.

        Notes
        -----
        Only minimal DICOM attributes are read using ``stop_before_pixels=True``,
        which makes scanning large directories significantly faster.

        Any unreadable or non-DICOM files are silently skipped.

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
