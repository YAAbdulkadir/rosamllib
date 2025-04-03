import os
import tempfile
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from collections import defaultdict
from rosamllib.dicoms import DICOMImage
from rosamllib.utils import sort_by_image_position_patient
from rosamllib.constants import non_common_tags


class DICOMImageReader:

    def __init__(self, dicom_input):
        """
        Initializes the DICOMImageReader with a path to the DICOM files, a single DICOM file,
        or a list of DICOM datasets.

        Parameters
        ----------
        dicom_input : str, list of pydicom.Dataset, or pydicom.Dataset
            The path to the directory containing DICOM files, a single DICOM file, or a list of
            pydicom Dataset objects.
        """

        self.datasets = None
        self._pixel_to_physical_matrix = None
        self._physical_to_pixel_matrix = None

        if isinstance(dicom_input, list):
            if isinstance(dicom_input[0], (str, Path)):
                self.input_type = "files"
                self.file_names = [str(file) for file in dicom_input]
            else:
                self.input_type = "datasets"
                self.datasets = dicom_input
        elif isinstance(dicom_input, (str, Path)):
            dicom_input = str(dicom_input)
            if os.path.isfile(dicom_input):
                # Handle a single DICOM file
                self.input_type = "single_file"
                self.single_file = dicom_input
            else:
                self.input_type = "dir"
                self.dicom_path = dicom_input
        elif isinstance(dicom_input, Dataset):
            # Handle a single DICOM dataset
            self.input_type = "single_dataset"
            self.datasets = [dicom_input]

        else:
            raise ValueError(
                "Invalid input. Provide a DICOM file path, a directory, a list of datasets "
                "or a sinlge pydicom.Dataset."
            )

    def read(self):
        """
        Reads DICOM image files, a single file, or datasets, and loads them as a SimpleITK image.

        This method filters the files or datasets based on the specified modality and
        SeriesInstanceUID, sorts them according to their position along the imaging axis, and reads
        them into a SimpleITK image.

        Raises
        ------
        ValueError
            If multiple SeriesInstanceUIDs are found in the provided input.

        Notes
        -----
        If the input is a directory, all DICOM files in that directory will be read.
        If the input is a single file or dataset, only that image is read.
        For a list of datasets, each dataset is saved to a temporary file and sorted before being
        processed by `SimpleITK.ImageSeriesReader`.
        """
        series_file_names = []

        if self.input_type == "datasets":
            # Process the list of pydicom datasets
            return self._process_datasets()
        elif self.input_type == "dir":
            # Process the directory of DICOM files
            series_file_names = self._process_dicom_directory()
        elif self.input_type == "files":
            series_file_names = self.file_names
        elif self.input_type == "single_file":
            series_file_names = [self.single_file]
        elif self.input_type == "single_dataset":
            # Handle a single DICOM dataset
            return self._read_single_dataset(self.datasets[0])

        # Sort the file names or datasets based on ImagePositionPatient along the imaging axis
        series_file_names_sorted = sort_by_image_position_patient(series_file_names)

        # Set filenames manually for SimpleITK reader
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names_sorted)
        image = reader.Execute()
        # image_orientation_patient = dcmread(
        #     series_file_names[0], stop_before_pixels=True
        # ).ImageOrientationPatient
        # row_cosines = np.array(image_orientation_patient[:3]).astype(float)
        # col_cosines = np.array(image_orientation_patient[3:]).astype(float)
        # imaging_axis = np.cross(row_cosines, col_cosines)
        # direction = (*row_cosines[:], *col_cosines[:], *imaging_axis[:])
        # image.SetDirection(direction)

        # Set the common dicom tags by reading the first file
        single_slice = sitk.ReadImage(series_file_names_sorted[0])

        for key in single_slice.GetMetaDataKeys():
            if key not in non_common_tags:
                value = single_slice.GetMetaData(key)
                image.SetMetaData(key, value)

        return DICOMImage(image)

    def _read_single_dataset(self, dataset):
        """
        Reads and processes a single pydicom.Dataset object into a SimpleITK image.

        Parameters
        ----------
        dataset : pydicom.Dataset
            The pydicom dataset representing a single DICOM file.

        Returns
        -------
        SimpleITK.Image
            The DICOM image loaded from the dataset.

        Notes
        -----
        The dataset must have the `filename` attribute for this method to work, as it uses the
        `SimpleITK.ImageFileReader` to read the file.
        """
        reader = sitk.ImageFileReader()
        reader.SetImageIO("GDCMImageIO")
        reader.SetFileName(dataset.filename)
        image = reader.Execute()

        return DICOMImage(image)

    def _process_dicom_directory(self):
        """
        Processes a directory of DICOM files to filter by modality and SeriesInstanceUID.

        Returns
        -------
        list of str
            The filtered and sorted list of DICOM file paths.

        Raises
        ------
        ValueError
            If multiple SeriesInstanceUIDs are found.
        """
        all_files = [
            os.path.join(self.dicom_path, f)
            for f in os.listdir(self.dicom_path)
            if os.path.isfile(os.path.join(self.dicom_path, f))
        ]

        files_by_series = defaultdict(list)

        for file in all_files:
            try:
                ds = dcmread(file, stop_before_pixels=True)
                files_by_series[ds.SeriesInstanceUID].append(file)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

        if not files_by_series:
            raise ValueError("No Valid DICOM series found in the directory.")

        if len(files_by_series) > 1:
            raise ValueError("Multiple series found.")

        # Use the only available SeriesInstanceUID
        series_file_names = next(iter(files_by_series.values()))

        return series_file_names

    def _process_datasets(self):
        """
        Processes a list of pydicom.Dataset objects to filter by modality and SeriesInstanceUID.

        Converts the pydicom.Dataset objects to temporary DICOM files and loads them using
        SimpleITK.

        Returns
        -------
        SimpleITK.Image
            The SimpleITK image loaded from the datasets.

        Raises
        ------
        ValueError
            If no matching series is found in the provided datasets.

        Notes
        -----
        The datasets are temporarily saved to disk to be processed by SimpleITK, as it expects file
        paths for reading.
        """

        datasets_by_series = defaultdict(list)

        for ds in self.datasets:
            datasets_by_series[ds.SeriesInstanceUID].append(ds)

        if len(datasets_by_series) > 1:
            raise ValueError("Multiple SeriesInstanceUIDs found in the provided datasets.")

        # Use the only available SeriesInstanceUID
        filtered_datasets = next(iter(datasets_by_series.values()))

        # Create temporary files for each dataset
        temp_file_paths = []
        for ds in filtered_datasets:
            # Create a temporary file to store the dataset
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
            ds.save_as(temp_file.name)
            temp_file_paths.append(temp_file.name)

        # Sort temporary file paths by ImagePositionPatient
        temp_file_paths_sorted = sort_by_image_position_patient(temp_file_paths)

        # Load the temporary files into SimpleITK
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(temp_file_paths_sorted)
        image = reader.Execute()

        # Set the common dicom tags by reading the first file
        single_slice = sitk.ReadImage(temp_file_paths_sorted[0])

        for key in single_slice.GetMetaDataKeys():
            if key not in non_common_tags:
                value = single_slice.GetMetaData(key)
                image.SetMetaData(key, value)

        # Clean up temporary files
        for temp_file in temp_file_paths:
            os.remove(temp_file)

        return DICOMImage(image)
