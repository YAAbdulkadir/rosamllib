import pydicom
import numpy as np
import SimpleITK as sitk
from rosamllib.dicoms.dicom_image import DICOMImage


class RTDose(sitk.Image):
    """
    A class representing RTDOSE DICOM images, extending SimpleITK.Image and providing additional
    methods for DICOM metadata handling and dose grid management.

    This class allows accessing and setting DICOM metadata using dot notation, and resampling
    the dose grid to match a reference DICOM image. It also provides functionality to retrieve
    and scale dose values based on the dose grid scaling factor.

    Parameters
    ----------
    dose : SimpleITK.Image
        A SimpleITK image object representing the RTDOSE DICOM image.

    Attributes
    ----------
    _dose_grid_scaling : float or None
        The scaling factor for the dose grid, extracted from the DICOM metadata.

    Examples
    --------
    >>> dose_reader = RTDoseReader("path/to/RTDOSE.dcm")
    >>> rtdose = dose_reader.read()
    >>> print(rtdose.PatientID)  # Access metadata using dot notation
    >>> dose_array = rtdose.get_dose_array()  # Get dose array with scaling applied
    >>> rtdose.PatientID = "123456"  # Set metadata using dot notation

    See Also
    --------
    DICOMImage : The base class providing additional DICOM metadata handling functionality.
    """

    def __init__(self, dose):
        """
        Initialize the RTDose object.

        Parameters
        ----------
        dose : SimpleITK.Image
            The SimpleITK image representing the RTDOSE DICOM image.
        """
        super().__init__(dose)
        self._dose_grid_scaling = None

    def __getattr__(self, attr):
        """
        Allows attribute access for DICOM metadata via dot notation.

        Parameters
        ----------
        attr : str
            The attribute being accessed (e.g., 'PatientID').

        Returns
        -------
        str
            The value of the requested DICOM metadata tag.

        Raises
        ------
        AttributeError
            If the attribute is not a valid DICOM keyword or metadata is not found.
        """
        # Try to convert the attribute (keyword) to a DICOM tag
        try:
            tag = pydicom.datadict.tag_for_keyword(attr)
            if tag is None:
                raise ValueError("Not a DICOM metadata.")

            # Convert the integer tag into a pydicom Tag object (this separates group and element)
            tag_obj = pydicom.tag.Tag(tag)

            # Format the tag into the "group|element" format that SimpleITK expects
            tag_str = f"{tag_obj.group:04X}|{tag_obj.element:04X}"

            tag_str = tag_str.lower()
            if self.HasMetaDataKey(tag_str):
                # Fetch the metadata using the SimpleITK GetMetaData method
                return self.GetMetaData(tag_str)
            else:
                raise AttributeError(f"'RTDose' object has no attribute '{attr}'")

        except ValueError:
            # not a valid DICOM metadata, check if parent has the method
            try:
                return super().__getattr__(attr)
            except Exception:
                raise AttributeError(f"'RTDose' object has no attribute '{attr}'")

        except Exception:
            raise AttributeError(f"'RTDose' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        """
        Allows setting DICOM metadata attributes via dot notation.

        Parameters
        ----------
        attr : str
            The attribute being set (e.g., 'PatientID').
        value : str
            The value to be assigned to the attribute.

        Raises
        ------
        AttributeError
            If the attribute is not a valid DICOM metadata keyword.
        """
        try:
            tag = pydicom.datadict.tag_for_keyword(attr)
            if tag is not None:
                tag_obj = pydicom.tag.Tag(tag)
                tag_str = f"{tag_obj.group:04X}|{tag_obj.element:04X}"
                tag_str = tag_str.lower()
                self.SetMetaData(tag_str, str(value))
            else:
                super().__setattr__(attr, value)
        except Exception:
            super().__setattr__(attr, value)

    def __dir__(self):
        """
        Returns a list of attributes and DICOM metadata keywords.

        This method dynamically combines standard attributes and the available DICOM metadata
        keywords, providing a similar behavior to pydicom.Dataset's `dir()` method.

        Returns
        -------
        list
            A combined list of attributes and DICOM metadata keywords.
        """
        default_dir = super().__dir__()
        metadata_keys = self.GetMetaDataKeys()
        dicom_keywords = []
        for key in metadata_keys:
            try:
                group, element = key.split("|")
                tag = pydicom.tag.Tag(int(group, 16), int(element, 16))
                keyword = pydicom.datadict.keyword_for_tag(tag)
                if keyword:
                    dicom_keywords.append(keyword)
            except ValueError:
                # not a dicom tag
                pass

        # combine the default attributes with the DICOM keywords
        return default_dir + dicom_keywords

    def dir(self):
        """
        Custom `dir` method to return a list of available attributes and DICOM metadata keywords.

        This method is similar to `__dir__()` and provides a dynamic list of attributes and
        metadata for easier access.

        Returns
        -------
        list of str
            List of all attributes, including DICOM metadata keywords.
        """
        return self.__dir__()

    def get_dose_array(self):
        """
        Retrieves the dose array with the appropriate scaling factor applied, if present.

        This method returns the dose array stored in the DICOM image, with the dose grid scaling
        factor applied if the metadata key is present.

        Returns
        -------
        numpy.ndarray
            The dose array with scaling applied, or the unscaled array if no scaling is found.
        """
        scaling_factor = self.dose_grid_scaling
        dose_array = np.squeeze(sitk.GetArrayFromImage(self)).astype(np.float64)
        if self.dose_grid_scaling is not None:
            return dose_array * scaling_factor
        else:
            return dose_array

    @property
    def dose_grid_scaling(self):
        """
        Property to retrieve the dose grid scaling factor from the DICOM metadata.

        This property lazily loads the scaling factor from the DICOM metadata the first time it
        is accessed, and caches the value for future use.

        Returns
        -------
        float or None
            The dose grid scaling factor, or None if not available.
        """
        if self._dose_grid_scaling is None:
            if self.HasMetaDataKey("3004|000e"):
                self._dose_grid_scaling = float(self.GetMetaData("3004|000e"))
        return self._dose_grid_scaling

    def resample_dose_to_image_grid(self, referenced_image):
        """
        Resamples the dose grid to match the grid of a referenced DICOM image.

        This method aligns the dose grid with the image grid of a referenced DICOM image by
        resampling it. This is particularly useful when visualizing or analyzing dose data
        in the context of a DICOM image. The dose grid scaling is preserved during the
        resampling process.

        Parameters
        ----------
        referenced_image : DICOMImage
            The DICOM image to which the dose grid will be resampled.

        Returns
        -------
        RTDose
            A new `RTDose` object containing the resampled dose image.

        Raises
        ------
        TypeError
            If the referenced image is not of type `DICOMImage`.

        Notes
        -----
        - The `DoseGridScaling` value is preserved in the resampled dose.
        - Resampling can be useful when the CT image and the RTDOSE file grids are not aligned.

        See Also
        --------
        DICOMImage : The DICOM image class to which the dose is aligned.
        """
        if not isinstance(referenced_image, DICOMImage):
            raise TypeError("The referenced image must be a `DICOMImage` object.")

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(referenced_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(referenced_image.GetSpacing())
        resampler.SetOutputOrigin(referenced_image.GetOrigin())
        resampler.SetOutputDirection(referenced_image.GetDirection())
        resampler.SetSize(referenced_image.GetSize())
        resampler.SetDefaultPixelValue(
            0
        )  # Set default value for regions outside the original dose grid

        resampled_dose = resampler.Execute(self)
        # resampled_dose.SetMetaData("3004|000e") = self.dose_grid_scaling
        resampled_dose = RTDose(resampled_dose)
        resampled_dose.DoseGridScaling = self.dose_grid_scaling
        resampled_dose.DoseUnits = self.DoseUnits

        return resampled_dose
