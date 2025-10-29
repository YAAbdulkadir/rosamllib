import pydicom
from pydicom.dataset import Dataset


class RTPlan:
    def __init__(self, plan_dataset: Dataset):
        """
        Initialize the RTPlan object with a DICOM RTPLAN dataset.

        Parameters
        ----------
        plan_dataset : pydicom.Dataset
            The DICOM RTPLAN dataset representing the radiation treatment plan.
        """
        self.plan_dataset = plan_dataset

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
        if attr in self.dir():
            return getattr(self.plan_dataset, attr)

        # Fallback to the parent class method if not a valid DICOM attribute
        try:
            return super().__getattr__(attr)
        except Exception:
            raise AttributeError(f"'RTPlan' object has no attribute '{attr}'")

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
            If the attribute is not a valid DICOM keyword or if setting fails.
        """
        if attr == "plan_dataset":
            super().__setattr__(attr, value)
            return

        try:
            # Try to find the corresponding DICOM tag for the keyword
            tag = pydicom.datadict.tag_for_keyword(attr)
            if tag is not None:
                tag_obj = pydicom.tag.Tag(tag)
                tag_str = f"{tag_obj.group:04X}|{tag_obj.element:04X}"
                self.plan_dataset[tag_str].value = value
            else:
                # If it's not a valid DICOM tag, set it as a regular attribute
                super().__setattr__(attr, value)
        except Exception:
            # If anything goes wrong, fall back to setting the attribute normally
            super().__setattr__(attr, value)

    def __dir__(self):
        """
        Returns a list of attributes and DICOM metadata keywords.

        Returns
        -------
        list
            A combined list of attributes and DICOM metadata keywords.
        """
        default_dir = super().__dir__()
        dicom_keywords = [
            pydicom.datadict.keyword_for_tag(tag) for tag in self.plan_dataset.keys()
        ]
        dicom_keywords = [keyword for keyword in dicom_keywords if keyword]  # Filter out None

        # Combine the default attributes with the DICOM keywords
        return default_dir + dicom_keywords

    def dir(self):
        """
        Custom dir method to return a list of available attributes and DICOM metadata keywords.

        Returns
        -------
        list of str
            List of all attributes, including DICOM metadata keywords.
        """
        return self.__dir__()

    def __contains__(self, key):
        return key in self.plan_dataset

    def __getitem__(self, key):
        return self.plan_dataset[key]

    def keys(self):
        return self.plan_dataset.keys()
