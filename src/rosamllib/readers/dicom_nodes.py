class DatasetNode:
    """
    Represents a dataset or collection of patients in the DICOM hierarchy.

    This class serves as a container for all patients within a particular dataset,
    grouping them under a single node. This can represent an institution, study group,
    or any higher-level categorization above individual patients.

    Parameters
    ----------
    dataset_id : str
        The unique identifier for the dataset or collection.
    dataset_name : str, optional
        The name or description of the dataset (e.g., institution name). Default is None.

    Attributes
    ----------
    dataset_id : str
        The unique identifier for the dataset.
    dataset_name : str or None
        The name or description of the dataset.
    patients : dict of str to PatientNode
        A dictionary containing `PatientNode` objects associated with this dataset.
        Keys are PatientIDs (str), and values are `PatientNode` instances.

    Methods
    -------
    add_patient(patient_node)
        Adds a `PatientNode` to the dataset.
    get_patient(patient_id)
        Retrieves a `PatientNode` from the dataset by PatientID.

    Examples
    --------
    >>> dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    >>> dataset.dataset_id
    'Institution_123'
    >>> dataset.dataset_name
    'XYZ Medical Center'
    >>> dataset.patients
    {}

    >>> dataset.add_patient(PatientNode(patient_id="12345", patient_name="John Doe"))
    >>> patient = dataset.get_patient("12345")
    >>> print(patient.PatientID, patient.PatientName)
    '12345', 'John Doe'
    """

    def __init__(self, dataset_id, dataset_name=None):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.patients = {}  # Key: PatientID

    def add_patient(self, patient_node):
        """
        Adds a PatientNode to the dataset.

        Parameters
        ----------
        patient_node : PatientNode
            The `PatientNode` to add to this dataset.

        Examples
        --------
        >>> dataset.add_patient(patient_node)
        """
        self.patients[patient_node.PatientID] = patient_node

    def get_patient(self, patient_id):
        """
        Retrieves a PatientNode from the dataset based on the provided PatientID.

        Parameters
        ----------
        patient_id : str
            The unique identifier for the patient (PatientID) to retrieve.

        Returns
        -------
        PatientNode or None
            The `PatientNode` instance associated with the given PatientID if found,
            otherwise None.

        Examples
        --------
        >>> patient = dataset.get_patient("12345")
        >>> if patient:
        ...     print(patient.PatientID, patient.PatientName)
        ... else:
        ...     print("Patient not found.")
        """
        return self.patients.get(patient_id)

    def __len__(self):
        """
        Returns the number of patients in the dataset.

        Returns
        -------
        int
            The total number of `PatientNode` instances in the dataset.
        """
        return len(self.patients)

    def __iter__(self):
        """
        Iterates over all `PatientNode` objects in the dataset.

        Yields
        ------
        PatientNode
            Each patient node in the dataset.
        """
        return iter(self.patients.values())

    def __repr__(self):
        """
        Returns a string representation of the `DatasetNode`, including the dataset ID,
        name, and number of patients.

        Returns
        -------
        str
            A string representation of the `DatasetNode` object.
        """
        return (
            f"DatasetNode(dataset_id={self.dataset_id}, "
            f"dataset_name={self.dataset_name}, "
            f"NumPatients={len(self)})"
        )


class PatientNode:
    """
    Represents a patient in the DICOM hierarchy.

    This class serves as a container for all studies associated with a patient.
    It stores the patient's unique identifier and name, along with a dictionary
    of `StudyNode` objects that represent the studies within the patient.

    Parameters
    ----------
    patient_id : str
        The unique identifier for the patient (PatientID).
    patient_name : str, optional
        The name of the patient (PatientName). Default is None.
    parent_dataset : DatasetNode, optional
        The `DatasetNode` this patient belongs to. Default is None.

    Attributes
    ----------
    PatientID : str
        The unique identifier for the patient.
    PatientName : str or None
        The name of the patient, if available.
    studies : dict of str to StudyNode
        A dictionary containing `StudyNode` objects associated with the patient.
        Keys are `StudyInstanceUID`s (str), and values are `StudyNode` instances.
    parent_dataset : DatasetNode or None
        Reference to the `DatasetNode` that this patient belongs to.

    Methods
    -------
    add_study(study_node)
        Adds a `StudyNode` to this patient's studies.
    get_study(study_uid)
        Retrieves a `StudyNode` by its `StudyInstanceUID`.

    Examples
    --------
    >>> patient = PatientNode(
                    patient_id="12345",
                    patient_name="John Doe",
                    parent_dataset=dataset
                    )
    >>> patient.PatientID
    '12345'
    >>> patient.PatientName
    'John Doe'
    >>> patient.studies
    {}
    >>> patient.add_study(StudyNode(study_uid="1.2.3.4.5", study_description="CT Chest"))
    >>> study = patient.get_study("1.2.3.4.5")
    >>> study.StudyInstanceUID
    '1.2.3.4.5'
    """

    def __init__(self, patient_id, patient_name=None, parent_dataset=None):
        self.PatientID = patient_id
        self.PatientName = patient_name
        self.studies = {}  # Key: StudyInstanceUID
        self.parent_dataset = parent_dataset

    def add_study(self, study_node):
        """
        Adds a StudyNode to the patient's studies.

        Parameters
        ----------
        study_node : StudyNode
            The `StudyNode` object to add to this patient.

        Examples
        --------
        >>> study = StudyNode(study_uid="1.2.3.4.5", study_description="CT Chest")
        >>> patient.add_study(study)
        """
        self.studies[study_node.StudyInstanceUID] = study_node
        study_node.parent_patient = self

    def get_study(self, study_uid):
        """
        Retrieves a StudyNode by its StudyInstanceUID.

        Parameters
        ----------
        study_uid : str
            The unique identifier of the study to retrieve.

        Returns
        -------
        StudyNode or None
            The `StudyNode` instance if found, or None if not present.

        Examples
        --------
        >>> study = patient.get_study("1.2.3.4.5")
        >>> print(study.StudyInstanceUID)
        '1.2.3.4.5'
        """
        return self.studies.get(study_uid)

    def __getattr__(self, name):
        """
        Delegates attribute access to the parent_dataset if the attribute
        is not found in the PatientNode itself.

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The value of the requested attribute if it exists, or raises AttributeError.

        Raises
        ------
        AttributeError
            If the attribute does not exist in either PatientNode or parent_dataset.
        """
        if self.parent_dataset is not None:
            return getattr(self.parent_dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        """
        Returns the number of studies associated with the patient.

        Returns
        -------
        int
            The total number of `StudyNode` instances in the patient's studies.
        """
        return len(self.studies)

    def __iter__(self):
        """
        Iterates over all `StudyNode` objects associated with the patient.

        Yields
        ------
        StudyNode
            Each study node associated with the patient.
        """
        return iter(self.studies.values())

    def __repr__(self):
        """
        Returns a string representation of the `PatientNode`, including the PatientID,
        PatientName, and the number of studies.

        Returns
        -------
        str
            A string representation of the `PatientNode` object.
        """
        return (
            f"PatientNode(PatientID={self.PatientID}, "
            f"PatientName={self.PatientName}, "
            f"NumStudies={len(self)})"
        )


class StudyNode:
    """
    Represents a study in the DICOM hierarchy.

    This class serves as a container for all the series associated with a study.
    It stores the study's unique identifier and description, along with a dictionary
    of `SeriesNode` objects that represent the series within the study. Each study is
    also linked to a parent patient, which can be accessed through the `parent_patient`
    attribute.

    Parameters
    ----------
    study_uid : str
        The unique identifier for the study (StudyInstanceUID).
    study_description : str, optional
        A description of the study (StudyDescription). Default is None.
    parent_patient : PatientNode, optional
        The `PatientNode` instance associated with this study. Default is None.

    Attributes
    ----------
    StudyInstanceUID : str
        The unique identifier for the study.
    StudyDescription : str or None
        The description of the study, if available.
    series : dict of str to SeriesNode
        A dictionary containing `SeriesNode` objects associated with the study.
        Keys are `SeriesInstanceUID`s (str), and values are `SeriesNode` instances.
    parent_patient : PatientNode or None
        The `PatientNode` instance that this study is associated with, providing access
        to the parent patient's data.

    Methods
    -------
    add_series(series_node)
        Adds a `SeriesNode` to this study's series.
    get_series(series_uid)
        Retrieves a `SeriesNode` by its `SeriesInstanceUID`.

    Examples
    --------
    >>> study = StudyNode(
                    study_uid='1.2.840.113619.2.55.3.604688654.783.1590531004.467',
                    study_description='CT Chest'
                    )
    >>> study.StudyInstanceUID
    '1.2.840.113619.2.55.3.604688654.783.1590531004.467'
    >>> study.StudyDescription
    'CT Chest'
    >>> study.series
    {}
    >>> study.parent_patient
    None
    >>> study.add_series(SeriesNode(series_uid="1.2.840.113619.2.55.4"))
    >>> series = study.get_series("1.2.840.113619.2.55.4")
    >>> series.SeriesInstanceUID
    '1.2.840.113619.2.55.4'
    """

    def __init__(self, study_uid, study_description=None, parent_patient=None):
        """
        Initialize a StudyNode with the given study instance UID, optional study description,
        and optional parent patient.
        """
        self.StudyInstanceUID = study_uid
        self.StudyDescription = study_description
        self.series = {}  # Key: SeriesInstanceUID
        self.parent_patient = parent_patient

    def add_series(self, series_node):
        """
        Adds a SeriesNode to the study's series.

        Parameters
        ----------
        series_node : SeriesNode
            The `SeriesNode` object to add to this study.

        Examples
        --------
        >>> series = SeriesNode(series_uid="1.2.840.113619.2.55.4")
        >>> study.add_series(series)
        """
        self.series[series_node.SeriesInstanceUID] = series_node
        series_node.parent_study = self

    def get_series(self, series_uid):
        """
        Retrieves a SeriesNode by its SeriesInstanceUID.

        Parameters
        ----------
        series_uid : str
            The unique identifier of the series to retrieve.

        Returns
        -------
        SeriesNode or None
            The `SeriesNode` instance if found, or None if not present.

        Examples
        --------
        >>> series = study.get_series("1.2.840.113619.2.55.4")
        >>> print(series.SeriesInstanceUID)
        '1.2.840.113619.2.55.4'
        """
        return self.series.get(series_uid)

    def __getattr__(self, name):
        """
        Delegates attribute access to the parent_patient if the attribute
        is not found in the StudyNode itself.

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The value of the requested attribute if it exists, or raises AttributeError.

        Raises
        ------
        AttributeError
            If the attribute does not exist in either StudyNode or parent_patient.
        """
        if self.parent_patient is not None:
            return getattr(self.parent_patient, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        """
        Returns the number of series associated with the study.

        Returns
        -------
        int
            The total number of `SeriesNode` instances in the study.
        """
        return len(self.series)

    def __iter__(self):
        """
        Iterates over all `SeriesNode` objects associated with the study.

        Yields
        ------
        SeriesNode
            Each series node associated with the study.
        """
        return iter(self.series.values())

    def __repr__(self):
        """
        Returns a string representation of the `StudyNode`, including the StudyInstanceUID,
        StudyDescription, and the number of series.

        Returns
        -------
        str
            A string representation of the `StudyNode` object.
        """
        return (
            f"StudyNode(StudyInstanceUID={self.StudyInstanceUID}, "
            f"StudyDescription={self.StudyDescription}, "
            f"NumSeries={len(self)})"
        )


class SeriesNode:
    """
    Represents a series in the DICOM hierarchy.

    This class serves as a container for all the instances associated with a series.
    It stores metadata related to the series and provides methods to interact with the instances.
    Each series is also linked to a parent study, which can be accessed through the `parent_study`
    attribute.

    Parameters
    ----------
    series_uid : str
        The unique identifier for the DICOM series (SeriesInstanceUID).
    parent_study : StudyNode, optional
        The `StudyNode` instance associated with this series. Default is None.

    Attributes
    ----------
    SeriesInstanceUID : str
        The unique identifier for the series.
    Modality : str or None
        The modality of the series (e.g., 'CT', 'MR').
    SeriesDescription : str or None
        A description of the series, if available.
    FrameOfReferenceUID : str or None
        The Frame of Reference UID for the series.
    SOPInstances : list of str
        A list of SOP Instance UIDs associated with the series.
    instances : dict of str to InstanceNode
        A dictionary of `InstanceNode` objects in the series.
        Keys are SOPInstanceUIDs (str), values are `InstanceNode` instances.
    instance_paths : list of str
        A list of file paths to the DICOM instances.
    referencing_series : list of SeriesNode
        A list of `SeriesNode` objects that reference this series.
    referenced_series : list of SeriesNode
        A list of `SeriesNode` objects referenced by this series.
    referenced_sids : list of str
        SeriesInstanceUIDs referenced by this series.
    referencing_sids : list of str
        SeriesInstanceUIDs that reference this series.
    frame_of_reference_registered : list of SeriesNode
        A list of other `SeriesNode` objects registered to the same frame of reference.
    is_embedded_in_raw : bool
        Indicates whether the series is embedded within a RAW series.
    raw_series_reference : SeriesNode or None
        Reference to the RAW series in which this series is embedded.
    parent_study : StudyNode or None
        The `StudyNode` that this series is associated with, providing access to the parent study.

    Methods
    -------
    add_instance(instance)
        Adds an `InstanceNode` to the series.
    get_instance(sop_instance_uid)
        Retrieves an `InstanceNode` by its SOPInstanceUID.

    Examples
    --------
    >>> series = SeriesNode("1.2.840.113619.2.55.3")
    >>> series.SeriesInstanceUID
    '1.2.840.113619.2.55.3'
    >>> len(series)
    0
    >>> series.add_instance(InstanceNode("1.2.3.4.5.6.7", "/path/to/file.dcm"))
    >>> len(series)
    1
    >>> instance = series.get_instance("1.2.3.4.5.6.7")
    >>> instance.sop_instance_uid
    '1.2.3.4.5.6.7'
    """

    def __init__(self, series_uid, parent_study=None):
        """
        Initializes the Series object with the given SeriesInstanceUID and sets
        the default values for the other attributes.

        Parameters
        ----------
        series_uid : str
            The unique identifier for the DICOM series (SeriesInstanceUID).
        parent_study : StudyNode, optional
            The `StudyNode` instance associated with this series. Default is None.
        """
        self.SeriesInstanceUID = series_uid
        self.Modality = None
        self.SeriesDescription = None
        self.FrameOfReferenceUID = None
        self.SOPInstances = []
        self.instances = {}  # Key: SOPInstanceUID
        self.instance_paths = []
        self.referencing_series = []
        self.referenced_series = []
        self.referenced_sids = []
        self.referencing_sids = []
        self.frame_of_reference_registered = []
        self.is_embedded_in_raw = False
        self.raw_series_reference = None
        self.parent_study = parent_study

    def add_instance(self, instance):
        """
        Adds an `InstanceNode` to the series.

        Parameters
        ----------
        instance : InstanceNode
            The `InstanceNode` object to be added to the series.
        """
        self.SOPInstances.append(instance.SOPInstanceUID)
        self.instances[instance.SOPInstanceUID] = instance
        self.instance_paths.append(instance.FilePath)

    def get_instance(self, sop_instance_uid):
        """
        Retrieves an InstanceNode by its SOPInstanceUID.

        Parameters
        ----------
        sop_instance_uid : str
            The unique identifier of the instance to retrieve.

        Returns
        -------
        InstanceNode or None
            The `InstanceNode` instance if found, or None if not present.

        Examples
        --------
        >>> instance = series.get_instance("1.2.3.4.5.6.7")
        >>> print(instance.sop_instance_uid)
        '1.2.3.4.5.6.7'
        """
        return self.instances.get(sop_instance_uid)

    def __getattr__(self, name):
        """
        Delegates attribute access to the parent study. If not found, the parent study's
        `__getattr__` will check its parent (patient).

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The value of the requested attribute if it exists, or raises AttributeError.

        Raises
        ------
        AttributeError
            If the attribute does not exist in this SeriesNode, the parent study, or
            the parent patient.
        """
        if self.parent_study is not None:
            return getattr(self.parent_study, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        """
        Returns the number of instances (files) in the series.

        Returns
        -------
        int
            The number of instances in the series.
        """
        return len(self.instances)

    def __iter__(self):
        """
        Iterates over all `InstanceNode` objects in the series.

        Yields
        ------
        InstanceNode
            Each instance in the series.
        """
        return iter(self.instances.values())

    def __repr__(self):
        """
        Returns a string representation of the `SeriesNode`, including its UID, modality, and
        number of instances, to provide a quick summary in debug outputs.

        Returns
        -------
        str
            A string representation of the `SeriesNode` object.
        """
        return (
            f"SeriesNode(SeriesInstanceUID='{self.SeriesInstanceUID}', "
            f"Modality='{self.Modality}', "
            f"SeriesDescription='{self.SeriesDescription}', "
            f"NumInstances={len(self)})"
        )


class InstanceNode:
    """
    Represents a DICOM instance (SOP instance) in the DICOM hierarchy.

    This class stores metadata and relationships associated with a DICOM instance,
    such as references to other instances or series.

    Parameters
    ----------
    SOPInstanceUID : str
        The unique identifier for the DICOM instance (SOPInstanceUID).
    FilePath : str
        The file path to the DICOM file.
    modality : str, optional
        The modality of the instance (e.g., 'CT', 'MR'). Default is None.
    parent_series : SeriesNode, optional
        The `SeriesNode` this instance belongs to. Default is None.

    Attributes
    ----------
    SOPInstanceUID : str
        The unique identifier for the DICOM instance.
    FilePath : str
        The file path to the DICOM file.
    Modality : str or None
        The modality of the instance.
    references : list
        A list of references from this instance to other instances or series.
    referenced_sop_instance_uids : list of str
        List of SOPInstanceUIDs referenced by this instance.
    referenced_sids : list of str
        List of SeriesInstanceUIDs referenced by this instance.
    referenced_series : list of SeriesNode
        A list of `SeriesNode` objects referenced by this instance.
    other_referenced_sids : list of str
        A list of additional SeriesInstanceUIDs referenced by this instance,
        for cases with multiple references (e.g., moving image in registration).
    other_referenced_series : list of SeriesNode
        A list of additional `SeriesNode` objects referenced by this instance.
    referenced_instances : list of InstanceNode
        List of `InstanceNode` objects referenced by this instance.
    referencing_instances : list of InstanceNode
        List of `InstanceNode` objects that reference this instance.
    parent_series : SeriesNode or None
        The `SeriesNode` that this instance belongs to.

    Examples
    --------
    >>> instance = InstanceNode("1.2.3.4.5.6.7", "/path/to/file.dcm", modality="CT")
    >>> instance.SOPInstanceUID
    '1.2.3.4.5.6.7'
    >>> instance.Modality
    'CT'
    >>> instance.FilePath
    '/path/to/file.dcm'
    """

    def __init__(self, SOPInstanceUID, FilePath, modality=None, parent_series=None):
        self.SOPInstanceUID = SOPInstanceUID
        self.FilePath = FilePath
        self.Modality = modality
        self.references = []
        self.referenced_sop_instance_uids = []
        self.referenced_sids = []
        self.referenced_series = []
        self.other_referenced_sids = []
        self.other_referenced_series = []
        self.referenced_instances = []
        self.referencing_instances = []
        self.parent_series = parent_series

    def __getattr__(self, name):
        """
        Delegates attribute access to the parent series. If not found, the parent series'
        `__getattr__` will handle delegation up to the parent study and patient.

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The value of the requested attribute if it exists, or raises AttributeError.

        Raises
        ------
        AttributeError
            If the attribute does not exist in this InstanceNode or the parent hierarchy.
        """
        if self.parent_series is not None:
            return getattr(self.parent_series, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        """
        Returns a string representation of the `InstanceNode`, including the SOPInstanceUID,
        Modality, and file path.

        Returns
        -------
        str
            A string representation of the `InstanceNode` object.
        """
        return (
            f"InstanceNode(SOPInstanceUID={self.SOPInstanceUID}, "
            f"Modality={self.Modality}, FilePath={self.FilePath})"
        )
