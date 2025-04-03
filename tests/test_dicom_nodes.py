import pytest
from rosamllib.readers.dicom_nodes import (
    DatasetNode,
    PatientNode,
    StudyNode,
    SeriesNode,
    InstanceNode,
)


@pytest.fixture
def dicom_hierarchy():
    """Fixture to create a DICOM hierarchy for testing."""
    patient = PatientNode(patient_id="12345", patient_name="John Doe")
    study = StudyNode(study_uid="1.2.3.4", study_description="CT Abdomen", parent_patient=patient)
    series = SeriesNode(series_uid="5.6.7.8", parent_study=study)
    instance = InstanceNode(
        sop_instance_uid="1.2.3.4.5.6.7",
        filepath="/path/to/file.dcm",
        modality="CT",
        parent_series=series,
    )
    return {
        "patient": patient,
        "study": study,
        "series": series,
        "instance": instance,
    }


def test_instance_initialization(dicom_hierarchy):
    """Test the initialization of InstanceNode."""
    instance = dicom_hierarchy["instance"]
    assert instance.SOPInstanceUID == "1.2.3.4.5.6.7"
    assert instance.filepath == "/path/to/file.dcm"
    assert instance.Modality == "CT"
    assert instance.parent_series == dicom_hierarchy["series"]


def test_instance_direct_attributes(dicom_hierarchy):
    """Test direct attributes of InstanceNode."""
    instance = dicom_hierarchy["instance"]
    assert instance.SOPInstanceUID == "1.2.3.4.5.6.7"
    assert instance.filepath == "/path/to/file.dcm"


def test_hierarchical_access(dicom_hierarchy):
    """Test hierarchical access via __getattr__."""
    instance = dicom_hierarchy["instance"]

    # Access attributes from SeriesNode
    assert instance.SeriesInstanceUID == "5.6.7.8"

    # Access attributes from StudyNode
    assert instance.StudyInstanceUID == "1.2.3.4"
    assert instance.StudyDescription == "CT Abdomen"

    # Access attributes from PatientNode
    assert instance.PatientID == "12345"
    assert instance.PatientName == "John Doe"


def test_attribute_error(dicom_hierarchy):
    """Test that accessing a non-existent attribute raises AttributeError."""
    instance = dicom_hierarchy["instance"]
    with pytest.raises(AttributeError):
        _ = instance.NonExistentAttribute


def test_instance_no_parent_series():
    """Test InstanceNode with no parent series."""
    instance = InstanceNode(sop_instance_uid="1.2.3.4.5.6.7", filepath="/path/to/file.dcm")
    with pytest.raises(AttributeError):
        _ = instance.SeriesInstanceUID  # Should raise AttributeError since parent_series is None


def test_series_initialization(dicom_hierarchy):
    """Test the initialization of SeriesNode."""
    series = dicom_hierarchy["series"]
    assert series.SeriesInstanceUID == "5.6.7.8"
    assert series.Modality is None
    assert series.SeriesDescription is None
    assert series.FrameOfReferenceUID is None
    assert series.parent_study == dicom_hierarchy["study"]


def test_series_direct_attributes(dicom_hierarchy):
    """Test direct attributes of SeriesNode."""
    series = dicom_hierarchy["series"]
    assert series.SeriesInstanceUID == "5.6.7.8"
    assert series.Modality is None
    assert series.SeriesDescription is None


def test_series_hierarchical_access(dicom_hierarchy):
    """Test hierarchical access via __getattr__."""
    series = dicom_hierarchy["series"]

    # Access attributes from StudyNode
    assert series.StudyInstanceUID == "1.2.3.4"
    assert series.StudyDescription == "CT Abdomen"

    # Access attributes from PatientNode
    assert series.PatientID == "12345"
    assert series.PatientName == "John Doe"


def test_series_add_instance(dicom_hierarchy):
    """Test adding an instance to the SeriesNode."""
    series = dicom_hierarchy["series"]
    instance = dicom_hierarchy["instance"]

    # Add an instance and verify it is added
    series.add_instance(instance)
    assert len(series) == 1
    assert instance.SOPInstanceUID in series.instances
    assert series.instances[instance.SOPInstanceUID] == instance
    assert instance.parent_series == series


def test_series_no_parent_study():
    """Test SeriesNode with no parent study."""
    series = SeriesNode(series_uid="5.6.7.8")
    with pytest.raises(AttributeError):
        _ = series.StudyInstanceUID  # Should raise AttributeError since parent_study is None


def test_series_duplicate_instance(dicom_hierarchy):
    """Test adding duplicate instances to a SeriesNode."""
    series = dicom_hierarchy["series"]
    instance = dicom_hierarchy["instance"]

    series.add_instance(instance)  # Add first instance
    series.add_instance(instance)  # Add duplicate
    assert len(series) == 1  # Should still have only one unique instance


def test_series_attribute_error(dicom_hierarchy):
    """Test that accessing a non-existent attribute raises AttributeError."""
    series = dicom_hierarchy["series"]
    with pytest.raises(AttributeError):
        _ = series.NonExistentAttribute


def test_study_initialization(dicom_hierarchy):
    """Test the initialization of StudyNode."""
    study = dicom_hierarchy["study"]
    assert study.StudyInstanceUID == "1.2.3.4"
    assert study.StudyDescription == "CT Abdomen"
    assert study.parent_patient == dicom_hierarchy["patient"]
    assert len(study.series) == 0


def test_study_direct_attributes(dicom_hierarchy):
    """Test direct attributes of StudyNode."""
    study = dicom_hierarchy["study"]
    assert study.StudyInstanceUID == "1.2.3.4"
    assert study.StudyDescription == "CT Abdomen"


def test_study_hierarchical_access(dicom_hierarchy):
    """Test hierarchical access via __getattr__."""
    study = dicom_hierarchy["study"]

    # Access attributes from PatientNode
    assert study.PatientID == "12345"
    assert study.PatientName == "John Doe"


def test_study_add_series(dicom_hierarchy):
    """Test adding a series to the StudyNode."""
    study = dicom_hierarchy["study"]
    series = dicom_hierarchy["series"]

    # Add a series and verify it is added
    study.add_series(series)
    assert len(study) == 1
    assert series.SeriesInstanceUID in study.series
    assert study.series[series.SeriesInstanceUID] == series
    assert series.parent_study == study


def test_study_get_series(dicom_hierarchy):
    """Test retrieving a series by SeriesInstanceUID."""
    study = dicom_hierarchy["study"]
    series = dicom_hierarchy["series"]

    # Add a series and retrieve it
    study.add_series(series)
    retrieved_series = study.get_series(series.SeriesInstanceUID)
    assert retrieved_series == series


def test_study_no_parent_patient():
    """Test StudyNode with no parent patient."""
    study = StudyNode(study_uid="1.2.3.4")
    with pytest.raises(AttributeError):
        _ = study.PatientID  # Should raise AttributeError since parent_patient is None


def test_study_get_missing_series(dicom_hierarchy):
    """Test retrieving a missing series from StudyNode."""
    study = dicom_hierarchy["study"]
    assert study.get_series("non-existent-uid") is None  # Should return None for missing series


def test_study_attribute_error(dicom_hierarchy):
    """Test that accessing a non-existent attribute raises AttributeError."""
    study = dicom_hierarchy["study"]
    with pytest.raises(AttributeError):
        _ = study.NonExistentAttribute


def test_patient_initialization(dicom_hierarchy):
    """Test the initialization of PatientNode."""
    patient = dicom_hierarchy["patient"]
    assert patient.PatientID == "12345"
    assert patient.PatientName == "John Doe"
    assert len(patient.studies) == 0


def test_patient_direct_attributes(dicom_hierarchy):
    """Test direct attributes of PatientNode."""
    patient = dicom_hierarchy["patient"]
    assert patient.PatientID == "12345"
    assert patient.PatientName == "John Doe"


def test_patient_add_study(dicom_hierarchy):
    """Test adding a study to the PatientNode."""
    patient = dicom_hierarchy["patient"]
    study = dicom_hierarchy["study"]

    # Add a study and verify it is added
    patient.add_study(study)
    assert len(patient) == 1
    assert study.StudyInstanceUID in patient.studies
    assert patient.studies[study.StudyInstanceUID] == study
    assert study.parent_patient == patient


def test_patient_get_study(dicom_hierarchy):
    """Test retrieving a study by StudyInstanceUID."""
    patient = dicom_hierarchy["patient"]
    study = dicom_hierarchy["study"]

    # Add a study and retrieve it
    patient.add_study(study)
    retrieved_study = patient.get_study(study.StudyInstanceUID)
    assert retrieved_study == study


def test_patient_iterate_studies(dicom_hierarchy):
    """Test iteration over studies in the PatientNode."""
    patient = dicom_hierarchy["patient"]
    study = dicom_hierarchy["study"]

    # Add a study and iterate over it
    patient.add_study(study)
    studies = list(patient)
    assert len(studies) == 1
    assert studies[0] == study


def test_patient_attribute_error(dicom_hierarchy):
    """Test that accessing a non-existent attribute raises AttributeError."""
    patient = dicom_hierarchy["patient"]
    with pytest.raises(AttributeError):
        _ = patient.NonExistentAttribute


def test_patient_duplicate_study(dicom_hierarchy):
    """Test adding duplicate studies to a PatientNode."""
    patient = dicom_hierarchy["patient"]
    study = dicom_hierarchy["study"]

    patient.add_study(study)  # Add first study
    patient.add_study(study)  # Add duplicate
    assert len(patient) == 1  # Should still have only one unique study


def test_patient_iterate_no_studies():
    """Test iterating over a PatientNode with no studies."""
    patient = PatientNode(patient_id="12345", patient_name="John Doe")
    studies = list(patient)
    assert len(studies) == 0  # Should be an empty list


def test_dataset_initialization():
    """Test the initialization of DatasetNode."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    assert dataset.dataset_id == "Institution_123"
    assert dataset.dataset_name == "XYZ Medical Center"
    assert len(dataset.patients) == 0


def test_dataset_direct_attributes():
    """Test direct attributes of DatasetNode."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    assert dataset.dataset_id == "Institution_123"
    assert dataset.dataset_name == "XYZ Medical Center"


def test_dataset_add_patient(dicom_hierarchy):
    """Test adding a patient to the DatasetNode."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    patient = dicom_hierarchy["patient"]

    # Add a patient and verify it is added
    dataset.add_patient(patient)
    assert len(dataset) == 1
    assert patient.PatientID in dataset.patients
    assert dataset.patients[patient.PatientID] == patient


def test_dataset_get_patient(dicom_hierarchy):
    """Test retrieving a patient by PatientID."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    patient = dicom_hierarchy["patient"]

    # Add a patient and retrieve it
    dataset.add_patient(patient)
    retrieved_patient = dataset.get_patient(patient.PatientID)
    assert retrieved_patient == patient


def test_dataset_iterate_patients(dicom_hierarchy):
    """Test iteration over patients in the DatasetNode."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    patient = dicom_hierarchy["patient"]

    # Add a patient and iterate over it
    dataset.add_patient(patient)
    patients = list(dataset)
    assert len(patients) == 1
    assert patients[0] == patient


def test_dataset_attribute_error():
    """Test that accessing a non-existent attribute raises AttributeError."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    with pytest.raises(AttributeError):
        _ = dataset.NonExistentAttribute


def test_dataset_iterate_no_patients():
    """Test iterating over a DatasetNode with no patients."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    patients = list(dataset)
    assert len(patients) == 0  # Should be an empty list


def test_dataset_duplicate_patient(dicom_hierarchy):
    """Test adding duplicate patients to a DatasetNode."""
    dataset = DatasetNode(dataset_id="Institution_123", dataset_name="XYZ Medical Center")
    patient = dicom_hierarchy["patient"]

    dataset.add_patient(patient)  # Add first patient
    dataset.add_patient(patient)  # Add duplicate
    assert len(dataset) == 1  # Should still have only one unique patient
