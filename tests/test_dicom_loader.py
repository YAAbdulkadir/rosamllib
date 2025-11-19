# import pytest
# import pandas as pd
# from unittest.mock import MagicMock
# from rosamllib.readers import DICOMLoader


# @pytest.fixture
# def dicom_loader(tmp_path):
#     """Fixture to create a DICOMLoader instance."""
#     return DICOMLoader(str(tmp_path))


# def dummy_function(instance):
#     print(f"Processing instance: {instance}")
#     return {"instance": instance}


# def test_initialization(dicom_loader):
#     """Test the initialization of the DICOMLoader class."""
#     assert dicom_loader.path
#     assert isinstance(dicom_loader.dicom_files, dict)
#     assert dicom_loader.dataset is None


# def test_load_with_invalid_path():
#     """Test loading with an invalid path."""
#     loader = DICOMLoader("invalid/path")
#     with pytest.raises(Exception):
#         loader.load()


# def test_load_from_directory(dicom_loader, tmp_path, monkeypatch):
#     """Test loading DICOM files from a directory."""
#     # Create a mock DICOM file
#     dicom_file_path = tmp_path / "sample.dcm"
#     dicom_file_path.write_bytes(b"Mock DICOM data")

#     # Create a mock for validate_dicom_path to prevent validation errors
#     def mock_validate_path(*args, **kwargs):
#         return True

#     monkeypatch.setattr("rosamllib.utils.validate_dicom_path", mock_validate_path)

#     # Create a mock for tqdm to avoid progress bar in tests
#     def mock_tqdm(iterable, *args, **kwargs):
#         return iterable

#     monkeypatch.setattr("tqdm.tqdm", mock_tqdm)

#     # Mock the _load_files method since that's what actually processes the files
#     def mock_load_files(self, files, tags_to_index=None):
#         self.dicom_files = {
#             "12345": {
#                 "1.2.3.4": MagicMock(Modality="CT", SeriesInstanceUID="1.2.3.4", PatientID="12345")
#             }
#         }

#     monkeypatch.setattr(DICOMLoader, "_load_files", mock_load_files)

#     # Invoke the load_from_directory method
#     dicom_loader.load_from_directory(tmp_path)

#     # Validate results
#     assert len(dicom_loader.dicom_files) > 0
#     assert "12345" in dicom_loader.dicom_files
#     assert "1.2.3.4" in dicom_loader.dicom_files["12345"]


# def test_load_file(dicom_loader, tmp_path, monkeypatch):
#     """Test loading a single DICOM file."""

#     # Create a mock DICOM file
#     dicom_file_path = tmp_path / "sample.dcm"
#     dicom_file_path.write_bytes(b"Mock DICOM data")

#     # Mock validate_dicom_path to prevent actual validation
#     def mock_validate_dicom_path(path):
#         assert path == str(dicom_file_path)

#     monkeypatch.setattr("rosamllib.utils.validate_dicom_path", mock_validate_dicom_path)

#     # Mock _load_files to verify it's called correctly
#     def mock_load_files(file_paths, tags_to_index):
#         assert file_paths == [str(dicom_file_path)]
#         assert tags_to_index is None

#     monkeypatch.setattr(dicom_loader, "_load_files", mock_load_files)

#     # Call the method with mocked dependencies
#     dicom_loader.load_file(str(dicom_file_path))


# def test_get_summary(dicom_loader):
#     """Test getting a summary of the dataset."""
#     dicom_loader.dataset = MagicMock()
#     dicom_loader.dataset.__iter__.return_value = []
#     summary = dicom_loader.get_summary()
#     assert summary == {
#         "total_patients": 0,
#         "total_studies": 0,
#         "total_series": 0,
#         "total_instances": 0,
#     }


# def test_get_patient_summary(dicom_loader):
#     """Test getting a summary of a patient's studies and series."""

#     # Mock instances
#     instance_mock = MagicMock(SOPInstanceUID="1.2.3", Modality="CT", filepath="path/to/file")

#     # Mock series
#     series_mock = MagicMock(
#         SeriesInstanceUID="series1",
#         SeriesDescription="Series Description",
#         Modality="CT",
#         __iter__=lambda self: iter([instance_mock]),
#     )

#     # Mock study
#     study_mock = MagicMock(
#         StudyInstanceUID="study1",
#         StudyDescription="Study Description",
#         __iter__=lambda self: iter([series_mock]),
#     )

#     # Mock patient
#     patient_mock = MagicMock(
#         PatientID="patient1",
#         PatientName="Test Patient",
#         __iter__=lambda self: iter([study_mock]),  # Ensure iter returns mocked studies
#     )

#     # Mock dataset
#     dataset_mock = MagicMock()
#     dataset_mock.patients = {"patient1": patient_mock}
#     dataset_mock.get_patient = lambda pid: patient_mock if pid == "patient1" else None

#     # Inject mock dataset into dicom_loader
#     dicom_loader.dataset = dataset_mock

#     # Call the method
#     summary = dicom_loader.get_patient_summary("patient1")

#     # Validate the summary structure
#     assert summary["patient_id"] == "patient1"
#     assert summary["patient_name"] == "Test Patient"
#     assert len(summary["studies"]) == 1
#     study_summary = summary["studies"][0]
#     assert study_summary["study_uid"] == "study1"


# def test_get_study_summary(dicom_loader):
#     """Test getting a summary of a study's series and instances."""

#     # Mock instances
#     instance_mock = MagicMock(SOPInstanceUID="1.2.3", Modality="CT", filepath="path/to/file")

#     # Mock series
#     series_mock = MagicMock(
#         SeriesInstanceUID="series1",
#         SeriesDescription="Series Description",
#         Modality="CT",
#         __iter__=lambda self: iter([instance_mock]),
#     )

#     # Mock study
#     study_mock = MagicMock(
#         StudyInstanceUID="study1",
#         StudyDescription="Study Description",
#         __iter__=lambda self: iter([series_mock]),
#         series={"series1": series_mock},
#         get_series=lambda uid: series_mock if uid == "series1" else None,
#     )

#     # Mock patient
#     patient_mock = MagicMock(
#         PatientID="patient1",
#         PatientName="Test Patient",
#         get_study=lambda uid: study_mock if uid == "study1" else None,
#         studies={"study1": study_mock},
#     )

#     # Mock dataset
#     dicom_loader.dataset = [patient_mock]

#     # Call the method
#     summary = dicom_loader.get_study_summary("study1")

#     # Validate the summary
#     assert summary["study_uid"] == "study1"
#     assert summary["study_description"] == "Study Description"
#     assert len(summary["series"]) == 1
#     series_summary = summary["series"][0]
#     assert series_summary["series_uid"] == "series1"
#     assert len(series_summary["instances"]) == 1
#     assert series_summary["instances"][0]["sop_instance_uid"] == "1.2.3"


# def test_get_series_summary(dicom_loader):
#     """Test getting a summary of a series."""

#     # Mock instances
#     instance_mock = MagicMock(SOPInstanceUID="1.2.3", Modality="CT", filepath="path/to/file")

#     # Mock series
#     series_mock = MagicMock(
#         SeriesInstanceUID="series1",
#         SeriesDescription="Series Description",
#         Modality="CT",
#         __iter__=lambda self: iter([instance_mock]),
#     )

#     # Mock study
#     study_mock = MagicMock(
#         series={"series1": series_mock},
#         get_series=lambda uid: series_mock if uid == "series1" else None,
#     )

#     # Mock patient
#     patient_mock = MagicMock(
#         PatientID="patient1",
#         PatientName="Test Patient",
#         __iter__=lambda self: iter([study_mock]),
#     )

#     # Mock dataset
#     dicom_loader.dataset = [patient_mock]

#     # Call the method
#     summary = dicom_loader.get_series_summary("series1")

#     # Validate the summary
#     assert summary["series_uid"] == "series1"
#     assert len(summary["instances"]) == 1
#     instance_summary = summary["instances"][0]
#     assert instance_summary["sop_instance_uid"] == "1.2.3"


# def test_get_modality_distribution(dicom_loader):
#     """Test getting the modality distribution in the dataset."""
#     # Mock the dataset structure
#     instance_mock = MagicMock(SOPInstanceUID="1.2.3", Modality="CT")
#     series_mock = MagicMock(
#         Modality="CT",
#         __iter__=lambda self: iter([instance_mock]),
#     )
#     study_mock = MagicMock(
#         __iter__=lambda self: iter([series_mock]),
#     )
#     patient_mock = MagicMock(
#         __iter__=lambda self: iter([study_mock]),
#     )
#     dicom_loader.dataset = [patient_mock]

#     # Call the method
#     distribution = dicom_loader.get_modality_distribution()

#     # Validate the distribution
#     assert distribution["CT"] == 1


# def test_query_with_filters(dicom_loader):
#     """Test querying metadata with filters."""
#     # Create a mock DataFrame with required columns
#     dicom_loader.metadata_df = pd.DataFrame(
#         {
#             "Modality": ["CT", "MR", "PT"],
#             "PatientID": ["patient1", "patient2", "patient3"],
#             "SOPInstanceUID": ["1.2.3", "4.5.6", "7.8.9"],
#             "StudyInstanceUID": ["1.1.1", "2.2.2", "3.3.3"],
#             "SeriesInstanceUID": ["1.2.1", "2.3.1", "3.4.1"],
#         }
#     )

#     # Perform the query
#     results = dicom_loader.query(Modality="CT")

#     # Verify the results
#     assert not results.empty
#     assert all(uid in ["1.2.1"] for uid in results["SeriesInstanceUID"].values)


# def test_get_patient_ids(dicom_loader):
#     """Test getting all patient IDs."""
#     dicom_loader.dataset = MagicMock()
#     dicom_loader.dataset.patients = {"patient1": MagicMock()}
#     patient_ids = dicom_loader.get_patient_ids()
#     assert patient_ids == ["patient1"]


# def test_get_patient(dicom_loader):
#     """Test retrieving a patient by PatientID."""
#     # Mock the dataset
#     patient_mock = MagicMock(PatientID="patient1")
#     dicom_loader.dataset = MagicMock()
#     dicom_loader.dataset.get_patient.return_value = patient_mock

#     # Test getting the patient
#     result = dicom_loader.get_patient("patient1")
#     assert result == patient_mock

#     # Test when patient does not exist
#     dicom_loader.dataset.get_patient.return_value = None
#     result = dicom_loader.get_patient("nonexistent")
#     assert result is None


# def test_get_study(dicom_loader):
#     """Test retrieving a study by StudyInstanceUID."""
#     # Mock the dataset
#     study_mock = MagicMock(StudyInstanceUID="study1")
#     patient_mock = MagicMock()
#     patient_mock.get_study.return_value = study_mock
#     dicom_loader.dataset = [patient_mock]

#     # Test getting the study
#     result = dicom_loader.get_study("study1")
#     assert result == study_mock

#     # Test when study does not exist
#     patient_mock.get_study.return_value = None
#     result = dicom_loader.get_study("nonexistent")
#     assert result is None


# def test_get_series(dicom_loader):
#     """Test retrieving a series by SeriesInstanceUID."""
#     # Mock the dataset
#     series_mock = MagicMock(SeriesInstanceUID="series1")
#     study_mock = MagicMock()
#     study_mock.get_series.return_value = series_mock
#     patient_mock = MagicMock()
#     patient_mock.__iter__.return_value = [study_mock]
#     dicom_loader.dataset = [patient_mock]

#     # Test getting the series
#     result = dicom_loader.get_series("series1")
#     assert result == series_mock

#     # Test when series does not exist
#     study_mock.get_series.return_value = None
#     result = dicom_loader.get_series("nonexistent")
#     assert result is None


# def test_get_instance(dicom_loader):
#     """Test retrieving an instance by SOPInstanceUID."""
#     # Mock the dataset
#     instance_mock = MagicMock(SOPInstanceUID="instance1")
#     series_mock = MagicMock()
#     series_mock.get_instance.return_value = instance_mock
#     study_mock = MagicMock()
#     study_mock.__iter__.return_value = [series_mock]
#     patient_mock = MagicMock()
#     patient_mock.__iter__.return_value = [study_mock]
#     dicom_loader.dataset = [patient_mock]

#     # Test getting the instance
#     result = dicom_loader.get_instance("instance1")
#     assert result == instance_mock

#     # Test when instance does not exist
#     series_mock.get_instance.return_value = None
#     result = dicom_loader.get_instance("nonexistent")
#     assert result is None


# def test_read_series(dicom_loader, monkeypatch):
#     """Test reading a series."""

#     # Mock the DICOMImageReader.read method
#     def mock_DICOMImageReader_read(*args, **kwargs):
#         return "Mock Image"

#     monkeypatch.setattr("rosamllib.readers.DICOMImageReader.read", mock_DICOMImageReader_read)

#     # Create a mock series, study, and patient structure
#     series_mock = MagicMock(
#         Modality="CT", instance_paths=["path/to/file"], is_embedded_in_raw=False
#     )
#     study_mock = MagicMock()
#     study_mock.get_series.return_value = series_mock
#     patient_mock = MagicMock()
#     patient_mock.__iter__.return_value = [study_mock]  # Mock iteration for studies
#     dicom_loader.dataset = [patient_mock]  # Mock the dataset as a list of patients

#     # Call the method
#     result = dicom_loader.read_series("series1")

#     # Validate the results
#     assert result == ["Mock Image"]


# def test_read_instance(dicom_loader, monkeypatch):
#     """Test reading a single instance by SOPInstanceUID."""

#     # Mock the appropriate reader methods
#     def mock_DICOMImageReader_read(*args, **kwargs):
#         return "Mock CT Image"

#     def mock_RTStructReader_read(*args, **kwargs):
#         return "Mock RTSTRUCT"

#     monkeypatch.setattr("rosamllib.readers.DICOMImageReader.read", mock_DICOMImageReader_read)
#     monkeypatch.setattr("rosamllib.readers.RTStructReader.read", mock_RTStructReader_read)

#     # Create mock instance, series, study, and patient
#     instance_mock = MagicMock(SOPInstanceUID="instance1", Modality="CT", filepath="path/to/file")
#     series_mock = MagicMock()
#     series_mock.get_instance.return_value = instance_mock
#     study_mock = MagicMock()
#     study_mock.__iter__.return_value = [series_mock]
#     patient_mock = MagicMock()
#     patient_mock.__iter__.return_value = [study_mock]
#     dicom_loader.dataset = [patient_mock]

#     # Call the method with a valid instance
#     result = dicom_loader.read_instance("instance1")
#     assert result == "Mock CT Image"

#     # Modify the instance to test a different modality
#     instance_mock.Modality = "RTSTRUCT"
#     result = dicom_loader.read_instance("instance1")
#     assert result == "Mock RTSTRUCT"

#     # Test for an instance not found
#     series_mock.get_instance.return_value = None
#     with pytest.raises(ValueError, match="Instance with SOPInstanceUID 'nonexistent' not found."):
#         dicom_loader.read_instance("nonexistent")

#     # Test for a modality without an implemented reader
#     instance_mock.Modality = "UNKNOWN"
#     series_mock.get_instance.return_value = instance_mock
#     with pytest.raises(
#         NotImplementedError, match="A reader for UNKNOWN type is not implemented yet."
#     ):
#         dicom_loader.read_instance("instance1")


# def test_process_in_parallel(dicom_loader):
#     """Test parallel processing of instances."""
#     # Mock the dataset with patient, study, and series structure
#     instance_mock = {"id": "InstanceMock"}
#     series_mock = [instance_mock]
#     study_mock = [series_mock]
#     patient_mock = [study_mock]
#     dicom_loader.dataset = [patient_mock]

#     # Call process_in_parallel
#     results, errors = dicom_loader.process_in_parallel(dummy_function)

#     # Validate results
#     assert len(results) == 1
#     assert results[0]["instance"] == instance_mock
#     assert len(errors) == 0


# def test_process_in_parallel_threads(dicom_loader):
#     """Test parallel processing of instances."""
#     # Mock the dataset with patient, study, and series structure
#     instance_mock = {"id": "InstanceMock"}
#     series_mock = [instance_mock]
#     study_mock = [series_mock]
#     patient_mock = [study_mock]
#     dicom_loader.dataset = [patient_mock]

#     # Call process_in_parallel
#     results, errors = dicom_loader.process_in_parallel_threads(dummy_function)

#     # Validate results
#     assert len(results) == 1
#     assert results[0]["instance"] == instance_mock
#     assert len(errors) == 0


# def test_visualize_series_references(monkeypatch, dicom_loader):
#     """Test visualizing series references."""

#     # Create a mock Digraph class
#     class MockDigraph:
#         def __init__(self, *args, **kwargs):
#             self.nodes = []
#             self.edges = []
#             self.attributes = {}
#             self.subgraphs = []
#             self.source = "digraph G {}"
#             self.rendered_path = "mock_rendered_path.png"  # Mock path for render

#         def node(self, *args, **kwargs):
#             self.nodes.append((args, kwargs))

#         def edge(self, *args, **kwargs):
#             self.edges.append((args, kwargs))

#         def attr(self, *args, **kwargs):
#             self.attributes.update(kwargs)

#         def subgraph(self, *args, **kwargs):
#             sub = MockDigraph()
#             self.subgraphs.append((args, kwargs, sub))
#             return sub

#         def render(self, filename, format=None, **kwargs):
#             self.rendered_path = f"{filename}.{format}" if format else filename
#             return self.rendered_path

#         def __enter__(self):
#             return self

#         def __exit__(self, exc_type, exc_value, traceback):
#             pass

#     monkeypatch.setattr("graphviz.Digraph", MockDigraph)

#     # Set up test data
#     dicom_loader.dicom_files = {
#         "patient1": {
#             "series1": MagicMock(
#                 Modality="CT",
#                 SeriesInstanceUID="series1",
#                 SeriesDescription="Test Series",
#                 StudyDescription="Test Study",
#                 ReferenceSeriesSequence=[],  # Add any reference sequences if needed
#             )
#         }
#     }

#     # Test the visualization
#     dicom_loader.visualize_series_references(patient_id="patient1", view=False)

#     # Assert that the mock render path was used
#     mock_graph = MockDigraph()
#     assert mock_graph.rendered_path.endswith(".png")
