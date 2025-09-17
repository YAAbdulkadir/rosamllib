import os
import time
import traceback
from typing import List, Optional, Union
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
from functools import partial
from pydicom import dcmread
from pydicom.tag import Tag
from pydicom.sequence import Sequence
from pydicom.datadict import keyword_for_tag, tag_for_keyword, dictionary_VR
from rosamllib.readers import (
    DICOMImageReader,
    RTStructReader,
    RTDoseReader,
    REGReader,
    DICOMRawReader,
    RTPlanReader,
    RTRecordReader,
    SEGReader,
)
from rosamllib.constants import VR_TO_DTYPE
from rosamllib.readers.dicom_nodes import (
    DatasetNode,
    PatientNode,
    StudyNode,
    SeriesNode,
    InstanceNode,
)
from rosamllib.utils import validate_dicom_path, query_df, parse_vr_value
from concurrent.futures import ThreadPoolExecutor, as_completed


def in_jupyter():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except Exception:
        return False


def apply_vscode_theme():
    """Automatically detect VS Code and apply styling."""
    if "VSCODE_PID" in os.environ:
        style = """
        <style>
            .cell-output-ipywidget-background {
                background-color: transparent !important;
            }
            :root {
                --jp-widgets-color: var(--vscode-editor-foreground);
                --jp-widgets-font-size: var(--vscode-editor-font-size);
            }
        </style>
        """
        display(HTML(style))


if in_jupyter():
    from tqdm.notebook import tqdm
    from IPython.display import display, HTML

    # Apply theme automatically if running in VS Code Jupyter
    apply_vscode_theme()
    time.sleep(0.5)
else:
    from tqdm import tqdm


def get_referencing_items(node, modality=None, level="INSTANCE"):
    """
    Retrieves all referencing items (instances or series) for a given node.

    Parameters
    ----------
    node : InstanceNode or SeriesNode
        The starting node for searching referencing items.
    modality : str, optional
        Modality to filter the referencing items (e.g., 'CT', 'MR').
    level : str, {'INSTANCE', 'SERIES'}
        Level at which to retrieve referencing items.

    Returns
    -------
    list
        A list of referencing InstanceNode or SeriesNode objects.
    """
    if level not in {"INSTANCE", "SERIES"}:
        raise ValueError("level must be either 'INSTANCE' or 'SERIES'")

    referencing_items = []

    if isinstance(node, InstanceNode):
        # Get instances that reference this instance
        if level == "INSTANCE":
            referencing_items = node.referencing_instances
        else:  # level == "SERIES"
            referencing_items = node.parent_series.referencing_series if node.parent_series else []

    elif isinstance(node, SeriesNode):
        # Get referencing instances
        referencing_instances = []
        for inst in node.instances.values():
            referencing_instances.extend(inst.referencing_instances)
        referencing_instances = list(set(referencing_instances))
        if level == "INSTANCE":
            referencing_items = referencing_instances
        else:  # level == "SERIES"
            referencing_items = list(set([item.parent_series for item in referencing_instances]))

    else:
        raise TypeError("Expected an InstanceNode or SeriesNode.")

    # Filter by modality if specified
    if modality:
        referencing_items = [
            item for item in referencing_items if getattr(item, "Modality", None) == modality
        ]

    return referencing_items


def get_referenced_items(node, modality=None, level="INSTANCE"):
    """
    Retrieves all referenced items (instances or series) for a given node.

    Parameters
    ----------
    node : InstanceNode or SeriesNode
        The starting node for searching referenced items.
    modality : str, optional
        Modality to filter the referenced items (e.g., 'CT', 'MR').
    level : str, {'INSTANCE', 'SERIES'}
        Level at which to retrieve referenced items.

    Returns
    -------
    list
        A list of referenced InstanceNode or SeriesNode objects.
    """
    if level not in {"INSTANCE", "SERIES"}:
        raise ValueError("level must be either 'INSTANCE' or 'SERIES'")

    referenced_items = []
    if isinstance(node, InstanceNode):
        # Get directly referenced instances or series
        if level == "INSTANCE":
            referenced_items = node.referenced_instances
        else:  # level == "SERIES"
            referenced_items = node.referenced_series

    elif isinstance(node, SeriesNode):
        # Get referenced series
        if level == "INSTANCE":
            referenced_items = []
            for series in node.referenced_series:
                referenced_items.extend(series.instances.values())
        else:  # level == "SERIES"
            referenced_items = node.referenced_series

    else:
        raise TypeError("Expected an InstanceNode or SeriesNode.")

    # Filter by modality if specified
    if modality:
        referenced_items = [
            item for item in referenced_items if getattr(item, "Modality", None) == modality
        ]

    return referenced_items


def get_referenced_sop_instance_uids(ds):
    """
    Extracts referenced SOPInstanceUIDs from RTSTRUCT, RTPLAN, and RTDOSE DICOM files.

    This method scans the DICOM dataset for references to other DICOM instances and returns
    the list of referenced SOPInstanceUIDs.

    Parameters
    ----------
    ds : pydicom.Dataset
        The DICOM dataset to extract references from.

    Returns
    -------
    list of str
        A list of referenced SOPInstanceUIDs from the DICOM dataset.

    Examples
    --------
    >>> uids = DICOMLoader._get_referenced_sop_instance_uids(ds)
    >>> print(uids)
    ['1.2.3.4.5.6.7', '1.2.3.4.5.6.8']
    """
    referenced_uids = set()
    if ds.Modality == "RTSTRUCT":
        if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
            for item in ds.ReferencedFrameOfReferenceSequence:
                if hasattr(item, "RTReferencedStudySequence"):
                    for study_item in item.RTReferencedStudySequence:
                        if hasattr(study_item, "RTReferencedSeriesSequence"):
                            for series_item in study_item.RTReferencedSeriesSequence:
                                if hasattr(series_item, "ContourImageSequence"):
                                    for contour_item in series_item.ContourImageSequence:
                                        referenced_uids.add(contour_item.ReferencedSOPInstanceUID)
        if hasattr(ds, "ROIContourSequence"):
            for roi_item in ds.ROIContourSequence:
                if hasattr(roi_item, "ContourSequence"):
                    for contour_seq in roi_item.ContourSequence:
                        if hasattr(contour_seq, "ContourImageSequence"):
                            for image_seq in contour_seq.ContourImageSequence:
                                referenced_uids.add(image_seq.ReferencedSOPInstanceUID)
    elif ds.Modality == "SEG":
        if hasattr(ds, "ReferencedSeriesSequence"):
            for item in ds.ReferencedSeriesSequence:
                if hasattr(item, "ReferencedInstanceSequence"):
                    for ref_seq in item.ReferencedInstanceSequence:
                        referenced_uids.add(ref_seq.ReferencedSOPInstanceUID)

    else:
        if hasattr(ds, "ReferencedStructureSetSequence"):
            for item in ds.ReferencedStructureSetSequence:
                if hasattr(item, "ReferencedSOPInstanceUID"):
                    referenced_uids.add(item.ReferencedSOPInstanceUID)

        if hasattr(ds, "ReferencedDoseSequence"):
            for item in ds.ReferencedDoseSequence:
                if hasattr(item, "ReferencedSOPInstanceUID"):
                    referenced_uids.add(item.ReferencedSOPInstanceUID)

        if hasattr(ds, "ReferencedRTPlanSequence"):
            for item in ds.ReferencedRTPlanSequence:
                if hasattr(item, "ReferencedSOPInstanceUID"):
                    referenced_uids.add(item.ReferencedSOPInstanceUID)

    return list(referenced_uids)


def get_metadata(ds, tags_to_index):
    metadata = {
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
        "Modality": getattr(ds, "Modality", None),
        "SeriesDescription": getattr(ds, "SeriesDescription", ""),
        "FrameOfReferenceUID": getattr(ds, "FrameOfReferenceUID", None),
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
        "StudyDescription": getattr(ds, "StudyDescription", ""),
        "PatientID": getattr(ds, "PatientID", None),
        "PatientName": getattr(ds, "PatientName", None),
    }
    for tag in tags_to_index:
        try:
            tag_obj = Tag(tag)
            vr = dictionary_VR(tag_obj)
            value = ds[tag_obj].value if tag_obj in ds else None
            if isinstance(value, Sequence) and vr == "SQ":
                metadata[keyword_for_tag(tag) or tag] = (ds[tag_obj].to_json()) if value else None
            else:
                metadata[keyword_for_tag(tag) or tag] = parse_vr_value(vr, value)
        except Exception:
            metadata[tag] = None

    return metadata


def process_standard_dicom(ds, filepath, tags_to_index):
    modality = getattr(ds, "Modality", None)
    metadata = get_metadata(ds, tags_to_index)
    instance_dict = {"FilePath": filepath, **metadata}

    if modality in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
        instance_dict["ReferencedSOPInstanceUIDs"] = get_referenced_sop_instance_uids(ds)

    return instance_dict


def process_reg_file(filepath, tags_to_index):
    reg = REGReader(filepath).read()
    metadata = get_metadata(reg, tags_to_index)
    instance_dict = {
        "FilePath": filepath,
        "ReferencedSeriesUIDs": reg.get_fixed_image_info()["SeriesInstanceUID"],
        "OtherReferencedSeriesUIDs": reg.get_moving_image_info()["SeriesInstanceUID"],
        **metadata,
    }

    return instance_dict


def process_raw_file(filepath, tags_to_index):
    raw_reader = DICOMRawReader(filepath)
    raw_reader.read()
    ds = raw_reader.dataset
    metadata = get_metadata(ds, tags_to_index)
    instance_dict = {
        "FilePath": filepath,
        **metadata,
    }
    embedded_instances = []
    try:
        embedded_datasets = raw_reader.get_embedded_datasets()

        for embedded_ds in embedded_datasets:
            embedded_metadata = get_metadata(embedded_ds, tags_to_index)
            embedded_instance_dict = {
                **embedded_metadata,
                "FilePath": filepath,
                "is_embedded_in_raw": True,
                "raw_series_reference_uid": instance_dict["SeriesInstanceUID"],
            }
            if embedded_instance_dict["Modality"] in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
                embedded_instance_dict["ReferencedSOPInstanceUIDs"] = (
                    get_referenced_sop_instance_uids(embedded_ds)
                )
            elif embedded_instance_dict["Modality"] == "REG":
                embedded_reg = REGReader(embedded_ds).read()
                embedded_instance_dict["ReferencedSeriesUIDs"] = (
                    embedded_reg.get_fixed_image_info()["SeriesInstanceUID"]
                )
                embedded_instance_dict["OtherReferencedSeriesUIDs"] = (
                    embedded_reg.get_moving_image_info()["SeriesInstanceUID"]
                )
            embedded_instances.append(embedded_instance_dict)
    except Exception:
        pass

    return instance_dict, embedded_instances


def process_seg_file(filepath, tags_to_index):
    seg = SEGReader(filepath).read()
    metadata = get_metadata(seg, tags_to_index)
    instance_dict = {"FilePath": filepath, **metadata}
    if hasattr(seg, "ReferencedSeriesSequence"):
        ref_seq = seg.ReferencedSeriesSequence[0]
        if hasattr(ref_seq, "SeriesInstanceUID"):
            instance_dict["ReferencedSeriesUIDs"] = ref_seq.SeriesInstanceUID
    instance_dict["ReferencedSOPInstanceUIDs"] = get_referenced_sop_instance_uids(seg)

    return instance_dict


def process_file(filepath, tags_to_index):
    try:
        ds = dcmread(filepath, stop_before_pixels=True)
        modality = getattr(ds, "Modality", None)
        embedded_instances = []
        if modality in ["CT", "MR", "PT", "RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
            instance_dict = process_standard_dicom(ds, filepath, tags_to_index)
        elif modality == "REG":
            instance_dict = process_reg_file(filepath, tags_to_index)
        elif modality == "RAW":
            instance_dict, embedded_instances = process_raw_file(filepath, tags_to_index)
        elif modality == "SEG":
            instance_dict = process_seg_file(filepath, tags_to_index)
        else:
            return []

        metadata_list = [instance_dict]
        for embedded_inst_dict in embedded_instances:
            metadata_list.append(embedded_inst_dict)
        return metadata_list
    except Exception:
        return []


class DICOMLoader:
    """
    A class for loading, organizing, and managing DICOM files in a hierarchical structure.

    The `DICOMLoader` class provides methods to load DICOM files from a specified path, organize
    them into a hierarchical structure of patients, studies, series, and instances, and retrieve
    information at each level. Additionally, it offers functionalities to summarize, visualize, and
    read DICOM data based on specific modalities. It is designed to handle large datasets and
    supports the extraction of metadata as well as the reading and visualization of DICOM series.

    Parameters
    ----------
    path : str
        The directory or file path where DICOM files are located.

    Attributes
    ----------
    path : str
        The directory or file path provided during initialization, used to locate DICOM files.
    dicom_files : dict
        A dictionary that stores DICOM files grouped by PatientID and SeriesInstanceUID.
    dataset : DatasetNode
        The top-level node containing all patients, organized into a dataset structure.

    Methods
    -------
    load()
        Loads DICOM files from the specified path and organizes them into a structured dataset.
    load_from_directory(path)
        Recursively loads all DICOM files in the given directory.
    get_summary()
        Provides a summary count of patients, studies, series, and instances.
    get_patient_summary(patient_id)
        Retrieves a detailed summary of all studies and series for a given patient.
    get_study_summary(study_uid)
        Retrieves a summary of series and instances within a specified study.
    get_series_summary(series_uid)
        Retrieves detailed information about a series, including instance paths.
    get_modality_distribution()
        Returns the distribution of modalities present in the dataset.
    get_patient_ids()
        Returns a list of all PatientIDs within the dataset.
    get_study_uids(patient_id)
        Returns a list of StudyInstanceUIDs for a specified patient.
    get_series_uids(study_uid)
        Returns a list of SeriesInstanceUIDs for a specified study.
    get_series_paths(patient_id, series_uid)
        Retrieves file paths for all instances within a specific series.
    get_patient(patient_id)
        Retrieves a PatientNode by its PatientID.
    get_study(study_uid)
        Retrieves a StudyNode by its StudyInstanceUID.
    get_series(series_uid)
        Retrieves a SeriesNode by its SeriesInstanceUID.
    get_instance(sop_instance_uid)
        Retrieves an InstanceNode by its SOPInstanceUID.
    read_series(series_uid)
        Reads and returns data for a series based on its SeriesInstanceUID.
    read_instance(sop_instance_uid)
        Reads and returns data for a specific instance based on its SOPInstanceUID.
    visualize_series_references(patient_id, output_file, view, per_patient, exclude_modalities,
                                exclude_series, include_uid, rankdir)
        Visualizes the series-level associations for all or specific patients using Graphviz.

    Examples
    --------
    >>> loader = DICOMLoader("/path/to/dicom/files")
    >>> loader.load()
    >>> summary = loader.get_summary()
    >>> print(summary)
    {'total_patients': 10, 'total_studies': 50, 'total_series': 200, 'total_instances': 5000}

    >>> patient_summary = loader.get_patient_summary("12345")
    >>> print(patient_summary)
    {'patient_id': '12345', 'patient_name': 'John Doe', 'studies': [{'study_uid': '1.2.3', ...}]}

    >>> series_paths = loader.get_series_paths("12345", "1.2.3.4.5")
    >>> print(series_paths)
    ['/path/to/file1.dcm', '/path/to/file2.dcm']
    """

    def __init__(self, path):
        """
        Initializes the DICOMLoader with the specified path.

        Parameters
        ----------
        path : str
            The directory or file path where DICOM files are located.
        """
        self.path = path
        self.dicom_files = {}
        self.dataset = None
        self.metadata_df = None

    def load(self, tags_to_index=None):
        """
        Loads the DICOM files from the specified path.

        This method validates the provided path, reads the DICOM files, and organizes them
        by patient and series. The method also associates referenced DICOMs using SOPInstanceUID
        and SeriesInstanceUID.

        Parameters
        ----------
        tags_to_index : list of str, optional
            A list of DICOM tags (keywords) to index during loading.

        Raises
        ------
        Exception
            If there is an error loading or processing the DICOM files.

        Examples
        --------
        >>> loader = DICOMLoader("/path/to/dicom/files")
        >>> loader.load()
        """
        default_tags = [
            "SOPInstanceUID",
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "PatientID",
            "SOPClassUID",
            "Modality",
        ]
        default_tags = {self._normalize_tag(tag) for tag in default_tags}
        if tags_to_index:
            tags_to_index = [self._normalize_tag(tag) for tag in tags_to_index]
            tags_to_index = {tag for tag in tags_to_index if tag}
            tags_to_index = list(default_tags | tags_to_index)
        else:
            tags_to_index = list(default_tags)

        validate_dicom_path(self.path)
        try:
            if os.path.isdir(self.path):
                self.load_from_directory(self.path, tags_to_index)
            else:
                self.load_file(self.path, tags_to_index)
            self._build_hierarchical_structure()

        except Exception as e:
            print(f"Error loading DICOM files: {e}")
            print(traceback.format_exc())

    def load_from_directory(self, path, tags_to_index=None):
        """
        Loads all DICOM files from a directory, including subdirectories.

        This method recursively searches the specified directory for DICOM files,
        reads their metadata, and organizes them by patient and series.

        Parameters
        ----------
        path : str
            The directory path to load DICOM files from.
        tags_to_index : list of str, optional
            A list of DICOM tags (keywords) to index during loading.

        Returns
        -------
        dict
            A dictionary where the keys are PatientIDs and the values are dictionaries
            of Series objects indexed by SeriesInstanceUID.

        Raises
        ------
        Exception
            If there is an error reading DICOM files.

        Examples
        --------
        >>> dicom_files = DICOMLoader.load_from_directory("/path/to/dicom/files")
        """
        validate_dicom_path(path)
        all_files = []
        for root, _, files in tqdm(os.walk(path), desc="Scanning directories"):
            for file in files:
                all_files.append(os.path.join(root, file))
        print(f"Found {len(all_files)} files.")
        self._load_files(all_files, tags_to_index)

    def load_file(self, path, tags_to_index=None):
        """
        Loads a single DICOM file and returns the Series object it belongs to.

        Parameters
        ----------
        path : str
            The file path to the DICOM file.
        tags_to_index : list of str, optional
            A list of DICOM tags (keywords) to index during loading.

        Returns
        -------
        dict
            A dictionary containing the DICOM data organized by PatientID and SeriesInstanceUID.

        Raises
        ------
        Exception
            If there is an error reading the DICOM file.

        Examples
        --------
        >>> dicom_file = DICOMLoader.load_file("/path/to/file.dcm")
        """
        validate_dicom_path(path)
        self._load_files([path], tags_to_index)

    def _load_files(self, files, tags_to_index=None):
        process_file_with_tags = partial(process_file, tags_to_index=tags_to_index)
        with ThreadPoolExecutor() as executor:
            # with ProcessPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(process_file_with_tags, files),
                    total=len(files),
                    desc="Loading DICOM files",
                    unit="file",
                )
            )

        exclude_keys = [
            "FilePath",
            "ReferencedSOPInstanceUIDs",
            "ReferencedSeriesUIDs",
            "OtherReferencedSeriesUIDs",
            "is_embedded_in_raw",
            "raw_series_reference_uid",
        ]
        metadata_list = []
        for result in results:
            for inst_dict in result:
                sop_instance_uid = inst_dict["SOPInstanceUID"]
                patient_id = inst_dict["PatientID"]
                series_uid = inst_dict["SeriesInstanceUID"]
                modality = inst_dict["Modality"]
                filepath = inst_dict["FilePath"]
                if patient_id not in self.dicom_files:
                    self.dicom_files[patient_id] = {}
                if series_uid not in self.dicom_files[patient_id]:
                    series = SeriesNode(series_uid)
                    series.PatientID = patient_id
                    series.Modality = modality
                    series.PatientName = inst_dict["PatientName"]
                    series.StudyInstanceUID = inst_dict["StudyInstanceUID"]
                    series.StudyDescription = inst_dict["StudyDescription"]
                    series.SeriesDescription = inst_dict["SeriesDescription"]
                    series.FrameOfReferenceUID = inst_dict["FrameOfReferenceUID"]
                    self.dicom_files[patient_id][series_uid] = series
                series = self.dicom_files[patient_id][series_uid]
                instance_node = InstanceNode(
                    sop_instance_uid, filepath, modality=modality, parent_series=series
                )
                series.add_instance(instance_node)
                if inst_dict.get("ReferencedSOPInstanceUIDs"):
                    instance_node.referenced_sop_instance_uids = inst_dict[
                        "ReferencedSOPInstanceUIDs"
                    ]
                if modality in ["REG", "SEG"]:
                    instance_node.referenced_sids.append(inst_dict["ReferencedSeriesUIDs"])
                    if modality == "REG":
                        instance_node.other_referenced_sids.append(
                            inst_dict["OtherReferencedSeriesUIDs"]
                        )
                if inst_dict.get("is_embedded_in_raw"):
                    series.is_embedded_in_raw = True
                    raw_series_reference_uid = inst_dict["raw_series_reference_uid"]
                    series.raw_series_reference = self.dicom_files[patient_id][
                        raw_series_reference_uid
                    ]
                metadata_list.append(
                    {key: val for key, val in inst_dict.items() if key not in exclude_keys}
                )
        DICOMLoader._associate_dicoms(self.dicom_files)

        self.metadata_df = pd.DataFrame(metadata_list)

        for col in self.metadata_df.columns:
            try:
                vr = dictionary_VR(Tag(tag_for_keyword(col))) if col != "InstanceNode" else None
            except TypeError:
                vr = dictionary_VR(Tag(col)) if col != "InstanceNode" else None
            dtype = VR_TO_DTYPE.get(vr, object)
            if dtype == "date":
                self.metadata_df[col] = pd.to_datetime(self.metadata_df[col], errors="coerce")
            elif dtype == "time":
                self.metadata_df[col] = pd.to_datetime(
                    self.metadata_df[col], format="%H:%M:%S", errors="coerce"
                ).dt.time
            elif dtype == "datetime":
                self.metadata_df[col] = pd.to_datetime(self.metadata_df[col], errors="coerce")
            else:
                self.metadata_df[col] = self.metadata_df[col].astype(dtype, errors="ignore")

    @staticmethod
    def _associate_dicoms(dicom_files):
        """
        Associates DICOM files based on referenced SOPInstanceUIDs and SeriesInstanceUIDs.

        This method builds lookup tables for SOPInstanceUIDs and SeriesInstanceUIDs and
        associates referenced DICOM instances and series by establishing connections between
        related DICOMs.

        Parameters
        ----------
        dicom_files : dict
            A dictionary where the processed DICOM data is stored, indexed by PatientID and
            SeriesInstanceUID.

        Examples
        --------
        >>> DICOMLoader._associate_dicoms(dicom_files)
        """
        # Create a lookup table for all SOPInstanceUIDs and SeriesInstanceUIDs across all patients
        sop_instance_uid_map = {}
        series_uid_map = {}
        frame_of_reference_uid_map = {}

        # Build the lookup maps
        for patient_id, series_dict in dicom_files.items():
            for series_uid, series in series_dict.items():
                series_uid_map[series_uid] = series
                for sop_instance_uid, instance_node in series.instances.items():
                    sop_instance_uid_map[sop_instance_uid] = instance_node

                # Map each FrameOfReferenceUID to a list of series sharing the same
                # FrameOfReferenceUID
                if series.FrameOfReferenceUID:
                    if series.FrameOfReferenceUID not in frame_of_reference_uid_map:
                        frame_of_reference_uid_map[series.FrameOfReferenceUID] = []
                    frame_of_reference_uid_map[series.FrameOfReferenceUID].append(series)

        # Now, associate instances based on their references
        for patient_id, series_dict in dicom_files.items():
            for series_uid, series in series_dict.items():
                for sop_uid, instance in series.instances.items():
                    modality = instance.Modality

                    # Initialize referenced_instances list
                    instance.referenced_instances = []

                    # General handling for referenced SOPInstanceUIDs
                    for ref_sop_uid in instance.referenced_sop_instance_uids:
                        ref_instance = sop_instance_uid_map.get(ref_sop_uid)
                        if ref_instance:
                            if ref_instance not in instance.referenced_instances:
                                instance.referenced_instances.append(ref_instance)
                            if ref_instance.parent_series not in instance.referenced_series:
                                instance.referenced_series.append(ref_instance.parent_series)
                            if instance not in ref_instance.referencing_instances:
                                ref_instance.referencing_instances.append(instance)

                        else:
                            # Reference to an instance not in dataset
                            pass

                    # Modality-specific associations
                    if modality in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
                        # RTSTRUCT references images via referenced_sop_instance_uids
                        referenced_series_uids = set()

                        for ref_instance in instance.referenced_instances:
                            ref_series_uid = ref_instance.parent_series.SeriesInstanceUID
                            referenced_series_uids.add(ref_series_uid)

                        for ref_sid in referenced_series_uids:
                            instance.referenced_sids.append(ref_sid)
                            ref_series = series_uid_map.get(ref_sid)
                            if ref_series:
                                if ref_series not in instance.referenced_series:
                                    instance.referenced_series.append(ref_series)

                    elif modality == "REG":
                        # REG references fixed image (referenced_sids) and
                        # moving image (other_referenced_sids)
                        ref_sids = instance.referenced_sids
                        other_ref_sids = instance.other_referenced_sids

                        if ref_sids:
                            for ref_sid in ref_sids:
                                ref_series = series_uid_map.get(ref_sid)
                                if ref_series:
                                    instance.referenced_series.append(ref_series)

                        if other_ref_sids:
                            for other_ref_sid in other_ref_sids:
                                other_ref_series = series_uid_map.get(other_ref_sid)
                                if other_ref_series:
                                    instance.other_referenced_series.append(other_ref_series)

                    elif modality == "SEG":
                        ref_sids = instance.referenced_sids
                        if ref_sids:
                            for ref_sid in ref_sids:
                                ref_series = series_uid_map.get(ref_sid)
                                if ref_series and ref_series not in instance.referenced_series:
                                    instance.referenced_series.append(ref_series)

                # Associate by FrameOfReferenceUID
                if series.FrameOfReferenceUID:
                    # Get all series sharing the same FrameOfReferenceUID
                    frame_of_reference_series = frame_of_reference_uid_map.get(
                        series.FrameOfReferenceUID, []
                    )
                    series.frame_of_reference_registered = [
                        s
                        for s in frame_of_reference_series
                        if s.SeriesInstanceUID != series.SeriesInstanceUID
                    ]

    def _build_hierarchical_structure(self):
        """
        Builds a hierarchical structure of DatasetNode, PatientNode, StudyNode, SeriesNode,
        and InstanceNode from the existing self.dicom_files.

        This method populates self.dataset with a DatasetNode instance containing PatientNode
        instances, and subsequently, the entire hierarchical structure.

        Returns
        -------
        None
        """
        # Initialize the DatasetNode as the root of the hierarchy
        dataset_id = "DICOM_Dataset"
        dataset_name = "DICOM Collection"
        self.dataset = DatasetNode(dataset_id, dataset_name)

        for patient_id, series_dict in self.dicom_files.items():
            # Create or get the PatientNode and add it to the DatasetNode
            if not self.dataset.get_patient(patient_id):
                any_series = next(iter(series_dict.values()))
                patient_name = any_series.PatientName
                patient_node = PatientNode(patient_id, patient_name, parent_dataset=self.dataset)
                self.dataset.add_patient(patient_node)

            if patient_id not in self.dataset.patients:
                # Assuming that PatientName is stored in one of the SeriesNodes
                any_series = next(iter(series_dict.values()))
                patient_name = any_series.PatientName
                patient_node = PatientNode(patient_id, patient_name, parent_dataset=self.dataset)
                self.dataset.add_patient(patient_node)
            else:
                patient_node = self.dataset.get_patient(patient_id)

            for series_uid, series_node in series_dict.items():
                # Retrieve StudyInstanceUID and StudyDescription from SeriesNode
                study_uid = series_node.StudyInstanceUID
                study_description = series_node.StudyDescription

                # Create or get the StudyNode and add it to the PatientNode
                if not patient_node.get_study(study_uid):
                    study_node = StudyNode(
                        study_uid, study_description, parent_patient=patient_node
                    )
                    patient_node.add_study(study_node)
                else:
                    study_node = patient_node.get_study(study_uid)

                # Add the SeriesNode to the StudyNode
                study_node.add_series(series_node)

                # Update SeriesNode attributes if necessary
                series_node.PatientID = patient_id
                series_node.PatientName = patient_node.PatientName
                series_node.StudyInstanceUID = study_uid
                series_node.StudyDescription = study_description

                # Link SeriesNode to the parent study node
                series_node.parent_study = study_node

    def _get_metadata(self, ds, tags_to_index, instance_node):
        """
        Extract metadata from the DICOM dataset for specified tags.

        Parameters
        ----------
        ds : pydicom.Dataset
            The DICOM dataset.
        tags_to_index : list
            List of DICOM tags to extract.
        instance_node : InstanceNode
            The associated instance node.

        Returns
        -------
        dict
            Metadata dictionary with normalized values.
        """
        metadata = {"InstanceNode": instance_node}
        for tag in tags_to_index:
            try:
                tag_obj = Tag(tag)
                vr = dictionary_VR(tag_obj)
                value = ds[tag_obj].value if tag_obj in ds else None
                if isinstance(value, Sequence) and vr == "SQ":
                    metadata[keyword_for_tag(tag) or tag] = (
                        (ds[tag_obj].to_json()) if value else None
                    )
                else:
                    metadata[keyword_for_tag(tag) or tag] = parse_vr_value(vr, value)
            except Exception as e:
                print(f"Exception occured:+: {e}")
                metadata[tag] = None

        return metadata

    def _normalize_tag(self, tag):
        """
        Normalizes a DICOM tag to its (group, element) representation.

        Parameters
        ----------
        tag : str or tuple
            The tag in either keyword (e.g., "PatientID") or (group, element) notation.

        Returns
        -------
        tuple
            The (group, element) representation of the tag if valid, otherwise None.
        """
        try:
            tag_obj = Tag(tag_for_keyword(tag))
            group_element = (f"{tag_obj.group:04X}", f"{tag_obj.element:04X}")
            return group_element
        except Exception:
            print(f"Unknown keyword '{tag}' ignored.")
            return None

    def _get_column_dtype(self, tag):
        """
        Determines the Pandas dtype for a given DICOM tag based on its VR.

        Parameters
        ----------
        tag : tuple
            The DICOM tag in (group, element) format.

        Returns
        -------
        type or str
            The corresponding Pandas dtype, or `object` if the VR is unknown.
        """
        try:
            vr = dictionary_VR(tag)
            return VR_TO_DTYPE.get(vr, object)
        except KeyError:
            return object

    def reindex(self, tags_to_index, add_to_existing=True):
        """
        Reindexes metadata with specified tags.

        Parameters
        ----------
        tags_to_index : list of str
            A list of DICOM tags (keywords) to index.
        add_to_existing : bool, optional
            If True, adds to the existing tag_index. If False, clears the tag_index first.
        """
        tags_to_index = [self._normalize_tag(tag) for tag in tags_to_index]
        tags_to_index = {tag for tag in tags_to_index if tag}

        if add_to_existing:
            existing_tags = list(self.metadata_df.columns)
            existing_tags.remove("InstanceNode")
            existing_tags = set(existing_tags)
            tags_to_index = list(existing_tags | tags_to_index)

        self.dicom_files = {}
        self.dataset = None
        self.metadata_df = None
        self.load(tags_to_index)

    def query(self, query_level="INSTANCE", **filters):
        """
        Queries the metadata DataFrame based on specified filters and query level,
        supporting advanced matching including wildcards, ranges, lists, regular expressions,
        and inverse regular expressions.

        Parameters
        ----------
        query_level : str, optional
            The hierarchical level to query within the DICOM metadata.
            One of {"PATIENT", "STUDY", "SERIES", "INSTANCE"}:
            - "PATIENT": Returns unique patient-level metadata.
            - "STUDY": Returns unique study-level metadata for each patient.
            - "SERIES": Returns unique series-level metadata for each study.
            - "INSTANCE": Returns individual instance-level metadata.
            Defaults to "INSTANCE".

        filters : dict
            Key-value pairs representing the query conditions. Each key is a column name
            (corresponding to a DICOM attribute), and its value is a condition.
            Supports the following types of filters:
            - **Exact Match**: {"column": "value"}
            Matches rows where the column equals the given value.
            - **Wildcard Matching**: {"column": "value*"} or {"column": "val?e"}
            Uses `*` to match multiple characters, `?` to match a single character.
            Escaped wildcards can be matched with `\*` or `\?`.
            - **Ranges**: {"column": {"gte": min_value, "lte": max_value}}
            Supports range operators:
            - `gte`: Greater than or equal to
            - `lte`: Less than or equal to
            - `gt`: Greater than
            - `lt`: Less than
            - `eq`: Equal to (alias for exact match)
            - `neq`: Not equal to
            - **Lists**: {"column": ["value1", "value2", "value3"]}
            Matches rows where the column value is in the provided list.
            - **Regular Expressions**: {"column": {"RegEx": "pattern"}}
            Matches rows where the column value matches the given regular expression.
            - **Inverse Regular Expressions**: {"column": {"NotRegEx": "pattern"}}
            Matches rows where the column value does not match the given regular expression.

        Returns
        -------
        pd.DataFrame
            A filtered DataFrame containing the matching results, restricted to the columns
            relevant for the specified query level, along with any filter-referenced columns.

        Notes
        -----
        - Wildcard filtering supports both `*` (matches zero or more characters) and `?`
        (matches exactly one character). For literal `*` or `?`, escape them with a backslash.
        - Range filters can be combined with wildcards, e.g., {"column": {"neq": "value*"}}.
        - Regular expressions provide precise pattern matching using RegEx syntax.
        - The output DataFrame includes columns specific to the query level:
            - "PATIENT": ["PatientID"]
            - "STUDY": ["PatientID", "StudyInstanceUID"]
            - "SERIES": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]
            - "INSTANCE": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
        Additional columns used in filters are also included in the result.

        Examples
        --------
        # Example 1: Query for all series with a specific modality
        >>> loader.query(query_level="SERIES", Modality="CT")
        PatientID  StudyInstanceUID  SeriesInstanceUID
        0       123        A001               S001
        1       456        A002               S002

        # Example 2: Wildcard and exact match query
        >>> loader.query(query_level="PATIENT", PatientID="1*")
        PatientID
        0       123
        1       101

        # Example 3: Query with a date range filter
        >>> loader.query(query_level="STUDY", StudyDate={"gte": "2023-01-01", "lte": "2023-06-30"})
        PatientID  StudyInstanceUID
        0       123        A001
        1       789        A003

        # Example 4: Combine range, wildcard, and list filters
        >>> loader.query(
        ...     query_level="INSTANCE",
        ...     Modality=["CT", "MR"],
        ...     StudyDate={"gte": "2023-01-01", "lte": "2023-12-31"},
        ...     SOPInstanceUID="I*"
        ... )
        PatientID  StudyInstanceUID  SeriesInstanceUID  SOPInstanceUID
        0       123        A001               S001             I001
        1       456        A002               S002             I002

        # Example 5: Escape literal wildcards
        >>> loader.query(query_level="PATIENT", PatientID="123\\*")
        PatientID
        0    123*

        # Example 5: RegEx query
        >>> loader.query(query_level="PATIENT", PatientID={"RegEx": "^1\\d{2}$"})
        PatientID
        0       123
        1       101

        # Example 6: Inverse RegEx query
        >>> loader.query(query_level="INSTANCE", SOPInstanceUID={"NotRegEx": "I\\d{3}"})
        PatientID  StudyInstanceUID  SeriesInstanceUID  SOPInstanceUID
        0       123        A001               S001          X001

        # Example 7: Combine RegEx with range filters
        >>> loader.query(
        ...     query_level="STUDY",
        ...     StudyDate={"gte": "2023-01-01", "lte": "2023-12-31"},
        ...     StudyInstanceUID={"RegEx": "^A.*"}
        ... )
        PatientID  StudyInstanceUID
        0       123        A001
        1       789        A003

        See Also
        --------
        query_df : Provides the underlying filtering functionality for DataFrames.
        """

        levels = {
            "PATIENT": ["PatientID"],
            "STUDY": ["PatientID", "StudyInstanceUID"],
            "SERIES": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
            "INSTANCE": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"],
        }

        query_level = query_level.upper()
        if query_level not in levels:
            raise ValueError(
                f"Invalid query level '{query_level}'. Must be one of {list(levels.keys())}."
            )

        # Validate and filter metadata using query_df
        filtered_df = query_df(self.metadata_df, **filters)

        # Retain only relevant columns for the specified query level
        relevant_columns = levels[query_level]
        filter_columns = [col for col in filters.keys() if col in filtered_df.columns]
        result_columns = list(set(relevant_columns + filter_columns))

        df_result = filtered_df[result_columns].copy()

        # Convert unhashable types to hashable ones for deduplication
        for col in df_result.columns:
            if df_result[col].apply(lambda x: isinstance(x, list)).any():
                df_result[col] = df_result[col].apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )

        return df_result.drop_duplicates().reset_index(drop=True)

    def process_in_parallel(self, func, level="INSTANCE", num_workers=None):
        """
        Applies a user-defined function to all patients, studies, series, or instances
        in the dataset in parallel.

        Parameters
        ----------
        func : callable
            A function to apply to each patient, study, series, or instance.
            It must accept a single argument.
        level : str, optional
            The level to apply the function: "PATIENT", "STUDY", "SERIES", or "INSTANCE".
             Default is "instance".
        num_workers : int or None, optional
            The number of parallel workers to use. If None, uses all available CPU cores.
            Default is None.

        Returns
        -------
        list
            A list of results from applying the function.

        Examples
        --------
        >>> def process_instance(instance):
        ...     return {"SOPInstanceUID": instance.SOPInstanceUID, "FilePath": instance.filepath}
        >>> results = loader.process_in_parallel(process_instance, level="instance", num_workers=8)
        """
        from multiprocessing import Pool

        # Determine the number of workers to use
        num_workers = num_workers or os.cpu_count() or 1

        # Get items to process based on the specified level
        items = []
        if level.upper() == "INSTANCE":
            for patient in self.dataset:
                for study in patient:
                    for series in study:
                        items.extend(series)
        elif level.upper() == "SERIES":
            for patient in self.dataset:
                for study in patient:
                    items.extend(study)
        elif level.upper() == "STUDY":
            for patient in self.dataset:
                items.extend(patient)
        elif level.upper() == "PATIENT":
            items.extend(self.dataset)
        else:
            raise ValueError(
                f"Invalid level '{level}'. "
                "Must be one of: 'PATIENT', 'STUDY', 'SERIES', or 'INSTANCE'."
            )

        if not items:
            return []

        results = []
        errors = []
        with Pool(num_workers) as pool:
            for item in tqdm(items, total=len(items), desc=f"Processing {level}s", unit=level):
                try:
                    result = pool.apply(func, (item,))
                    results.append(result)
                except Exception as e:
                    errors.append((item, str(e)))

        return results, errors

    def process_in_parallel_threads(self, func, level="INSTANCE", num_workers=None):
        """
        Applies a user-defined function to all patients, studies, series, or instances
        in the dataset in parallel using multi-threading.

        Parameters
        ----------
        func : callable
            A function to apply to each patient, study, series, or instance.
            It must accept a single argument.
        level : str, optional
            The level to apply the function: "PATIENT", "STUDY", "SERIES", or "INSTANCE".
            Default is "instance".
        num_workers : int or None, optional
            The number of parallel workers to use. If None, uses the default ThreadPoolExecutor
            settings. Default is None.

        Returns
        -------
        list
            A list of results from applying the function.

        Examples
        --------
        >>> def process_instance(instance):
        ...     return {"SOPInstanceUID": instance.SOPInstanceUID, "FilePath": instance.FilePath}
        >>> results = loader.process_in_parallel_threads(
        ...     process_instance, level="instance", num_workers=8
        ... )
        """
        # Get items to process based on the specified level
        items = []
        if level.upper() == "INSTANCE":
            for patient in self.dataset:
                for study in patient:
                    for series in study:
                        items.extend(series)
        elif level.upper() == "SERIES":
            for patient in self.dataset:
                for study in patient:
                    items.extend(study)
        elif level.upper() == "STUDY":
            for patient in self.dataset:
                items.extend(patient)
        elif level.upper() == "PATIENT":
            items.extend(self.dataset)
        else:
            raise ValueError(
                f"Invalid level '{level}'."
                "Must be one of: 'PATIENT', 'STUDY', 'SERIES', or 'INSTANCE'."
            )

        if not items:
            return []

        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in items}

            for future in tqdm(
                as_completed(future_to_item),
                total=len(items),
                desc=f"Processing {level}s",
                unit=level,
            ):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append((item, str(e)))

        return results, errors

    def get_summary(self):
        """
        Returns a summary of the entire DICOM dataset.

        Returns
        -------
        dict
            A dictionary containing the total counts of patients, studies, series, and instances.
        """
        if not self.dataset:
            return {
                "total_patients": 0,
                "total_studies": 0,
                "total_series": 0,
                "total_instances": 0,
            }

        num_patients = len(self.dataset)
        num_studies = 0
        num_series = 0
        num_instances = 0

        for patient in self.dataset:
            num_studies += len(patient)
            for study in patient:
                num_series += len(study)
                for series in study:
                    num_instances += len(series)

        summary = {
            "total_patients": num_patients,
            "total_studies": num_studies,
            "total_series": num_series,
            "total_instances": num_instances,
        }

        return summary

    def get_patient_summary(self, patient_id):
        """
        Returns a summary of all studies and series for the specified patient.

        Parameters
        ----------
        patient_id : str
            The PatientID of the patient to summarize.

        Returns
        -------
        dict or None
            A dictionary containing the patient's studies and series information,
            or None if the patient_id is not found.
        """
        if not self.dataset or patient_id not in self.dataset.patients:
            return None

        patient_node = self.dataset.get_patient(patient_id)
        patient_summary = {
            "patient_id": patient_node.PatientID,
            "patient_name": patient_node.PatientName,
            "studies": [],
        }

        for study_node in patient_node:
            # Use get_study_summary to get detailed study information
            study_summary = self.get_study_summary(study_node.StudyInstanceUID)
            if study_summary:
                patient_summary["studies"].append(study_summary)

        return patient_summary

    def get_study_summary(self, study_uid):
        """
        Returns a summary of all series and instances within the specified study.

        Parameters
        ----------
        study_uid : str
            The StudyInstanceUID of the study to summarize.

        Returns
        -------
        dict or None
            A dictionary containing the study's series and instances information,
            or None if the study_uid is not found.
        """
        for patient_node in self.dataset:
            if study_uid in patient_node.studies:
                study_node = patient_node.get_study(study_uid)
                study_summary = {
                    "patient_id": patient_node.PatientID,
                    "patient_name": patient_node.PatientName,
                    "study_uid": study_node.StudyInstanceUID,
                    "study_description": study_node.StudyDescription,
                    "series": [],
                }

                for series_node in study_node:
                    # Use get_series_summary to get detailed series information
                    series_summary = self.get_series_summary(series_node.SeriesInstanceUID)
                    if series_summary:
                        study_summary["series"].append(series_summary)

                return study_summary

        return None

    def get_series_summary(self, series_uid):
        """
        Returns detailed information about the specified series, including its instances.

        Parameters
        ----------
        series_uid : str
            The SeriesInstanceUID of the series to summarize.

        Returns
        -------
        dict or None
            A dictionary containing the series information and its instances,
            or None if the series_uid is not found.
        """
        for patient_node in self.dataset:
            for study_node in patient_node:
                if series_uid in study_node.series:
                    series_node = study_node.get_series(series_uid)
                    series_summary = {
                        "PatientID": patient_node.PatientID,
                        "PatientName": patient_node.PatientName,
                        "StudyInstanceUID": study_node.StudyInstanceUID,
                        "StudyDescription": study_node.StudyDescription,
                        "SeriesInstanceUID": series_node.SeriesInstanceUID,
                        "SeriesDescription": series_node.SeriesDescription,
                        "Modality": series_node.Modality,
                        "NumInstances": len(series_node),
                        "Instances": [],
                    }

                    for instance_node in series_node:
                        instance_info = {
                            "SOPInstanceUID": instance_node.SOPInstanceUID,
                            "Modality": instance_node.Modality,
                            "FilePath": instance_node.FilePath,
                        }
                        series_summary["Instances"].append(instance_info)

                    return series_summary

        return None

    def get_modality_distribution(self):
        """
        Returns the distribution of modalities in the dataset, with special handling for certain
        modalities.

        This method iterates over all `SeriesNode` objects in the dataset and calculates the
        distribution of modalities. For modalities like `RTPLAN`, `RTDOSE`, `RTSTRUCT`, and
        `RTRECORD`, the counts are based on the number of `InstanceNode` objects within those
        series. For other modalities, the count is based on the number of `SeriesNode` objects.

        Returns
        -------
        dict
            A dictionary where keys are modalities and values are counts. For `RTPLAN`, `RTDOSE`,
            `RTSTRUCT`, and `RTRECORD`, the values represent the total number of instances.
            For all other modalities, the values represent the number of series.

        Examples
        --------
        >>> distribution = loader.get_modality_distribution()
        >>> print(distribution)
        {'CT': 10, 'MR': 5, 'RTPLAN': 3, 'RTSTRUCT': 8, 'RTDOSE': 5, 'Unknown': 2}
        """
        modality_counts = {}

        for patient_node in self.dataset:
            for study_node in patient_node:
                for series_node in study_node:
                    modality = series_node.Modality or "Unknown"
                    if modality in ["RTPLAN", "RTDOSE", "RTSTRUCT", "RTRECORD"]:
                        for instance_node in series_node:
                            modality_counts[modality] = modality_counts.get(modality, 0) + 1
                    else:
                        modality_counts[modality] = modality_counts.get(modality, 0) + 1

        return modality_counts

    def get_patient_ids(self):
        """
        Returns a list of all PatientIDs in the dataset.

        Returns
        -------
        list of str
            A list of PatientIDs.
        """
        return list(self.dataset.patients.keys())

    def get_study_uids(self, patient_id):
        """
        Returns a list of StudyInstanceUIDs for the specified patient.

        Parameters
        ----------
        patient_id : str
            The PatientID of the patient.

        Returns
        -------
        list of str
            A list of StudyInstanceUIDs, or an empty list if the patient is not found.
        """
        patient_node = self.dataset.get_patient(patient_id)
        if patient_node is None:
            return []
        return list(patient_node.studies.keys())

    def get_series_uids(self, study_uid):
        """
        Returns a list of SeriesInstanceUIDs for the specified study.

        Parameters
        ----------
        study_uid : str
            The StudyInstanceUID of the study.

        Returns
        -------
        list of str
            A list of SeriesInstanceUIDs, or an empty list if the study is not found.
        """
        for patient_node in self.dataset:
            study_node = patient_node.get_study(study_uid)
            if study_node:
                return list(study_node.series.keys())
        return []

    def get_series_paths(self, patient_id, series_uid):
        """
        Returns the file paths for all instances in a specific series.

        Parameters
        ----------
        patient_id : str
            The PatientID of the series to retrieve.
        series_uid : str
            The SeriesInstanceUID of the series to retrieve.

        Returns
        -------
        list of str
            A list of file paths for the specified series.

        Raises
        ------
        ValueError
            If the specified series is not found for the given patient.
        """
        patient_node = self.dataset.get_patient(patient_id)
        if patient_node is None:
            raise ValueError(f"Patient {patient_id} not found.")

        for study_node in patient_node:
            series_node = study_node.get_series(series_uid)
            if series_node:
                return series_node.instance_paths

        raise ValueError(f"Series {series_uid} for Patient {patient_id} not found.")

    def get_patient(self, patient_id):
        """
        Retrieves a PatientNode by its PatientID.

        Parameters
        ----------
        patient_id : str
            The PatientID of the patient to retrieve.

        Returns
        -------
        PatientNode or None
            The `PatientNode` associated with the given patient_id, or None if not found.
        """
        return self.dataset.get_patient(patient_id) if self.dataset else None

    def get_study(self, study_uid):
        """
        Retrieves a StudyNode by its StudyInstanceUID.

        Parameters
        ----------
        study_uid : str
            The StudyInstanceUID of the study to retrieve.

        Returns
        -------
        StudyNode or None
            The `StudyNode` associated with the given study_uid, or None if not found.
        """
        for patient_node in self.dataset:
            study_node = patient_node.get_study(study_uid)
            if study_node:
                return study_node
        return None

    def get_series(self, series_uid):
        """
        Retrieves a SeriesNode by its SeriesInstanceUID.

        Parameters
        ----------
        series_uid : str
            The SeriesInstanceUID of the series to retrieve.

        Returns
        -------
        SeriesNode or None
            The `SeriesNode` associated with the given series_uid, or None if not found.
        """
        for patient_node in self.dataset:
            for study_node in patient_node:
                series_node = study_node.get_series(series_uid)
                if series_node:
                    return series_node
        return None

    def get_instance(self, sop_instance_uid):
        """
        Retrieves an InstanceNode by its SOPInstanceUID.

        Parameters
        ----------
        sop_instance_uid : str
            The SOPInstanceUID of the instance to retrieve.

        Returns
        -------
        InstanceNode or None
            The `InstanceNode` associated with the given sop_instance_uid, or None if not found.
        """
        for patient_node in self.dataset:
            for study_node in patient_node:
                for series_node in study_node:
                    instance_node = series_node.get_instance(sop_instance_uid)
                    if instance_node:
                        return instance_node
        return None

    def read_series(self, series_uid):
        """
        Reads a DICOM series based on its SeriesInstanceUID and returns an appropriate
        representation of the series using modality-specific readers.

        This method first searches for the series with the given SeriesInstanceUID in the
        loaded DICOM data within the dataset graph. It then selects the appropriate reader
        based on the modality of the series and reads the data accordingly. If the series
        is embedded in a RAW file, it extracts the embedded datasets and reads them.

        Parameters
        ----------
        series_uid : str
            The unique SeriesInstanceUID of the series to be read.

        Returns
        -------
        list
            A list of objects representing the series. For DICOM-RT modalities
            (e.g., RTSTRUCT, RTDOSE), each instance is read separately, and the
            results are returned as a list of objects. For embedded series in RAW files,
            the embedded datasets are extracted and returned as a list. If the series has
            only one instance, a list containing one object is returned.

        Raises
        ------
        ValueError
            If no series with the given SeriesInstanceUID is found in the loaded DICOM files.
        NotImplementedError
            If a reader for this modality type is not implemented yet.

        Examples
        --------
        >>> loader = DICOMLoader("/path/to/dicom/files")
        >>> loader.load()
        >>> dicom_image = loader.read_series("1.2.840.113619.2.55.3")[0]
        >>> rtstruct = loader.read_series("1.2.840.113619.2.55.4")[0]
        """
        # Retrieve the series using the get_series method
        found_series = self.get_series(series_uid)
        if not found_series:
            raise ValueError(f"Series with SeriesInstanceUID '{series_uid}' not found.")

        if found_series is None:
            raise ValueError(f"Series with SeriesInstanceUID '{series_uid}' not found.")

        # Determine the modality and handle accordingly
        modality = found_series.Modality

        if found_series.is_embedded_in_raw:
            raw_series_reference = found_series.raw_series_reference
            embedded_datasets = (
                DICOMRawReader(raw_series_reference.SOPInstances[0]).read().get_embedded_datasets()
            )
            embedded_series = [
                self._read_embedded(dataset)
                for dataset in embedded_datasets
                if dataset.SeriesInstanceUID == series_uid
            ]
            return embedded_series

        if modality in ["CT", "MR", "PT"]:
            return [DICOMImageReader(found_series.instance_paths).read()]

        elif modality == "RTSTRUCT":
            return [
                RTStructReader(instance_path).read()
                for instance_path in found_series.instance_paths
            ]

        elif modality == "RTDOSE":
            return [
                RTDoseReader(instance_path).read() for instance_path in found_series.instance_paths
            ]

        elif modality == "REG":
            return [
                REGReader(instance_path).read() for instance_path in found_series.instance_paths
            ]

        elif modality == "RTPLAN":
            return [
                RTPlanReader(instance_path).read() for instance_path in found_series.instance_paths
            ]

        elif modality == "RTRECORD":
            return [
                RTRecordReader(instance_path).read()
                for instance_path in found_series.instance_paths
            ]

        elif modality == "SEG":
            return [
                SEGReader(instance_path).read() for instance_path in found_series.instance_paths
            ]

        else:
            raise NotImplementedError(f"A reader for {modality} type is not implemented yet.")

    def read_instance(self, sop_instance_uid):
        """
        Reads a single DICOM instance based on its SOPInstanceUID and returns an appropriate
        representation of the instance using modality-specific readers.

        This method searches within the dataset graph to locate the instance with the given
        SOPInstanceUID. It then selects the appropriate reader based on the modality of the
        series to which the instance belongs and reads the data accordingly.

        Parameters
        ----------
        sop_instance_uid : str
            The unique SOPInstanceUID of the instance to be read.

        Returns
        -------
        object
            An object representing the instance. This object type depends on the modality of
            the instance (e.g., RTStruct, RTDose, DICOMImage).



        Raises
        ------
        ValueError
            If no instance with the given SOPInstanceUID is found in the loaded DICOM files.
        NotImplementedError
            If a reader for this modality type is not implemented yet.

        Examples
        --------
        >>> loader = DICOMLoader("/path/to/dicom/files")
        >>> loader.load()
        >>> instance = loader.read_instance("1.2.840.113619.2.55.3.1234")
        >>> print(instance)
        """
        # Retrieve the instance
        found_instance = self.get_instance(sop_instance_uid)

        if not found_instance:
            raise ValueError(f"Instance with SOPInstanceUID '{sop_instance_uid}' not found.")

        # Determine the modality and use the appropriate reader
        modality = found_instance.Modality
        filepath = found_instance.FilePath

        if modality in ["CT", "MR", "PT"]:
            return DICOMImageReader(filepath).read()

        elif modality == "RTSTRUCT":
            return RTStructReader(filepath).read()

        elif modality == "RTDOSE":
            return RTDoseReader(filepath).read()

        elif modality == "REG":
            return REGReader(filepath).read()

        elif modality == "RTPLAN":
            return RTPlanReader(filepath).read()

        elif modality == "RTRECORD":
            return RTRecordReader(filepath).read()

        elif modality == "SEG":
            return SEGReader(filepath).read()

        else:
            raise NotImplementedError(f"A reader for {modality} type is not implemented yet.")

    def _read_embedded(self, dataset):
        """
        Reads an embedded DICOM dataset from a RAW file based on its modality and returns
        the appropriate object using modality-specific readers.

        This method is used internally to handle embedded datasets in RAW files. It selects
        the appropriate reader based on the modality of the embedded dataset and reads the
        data accordingly.

        Parameters
        ----------
        dataset : pydicom.Dataset
            The embedded DICOM dataset to be read. This dataset is typically extracted
            from a RAW file.

        Returns
        -------
        object
            The appropriate representation of the embedded dataset based on its modality.
            For example, if the dataset represents a CT image, it returns a `DICOMImage`
            object. If the dataset represents an RTSTRUCT, it returns an `RTStruct` object.
        """
        if dataset.Modality in ["CT", "MR", "PT"]:
            return DICOMImageReader(dataset).read()
        elif dataset.Modality == "RTSTRUCT":
            return RTStructReader(dataset).read()
        elif dataset.Modality == "RTDOSE":
            return RTDoseReader(dataset).read()
        elif dataset.Modality == "REG":
            return REGReader(dataset)
        elif dataset.Modality == "RTPLAN":
            return RTPlanReader(dataset)
        elif dataset.Modality == "RTRECORD":
            return RTRecordReader(dataset)

    @staticmethod
    def get_referencing_items(node, modality=None, level="INSTANCE"):
        """
        Retrieves all referencing items (instances or series) for a given node.

        Parameters
        ----------
        node : InstanceNode or SeriesNode
            The starting node for searching referencing items.
        modality : str, optional
            Modality to filter the referencing items (e.g., 'CT', 'MR').
        level : str, {'INSTANCE', 'SERIES'}
            Level at which to retrieve referencing items.

        Returns
        -------
        list
            A list of referencing InstanceNode or SeriesNode objects.
        """
        if level not in {"INSTANCE", "SERIES"}:
            raise ValueError("level must be either 'INSTANCE' or 'SERIES'")

        referencing_items = []

        if isinstance(node, InstanceNode):
            # Get instances that reference this instance
            if level == "INSTANCE":
                referencing_items = node.referencing_instances
            else:  # level == "SERIES"
                referencing_items = (
                    node.parent_series.referencing_series if node.parent_series else []
                )

        elif isinstance(node, SeriesNode):
            # Get referencing instances
            referencing_instances = []
            for inst in node.instances.values():
                referencing_instances.extend(inst.referencing_instances)
            referencing_instances = list(set(referencing_instances))
            if level == "INSTANCE":
                referencing_items = referencing_instances
            else:  # level == "SERIES"
                referencing_items = list(
                    set([item.parent_series for item in referencing_instances])
                )

        else:
            raise TypeError("Expected an InstanceNode or SeriesNode.")

        # Filter by modality if specified
        if modality:
            referencing_items = [
                item for item in referencing_items if getattr(item, "Modality", None) == modality
            ]

        return referencing_items

    @staticmethod
    def get_referenced_nodes(
        node: Union[SeriesNode, InstanceNode],
        modality: Optional[str] = None,
        level: str = "INSTANCE",
        recursive: bool = True,
    ) -> List[Union[SeriesNode, InstanceNode]]:
        """
        Retrieves referenced nodes of a specified level and modality from a
        SeriesNode or InstanceNode.

        This function returns directly referenced nodes by default, and can also traverse
        the reference graph recursively to find indirectly referenced nodes if `recursive=True`.

        Parameters
        ----------
        node : InstanceNode or SeriesNode
            The node to start from (e.g., RTDOSE instance, CT series).

        modality : str, optional
            If specified, filters returned nodes by Modality (e.g., "CT", "RTSTRUCT").
            If None, all modalities are included.

        level : str
            One of {"INSTANCE", "SERIES"} (case-insensitive).
            Determines the type of nodes to return:
            - "INSTANCE": Returns InstanceNode objects
            - "SERIES": Returns SeriesNode objects

        recursive : bool, optional (default: True)
            If True, traverses both direct and indirect references recursively.
            If False, only returns directly referenced nodes.

        Returns
        -------
        List[InstanceNode or SeriesNode]
            A list of referenced nodes matching the given criteria.

        Raises
        ------
        ValueError
            If the level is not one of {"INSTANCE", "SERIES"}.

        Examples
        --------
        >>> # Get all CT series (direct + indirect) from an RTDOSE instance
        >>> get_referenced_nodes(rtdose_instance, modality="CT", level="series")

        >>> # Get only directly referenced RTPLAN instances
        >>> get_referenced_nodes(
        >>>     dose_inst,
        >>>     modality="RTPLAN",
        >>>     level="instance",
        >>>     recursive=False
        >>> )
        """
        level = level.upper()
        if level not in {"INSTANCE", "SERIES"}:
            raise ValueError("level must be 'INSTANCE' or 'SERIES'")

        visited = set()
        results = []

        def traverse(n):
            if id(n) in visited:
                return
            visited.add(id(n))

            # Yield the node if it's of the right level and matches modality
            if level == "INSTANCE" and isinstance(n, InstanceNode):
                if modality is None or getattr(n, "Modality", None) == modality:
                    results.append(n)

            elif level == "SERIES" and isinstance(n, SeriesNode):
                if modality is None or getattr(n, "Modality", None) == modality:
                    results.append(n)

            # Traverse deeper if recursive is enabled
            if recursive:
                # Always follow both instance and series links regardless of output level
                if isinstance(n, SeriesNode):
                    for instance in n.instances.values():
                        for ref in instance.referenced_instances:
                            traverse(ref)
                        for ref_series in instance.referenced_series:
                            traverse(ref_series)
                elif isinstance(n, InstanceNode):
                    for ref in n.referenced_instances:
                        traverse(ref)
                    for ref_series in n.referenced_series:
                        traverse(ref_series)

        traverse(node)
        return results

    @staticmethod
    def get_nodes_for_patient(
        patient_node,
        level="SERIES",
        modality=None,
        uid=None,
    ):
        """
        Retrieves StudyNode, SeriesNode, or InstanceNode objects from a given PatientNode.

        Parameters
        ----------
        patient_node : PatientNode
            The patient node to search under.

        level : str, optional
            One of {"STUDY", "SERIES", "INSTANCE"} (case-insensitive).
            Determines which level of nodes to return. Default is "SERIES".

        modality : str, optional
            If specified, filters nodes by Modality (only applicable for SERIES/INSTANCE levels).

        uid : str, optional
            If specified, filters for a specific UID:
            - For level="STUDY": matches StudyInstanceUID
            - For level="SERIES": matches SeriesInstanceUID
            - For level="INSTANCE": matches SOPInstanceUID

        Returns
        -------
        List[StudyNode | SeriesNode | InstanceNode]
            A list of matching nodes at the requested level.
            If `uid` is specified, returns at most one element.

        Raises
        ------
        ValueError
            If `level` is not one of {"STUDY", "SERIES", "INSTANCE"}.
        """
        level = level.upper()
        if level not in {"STUDY", "SERIES", "INSTANCE"}:
            raise ValueError("level must be 'STUDY', 'SERIES', or 'INSTANCE'")

        results = []

        for study_node in patient_node:
            if level == "STUDY":
                if uid and study_node.StudyInstanceUID != uid:
                    continue
                results.append(study_node)

            elif level == "SERIES":
                for series_node in study_node:
                    if uid and series_node.SeriesInstanceUID != uid:
                        continue
                    if modality and series_node.Modality != modality:
                        continue
                    results.append(series_node)

            elif level == "INSTANCE":
                for series_node in study_node:
                    for instance_node in series_node:
                        if uid and instance_node.SOPInstanceUID != uid:
                            continue
                        if modality and instance_node.Modality != modality:
                            continue
                        results.append(instance_node)

        return results

    def visualize_series_references(
        self,
        patient_id=None,
        output_file=None,
        view=True,
        per_patient=False,
        exclude_modalities=None,
        exclude_series=[],
        include_uid=False,
        rankdir="BT",
    ):
        """
        Visualizes the series-level associations for all patients or a specific patient using
        Graphviz. Each series is represented as a box, and an edge is drawn from a series to its
        referenced series. The patient ID will be the top node, followed by root series (e.g., CT)
        and referenced series (e.g., RTDOSE).

        Parameters
        ----------
        patient_id : str or None, optional
            If provided, only generates the graph for the specified patient. This takes priority
            over `per_patient`.
        output_file : str or None, optional
            The name of the output file for the graph visualization. If None, the graph will not
            be saved. If `per_patient=True`, this will serve as a prefix for the patient-specific
            files.
        view : bool, optional
            Whether to automatically view the graph after it's generated using `matplotlib` or
            another viewer.
        per_patient : bool, optional
            Whether to create separate graphs for each patient. If False, all patients are
            visualized in one graph.
        exclude_modalities : list of str, optional
            A list of modalities to exclude from the visualization. If None, all modalities are
            included.
        exclude_series : list of str, optional
            A list of SeriesInstanceUIDs to exclude from the graph. If None or empty, no series
            are excluded.
        include_uid : bool, optional
            Whether to include the (SOP/Series)InstanceUID in the label for each node.
        rankdir : str, optional
            The direction of the graph layout. Must be one of ['RL', 'LR', 'BT', 'TB'].


        Returns
        -------
        None
        """
        if rankdir not in ["RL", "LR", "BT", "TB"]:
            raise ValueError(f"{rankdir} is not a valid option for rankdir")

        # define color mappings based on modality
        modality_colors = {
            "CT": "lightsteelblue",
            "MR": "lightseagreen",
            "PT": "lightcoral",
            "RTSTRUCT": "navajowhite",
            "RTPLAN": "lightgoldenrodyellow",
            "RTDOSE": "lightpink",
            "RTRECORD": "lavender",
            "REG": "thistle",
            "SEG": "peachpuff",
            "DEFAULT": "lightgray",
        }
        patient_color = "dodgerblue"
        raw_subgraph_color = "lightcyan"

        def study_color_generator():
            study_subgraph_colors = [
                "honeydew",
                "lavenderblush",
                "azure",
                "seashell",
                "mintcream",
                "mistyrose",
                "aliceblue",
                "powderblue",
                "oldlace",
            ]
            while True:
                for color in study_subgraph_colors:
                    yield color

        def get_modality_color(modality):
            """
            Helper function to get the background color based on the modality.
            """
            return modality_colors.get(modality, modality_colors["DEFAULT"])

        def get_referenced_series(series):
            referenced_series = set()
            for sop_uid, instance in series.instances.items():
                if instance.referenced_series:
                    for ref_series in instance.referenced_series:
                        referenced_series.add(ref_series)

            return referenced_series

        def get_other_referenced_series(series):
            referenced_series = set()
            for sop_uid, instance in series.instances.items():
                if instance.other_referenced_series:
                    for ref_series in instance.other_referenced_series:
                        referenced_series.add(ref_series)

            return referenced_series

        def get_frame_registered_image_series(series):
            referenced_series = set()
            for series in series.frame_of_reference_registered:
                if series.Modality in ["CT", "MR", "PT"]:
                    referenced_series.add(series)
            return referenced_series

        def exclude_referenced(
            series, exclude_modalities=exclude_modalities, exclude_series=exclude_series
        ):
            if exclude_modalities and series.Modality in exclude_modalities:
                return True
            if exclude_series and series.SeriesInstanceUID in exclude_series:
                return True
            return False

        def create_graph(patient_id, series_dict, graph):
            """
            Helper function to create a graph for a specific patient.
            """
            # Add patient ID as the top node for each patient's graph
            graph.node(
                patient_id,
                label=(
                    f"Patient ID: {patient_id}\n"
                    f"{series_dict[list(series_dict.keys())[0]].PatientName}"
                ),
                fillcolor=patient_color,
                style="filled",
            )
            # group series based on their study instance uid
            grouped_series = {}
            for series_uid, series in series_dict.items():
                if series.StudyInstanceUID:
                    study_uid = series.StudyInstanceUID
                    if study_uid not in grouped_series:
                        grouped_series[study_uid] = {}
                    grouped_series[study_uid][series_uid] = series
                else:
                    if "UNK" not in grouped_series:
                        grouped_series["UNK"] = {}
                    grouped_series["UNK"][series_uid] = series

            # for each group draw subgraph
            all_nodes_set = set()
            referencing_nodes_set = set()
            color_cycle = study_color_generator()

            # first pass: create nodes only
            for study_uid, grouped in grouped_series.items():
                first_sid = next(iter(grouped))
                first_series = grouped[first_sid]
                study_desc = first_series.StudyDescription
                ct_mr_pt_nodes = []
                with graph.subgraph(name=f"cluster_{study_uid}") as study_graph:
                    if include_uid:
                        label_rg = (
                            f"StudyDescription: {study_desc}" f"\nStudyInstanceUID: {study_uid}"
                        )

                    else:
                        label_rg = f"StudyDescription: {study_desc}"

                    label_loc = "b" if rankdir == "BT" else "t"
                    # label_loc = "t"
                    study_subgraph_color = next(color_cycle)
                    study_graph.attr(
                        label=label_rg,
                        labelloc=label_loc,
                        color="black",
                        style="filled",
                        fillcolor=study_subgraph_color,
                    )
                    for series_uid, series in grouped.items():

                        # Exclude modalities if specified
                        if exclude_modalities and series.Modality in exclude_modalities:
                            continue

                        if series.SeriesInstanceUID in exclude_series:
                            continue

                        if series.Modality == "RAW":
                            continue

                        if exclude_modalities and "RAW" in exclude_modalities:
                            if series.is_embedded_in_raw:
                                continue

                        if series.Modality in ["CT", "MR", "PT"]:
                            ct_mr_pt_nodes.append(series.SeriesInstanceUID)

                        # get the color based on modality
                        node_color = get_modality_color(series.Modality)

                        # handle embedded series in RAW
                        if series.is_embedded_in_raw:
                            # create another subgraph for the embedded series within the RAW series
                            with study_graph.subgraph(
                                name=f"cluster_{series.raw_series_reference.SeriesInstanceUID}"
                            ) as raw_graph:
                                if include_uid:
                                    label_r = (
                                        f"MIM Session: "
                                        f"{series.raw_series_reference.SeriesDescription}"
                                        "\nSeriesInstanceUID: "
                                        f"{series.raw_series_reference.SeriesInstanceUID}"
                                    )
                                else:
                                    label_r = (
                                        "MIM Session: "
                                        f"{series.raw_series_reference.SeriesDescription}"
                                    )
                                raw_graph.attr(
                                    label=label_r,
                                    color="black",
                                    style="filled",
                                    fillcolor=raw_subgraph_color,
                                )

                                # italicize the embedded series
                                if include_uid:
                                    label = (
                                        f"{series.Modality}: {series.SeriesDescription}"
                                        f"\n{series.SeriesInstanceUID}"
                                    )
                                else:
                                    label = f"{series.Modality}: {series.SeriesDescription}"
                                raw_graph.node(
                                    series.SeriesInstanceUID,
                                    label=label,
                                    shape="box",
                                    style="filled",
                                    fontcolor="black",
                                    fontname="Times-Italic",
                                    fillcolor=node_color,
                                )
                                all_nodes_set.add(series.SeriesInstanceUID)
                        else:
                            if series.Modality in [
                                "RTSTRUCT",
                                "RTPLAN",
                                "RTDOSE",
                                "RTRECORD",
                                "SEG",
                            ]:
                                # Add each instance separately as a node
                                for sop_uid, instance in series.instances.items():
                                    if include_uid:
                                        label = (
                                            f"{series.Modality}: {series.SeriesDescription}"
                                            f"\nSOPInstanceUID: {sop_uid}"
                                        )
                                    else:
                                        label = f"{series.Modality}: {series.SeriesDescription}"
                                    node_color = get_modality_color(series.Modality)
                                    study_graph.node(
                                        sop_uid,
                                        label=label,
                                        style="filled",
                                        fillcolor=node_color,
                                    )
                                    all_nodes_set.add(sop_uid)

                            else:
                                # Add each series as a node (box)
                                if include_uid:
                                    label = (
                                        f"{series.Modality}: {series.SeriesDescription}"
                                        f"\nSeriesInstanceUID: {series.SeriesInstanceUID}"
                                    )
                                else:
                                    label = f"{series.Modality}: {series.SeriesDescription}"
                                node_color = get_modality_color(series.Modality)
                                study_graph.node(
                                    series.SeriesInstanceUID,
                                    label=label,
                                    style="filled",
                                    fillcolor=node_color,
                                )
                                all_nodes_set.add(series.SeriesInstanceUID)

                    # Enforce same rank for CT, MR, PT
                    if ct_mr_pt_nodes:
                        with study_graph.subgraph() as same_rank:
                            same_rank.attr(rank="same")
                            for node in ct_mr_pt_nodes:
                                same_rank.node(node)
            # second pass: add edges based on references
            for study_uid, grouped in grouped_series.items():
                if study_uid != "UNK":
                    # if True:
                    for series_uid, series in grouped.items():
                        # Exclude modalities if specified
                        if exclude_modalities and series.Modality in exclude_modalities:
                            continue

                        if series.SeriesInstanceUID in exclude_series:
                            continue

                        if series.Modality == "RAW":
                            continue

                        if exclude_modalities and "RAW" in exclude_modalities:
                            if series.is_embedded_in_raw:
                                continue

                        if series.is_embedded_in_raw:
                            continue

                        if series.Modality in [
                            "RTSTRUCT",
                            "RTPLAN",
                            "RTDOSE",
                            "RTRECORD",
                            "SEG",
                        ]:
                            # Add each instance separately as a node
                            for sop_uid, instance in series.instances.items():
                                # Check for direct references to other nodes
                                if series.Modality in ["RTSTRUCT", "SEG"]:
                                    referenced_series_list = instance.referenced_series
                                    if referenced_series_list:
                                        for referenced_series in referenced_series_list:
                                            if not exclude_referenced(referenced_series):
                                                referencing_nodes_set.add(instance.SOPInstanceUID)

                                                # Draw an edge pointing *upwards* from the
                                                # referenced node to the referencing node
                                                graph.edge(
                                                    instance.SOPInstanceUID,
                                                    referenced_series.SeriesInstanceUID,
                                                )
                                    else:
                                        # Check for FrameOfReference registeration
                                        if series.frame_of_reference_registered:
                                            for (
                                                frame_of_ref_series
                                            ) in series.frame_of_reference_registered:
                                                if frame_of_ref_series.Modality in [
                                                    "CT",
                                                    "MR",
                                                    "PT",
                                                ]:
                                                    if not exclude_referenced(frame_of_ref_series):
                                                        referencing_nodes_set.add(
                                                            instance.SOPInstanceUID
                                                        )

                                                        graph.edge(
                                                            instance.SOPInstanceUID,
                                                            frame_of_ref_series.SeriesInstanceUID,
                                                            style="dashed",
                                                        )
                                                        break
                                else:
                                    referenced_instances_list = instance.referenced_instances
                                    if referenced_instances_list:
                                        for referenced_instance in referenced_instances_list:
                                            if not exclude_referenced(
                                                referenced_instance.parent_series
                                            ):
                                                referencing_nodes_set.add(instance.SOPInstanceUID)

                                                # Draw an edge pointing *upwards* from the
                                                # referenced node to the referencing node
                                                graph.edge(
                                                    instance.SOPInstanceUID,
                                                    referenced_instance.SOPInstanceUID,
                                                )
                                    else:
                                        # Check if FrameOfReference registration
                                        if series.frame_of_reference_registered:
                                            for (
                                                frame_of_ref_series
                                            ) in series.frame_of_reference_registered:
                                                if frame_of_ref_series.Modality in [
                                                    "CT",
                                                    "MR",
                                                    "PT",
                                                ]:
                                                    if not exclude_referenced(frame_of_ref_series):
                                                        referencing_nodes_set.add(
                                                            instance.SOPInstanceUID
                                                        )
                                                        graph.edge(
                                                            instance.SOPInstanceUID,
                                                            frame_of_ref_series.SeriesInstanceUID,
                                                            style="dashed",
                                                        )
                                                        break
                        else:
                            # Check if the series references another series directly
                            referenced_series_set = get_referenced_series(series)
                            if referenced_series_set:
                                referenced_series = referenced_series_set.pop()
                                if not exclude_referenced(referenced_series):
                                    referenced_series_uid = referenced_series.SeriesInstanceUID
                                    referencing_nodes_set.add(series.SeriesInstanceUID)

                                    # Draw an edge pointing *upwards* from the referenced series
                                    # to the referencing series
                                    graph.edge(
                                        series.SeriesInstanceUID,
                                        referenced_series_uid,
                                    )

                            # Check for REG modality and moving image reference
                            # (other_referenced_sid)
                            if series.Modality == "REG":
                                other_referenced_series_set = get_other_referenced_series(series)
                                if other_referenced_series_set:
                                    other_referenced_series = other_referenced_series_set.pop()
                                    if not exclude_referenced(other_referenced_series):
                                        referencing_nodes_set.add(series.SeriesInstanceUID)
                                        # Draw a dashed blue edge for the REG moving image
                                        # reference
                                        graph.edge(
                                            series.SeriesInstanceUID,
                                            other_referenced_series.SeriesInstanceUID,
                                            style="dotted",
                                        )

            # Root nodes are those that don't reference other series
            root_nodes = all_nodes_set - referencing_nodes_set

            # Connect the patient node to the root series nodes
            for root in root_nodes:
                graph.edge(
                    root, patient_id, style="invis"
                )  # Root points to the patient (arrows go up)

            return graph

        def display_graph_with_matplotlib(dot_source, dpi=1000):
            """
            Displays the Graphviz graph using matplotlib, by converting SVG to PNG.
            """
            # Generate the PNG in memory
            graph_svg = graphviz.Source(dot_source)
            png_data = graph_svg.pipe(format="png")

            # Load the PNG into a Matplotlib plot
            img = mpimg.imread(BytesIO(png_data), format="png")

            # Display the PNG using matplotlib
            plt.figure(figsize=(12, 12), dpi=dpi)  # Adjust figure size for large graphs
            plt.imshow(img)
            plt.axis("off")
            plt.show()

        def display_graph_in_jupyter(dot_source):
            """
            Displays the graph inline in a Jupyter notebook using IPython's display and SVG.
            """
            from IPython.display import display, SVG

            graph_svg = graphviz.Source(dot_source)
            svg = graph_svg.pipe(format="svg").decode("utf-8")
            display(SVG(svg))

            # display(SVG(graphviz.Source(dot_source).pipe(format="svg")))

        is_jupyter = in_jupyter()

        # if patient_id is specified, only generate for that patient
        if patient_id is not None:
            series_dict = self.dicom_files.get(patient_id, {})
            if not series_dict:
                print(f"No data found for patient {patient_id}")
                return
            graph = graphviz.Digraph(comment=f"DICOM Series Associations for {patient_id}")
            graph.attr("node", shape="box", style="filled", fillcolor="lightgray", color="black")
            graph.attr(rankdir=rankdir)

            # Create a graph for the specified patient
            graph = create_graph(patient_id, series_dict, graph)

            # Render and view the graph for the specified patient
            if output_file:
                graph.render(f"{output_file}_{patient_id}", format="svg")

            if view:
                if is_jupyter:
                    display_graph_in_jupyter(graph.source)
                else:
                    display_graph_with_matplotlib(graph.source)

        elif per_patient:
            # Create separate graphs for each patient
            for patient_id, series_dict in self.dicom_files.items():
                graph = graphviz.Digraph(comment=f"DICOM Series Associations for {patient_id}")
                graph.attr(
                    "node", shape="box", style="filled", fillcolor="lightgray", color="black"
                )

                graph.attr(rankdir=rankdir)

                # Create a graph for each patient
                graph = create_graph(patient_id, series_dict, graph)

                # Render and view each patient's graph
                if output_file:
                    patient_output_file = f"{output_file}_{patient_id}.svg"
                    graph.render(patient_output_file, format="svg")

                if view:
                    if is_jupyter:
                        display_graph_in_jupyter(graph.source)
                    else:
                        display_graph_with_matplotlib(graph.source)

        else:
            # Create a combined graph for all patients
            graph = graphviz.Digraph(comment="DICOM Series Associations")
            graph.attr("node", shape="box", style="filled", fillcolor="lightgray", color="black")

            graph.attr(rankdir=rankdir)

            # Loop through all patients and their series
            for patient_id, series_dict in self.dicom_files.items():
                # Add each patient's series to the combined graph
                graph = create_graph(patient_id, series_dict, graph)

            # Render and view the combined graph
            if output_file:
                graph.render(output_file, format="svg")

            if view:
                if is_jupyter:
                    display_graph_in_jupyter(graph.source)
                else:
                    display_graph_with_matplotlib(graph.source)

    def __iter__(self):
        """
        Iterates over all loaded patients in the dataset.

        This method allows the DICOMLoader to be iterated over, yielding `PatientNode` instances.
        Each `PatientNode` contains studies (`StudyNode`s), which in turn contain series
        (`SeriesNode`s) and instances (`InstanceNode`s).

        Yields
        ------
        PatientNode
            The next `PatientNode` instance in the dataset.

        Examples
        --------
        >>> loader = DICOMLoader("/path/to/dicom/files")
        >>> loader.load()
        >>> for patient in loader:
        ...     print(patient.PatientName, patient.PatientID)
        'John Doe', 12345
        'Jane Smith', 67890
        """
        if self.dataset:
            yield from self.dataset

    def __repr__(self):
        """
        Returns a string representation of the `DICOMLoader` instance, including the dataset path,
        dataset ID, and the number of patients in the dataset.

        Returns
        -------
        str
            A string representation of the `DICOMLoader` object.
        """
        dataset_id = self.dataset.dataset_id if self.dataset else "None"
        num_patients = len(self.dataset) if self.dataset else 0
        return (
            f"DICOMLoader(path='{self.path}', "
            f"dataset_id='{dataset_id}', "
            f"NumPatients={num_patients})"
        )
