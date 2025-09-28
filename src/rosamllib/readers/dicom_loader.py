import os
import time
import traceback
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp
from io import BytesIO
from functools import partial, lru_cache
from typing import Iterable, List, Optional, Set, Union, Any, Dict
from collections import defaultdict, deque
from itertools import chain
from pydicom import dcmread
from pydicom.tag import Tag, BaseTag
from pydicom.datadict import keyword_for_tag, tag_for_keyword, dictionary_VR
from dataclasses import dataclass
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
    SeriesNode,
    InstanceNode,
)
from rosamllib.utils import (
    validate_dicom_path,
    query_df,
    parse_vr_value,
    get_referenced_sop_instance_uids,
    extract_rtstruct_for_uids,
    deprecated,
)
from rosamllib.readers.query_dicom import query_instances, QueryOptions
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from string import hexdigits


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


@lru_cache(maxsize=None)
def _tag_info(col: str):
    t = tag_for_keyword(col)
    if t:
        return (t, dictionary_VR(t), col)  # (Tag, VR, keyword)
    try:
        g, e = col.split(",")
        t = Tag((int(g, 16), int(e, 16)))
        return (t, dictionary_VR(t), col)
    except Exception:
        return (None, None, col)


# Core, always-needed keywords
CORE_TAGS = [
    "SOPClassUID",
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "PatientID",
    "PatientName",
    "StudyDescription",
    "SeriesDescription",
    "FrameOfReferenceUID",
    "Modality",
]

# Extra sequences needed for RT objects (plan/dose/record + shared)
RT_COMMON_SEQ = [
    # Common “Referenced*” containers frequently used across RT objects
    "ReferencedStructureSetSequence",
    "ReferencedDoseSequence",
    "ReferencedRTPlanSequence",
]

# Extra sequences specifically needed to fully resolve RTSTRUCT references & FoRs
RTSTRUCT_SEQ = [
    "ReferencedFrameOfReferenceSequence",
    "RTReferencedStudySequence",
    "RTReferencedSeriesSequence",
    "ROIContourSequence",
    "ContourSequence",
    "ContourImageSequence",
    # Some writers also embed in the above paths only;
    # listing them here ensures the full chain gets parsed.
]

# Extra sequences for SEG
SEG_SEQ = [
    "ReferencedSeriesSequence",
    "ReferencedInstanceSequence",
]


@dataclass(frozen=True)
class TagPlan:
    tag: Tag
    name: str
    vr: str
    is_sq: bool
    # parser: Any


def _parse_int_flex(x) -> int:
    """Parse int from diverse formats: int, '0x..', hex '0010', or decimal '16'."""
    if isinstance(x, int):
        return x
    s = str(x).strip().lower()
    # strip surrounding parentheses if any
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    # remove spaces
    s = s.replace(" ", "")
    # hex with 0x
    if s.startswith("0x"):
        return int(s, 16)
    # pure hex? (all chars hex and typical length <= 4)
    if all(c in hexdigits for c in s):
        try:
            return int(s, 16)
        except ValueError:
            pass
    # fallback decimal
    return int(s, 10)


def _to_tag(obj) -> Optional[Tag]:
    """
    Return a pydicom Tag for many input shapes, or None if unresolvable.
    Accepts keyword, Tag, int, (g,e) tuple, 'GGGG,EEEE', '(GGGG, EEEE)', '(16, 32)', '00100020'.
    """
    if obj is None:
        return None

    # already a Tag
    if isinstance(obj, BaseTag):
        return Tag(obj)

    # keyword?
    tnum = tag_for_keyword(str(obj))
    if tnum:
        return Tag(tnum)

    # tuple?
    if isinstance(obj, tuple) and len(obj) == 2:
        try:
            g = _parse_int_flex(obj[0])
            e = _parse_int_flex(obj[1])
            return Tag((g, e))
        except Exception:
            return None

    # string forms
    if isinstance(obj, str):
        s = obj.strip()
        # strip parentheses like "(16, 32)" or "(0010, 0020)"
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        s = s.strip()
        s_wo_spaces = s.replace(" ", "")

        # 'GGGG,EEEE' or '16,32'
        if "," in s_wo_spaces:
            try:
                g_str, e_str = s_wo_spaces.split(",", 1)
                g = _parse_int_flex(g_str)
                e = _parse_int_flex(e_str)
                return Tag((g, e))
            except Exception:
                return None

        # '00100020' (8 hex chars)
        if len(s_wo_spaces) == 8 and all(c in hexdigits for c in s_wo_spaces):
            try:
                g = int(s_wo_spaces[:4], 16)
                e = int(s_wo_spaces[4:], 16)
                return Tag((g, e))
            except Exception:
                return None

        # try as a single int (decimal or hex)
        try:
            return Tag(_parse_int_flex(s_wo_spaces))
        except Exception:
            return None

    # last resort: try Tag(obj)
    try:
        return Tag(obj)
    except Exception:
        return None


def _as_specific_key(tag_like: Any) -> str:
    t = _to_tag(tag_like)
    if t is not None:
        return f"{int(t.group):04X},{int(t.element):04X}"
    # fall back trying to preserve keyword if resolvable
    tnum = tag_for_keyword(str(tag_like))
    if tnum:
        return keyword_for_tag(tnum) or f"{int(Tag(tnum).group):04X},{int(Tag(tnum).element):04X}"
    # last resort: return original string
    return str(tag_like)


def _build_specific_tags_set(user_tags: Optional[Iterable[Any]]) -> set:
    wanted = set()
    # core basics
    for k in CORE_TAGS:
        t = _to_tag(k)
        if t is not None:
            wanted.add(t)
    # user-specified (can be Tag, TagPlan, keyword, etc.)
    if user_tags:
        for u in user_tags:
            t = getattr(u, "tag", None)  # TagPlan?
            if t is None:
                t = _to_tag(u)
            if t is not None:
                wanted.add(t)
    return wanted


def _specific_from_tag_plan(tag_plan: Optional[list[TagPlan]]) -> set[Tag]:
    if not tag_plan:
        return set()
    return {tp.tag for tp in tag_plan}


def _extend_for_modalities(
    tag_set: Set[Tag], *, rt_common=False, rtstruct=False, seg=False
) -> None:
    if rt_common:
        for k in RT_COMMON_SEQ:
            t = _to_tag(k)
            tag_set.add(t) if t else None
    if rtstruct:
        for k in RTSTRUCT_SEQ:
            t = _to_tag(k)
            tag_set.add(t) if t else None
    if seg:
        for k in SEG_SEQ:
            t = _to_tag(k)
            tag_set.add(t) if t else None


def _build_specific_tags(
    user_tags: Optional[Iterable[Any]] = None,
    *,
    include_rt_common: bool = False,
    include_rtstruct: bool = False,
    include_seg: bool = False,
) -> List[str]:
    """
    Compose a deduped list for dcmread(specific_tags=...).
    """
    wanted: Set[str] = set(_as_specific_key(t) for t in CORE_TAGS)

    if user_tags:
        for t in user_tags:
            wanted.add(_as_specific_key(t))

    if include_rt_common:
        for t in RT_COMMON_SEQ:
            wanted.add(_as_specific_key(t))
    if include_rtstruct:
        for t in RTSTRUCT_SEQ:
            wanted.add(_as_specific_key(t))
    if include_seg:
        for t in SEG_SEQ:
            wanted.add(_as_specific_key(t))

    return sorted(wanted)


def _max_workers_io():
    # IO-bound heuristic
    return min(32, (mp.cpu_count() or 4) * 5)


def _max_workers_cpu():
    return max(1, (os.cpu_count() or 4))


def _choose_executor(num_tasks: int, *, prefer: str = "auto", threshold: int = 3000):
    """
    Return (ExecutorClass, max_workers, use_processes) based on task count.
    - prefer='threads' or 'processes' to force a mode
    - prefer='auto' chooses processes when num_tasks >= threshold
    """
    if prefer == "threads":
        return ThreadPoolExecutor, _max_workers_io(), False
    if prefer == "processes":
        return ProcessPoolExecutor, _max_workers_cpu(), True
    # auto
    if num_tasks >= threshold:
        return ProcessPoolExecutor, _max_workers_cpu(), True
    else:
        return ThreadPoolExecutor, _max_workers_io(), False


def build_tag_plan(tags_to_index: list[tuple[int, int]]) -> list[TagPlan]:
    plan = []
    all_tags = []

    for k in CORE_TAGS:
        t = tag_for_keyword(k)
        if t:
            all_tags.append((Tag(t).group, Tag(t).element))
    if tags_to_index:
        for g, e in tags_to_index:
            all_tags.append((g, e))

    seen = set()
    uniq = []
    for ge in all_tags:
        if ge not in seen:
            seen.add(ge)
            uniq.append(ge)

    for g, e in uniq:
        t = Tag((g, e))
        vr = dictionary_VR(t)
        name = keyword_for_tag(t) or f"{g:04X},{e:04X}"
        is_sq = vr == "SQ"
        plan.append(TagPlan(tag=t, name=name, vr=vr, is_sq=is_sq))
    return plan


def get_metadata(ds, plan: list[TagPlan], seq_policy: str = "json") -> Dict[str, Any]:
    out = {
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
    for tp in plan:
        try:
            if tp.tag in ds:
                val = ds[tp.tag].value
                if tp.is_sq:
                    if seq_policy == "drop":
                        continue
                    elif seq_policy == "len":
                        out[tp.name] = len(val) if val is not None else 0
                    else:
                        out[tp.name] = ds[tp.tag].to_json() if val else None
                else:
                    if tp.vr == "UI":
                        out[tp.name] = str(val) if val is not None else None
                    else:
                        out[tp.name] = parse_vr_value(tp.vr, val)
            else:
                if tp.is_sq:
                    if seq_policy == "drop":
                        continue
                    elif seq_policy == "len":
                        out[tp.name] = 0
                    else:
                        out[tp.name] = None
                else:
                    out[tp.name] = None
        except Exception:
            out[tp.name] = None
    return out


def process_standard_dicom(ds, filepath, tag_plan, seq_policy):
    modality = getattr(ds, "Modality", None)
    metadata = get_metadata(ds, tag_plan, seq_policy)
    instance_dict = {"FilePath": filepath, **metadata}

    if modality in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
        refs_map = get_referenced_sop_instance_uids(ds)
        instance_dict["ReferencedSOPInstanceUIDs"] = list(chain.from_iterable(refs_map.values()))
        if modality == "RTSTRUCT":
            instance_dict["RTStructFoRUIDs"] = extract_rtstruct_for_uids(ds)

    return instance_dict


def process_reg_file(filepath, tag_plan, seq_policy):
    reg = REGReader(filepath).read()
    metadata = get_metadata(reg, tag_plan, seq_policy)
    instance_dict = {
        "FilePath": filepath,
        "ReferencedSeriesUIDs": reg.get_fixed_image_info()["SeriesInstanceUID"],
        "OtherReferencedSeriesUIDs": reg.get_moving_image_info()["SeriesInstanceUID"],
        **metadata,
    }

    return instance_dict


def process_raw_file(filepath, tag_plan, seq_policy):
    raw_reader = DICOMRawReader(filepath)
    raw_reader.read()
    ds = raw_reader.dataset
    metadata = get_metadata(ds, tag_plan, seq_policy)
    instance_dict = {
        "FilePath": filepath,
        **metadata,
    }
    embedded_instances = []
    try:
        embedded_datasets = raw_reader.get_embedded_datasets()

        for embedded_ds in embedded_datasets:
            embedded_metadata = get_metadata(embedded_ds, tag_plan, seq_policy)
            embedded_instance_dict = {
                **embedded_metadata,
                "FilePath": filepath,
                "is_embedded_in_raw": True,
                "raw_series_reference_uid": instance_dict["SeriesInstanceUID"],
            }
            if embedded_instance_dict["Modality"] in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
                refs_map = get_referenced_sop_instance_uids(embedded_ds)
                embedded_instance_dict["ReferencedSOPInstanceUIDs"] = list(
                    chain.from_iterable(refs_map.values())
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


def process_seg_file(filepath, tag_plan, seq_policy):
    seg = SEGReader(filepath).read()
    metadata = get_metadata(seg, tag_plan, seq_policy)
    instance_dict = {"FilePath": filepath, **metadata}
    if hasattr(seg, "ReferencedSeriesSequence"):
        ref_seq = seg.ReferencedSeriesSequence[0]
        if hasattr(ref_seq, "SeriesInstanceUID"):
            instance_dict["ReferencedSeriesUIDs"] = ref_seq.SeriesInstanceUID
    refs_map = get_referenced_sop_instance_uids(seg)
    instance_dict["ReferencedSOPInstanceUIDs"] = list(chain.from_iterable(refs_map.values()))

    return instance_dict


def process_file(filepath, tag_plan, seq_policy):
    try:
        # specific = _build_specific_tags_set(tag_plan)
        specific = _specific_from_tag_plan(tag_plan)
        ds = dcmread(filepath, stop_before_pixels=True, specific_tags=specific)

        modality = getattr(ds, "Modality", None)
        embedded_instances = []

        # RT/SEG: re-read headers fully so nested sequences are available
        if modality in ["RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD", "SEG"]:
            ds = dcmread(filepath, stop_before_pixels=True)

        if modality in ["CT", "MR", "PT", "RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"]:
            instance_dict = process_standard_dicom(ds, filepath, tag_plan, seq_policy)
        elif modality == "REG":
            instance_dict = process_reg_file(filepath, tag_plan, seq_policy)
        elif modality == "RAW":
            instance_dict, embedded_instances = process_raw_file(filepath, tag_plan, seq_policy)
        elif modality == "SEG":
            instance_dict = process_seg_file(filepath, tag_plan, seq_policy)
        else:
            return []

        out = [instance_dict]
        out.extend(embedded_instances)
        return out
    except Exception as e:
        print(e)
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
        self.metadata_df = None
        # Initialize the DatasetNode as the root of the hierarchy
        dataset_id = "DICOM_Dataset"
        dataset_name = "DICOM Collection"
        self.dataset = DatasetNode(dataset_id, dataset_name)
        self._sop_to_instance = {}  # fast lookup: SOPInstanceUID -> InstanceNode

    def load(self, tags_to_index=None, seq_policy="len"):
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
        tag_plan = build_tag_plan(tags_to_index)
        try:
            if os.path.isdir(self.path):
                self.load_from_directory(self.path, seq_policy=seq_policy, tag_plan=tag_plan)
            else:
                self.load_file(self.path, seq_policy=seq_policy, tag_plan=tag_plan)

        except Exception as e:
            print(f"Error loading DICOM files: {e}")
            print(traceback.format_exc())

    def load_from_directory(self, path, seq_policy, tag_plan=None):
        """
        Loads all DICOM files from a directory, including subdirectories.

        This method recursively searches the specified directory for DICOM files,
        reads their metadata, and organizes them by patient and series.

        Parameters
        ----------
        path : str
            The directory path to load DICOM files from.
        tag_plan : list of str, optional
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
        self._load_files(all_files, tag_plan=tag_plan, seq_policy=seq_policy)

    def load_file(self, path, seq_policy, tag_plan=None):
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
        self._load_files([path], seq_policy=seq_policy, tag_plan=tag_plan)

    def _load_files(self, files, seq_policy, tag_plan=None):
        process_file_with_tags = partial(process_file, tag_plan=tag_plan, seq_policy=seq_policy)

        unresolved_raw_links = []
        metadata_rows = []
        exclude_keys = {
            "FilePath",
            "ReferencedSOPInstanceUIDs",
            "ReferencedSeriesUIDs",
            "OtherReferencedSeriesUIDs",
            "is_embedded_in_raw",
            "raw_series_reference_uid",
            "RTStructFoRUIDs",
        }

        Exec, max_workers, is_proc = _choose_executor(len(files), prefer="auto", threshold=10000)

        with Exec(max_workers=max_workers) as ex, tqdm(
            total=len(files), desc="Loading DICOM files", unit="file"
        ) as pbar:

            futures = [ex.submit(process_file_with_tags, f) for f in files]
            for fut in as_completed(futures):
                result = fut.result()  # list[dict] or []
                for inst_dict in result:
                    if not inst_dict or "SOPInstanceUID" not in inst_dict:
                        continue

                    sop_instance_uid = inst_dict["SOPInstanceUID"]
                    patient_id = inst_dict.get("PatientID")
                    patient_name = inst_dict.get("PatientName")
                    study_uid = inst_dict.get("StudyInstanceUID")
                    study_desc = inst_dict.get("StudyDescription")
                    series_uid = inst_dict.get("SeriesInstanceUID")
                    series_desc = inst_dict.get("SeriesDescription")
                    modality = (inst_dict.get("Modality") or "").upper()
                    filepath = inst_dict.get("FilePath")

                    if patient_id is None or series_uid is None:
                        continue

                    if patient_id not in self.dicom_files:
                        self.dicom_files[patient_id] = {}

                    patient_node = self.dataset.get_or_create_patient(patient_id, patient_name)
                    study_node = patient_node.get_or_create_study(study_uid, study_desc)

                    if series_uid not in self.dicom_files[patient_id]:
                        series = study_node.get_or_create_series(series_uid, modality, series_desc)
                        series.Modality = modality
                        series.FrameOfReferenceUID = inst_dict.get("FrameOfReferenceUID")
                        self.dicom_files[patient_id][series_uid] = series
                    series = self.dicom_files[patient_id][series_uid]
                    instance_node = series.get_instance(sop_instance_uid)
                    if not instance_node:
                        instance_node = InstanceNode(
                            sop_instance_uid, filepath, Modality=modality, parent_series=series
                        )
                    if modality == "RTSTRUCT":
                        instance_node.FrameOfReferenceUIDs = list(
                            inst_dict.get("RTStructFoRUIDs") or []
                        )
                    series.add_instance(instance_node, overwrite=True)

                    refs = inst_dict.get("ReferencedSOPInstanceUIDs")
                    if refs:
                        instance_node.referenced_sop_instance_uids = refs

                    if modality in {"REG", "SEG"}:
                        ref_sid = inst_dict.get("ReferencedSeriesUIDs")
                        if ref_sid:
                            instance_node.referenced_sids.append(ref_sid)
                        if modality == "REG":
                            other_sid = inst_dict.get("OtherReferencedSeriesUIDs")
                            if other_sid:
                                instance_node.other_referenced_sids.append(other_sid)

                    if inst_dict.get("is_embedded_in_raw"):
                        series.is_embedded_in_raw = True
                        unresolved_raw_links.append(
                            (patient_id, series, inst_dict.get("raw_series_reference_uid"))
                        )

                    metadata_rows.append(
                        {k: v for k, v in inst_dict.items() if k not in exclude_keys}
                    )

                pbar.update(1)

        # Resolve RAW parents (after all series exist)
        for pid, child_series, raw_sid in unresolved_raw_links:
            if raw_sid:
                parent = self.dicom_files.get(pid, {}).get(raw_sid)
                if parent:
                    child_series.raw_series_reference = parent

        # Wire associations
        DICOMLoader._associate_dicoms(self.dicom_files)

        # Build SOP index
        self._sop_to_instance = {
            str(sop_uid): inst
            for _pid, sdict in self.dicom_files.items()
            for _sid, series in sdict.items()
            for sop_uid, inst in series.instances.items()
        }

        # Build DF
        self.metadata_df = pd.DataFrame(metadata_rows)

        # 8) dtype normalization driven by VR
        for col in self.metadata_df.columns:
            if col == "InstanceNode":
                continue
            # try keyword first
            tag, vr, _ = _tag_info(col)
            dtype = VR_TO_DTYPE.get(vr, object)
            if dtype == "date":
                self.metadata_df[col] = pd.to_datetime(self.metadata_df[col], errors="coerce")
            elif dtype == "time":
                # conservative: keep as string if parsing fails
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
        - Builds SOP/Series/FrameOfReference maps
        - Wires instance<->instance, instance->series, and series<->series edges
        - Populates reverse edges for efficient "referencing" queries
        - Ensures FoR connectivity is symmetric
        - NEW: RTSTRUCT InstanceNode.FrameOfReferenceUIDs is populated as the union of
            FoRs parsed from the DICOM and FoRs of referenced image series.
        """
        # sidecar state for fast identity de-dup per list
        _seen_map = {}

        def _append_unique(lst, obj):
            if obj is None:
                return
            lid = id(lst)
            s = _seen_map.get(lid)
            if s is None:
                s = set(id(x) for x in lst)
                _seen_map[lid] = s
            oid = id(obj)
            if oid not in s:
                lst.append(obj)
                s.add(oid)

        # Build lookup maps
        sop_instance_uid_map = {}
        series_uid_map = {}
        frame_of_reference_uid_map = {}

        for _pid, series_dict in dicom_files.items():
            for sid, series in series_dict.items():
                series_uid_map[sid] = series

                # ensure series containers exist
                series.referencing_series = getattr(series, "referencing_series", [])
                series.referenced_series = getattr(series, "referenced_series", [])
                series.frame_of_reference_registered = getattr(
                    series, "frame_of_reference_registered", []
                )

                for sop_uid, inst in series.instances.items():
                    sop_instance_uid_map[sop_uid] = inst

                    # ensure instance containers exist
                    inst.referenced_instances = getattr(inst, "referenced_instances", [])
                    inst.referencing_instances = getattr(inst, "referencing_instances", [])
                    inst.referenced_series = getattr(inst, "referenced_series", [])
                    inst.other_referenced_series = getattr(inst, "other_referenced_series", [])
                    inst.referenced_sop_instance_uids = getattr(
                        inst, "referenced_sop_instance_uids", []
                    )
                    inst.referenced_sids = getattr(inst, "referenced_sids", [])
                    inst.other_referenced_sids = getattr(inst, "other_referenced_sids", [])

                    # NEW: normalize multi-FoR field on instance (esp. RTSTRUCT)
                    inst.FrameOfReferenceUIDs = list(
                        getattr(inst, "FrameOfReferenceUIDs", []) or []
                    )

                # map FoR -> series list
                if getattr(series, "FrameOfReferenceUID", None):
                    frame_of_reference_uid_map.setdefault(series.FrameOfReferenceUID, []).append(
                        series
                    )

        # Wire references
        for _pid, series_dict in dicom_files.items():
            for sid, series in series_dict.items():
                for sop_uid, inst in series.instances.items():
                    modality = getattr(inst, "Modality", None)

                    # A) SOPInstanceUID links (instance -> instance)
                    for ref_sop_uid in getattr(inst, "referenced_sop_instance_uids", []):
                        ref_inst = sop_instance_uid_map.get(ref_sop_uid)
                        if not ref_inst:
                            continue

                        _append_unique(inst.referenced_instances, ref_inst)
                        _append_unique(ref_inst.referencing_instances, inst)

                        # promote to series-level edges
                        src_series = inst.parent_series
                        dst_series = getattr(ref_inst, "parent_series", None)
                        if src_series and dst_series and src_series is not dst_series:
                            _append_unique(src_series.referenced_series, dst_series)
                            _append_unique(dst_series.referencing_series, src_series)
                            _append_unique(inst.referenced_series, dst_series)

                    # B) Modality-specific aggregation of series links
                    if modality in {"RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD"}:
                        for ref_inst in inst.referenced_instances:
                            rs = ref_inst.parent_series
                            if rs:
                                _append_unique(inst.referenced_series, rs)
                                _append_unique(inst.referenced_sids, rs.SeriesInstanceUID)

                    if modality == "REG":
                        for ref_sid in getattr(inst, "referenced_sids", []):
                            rs = series_uid_map.get(ref_sid)
                            if rs:
                                _append_unique(inst.referenced_series, rs)
                        for other_sid in getattr(inst, "other_referenced_sids", []):
                            rs = series_uid_map.get(other_sid)
                            if rs:
                                _append_unique(inst.other_referenced_series, rs)
                                _append_unique(inst.referenced_series, rs)

                    if modality == "SEG":
                        for ref_sid in getattr(inst, "referenced_sids", []):
                            rs = series_uid_map.get(ref_sid)
                            if rs:
                                _append_unique(inst.referenced_series, rs)

                    # C) Promote instance->series to series<->series (reverse edges)
                    for rs in list(getattr(inst, "referenced_series", [])):
                        src_series = inst.parent_series
                        if src_series and rs and src_series is not rs:
                            _append_unique(src_series.referenced_series, rs)
                            _append_unique(rs.referencing_series, src_series)

                    for rs in list(getattr(inst, "other_referenced_series", [])):
                        src_series = inst.parent_series
                        if src_series and rs and src_series is not rs:
                            _append_unique(src_series.referenced_series, rs)
                            _append_unique(rs.referencing_series, src_series)

                    # D) NEW: RTSTRUCT multi-FoR reconciliation on the instance
                    if (modality or "").upper() == "RTSTRUCT":
                        # FoRs from referenced image series present in this dataset
                        fors_from_series = {
                            s.FrameOfReferenceUID
                            for s in getattr(inst, "referenced_series", [])
                            if getattr(s, "FrameOfReferenceUID", None)
                        }
                        # FoRs parsed from the RTSTRUCT dataset earlier (loader populated)
                        fors_from_ds = set(inst.FrameOfReferenceUIDs or [])
                        # Union and store back on the instance
                        inst.FrameOfReferenceUIDs = sorted(
                            {str(u) for u in (fors_from_series | fors_from_ds) if u}
                        )

        # --- EFFECTIVE FrameOfReference connectivity (symmetric, series-level) ---

        # Reset precomputed neighbors
        for _pid, series_dict in dicom_files.items():
            for _sid, s in series_dict.items():
                s.frame_of_reference_registered = getattr(s, "frame_of_reference_registered", [])
                s.frame_of_reference_registered[:] = []

        # Build FoR -> [series] buckets directly
        for_to_series = {}  # str -> list[SeriesNode]

        for _pid, series_dict in dicom_files.items():
            for _sid, s in series_dict.items():
                fors = set()

                # series-level FoR
                fo = getattr(s, "FrameOfReferenceUID", None)
                if fo:
                    fors.add(str(fo))

                # derive from instances + their referenced series
                for inst in getattr(s, "instances", {}).values():
                    for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                        if u:
                            fors.add(str(u))
                    for rs in getattr(inst, "referenced_series", []) or []:
                        u = getattr(rs, "FrameOfReferenceUID", None)
                        if u:
                            fors.add(str(u))

                # (optional) only set if the field exists on a non-slotted class
                if hasattr(s, "EffectiveFrameOfReferenceUIDs"):
                    try:
                        setattr(s, "EffectiveFrameOfReferenceUIDs", sorted(fors))
                    except Exception:
                        pass  # ignore if slotted

                # bucket this series by each FoR
                for u in fors:
                    for_to_series.setdefault(u, []).append(s)

        # Dedup helper (by identity)
        _seen_map = {}

        def _append_unique(lst, obj):
            if obj is None:
                return
            lid = id(lst)
            seen = _seen_map.get(lid)
            if seen is None:
                seen = set(id(x) for x in lst)
                _seen_map[lid] = seen
            oid = id(obj)
            if oid not in seen:
                lst.append(obj)
                seen.add(oid)

        # Symmetric neighbors within each effective FoR bucket
        for u, series_list in for_to_series.items():
            n = len(series_list)
            for i in range(n):
                si = series_list[i]
                for j in range(n):
                    if i == j:
                        continue
                    _append_unique(si.frame_of_reference_registered, series_list[j])

    def _normalize_tag(self, tag):
        """
        Normalize to a (group, element) tuple of ints.
        Accepts DICOM keyword ('PatientID') or any Tag-like input.
        """
        try:
            if isinstance(tag, tuple) and len(tag) == 2:
                # allow ('0008', '0018') or (0x0008, 0x0018) etc.
                g = int(tag[0], 16) if isinstance(tag[0], str) else int(tag[0])
                e = int(tag[1], 16) if isinstance(tag[1], str) else int(tag[1])
                return (g, e)
            # keyword
            t = Tag(tag_for_keyword(str(tag)))
            if t is None:
                raise ValueError(f"Unknown keyword '{tag}'")
            return (t.group, t.element)
        except Exception:
            print(f"Unknown tag/keyword '{tag}' ignored.")
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

    def query(
        self,
        query_level: str = "INSTANCE",
        *,
        include: Optional[Iterable[str]] = None,
        case_insensitive: bool = False,
        default_atol: float = 1e-6,
        default_rtol: float = 1e-6,
        sort_by: Optional[Iterable[str]] = None,
        **filters,
    ) -> pd.DataFrame:
        """
        Query the loaded DICOM metadata at a chosen hierarchy level.

        This is a convenience wrapper around :func:`query_df` that returns only the
        identifiers relevant to the requested level (plus any columns referenced in
        filters). All matching semantics (wildcards, regex, ranges, temporal ops,
        container membership, callable predicates, dot-path traversal, etc.) are
        delegated to ``query_df``.

        Parameters
        ----------
        query_level : {'PATIENT', 'STUDY', 'SERIES', 'INSTANCE'}, default 'INSTANCE'
            The granularity of the result:
            * ``'PATIENT'``  → columns: ``['PatientID']``
            * ``'STUDY'``    → columns: ``['PatientID', 'StudyInstanceUID']``
            * ``'SERIES'``   → columns: ``['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID']``
            * ``'INSTANCE'`` → columns: ``['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']``

            Any columns used in ``filters`` that exist in the metadata are also included.
        **filters :
            Passed through to :func:`query_df` and applied to ``self.metadata_df``.
            See ``query_df`` for the full specification of supported operators and
            behaviors (case insensitivity, temporal coercion, dot-paths, etc.).

        Returns
        -------
        pandas.DataFrame
            A de-duplicated DataFrame restricted to the identifier columns for the
            requested level plus any filter-referenced columns. Row order follows the
            underlying filtered metadata.

        Raises
        ------
        ValueError
            If ``query_level`` is not one of the supported values.
        KeyError
            If a filter references a column that does not exist in the metadata.

        Notes
        -----
        This method relies on ``self.metadata_df`` that is populated by ``load()``.
        For complex sequence/JSON columns, consider pre-extracting fields into their
        own columns if you will query them frequently.

        Examples
        --------
        Series by modality
        >>> loader.query(query_level="SERIES", Modality="CT")

        Patients with ID starting 'P1' (wildcard)
        >>> loader.query(query_level="PATIENT", PatientID="P1*")

        Studies in a date range
        >>> loader.query(query_level="STUDY", StudyDate={"gte": "2024-01-01", "lte": "2024-06-30"})

        Instances with SOPInstanceUID matching a pattern (regex)
        >>> loader.query(query_level="INSTANCE",
        ...              SOPInstanceUID={"RegEx": r"^1\\.2\\.840\\..*"})
        """
        LEVEL_COLS = {
            "PATIENT": ["PatientID"],
            "STUDY": ["PatientID", "StudyInstanceUID"],
            "SERIES": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
            "INSTANCE": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"],
        }

        lvl = str(query_level).upper()
        if lvl not in LEVEL_COLS:
            raise ValueError(
                f"Invalid query_level '{query_level}'. Must be one of {list(LEVEL_COLS)}."
            )

        # 1) Filter with the generic engine
        fdf = query_df(
            self.metadata_df,
            case_insensitive=case_insensitive,
            default_atol=default_atol,
            default_rtol=default_rtol,
            **filters,
        )

        # 2) Build the column set to return
        level_cols = LEVEL_COLS[lvl]
        # include any filter-referenced columns that actually exist
        filter_cols = [c for c in filters.keys() if c in fdf.columns]
        extra_cols = [c for c in (include or []) if c in fdf.columns]

        # preserve order: level IDs first, then filter cols, then extras (dedup by dict keys)
        ordered = list(dict.fromkeys(list(level_cols) + filter_cols + extra_cols))

        # 3) Sort and de-duplicate by the level ID columns
        sort_keys = list(sort_by) if sort_by else level_cols
        # (Only keep columns that exist to avoid KeyErrors if user passed unknown sort columns)
        sort_keys = [c for c in sort_keys if c in fdf.columns]
        if sort_keys:
            fdf = fdf.sort_values(sort_keys, kind="mergesort")  # stable

        if lvl == "INSTANCE":
            # At instance level we can keep every row (already unique if SOPInstanceUID is unique)
            out = fdf[ordered].copy()
        else:
            # Dedup ONLY on the level ID columns, keep the first row per group
            out = fdf.drop_duplicates(subset=level_cols, keep="first")[ordered].copy()

        # 4) Final tidy-up
        out = out.reset_index(drop=True)
        return out

    def advanced_query(
        self,
        query_level: str = "INSTANCE",
        *,
        df_filters: Optional[Dict[str, Union[str, List[Any], Dict[str, Any]]]] = None,
        dcm_filters: Optional[Dict[str, Union[str, List[Any], Dict[str, Any]]]] = None,
        dcm_options: Optional[QueryOptions] = None,
        return_instances: bool = False,  # include matching InstanceNodes
        return_paths: bool = False,  # if True, return file paths instead of nodes
    ):
        """
        Two-stage advanced query.

        Stage 1 (optional): DataFrame shortlist via `query`-style filters.
        Stage 2 (optional): Deep DICOM evaluation via pydicom:
            - wildcards/regex (including escaped '\*', '\?')
            - multi-value (VM>1) tags (ANY/ALL via QueryOptions)
            - nested sequences via dot-paths with [*] or [index]
            - date/time/datetime ranges for DA/TM/DT

        Parameters
        ----------
        query_level : {"PATIENT","STUDY","SERIES","INSTANCE"}
            Desired granularity for the returned DataFrame.
        df_filters : dict, optional
            Filters applied to self.metadata_df using your `query_df` semantics.
        dcm_filters : dict, optional
            Deep DICOM filters evaluated by opening the files (uses `query_instances`).
        dcm_options : QueryOptions, optional
            Controls deep-query behavior (e.g., any_vs_all_multi="any"/"all").
        return_instances : bool
            If True, also return a list of matching InstanceNodes (or paths if return_paths=True).
        return_paths : bool
            If True (and return_instances=True), return file paths instead of InstanceNodes.

        Returns
        -------
        If return_instances or return_paths is True:
            (hits: list[InstanceNode|str], df: pd.DataFrame)
        Else:
            df: pd.DataFrame
        """
        if self.metadata_df is None or self.metadata_df.empty:
            empty = self.metadata_df.iloc[0:0] if self.metadata_df is not None else pd.DataFrame()
            return ([], empty) if (return_instances or return_paths) else empty

        levels = {
            "PATIENT": ["PatientID"],
            "STUDY": ["PatientID", "StudyInstanceUID"],
            "SERIES": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
            "INSTANCE": ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"],
        }
        ql = query_level.upper()
        if ql not in levels:
            raise ValueError(
                f"Invalid query level '{query_level}'. Must be one of {list(levels.keys())}."
            )

        # ---------------- Stage 1: DataFrame shortlist ----------------
        shortlist_df = self.metadata_df
        if df_filters:
            shortlist_df = self.query(query_level, **df_filters)

        if dcm_filters is None:
            # No deep stage; just project/uniq at requested level and return
            level_cols = levels[ql]
            # carry any df_filter columns, when present in DF
            extra = [
                c
                for c in (df_filters or {}).keys()
                if c in shortlist_df.columns and c not in level_cols
            ]
            out_cols = list(dict.fromkeys(level_cols + extra))
            df_out = shortlist_df[out_cols].copy()
            for col in df_out.columns:
                if df_out[col].apply(lambda x: isinstance(x, list)).any():
                    df_out[col] = df_out[col].apply(
                        lambda x: tuple(x) if isinstance(x, list) else x
                    )
            df_out = df_out.drop_duplicates().reset_index(drop=True)

            if return_instances or return_paths:
                hits = []
                if "SOPInstanceUID" in df_out.columns:
                    for sop in df_out["SOPInstanceUID"].astype(str).tolist():
                        inst = self._sop_to_instance.get(sop)
                        if inst:
                            hits.append(inst.FilePath if return_paths else inst)
                return hits, df_out

            return df_out

        # ---------------- Stage 2: deep DICOM evaluation ----------------
        # Decide which SOPs to evaluate, based on the requested level
        if "SOPInstanceUID" in shortlist_df.columns:
            sop_series = shortlist_df["SOPInstanceUID"]
        elif "SeriesInstanceUID" in shortlist_df.columns:
            sop_series = self.metadata_df[
                self.metadata_df["SeriesInstanceUID"].isin(
                    shortlist_df["SeriesInstanceUID"].unique()
                )
            ]["SOPInstanceUID"]
        elif "StudyInstanceUID" in shortlist_df.columns:
            sop_series = self.metadata_df[
                self.metadata_df["StudyInstanceUID"].isin(
                    shortlist_df["StudyInstanceUID"].unique()
                )
            ]["SOPInstanceUID"]
        else:
            sop_series = self.metadata_df[
                self.metadata_df["PatientID"].isin(shortlist_df["PatientID"].unique())
            ]["SOPInstanceUID"]

        sop_list = sop_series.dropna().astype(str).unique().tolist()
        if not sop_list:
            empty = shortlist_df.iloc[0:0]
            return ([], empty) if (return_instances or return_paths) else empty

        # Build candidate list for deep evaluation
        candidates: List[Union[str, InstanceNode]] = []
        for sop in sop_list:
            inst = self._sop_to_instance.get(sop)
            if inst:
                candidates.append(inst.FilePath if return_paths else inst)

        if not candidates:
            empty = shortlist_df.iloc[0:0]
            return ([], empty) if (return_instances or return_paths) else empty

        # Evaluate deep filters
        dcm_options = dcm_options or QueryOptions()
        deep_hits = query_instances(candidates, options=dcm_options, **dcm_filters)

        # Map deep hits -> SOP list
        if return_paths:
            # Resolve path -> SOP via reverse lookup from node map
            path_to_sop = {
                inst.FilePath: str(inst.SOPInstanceUID) for inst in self._sop_to_instance.values()
            }
            hit_sops = [path_to_sop.get(p) for p in deep_hits]
            hit_sops = [s for s in hit_sops if s]
        else:
            hit_sops = [str(h.SOPInstanceUID) for h in deep_hits if hasattr(h, "SOPInstanceUID")]

        if not hit_sops:
            empty = shortlist_df.iloc[0:0]
            return (deep_hits, empty) if (return_instances or return_paths) else empty

        # Build the final DF for requested level using global metadata (stable)
        df_hits = self.metadata_df[self.metadata_df["SOPInstanceUID"].astype(str).isin(hit_sops)]

        level_cols = levels[ql]
        # Carry any referenced columns that happen to be in metadata_df;
        # deep-only keys won’t be present
        extra_cols = [
            c
            for c in (df_filters or {}).keys()
            if c in self.metadata_df.columns and c not in level_cols
        ]
        out_cols = list(dict.fromkeys(level_cols + extra_cols))
        df_out = df_hits[out_cols].copy()

        for col in df_out.columns:
            if df_out[col].apply(lambda x: isinstance(x, list)).any():
                df_out[col] = df_out[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        df_out = df_out.drop_duplicates().reset_index(drop=True)

        if return_instances or return_paths:
            return deep_hits, df_out

        return df_out

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
    @deprecated("get_referencing_nodes", remove_in="v0.6")
    def get_referencing_items(node, modality=None, level="INSTANCE"):
        return DICOMLoader.get_referencing_nodes(node, modality=modality, level=level)

    @staticmethod
    def get_referenced_nodes(
        node: Union[SeriesNode, InstanceNode],
        modality: Optional[Union[str, Iterable[str]]] = None,
        level: str = "INSTANCE",
        recursive: bool = True,
        include_start: bool = False,
    ) -> List[Union[SeriesNode, InstanceNode]]:
        """
        Return referenced nodes of a specified level (INSTANCE|SERIES), optionally filtered
        by modality.
        - Direct neighbors are always considered; if `recursive=False`, stop after depth=1.
        - If `recursive=True`, traverse further (no depth limit).
        - `include_start=False` by default (doesn't include the input `node` in results).
        """

        def norm_modalities(m) -> Optional[Set[str]]:
            if m is None:
                return None
            if isinstance(m, str):
                return {m.upper()}
            return {str(x).upper() for x in m}

        def modality_ok(obj) -> bool:
            if wanted is None:
                return True
            mod = getattr(obj, "Modality", None)
            return (mod or "").upper() in wanted

        def maybe_add(n):
            if level == "INSTANCE" and isinstance(n, InstanceNode) and modality_ok(n):
                out.append(n)
            elif level == "SERIES" and isinstance(n, SeriesNode) and modality_ok(n):
                out.append(n)

        level = level.upper()
        if level not in {"INSTANCE", "SERIES"}:
            raise ValueError("level must be 'INSTANCE' or 'SERIES'")

        wanted = norm_modalities(modality)
        out: List[Union[SeriesNode, InstanceNode]] = []
        seen: Set[int] = set()

        # BFS
        q = deque()
        # depth=0 is the start node; we still may include it if requested
        q.append((node, 0))
        if include_start:
            maybe_add(node)

        max_depth = None if recursive else 1

        while q:
            n, d = q.popleft()
            nid = id(n)
            if nid in seen:
                continue
            seen.add(nid)

            # collect (except the start if include_start=False)
            if d > 0:
                maybe_add(n)

            # stop expanding if we've hit the depth limit
            if max_depth is not None and d >= max_depth:
                continue

            # neighbors
            if isinstance(n, SeriesNode):
                # 1) direct series->series links (e.g., REG/SEG edges resolved at series level)
                for s in getattr(n, "referenced_series", []) or []:
                    q.append((s, d + 1))
                # 2) go through instances to follow instance-level links
                for inst in getattr(n, "instances", {}).values():
                    # instance->instance
                    for ref in getattr(inst, "referenced_instances", []) or []:
                        q.append((ref, d + 1))
                    # instance->series
                    for s in getattr(inst, "referenced_series", []) or []:
                        q.append((s, d + 1))

            elif isinstance(n, InstanceNode):
                # instance->instance
                for ref in getattr(n, "referenced_instances", []) or []:
                    q.append((ref, d + 1))
                # instance->series
                for s in getattr(n, "referenced_series", []) or []:
                    q.append((s, d + 1))

        # de-dup while preserving order (by id)
        seen_ids = set()
        deduped = []
        for x in out:
            xid = id(x)
            if xid not in seen_ids:
                seen_ids.add(xid)
                deduped.append(x)

        return deduped

    @staticmethod
    def get_referencing_nodes(
        node: Union[SeriesNode, InstanceNode],
        modality: Optional[Union[str, Iterable[str]]] = None,
        level: str = "INSTANCE",
        recursive: bool = True,
        include_start: bool = False,
    ) -> List[Union[SeriesNode, InstanceNode]]:
        """
        Return nodes that share the same FrameOfReferenceUID as the given node.

        Parameters
        ----------
        node : SeriesNode | InstanceNode
            Anchor node. If an InstanceNode is provided, its parent SeriesNode is used.
        level : {'SERIES', 'INSTANCE'}, default 'SERIES'
            - 'SERIES': return SeriesNode peers in the same Frame of Reference (FoR).
            - 'INSTANCE': return InstanceNode peers from all series in the same FoR.
            Note: with 'INSTANCE' and include_self=False, result can be empty if
            there are no peer series (i.e., anchor is the only series in its FoR).
        include_self : bool, default False
            Include the anchor in the results:
            - 'SERIES': include the anchor series.
            - 'INSTANCE': include all instances from the anchor series.
        modality : str or Iterable[str], optional
            Case-insensitive modality filter.
            - For level='SERIES', filters by `series.Modality` (e.g., 'CT', 'MR').
            - For level='INSTANCE', filters by `instance.Modality` (e.g., 'CT', 'RTDOSE').
            You may pass a single string (e.g., "CT") or an iterable (e.g., ["CT","MR"]).

        Returns
        -------
        list[SeriesNode] | list[InstanceNode]
            Peers in the same Frame of Reference, filtered by level/modality.

        Notes
        -----
        - This method prefers the precomputed `series.frame_of_reference_registered`
        filled during `_associate_dicoms`. If that list is empty, it falls back
        to scanning `self.dicom_files`.
        - Passing a single string for `modality` is supported and treated as a set
        with one element (e.g., "CT" -> {"CT"}).

        Examples
        --------
        >>> # Series peers (CT or MR) sharing the same FoR as a dose's CT
        >>> peers = loader.get_frame_registered_nodes(dose.parent_series,
        ...                                           level="SERIES",
        ...                                           modality=["CT","MR"])
        >>> # All RTDOSE instances within the same FoR (including the anchor series)
        >>> doses = loader.get_frame_registered_nodes(ct_series,
        ...                                           level="INSTANCE",
        ...                                           include_self=True,
        ...                                           modality="RTDOSE")
        """

        def norm_modalities(m) -> Optional[Set[str]]:
            if m is None:
                return None
            if isinstance(m, str):
                return {m.upper()}
            return {str(x).upper() for x in m}

        def modality_ok(obj) -> bool:
            if wanted is None:
                return True
            mod = getattr(obj, "Modality", None)
            return (mod or "").upper() in wanted

        def maybe_add(n):
            if level == "INSTANCE" and isinstance(n, InstanceNode) and modality_ok(n):
                out.append(n)
            elif level == "SERIES" and isinstance(n, SeriesNode) and modality_ok(n):
                out.append(n)

        def enqueue(nei, depth):
            if nei is None:
                return
            q.append((nei, depth))

        level = level.upper()
        if level not in {"INSTANCE", "SERIES"}:
            raise ValueError("level must be 'INSTANCE' or 'SERIES'")

        wanted = norm_modalities(modality)
        out: List[Union[SeriesNode, InstanceNode]] = []
        seen: Set[int] = set()
        q = deque()
        q.append((node, 0))
        if include_start:
            maybe_add(node)

        max_depth = None if recursive else 1

        while q:
            n, d = q.popleft()
            nid = id(n)
            if nid in seen:
                continue
            seen.add(nid)

            # collect (except depth 0 unless include_start)
            if d > 0:
                maybe_add(n)

            # stop expanding if depth cap
            if max_depth is not None and d >= max_depth:
                continue

            # ---- incoming neighbors ----
            if isinstance(n, InstanceNode):
                # instances that reference this instance
                for rin in getattr(n, "referencing_instances", []) or []:
                    enqueue(rin, d + 1)
                    # their parent series are also referrers at the series level
                    enqueue(getattr(rin, "parent_series", None), d + 1)

                # series that reference this instance directly (if you maintain such a list)
                # Not standard in your model; typically we discover series via the instances above.

                # the instance's parent series might be referenced by other series;
                # climb to series and continue
                ps = getattr(n, "parent_series", None)
                if ps is not None:
                    # series that reference this series (if populated)
                    for rs in getattr(ps, "referencing_series", []) or []:
                        enqueue(rs, d + 1)

            elif isinstance(n, SeriesNode):
                # series that reference this series (if populated)
                for rs in getattr(n, "referencing_series", []) or []:
                    enqueue(rs, d + 1)

                # instances that reference any instance within this series
                for inst in getattr(n, "instances", {}).values():
                    for rin in getattr(inst, "referencing_instances", []) or []:
                        enqueue(rin, d + 1)
                        enqueue(getattr(rin, "parent_series", None), d + 1)

        # stable de-dup by object id
        uniq_ids = set()
        deduped = []
        for x in out:
            xid = id(x)
            if xid not in uniq_ids:
                uniq_ids.add(xid)
                deduped.append(x)

        return deduped

    @staticmethod
    def get_frame_registered_nodes(
        node: Union[SeriesNode, InstanceNode],
        *,
        level: str = "SERIES",
        include_self: bool = False,
        modality: Optional[Union[str, Iterable[str]]] = None,
        dicom_files: Optional[Dict[str, Dict[str, SeriesNode]]] = None,
        derive_frame_from_references: bool = True,
    ) -> List[Union[SeriesNode, InstanceNode]]:
        """
        Return nodes that share at least one effective FrameOfReferenceUID with the anchor.

        Effective FoR of a series is the union of:
        - series.FrameOfReferenceUID
        - (if derive_frame_from_references) any inst.FrameOfReferenceUIDs
        - (if derive_frame_from_references) FoR of any series referenced by its instances
        """

        def _wanted_set(m):
            if m is None:
                return None
            return {m.upper()} if isinstance(m, str) else {str(x).upper() for x in m}

        def _series_mod_ok(s):
            return wanted is None or (getattr(s, "Modality", None) or "").upper() in wanted

        def _inst_mod_ok(i):
            return wanted is None or (getattr(i, "Modality", None) or "").upper() in wanted

        def _effective_fors(series: SeriesNode) -> set[str]:
            fors: set[str] = set()
            fo_direct = getattr(series, "FrameOfReferenceUID", None)
            if fo_direct:
                fors.add(str(fo_direct))
            if derive_frame_from_references:
                for inst in getattr(series, "instances", {}).values():
                    for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                        if u:
                            fors.add(str(u))
                    for rs in getattr(inst, "referenced_series", []) or []:
                        u = getattr(rs, "FrameOfReferenceUID", None)
                        if u:
                            fors.add(str(u))
            return fors

        lvl = str(level).upper()
        if lvl not in {"SERIES", "INSTANCE"}:
            raise ValueError("level must be 'SERIES' or 'INSTANCE'")

        wanted = _wanted_set(modality)

        # Anchor series + anchor FoR set
        anchor_series = (
            node if isinstance(node, SeriesNode) else getattr(node, "parent_series", None)
        )
        anchor_fors: set[str] = set()
        if isinstance(node, InstanceNode) and getattr(node, "FrameOfReferenceUIDs", None):
            anchor_fors |= {str(u) for u in (node.FrameOfReferenceUIDs or []) if u}
        if anchor_series and getattr(anchor_series, "FrameOfReferenceUID", None):
            anchor_fors.add(str(anchor_series.FrameOfReferenceUID))
        # If still empty and we’re allowed to derive, derive from anchor series’ instances
        if not anchor_fors and anchor_series and derive_frame_from_references:
            anchor_fors |= _effective_fors(anchor_series)

        if not anchor_fors:
            # no FoR context — return only self if requested
            if include_self:
                if lvl == "SERIES" and isinstance(node, SeriesNode) and _series_mod_ok(node):
                    return [node]
                if lvl == "INSTANCE" and isinstance(node, InstanceNode) and _inst_mod_ok(node):
                    return [node]
            return []

        # Collect peer series: intersection of effective FoRs with anchor_fors
        peer_series: list[SeriesNode] = []
        seen_sid: set[int] = set()

        # Prefer dicom_files when provided (covers RTSTRUCT/SEG/REG cases correctly)
        if dicom_files:
            for _pid, sdict in dicom_files.items():
                for s in sdict.values():
                    if anchor_series is not None and s is anchor_series:
                        continue
                    eff = _effective_fors(s)
                    if eff and (eff & anchor_fors):
                        if id(s) not in seen_sid:
                            peer_series.append(s)
                            seen_sid.add(id(s))
        else:
            # Fallback to precomputed FoR neighbors (series-level only; may miss RTSTRUCT)
            if anchor_series:
                for s in list(getattr(anchor_series, "frame_of_reference_registered", []) or []):
                    if id(s) in seen_sid:
                        continue
                    # verify intersection using effective FoRs to avoid false negatives
                    if _effective_fors(s) & anchor_fors:
                        peer_series.append(s)
                        seen_sid.add(id(s))

        if lvl == "SERIES":
            out = []
            if include_self and anchor_series and _series_mod_ok(anchor_series):
                out.append(anchor_series)
            out.extend([s for s in peer_series if _series_mod_ok(s)])
            # de-dup
            seen = set()
            dedup = []
            for x in out:
                if id(x) not in seen:
                    seen.add(id(x))
                    dedup.append(x)
            return dedup

        # INSTANCE level: return instances from anchor (optional) + peer series
        out_i: list[InstanceNode] = []

        def add_series(series: SeriesNode):
            for inst in getattr(series, "instances", {}).values():
                if _inst_mod_ok(inst):
                    out_i.append(inst)

        if include_self and anchor_series:
            add_series(anchor_series)
        for s in peer_series:
            add_series(s)

        seen = set()
        dedup = []
        for x in out_i:
            if id(x) not in seen:
                seen.add(id(x))
                dedup.append(x)
        return dedup

    def get_frame_registered_clusters(
        self,
        *,
        scope: str = "dataset",  # 'dataset' | 'patient' | 'study'
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
        modality: Optional[Union[str, Iterable[str]]] = None,
        include_missing_for: bool = False,
        derive_frame_from_references: bool = True,  # <— generic & True by default
        min_cluster_size: int = 1,
    ) -> Dict[str, List[SeriesNode]]:
        """
        Group series by FrameOfReferenceUID (FoR) within the chosen scope.

        Scope
        -----
        - 'dataset': consider all series in `self.dicom_files` (all patients/studies).
        - 'patient': consider only series under `patient_id`.
        - 'study'  : consider only series under (`patient_id`, `study_uid`).
                    Cross-study series with the same FoR are excluded by design.

        Parameters
        ----------
        scope : {'dataset','patient','study'}, default 'dataset'
        patient_id : str, optional
            Required when scope is 'patient' or 'study'.
        study_uid : str, optional
            Required when scope is 'study'.
        modality : str or Iterable[str], optional
            Case-insensitive modality filter applied to `series.Modality`.
            Accepts a single string (e.g., "CT") or an iterable (e.g., ["CT","MR"]).
        include_missing_for : bool, default False
            If True, include series with no resolvable FoR under key '<MISSING>'.
        derive_frame_from_references : bool, default True
            When True, derive additional FoR memberships for a series by inspecting:
            (1) each instance's `InstanceNode.FrameOfReferenceUIDs`, and
            (2) each instance's `referenced_series.FrameOfReferenceUID`.
            The final FoR set is the union of the series' own FoR (if any) and the derived FoRs.
            A series may therefore appear in multiple clusters (e.g., RTSTRUCT that references
            multiple FoRs). When False, only the series' own FoR is used.
        min_cluster_size : int, default 1
            Drop clusters smaller than this size (by number of series).

        Returns
        -------
        dict[str, list[SeriesNode]]
            Map FoR UID -> list of SeriesNodes (sorted by Modality then SeriesInstanceUID).
            If `include_missing_for` is True, includes key '<MISSING>' when no FoR is found.

        Notes
        -----
        - With `derive_frame_from_references=True` (default), RTSTRUCT or any modality that carries
        or references multiple FoRs can appear in multiple FoR clusters, which matches DICOM
        allowances.
        - With `derive_frame_from_references=False`, clustering is a strict series-level FoR
        grouping.

        Examples
        --------
        >>> # dataset-wide clusters (CT only)
        >>> clusters = loader.get_frame_registered_clusters(modality="CT")
        >>> # patient-scope clusters (CT or MR)
        >>> clusters_p = loader.get_frame_registered_clusters(scope="patient",
        ...                                                  patient_id="P001",
        ...                                                  modality=["CT","MR"])
        >>> # study-scope clusters (no cross-study FoR pulling)
        >>> clusters_s = loader.get_frame_registered_clusters(scope="study",
        ...                                                  patient_id="P001",
        ...                                                  study_uid="1.2.3")
        >>> # derive FoR from references (default True) and keep only clusters with ≥2 series
        >>> clusters_rt = loader.get_frame_registered_clusters(scope="patient",
        ...     patient_id="P001", min_cluster_size=2)
        """

        def _wanted_set(m: Optional[Union[str, Iterable[str]]]) -> Optional[Set[str]]:
            if m is None:
                return None
            if isinstance(m, str):
                return {m.upper()}
            return {str(x).upper() for x in m}

        def mod_ok(s: SeriesNode) -> bool:
            return wanted is None or (getattr(s, "Modality", None) or "").upper() in wanted

        wanted = _wanted_set(modality)

        scope_l = scope.lower()
        if scope_l == "dataset":
            series_iter = (
                s for _pid, sdict in (self.dicom_files or {}).items() for s in sdict.values()
            )
        elif scope_l == "patient":
            if not patient_id:
                raise ValueError("patient_id is required when scope='patient'")
            series_iter = iter((self.dicom_files.get(patient_id, {}) or {}).values())
        elif scope_l == "study":
            if not (patient_id and study_uid):
                raise ValueError("patient_id and study_uid are required when scope='study'")
            series_iter = (
                s
                for s in (self.dicom_files.get(patient_id, {}) or {}).values()
                if getattr(s, "StudyInstanceUID", None) == study_uid
            )
        else:
            raise ValueError("scope must be one of {'dataset','patient','study'}")

        groups: Dict[str, List[SeriesNode]] = defaultdict(list)

        for s in series_iter:
            if not mod_ok(s):
                continue

            # Start with the series' own FoR (if any)
            fors: Set[str] = set()
            fo_direct = getattr(s, "FrameOfReferenceUID", None)
            if fo_direct:
                fors.add(str(fo_direct))

            if derive_frame_from_references:
                # Add FoRs from instances
                for inst in getattr(s, "instances", {}).values():
                    # Instance-declared FoRs (multi-FoR friendly, e.g., RTSTRUCT)
                    for u in getattr(inst, "FrameOfReferenceUIDs", []) or []:
                        if u:
                            fors.add(str(u))
                    # FoRs from explicitly referenced series
                    for rs in getattr(inst, "referenced_series", []) or []:
                        u = getattr(rs, "FrameOfReferenceUID", None)
                        if u:
                            fors.add(str(u))

            if fors:
                for u in fors:
                    groups[u].append(s)
            else:
                if include_missing_for:
                    groups["<MISSING>"].append(s)

        # Sort and filter by min_cluster_size
        for k in list(groups.keys()):
            groups[k] = sorted(
                groups[k], key=lambda x: ((x.Modality or ""), (x.SeriesInstanceUID or ""))
            )
        if min_cluster_size > 1:
            groups = {k: v for k, v in groups.items() if len(v) >= min_cluster_size}

        return dict(groups)

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
            referenced_series = list()
            for sop_uid, instance in series.instances.items():
                if instance.referenced_sids:
                    for ref_sid in instance.referenced_sids:
                        ref_series = self.get_series(ref_sid)
                        if ref_series:
                            referenced_series.append(ref_series)

            return referenced_series

        def get_other_referenced_series(series):
            referenced_series = list()
            for sop_uid, instance in series.instances.items():
                if instance.other_referenced_sids:
                    for ref_sid in instance.other_referenced_sids:
                        ref_series = self.get_series(ref_sid)
                        if ref_series:
                            referenced_series.append(ref_series)

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
                                referenced_series = referenced_series_set[0]
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
                                    other_referenced_series = other_referenced_series_set[0]
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

    def __getitem__(self, patient_id):
        return self.dataset[patient_id]

    def __contains__(self, patient_id):
        return patient_id in self.dataset

    def __len__(self):
        return len(self.dataset)

    def iter_studies(self):
        self.dataset.iter_studies()

    def iter_series(self):
        self.dataset.iter_series()

    def iter_instances(self):
        self.dataset.iter_instances()

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
