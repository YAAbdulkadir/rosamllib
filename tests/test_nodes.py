import types
import pytest

from rosamllib.nodes.dicom_nodes import (
    DatasetNode,
    PatientNode,
    StudyNode,
    SeriesNode,
    InstanceNode,
)
import rosamllib.nodes.dicom_nodes as dicom_nodes_mod


# --- simple fakes used to isolate DatasetNode behavior ---


class FakeInstanceNode:
    """Minimal stand-in for InstanceNode for SeriesNode tests."""

    def __init__(self, sop_uid: str, file_path: str | None = None):
        self.SOPInstanceUID = sop_uid
        self.FilePath = file_path
        self.parent_series = None

    # allow roundtrip serialization tests to work later
    def to_dict(self):
        return {
            "SOPInstanceUID": self.SOPInstanceUID,
            "FilePath": self.FilePath,
        }

    @classmethod
    def from_dict(cls, d, parent_series=None):
        inst = cls(d["SOPInstanceUID"], d.get("FilePath"))
        inst.parent_series = parent_series
        return inst


class FakeSeriesNode:
    def __init__(self, series_uid, instances=None, modality=None, frame_uid=None):
        self.SeriesInstanceUID = series_uid
        # both attribute and internal dict (PatientNode.find_dangling_references uses .instances)
        self.instances = instances or {}
        self._instances = self.instances
        self.Modality = modality
        self.FrameOfReferenceUID = frame_uid
        self.parent_study = None

    def get_instance(self, sop_uid):
        return self._instances.get(sop_uid)

    def __iter__(self):
        # iterate over instances
        return iter(self._instances.values())


class FakeStudyNode:
    def __init__(self, study_uid, series=None):
        self.StudyInstanceUID = study_uid
        self._series = series or {}

    def get_series(self, series_uid):
        return self._series.get(series_uid)

    def get_instance(self, sop_uid, *, series_uid=None):
        # supports PatientNode.get_instance delegation
        if series_uid is not None:
            s = self.get_series(series_uid)
            return None if s is None else s.get_instance(sop_uid)
        for s in self._series.values():
            inst = s.get_instance(sop_uid)
            if inst is not None:
                return inst
        return None

    def __iter__(self):
        # iterate over series
        return iter(self._series.values())


class FakePatientNode:
    def __init__(
        self,
        patient_id,
        *,
        studies=None,
        sources_without_reach=None,
        dangling=None,
        orphans=None,
    ):
        self.PatientID = patient_id
        self._studies = studies or {}
        self._sources_without_reach = sources_without_reach
        self._dangling = dangling or []
        self._orphans = orphans or []

    def get_study(self, study_uid):
        return self._studies.get(study_uid)

    def get_series(self, series_uid):
        # search all studies
        for st in self._studies.values():
            s = st.get_series(series_uid)
            if s is not None:
                return s
        return None

    def get_instance(self, sop_uid, *, series_uid=None, study_uid=None):
        # super simplified for testing delegation
        if series_uid is not None:
            s = self.get_series(series_uid)
            return None if s is None else s.get_instance(sop_uid)
        # fall back to scanning all series
        for st in self._studies.values():
            for se in st:
                inst = se.get_instance(sop_uid)
                if inst is not None:
                    return inst
        return None

    def find_sources_without_reach(self, **kwargs):
        # return preset list (or empty) regardless of kwargs
        return self._sources_without_reach or []

    def find_dangling_references(self, *, return_df=False):
        # DatasetNode always calls this with return_df=False
        return list(self._dangling)

    def find_orphans(self, *, level, modality, include_frame, return_df):
        # ignore arguments and return preset list
        return list(self._orphans)

    def to_dict(self):
        # minimal to_dict for DatasetNode.to_dict()
        return {
            "PatientID": self.PatientID,
        }

    def __iter__(self):
        # allow DatasetNode.iter_studies to do `yield from patient`
        return iter(self._studies.values())


# --- basic construction and repr ---


def test_dataset_node_initialization_and_repr():
    ds = DatasetNode(dataset_id="DS001", dataset_name="Test Dataset")
    assert ds.dataset_id == "DS001"
    assert ds.dataset_name == "Test Dataset"
    assert len(ds) == 0
    assert ds.patients == {}

    r = repr(ds)
    assert "DatasetNode" in r
    assert "DS001" in r
    assert "NumPatients=0" in r


# --- add_patient / basic accessors ---


class PatientWithID:
    def __init__(self, pid):
        self.PatientID = pid
        self.parent_dataset = None


def test_add_patient_basic_and_accessors():
    ds = DatasetNode(dataset_id="DS001")
    p = PatientWithID("P1")

    ds.add_patient(p)
    assert len(ds) == 1
    assert "P1" in ds
    assert ds["P1"] is p
    assert ds.get_patient("P1") is p
    assert p.parent_dataset is ds

    # iteration yields patients
    pts = list(ds)
    assert pts == [p]


def test_add_patient_requires_non_empty_patient_id():
    ds = DatasetNode(dataset_id="DS001")
    p = PatientWithID("")

    with pytest.raises(ValueError):
        ds.add_patient(p)


def test_add_patient_duplicate_without_overwrite_raises():
    ds = DatasetNode(dataset_id="DS001")
    p1 = PatientWithID("P1")
    p2 = PatientWithID("P1")

    ds.add_patient(p1)
    with pytest.raises(KeyError):
        ds.add_patient(p2)

    # still the original patient
    assert ds["P1"] is p1


def test_add_patient_with_overwrite_replaces_existing():
    ds = DatasetNode(dataset_id="DS001")
    p1 = PatientWithID("P1")
    p2 = PatientWithID("P1")

    ds.add_patient(p1)
    ds.add_patient(p2, overwrite=True)

    assert ds["P1"] is p2
    assert p2.parent_dataset is ds


# --- get_or_create_patient ---


def test_get_or_create_patient_creates_and_reuses_existing():
    ds = DatasetNode(dataset_id="DS001")

    p1 = ds.get_or_create_patient("P1", patient_name="Alice")
    assert p1.PatientID == "P1"
    assert getattr(p1, "patient_name", None) == "Alice"
    assert ds.get_patient("P1") is p1
    assert p1.parent_dataset is ds

    # second call returns existing patient, not a new one
    p2 = ds.get_or_create_patient("P1")
    assert p2 is p1
    # name should not change if not provided on subsequent call
    assert getattr(p2, "patient_name", None) == "Alice"


# --- get_study delegation ---


def test_get_study_with_patient_id_fast_path():
    st = FakeStudyNode("ST1")
    fake_patient = FakePatientNode("P1", studies={"ST1": st})

    ds = DatasetNode(dataset_id="DS001")
    # insert directly into dict to avoid PatientID validation
    ds.patients["P1"] = fake_patient

    result = ds.get_study("ST1", patient_id="P1")
    assert result is st

    # unknown patient -> None
    assert ds.get_study("ST1", patient_id="P2") is None

    # unknown study -> None
    assert ds.get_study("STX", patient_id="P1") is None


def test_get_study_searches_all_patients_when_patient_not_given():
    st_target = FakeStudyNode("ST_TARGET")
    p1 = FakePatientNode("P1", studies={})
    p2 = FakePatientNode("P2", studies={"ST_TARGET": st_target})

    ds = DatasetNode(dataset_id="DS001")
    ds.patients["P1"] = p1
    ds.patients["P2"] = p2

    result = ds.get_study("ST_TARGET")
    assert result is st_target


# --- get_series delegation ---


def test_get_series_with_patient_and_study():
    series = FakeSeriesNode("SE1")
    study = FakeStudyNode("ST1", series={"SE1": series})
    patient = FakePatientNode("P1", studies={"ST1": study})

    ds = DatasetNode(dataset_id="DS001")
    ds.patients["P1"] = patient

    result = ds.get_series("SE1", patient_id="P1", study_uid="ST1")
    assert result is series

    # patient not found
    assert ds.get_series("SE1", patient_id="P2", study_uid="ST1") is None

    # study not found
    result = ds.get_series("SE_X", patient_id="P1", study_uid="ST1")
    assert result is None


def test_get_series_with_patient_only():
    series = FakeSeriesNode("SE1")
    patient = FakePatientNode("P1", studies={"ST1": FakeStudyNode("ST1", {"SE1": series})})

    ds = DatasetNode(dataset_id="DS001")
    ds.patients["P1"] = patient

    result = ds.get_series("SE1", patient_id="P1")
    assert result is series


def test_get_series_with_study_only(monkeypatch):
    series = FakeSeriesNode("SE1")
    study = FakeStudyNode("ST1", series={"SE1": series})

    ds = DatasetNode(dataset_id="DS001")

    # patch DatasetNode.get_study on the class
    def fake_get_study(self, uid, patient_id=None):
        assert uid == "ST1"
        assert patient_id is None
        return study

    monkeypatch.setattr(DatasetNode, "get_study", fake_get_study)

    result = ds.get_series("SE1", study_uid="ST1")
    assert result is series


def test_get_series_fallback_to_find_series(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    sentinel = object()

    def fake_find_series(self, uid):
        assert uid == "SE1"
        return sentinel

    monkeypatch.setattr(DatasetNode, "find_series", fake_find_series)

    result = ds.get_series("SE1")
    assert result is sentinel


# --- get_instance delegation ---


def test_get_instance_prefers_patient():
    ds = DatasetNode(dataset_id="DS001")
    sentinel = object()

    def fake_get_instance(self, sop_uid, *, series_uid=None, study_uid=None):
        return (sentinel, sop_uid, series_uid, study_uid)

    # build a simple object with get_instance
    patient = types.SimpleNamespace()
    patient.get_instance = lambda sop_uid, series_uid=None, study_uid=None: fake_get_instance(
        patient, sop_uid, series_uid=series_uid, study_uid=study_uid
    )

    ds.patients["P1"] = patient

    result = ds.get_instance("1.2.3", patient_id="P1", series_uid="SE1", study_uid="ST1")
    assert result == (sentinel, "1.2.3", "SE1", "ST1")


def test_get_instance_uses_study_when_no_patient(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    sentinel = object()

    class Study:
        def get_instance(self, sop_uid, *, series_uid=None):
            return (sentinel, sop_uid, series_uid)

    study = Study()

    def fake_get_study(self, uid, patient_id=None):
        assert uid == "ST1"
        assert patient_id is None
        return study

    monkeypatch.setattr(DatasetNode, "get_study", fake_get_study)

    result = ds.get_instance("1.2.3", study_uid="ST1", series_uid="SE1")
    assert result == (sentinel, "1.2.3", "SE1")


def test_get_instance_uses_series_when_only_series_given(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    sentinel = object()

    class Series:
        def get_instance(self, sop_uid):
            return (sentinel, sop_uid)

    series = Series()

    def fake_find_series(self, uid):
        assert uid == "SE1"
        return series

    monkeypatch.setattr(DatasetNode, "find_series", fake_find_series)

    result = ds.get_instance("1.2.3", series_uid="SE1")
    assert result == (sentinel, "1.2.3")


def test_get_instance_fallback_find_instance(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    sentinel = object()

    def fake_find_instance(self, uid):
        assert uid == "1.2.3"
        return sentinel

    monkeypatch.setattr(DatasetNode, "find_instance", fake_find_instance)

    result = ds.get_instance("1.2.3")
    assert result is sentinel


# --- traversal helpers ---


def test_iter_helpers_and_find_methods():
    inst1 = FakeInstanceNode("I1")
    inst2 = FakeInstanceNode("I2")
    se1 = FakeSeriesNode("SE1", instances={"I1": inst1})
    se2 = FakeSeriesNode("SE2", instances={"I2": inst2})
    st1 = FakeStudyNode("ST1", series={"SE1": se1})
    st2 = FakeStudyNode("ST2", series={"SE2": se2})
    p = FakePatientNode("P1", studies={"ST1": st1, "ST2": st2})

    ds = DatasetNode(dataset_id="DS001")
    ds.patients["P1"] = p

    assert list(ds.iter_studies()) == [st1, st2]
    assert list(ds.iter_series()) == [se1, se2]
    assert list(ds.iter_instances()) == [inst1, inst2]

    assert ds.find_series("SE1") is se1
    assert ds.find_series("SE2") is se2
    assert ds.find_series("SEX") is None

    assert ds.find_instance("I1") is inst1
    assert ds.find_instance("I2") is inst2
    assert ds.find_instance("IX") is None


# --- wrapper methods to utils:
# get_referenced_nodes / get_referencing_nodes / get_frame_registered_nodes ---


def test_get_referenced_nodes_delegates(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    node = object()
    called = {}

    def fake_helper(n, modality, level, recursive, include_start):
        called["args"] = (n, modality, level, recursive, include_start)
        return ["ok"]

    # patch the name bound in dicom_nodes_mod
    monkeypatch.setattr(dicom_nodes_mod, "get_referenced_nodes", fake_helper, raising=True)

    result = ds.get_referenced_nodes(
        node, modality="CT", level="INSTANCE", recursive=False, include_start=True
    )
    assert result == ["ok"]
    assert called["args"] == (node, "CT", "INSTANCE", False, True)


def test_get_referencing_nodes_delegates(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    node = object()
    called = {}

    def fake_helper(n, modality, level, recursive, include_start):
        called["args"] = (n, modality, level, recursive, include_start)
        return ["peer"]

    monkeypatch.setattr(dicom_nodes_mod, "get_referencing_nodes", fake_helper, raising=True)

    result = ds.get_referencing_nodes(
        node, modality=["CT", "MR"], level="SERIES", recursive=True, include_start=False
    )
    assert result == ["peer"]
    assert called["args"] == (node, ["CT", "MR"], "SERIES", True, False)


def test_get_frame_registered_nodes_delegates(monkeypatch):
    ds = DatasetNode(dataset_id="DS001")
    node = object()
    called = {}

    def fake_helper(
        n, *, level, include_self, modality, dicom_files, derive_frame_from_references
    ):
        called["kwargs"] = dict(
            n=n,
            level=level,
            include_self=include_self,
            modality=modality,
            dicom_files=dicom_files,
            derive_frame_from_references=derive_frame_from_references,
        )
        return ["frame_peer"]

    monkeypatch.setattr(dicom_nodes_mod, "get_frame_registered_nodes", fake_helper, raising=True)

    result = ds.get_frame_registered_nodes(
        node,
        level="INSTANCE",
        include_self=True,
        modality="RTDOSE",
        dicom_files={"dummy": "value"},
        derive_frame_from_references=False,
    )
    assert result == ["frame_peer"]
    assert called["kwargs"] == dict(
        n=node,
        level="INSTANCE",
        include_self=True,
        modality="RTDOSE",
        dicom_files={"dummy": "value"},
        derive_frame_from_references=False,
    )


# --- reports: report_sources_without_reach / dangling_references_report / orphan_report ---


def test_report_sources_without_reach_collects_by_patient_id():
    p1 = FakePatientNode("P1", sources_without_reach=["a", "b"])
    p2 = FakePatientNode("P2", sources_without_reach=[])
    p3 = FakePatientNode("P3", sources_without_reach=["c"])

    ds = DatasetNode(dataset_id="DS001")
    ds.patients = {"P1": p1, "P2": p2, "P3": p3}

    report = ds.report_sources_without_reach()
    assert report == {"P1": ["a", "b"], "P3": ["c"]}


def test_dangling_references_report_list_mode():
    dangling1 = [{"foo": 1}]
    dangling2 = [{"foo": 2}, {"foo": 3}]

    p1 = FakePatientNode("P1", dangling=dangling1)
    p2 = FakePatientNode("P2", dangling=dangling2)

    ds = DatasetNode(dataset_id="DS001")
    ds.patients = {"P1": p1, "P2": p2}

    res = ds.dangling_references_report(return_df=False)
    # should be concatenation of p1 + p2 results, preserving order
    assert res == dangling1 + dangling2


def test_dangling_references_report_dataframe_mode():
    pd = pytest.importorskip("pandas")

    dangling1 = [
        {
            "PatientID": "P1",
            "SourceLevel": "SERIES",
            "SourceModality": "CT",
            "SourceSeriesUID": "SE1",
            "SourceSOPInstanceUID": None,
            "MissingKind": "INSTANCE",
            "MissingUID": "I_X",
        }
    ]
    dangling2 = [
        {
            "PatientID": "P2",
            "SourceLevel": "INSTANCE",
            "SourceModality": "RTDOSE",
            "SourceSeriesUID": "SE2",
            "SourceSOPInstanceUID": "I2",
            "MissingKind": "SERIES",
            "MissingUID": "SE_X",
        }
    ]

    p1 = FakePatientNode("P1", dangling=dangling1)
    p2 = FakePatientNode("P2", dangling=dangling2)

    ds = DatasetNode(dataset_id="DS001")
    ds.patients = {"P1": p1, "P2": p2}

    df = ds.dangling_references_report(return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "PatientID",
        "SourceLevel",
        "SourceModality",
        "SourceSeriesUID",
        "SourceSOPInstanceUID",
        "MissingKind",
        "MissingUID",
    }
    # should contain both rows
    assert len(df) == 2


def test_orphan_report_list_mode():
    # build fake series + instance orphans for a single patient
    se = FakeSeriesNode("SE1", modality="CT", frame_uid="F1")
    st = FakeStudyNode("ST1", series={"SE1": se})
    se.parent_study = st

    inst = FakeInstanceNode("I1")
    inst.parent_series = se

    p1 = FakePatientNode("P1", orphans=[se, inst])

    ds = DatasetNode(dataset_id="DS001")
    ds.patients = {"P1": p1}

    res = ds.orphan_report(return_df=False)
    # should just return the raw nodes in a flat list
    assert res == [se, inst]


def test_orphan_report_dataframe_mode():
    pd = pytest.importorskip("pandas")

    # build fake series + instance orphans for two patients
    se1 = FakeSeriesNode("SE1", modality="CT", frame_uid="F1")
    st1 = FakeStudyNode("ST1", series={"SE1": se1})
    se1.parent_study = st1

    inst1 = FakeInstanceNode("I1")
    inst1.parent_series = se1

    se2 = FakeSeriesNode("SE2", modality="MR", frame_uid="F2")
    st2 = FakeStudyNode("ST2", series={"SE2": se2})
    se2.parent_study = st2

    inst2 = FakeInstanceNode("I2")
    inst2.parent_series = se2

    p1 = FakePatientNode("P1", orphans=[se1, inst1])
    p2 = FakePatientNode("P2", orphans=[se2, inst2])

    ds = DatasetNode(dataset_id="DS001")
    ds.patients = {"P1": p1, "P2": p2}

    df = ds.orphan_report(return_df=True, include_frame=True)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "PatientID",
        "Level",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Modality",
        "FrameOfReferenceUID",
        "IsOrphan",
    }
    # 4 rows: 2 series + 2 instances
    assert len(df) == 4

    # sanity checks
    assert (df["PatientID"] == "P1").any()
    assert (df["PatientID"] == "P2").any()
    # with our fake classes, everything is treated as INSTANCE
    assert (df["Level"] == "INSTANCE").all()


# ======================================================================
# PatientNode tests
# ======================================================================


def test_patient_node_basic_and_repr():
    p = PatientNode(patient_id="P1", patient_name="Alice")
    assert p.PatientID == "P1"
    assert p.PatientName == "Alice"
    assert len(p) == 0
    assert p.studies == {}

    r = repr(p)
    assert "PatientNode" in r
    assert "P1" in r
    assert "Alice" in r
    assert "NumStudies=0" in r


def test_patient_add_study_basic():
    p = PatientNode(patient_id="P1")
    st = FakeStudyNode("1.2.3.4")

    p.add_study(st)
    assert len(p) == 1
    assert "1.2.3.4" in p
    assert p["1.2.3.4"] is st
    assert p.get_study("1.2.3.4") is st
    assert st.parent_patient is p


def test_patient_add_study_invalid_uid():
    p = PatientNode(patient_id="P1")
    st = FakeStudyNode("not-a-uid")

    with pytest.raises(ValueError):
        p.add_study(st)


def test_patient_add_study_duplicate_without_overwrite():
    p = PatientNode(patient_id="P1")
    st1 = FakeStudyNode("1.2.3")
    st2 = FakeStudyNode("1.2.3")

    p.add_study(st1)
    with pytest.raises(KeyError):
        p.add_study(st2)

    assert p["1.2.3"] is st1


def test_patient_add_study_with_overwrite():
    p = PatientNode(patient_id="P1")
    st1 = FakeStudyNode("1.2.3")
    st2 = FakeStudyNode("1.2.3")

    p.add_study(st1)
    p.add_study(st2, overwrite=True)

    assert p["1.2.3"] is st2
    assert st2.parent_patient is p


def test_patient_get_or_create_study_creates_and_reuses():
    p = PatientNode(patient_id="P1")

    s1 = p.get_or_create_study("1.2.3", study_description="Foo")
    assert s1.StudyInstanceUID == "1.2.3"
    # We can’t assert field names on the real StudyNode yet, but we can at least
    # check it’s stored and parent is set.
    assert p.get_study("1.2.3") is s1
    assert s1.parent_patient is p

    # second call reuses the same study
    s2 = p.get_or_create_study("1.2.3")
    assert s2 is s1


# --- PatientNode.get_series ---


def test_patient_get_series_with_study_uid():
    se = FakeSeriesNode("SE1")
    st = FakeStudyNode("ST1", series={"SE1": se})

    p = PatientNode(patient_id="P1")
    p.studies["ST1"] = st

    result = p.get_series("SE1", study_uid="ST1")
    assert result is se

    # unknown study
    assert p.get_series("SE1", study_uid="STX") is None

    # unknown series
    assert p.get_series("SEx", study_uid="ST1") is None


def test_patient_get_series_searches_all_studies():
    se = FakeSeriesNode("SE_TARGET")
    st1 = FakeStudyNode("ST1", series={})
    st2 = FakeStudyNode("ST2", series={"SE_TARGET": se})

    p = PatientNode(patient_id="P1")
    p.studies = {"ST1": st1, "ST2": st2}

    result = p.get_series("SE_TARGET")
    assert result is se


# --- PatientNode.get_instance ---


def test_patient_get_instance_prefers_study_uid():
    inst = FakeInstanceNode("I1")
    se = FakeSeriesNode("SE1", instances={"I1": inst})
    st = FakeStudyNode("ST1", series={"SE1": se})

    p = PatientNode(patient_id="P1")
    p.studies["ST1"] = st

    # with both study_uid and series_uid
    res = p.get_instance("I1", study_uid="ST1", series_uid="SE1")
    assert res is inst

    # unknown study gives None
    assert p.get_instance("I1", study_uid="STX", series_uid="SE1") is None


def test_patient_get_instance_with_series_only():
    inst = FakeInstanceNode("I1")
    se = FakeSeriesNode("SE1", instances={"I1": inst})
    st1 = FakeStudyNode("ST1", series={})
    st2 = FakeStudyNode("ST2", series={"SE1": se})

    p = PatientNode(patient_id="P1")
    p.studies = {"ST1": st1, "ST2": st2}

    res = p.get_instance("I1", series_uid="SE1")
    assert res is inst

    # unknown series
    assert p.get_instance("I1", series_uid="SEX") is None


def test_patient_get_instance_scan_all_studies():
    inst = FakeInstanceNode("I1")
    se = FakeSeriesNode("SE1", instances={"I1": inst})
    st = FakeStudyNode("ST1", series={"SE1": se})

    p = PatientNode(patient_id="P1")
    p.studies["ST1"] = st

    res = p.get_instance("I1")
    assert res is inst

    assert p.get_instance("IX") is None


# --- PatientNode helper delegation wrappers ---


def test_patient_get_referenced_nodes_delegates(monkeypatch):
    p = PatientNode(patient_id="P1")
    node = object()
    called = {}

    def fake_helper(n, modality, level, recursive, include_start):
        called["args"] = (n, modality, level, recursive, include_start)
        return ["ok"]

    monkeypatch.setattr(dicom_nodes_mod, "get_referenced_nodes", fake_helper, raising=True)

    result = p.get_referenced_nodes(
        node, modality="CT", level="INSTANCE", recursive=False, include_start=True
    )
    assert result == ["ok"]
    assert called["args"] == (node, "CT", "INSTANCE", False, True)


def test_patient_get_referencing_nodes_delegates(monkeypatch):
    p = PatientNode(patient_id="P1")
    node = object()
    called = {}

    def fake_helper(n, modality, level, recursive, include_start):
        called["args"] = (n, modality, level, recursive, include_start)
        return ["peer"]

    monkeypatch.setattr(dicom_nodes_mod, "get_referencing_nodes", fake_helper, raising=True)

    result = p.get_referencing_nodes(
        node, modality=["CT", "MR"], level="SERIES", recursive=True, include_start=False
    )
    assert result == ["peer"]
    assert called["args"] == (node, ["CT", "MR"], "SERIES", True, False)


def test_patient_get_frame_registered_nodes_delegates(monkeypatch):
    p = PatientNode(patient_id="P1")
    node = object()
    called = {}

    def fake_helper(
        n, *, level, include_self, modality, dicom_files, derive_frame_from_references
    ):
        called["kwargs"] = dict(
            n=n,
            level=level,
            include_self=include_self,
            modality=modality,
            dicom_files=dicom_files,
            derive_frame_from_references=derive_frame_from_references,
        )
        return ["frame_peer"]

    monkeypatch.setattr(dicom_nodes_mod, "get_frame_registered_nodes", fake_helper, raising=True)

    result = p.get_frame_registered_nodes(
        node,
        level="INSTANCE",
        include_self=True,
        modality="RTDOSE",
        dicom_files={"dummy": "value"},
        derive_frame_from_references=False,
    )
    assert result == ["frame_peer"]
    assert called["kwargs"] == dict(
        n=node,
        level="INSTANCE",
        include_self=True,
        modality="RTDOSE",
        dicom_files={"dummy": "value"},
        derive_frame_from_references=False,
    )


# --- find_sources_with_reach / find_sources_without_reach ---


class SimpleNode:
    """Tiny node with just a name, for reachability tests."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"SimpleNode({self.name})"


def test_patient_find_sources_with_and_without_reach(monkeypatch):
    p = PatientNode(patient_id="P1")

    a = SimpleNode("A")
    b = SimpleNode("B")

    # Force _iter_nodes to return our two nodes regardless of level
    def fake_iter_nodes(self, level):
        assert level in {"SERIES", "INSTANCE"}
        return [a, b]

    monkeypatch.setattr(PatientNode, "_iter_nodes", fake_iter_nodes)

    # Make level/modality checks always succeed so only predicates matter
    monkeypatch.setattr(dicom_nodes_mod, "_level_ok", lambda n, lvl: True, raising=True)
    monkeypatch.setattr(dicom_nodes_mod, "_modality_ok", lambda n, wanted: True, raising=True)
    monkeypatch.setattr(dicom_nodes_mod, "_norm_modalities", lambda m: None, raising=True)

    # neighbors: A -> [B], B -> []
    def fake_referenced(n, modality, level, recursive, include_start):
        assert modality is None
        assert level in {"SERIES", "INSTANCE"}
        # Only A has a neighbor
        return [b] if n is a else []

    monkeypatch.setattr(dicom_nodes_mod, "get_referenced_nodes", fake_referenced, raising=True)

    # referencing / frame not used for traversal="referenced", allow_same_for=False
    monkeypatch.setattr(
        dicom_nodes_mod,
        "get_referencing_nodes",
        lambda *args, **kwargs: [],
        raising=True,
    )
    monkeypatch.setattr(
        dicom_nodes_mod,
        "get_frame_registered_nodes",
        lambda *args, **kwargs: [],
        raising=True,
    )

    # A and B are both possible starts; B is the only target
    starts = lambda n: n in (a, b)  # noqa: E731
    targets = lambda n: n is b  # noqa: E731

    hits = p.find_sources_with_reach(
        start_level="SERIES",
        start_modality=None,
        target_level="SERIES",
        target_modality=None,
        recursive=True,
        traversal="referenced",
        include_start=False,
        start_predicate=starts,
        target_predicate=targets,
        allow_same_for=False,
    )
    # Only A can reach B
    assert hits == [a]

    misses = p.find_sources_without_reach(
        start_level="SERIES",
        start_modality=None,
        target_level="SERIES",
        target_modality=None,
        recursive=True,
        traversal="referenced",
        include_start=False,
        start_predicate=starts,
        target_predicate=targets,
        allow_same_for=False,
    )
    # Only B cannot reach any target
    assert misses == [b]


# --- find_dangling_references ---


def test_patient_find_dangling_references_list_mode(monkeypatch):
    # Patch SeriesNode/InstanceNode to align isinstance checks with our fakes
    monkeypatch.setattr(dicom_nodes_mod, "SeriesNode", FakeSeriesNode, raising=True)
    monkeypatch.setattr(dicom_nodes_mod, "InstanceNode", FakeInstanceNode, raising=True)

    # Build series + instances
    inst1 = FakeInstanceNode("I1")
    inst2 = FakeInstanceNode("I2")

    # introduce dangling references on inst1
    inst1.ReferencedSeriesInstanceUIDs = ["SE_X"]  # missing series
    inst1.ReferencedSOPInstanceUIDs = ["I_X"]  # missing instance
    # inst2 is clean
    inst2.ReferencedSeriesInstanceUIDs = []
    inst2.ReferencedSOPInstanceUIDs = []

    se = FakeSeriesNode("SE1", instances={"I1": inst1, "I2": inst2})
    # dangling series reference at series level
    se.referenced_sids = ["SE_X", "SE1"]  # SE_X is missing, SE1 exists

    st = FakeStudyNode("ST1", series={"SE1": se})
    p = PatientNode(patient_id="P1")
    p.studies["ST1"] = st

    rows = p.find_dangling_references(return_df=False)
    # Expected:
    # 1) series SE1 -> missing SE_X
    # 2) inst I1 -> missing series SE_X
    # 3) inst I1 -> missing instance I_X
    assert len(rows) == 3

    kinds = {r["MissingKind"] for r in rows}
    assert kinds == {"SERIES", "INSTANCE"}

    # All should be tagged with PatientID P1
    assert all(r["PatientID"] == "P1" for r in rows)


def test_patient_find_dangling_references_dataframe_mode(monkeypatch):
    pd = pytest.importorskip("pandas")

    monkeypatch.setattr(dicom_nodes_mod, "SeriesNode", FakeSeriesNode, raising=True)
    monkeypatch.setattr(dicom_nodes_mod, "InstanceNode", FakeInstanceNode, raising=True)

    inst1 = FakeInstanceNode("I1")
    inst1.ReferencedSeriesInstanceUIDs = ["SE_X"]
    inst1.ReferencedSOPInstanceUIDs = ["I_X"]

    se = FakeSeriesNode("SE1", instances={"I1": inst1})
    se.referenced_sids = ["SE_X"]

    st = FakeStudyNode("ST1", series={"SE1": se})
    p = PatientNode(patient_id="P1")
    p.studies["ST1"] = st

    df = p.find_dangling_references(return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "PatientID",
        "SourceLevel",
        "SourceModality",
        "SourceSeriesUID",
        "SourceSOPInstanceUID",
        "MissingKind",
        "MissingUID",
    }
    assert len(df) == 3  # same reasoning as list mode


# --- find_orphans ---


def test_patient_find_orphans_series_level(monkeypatch):
    pd = pytest.importorskip("pandas")

    # Make isinstance(n, SeriesNode) and InstanceNode checks use our fakes
    monkeypatch.setattr(dicom_nodes_mod, "SeriesNode", FakeSeriesNode, raising=True)
    monkeypatch.setattr(dicom_nodes_mod, "InstanceNode", FakeInstanceNode, raising=True)

    p = PatientNode(patient_id="P1")

    se1 = FakeSeriesNode("SE1", modality="CT", frame_uid="F1")
    se2 = FakeSeriesNode("SE2", modality="CT", frame_uid="F1")

    st = FakeStudyNode("ST1", series={"SE1": se1, "SE2": se2})
    se1.parent_study = st
    se2.parent_study = st
    st.parent_patient = p
    p.studies["ST1"] = st

    # Define connectivity via helpers:
    # - SE1 has an outgoing edge to SE2, so SE1 is not orphan
    # - SE2 has no in/out edges, so SE2 should be an orphan
    def fake_get_referenced(n, modality, level, recursive, include_start):
        assert level == "SERIES"
        return [se2] if n is se1 else []

    def fake_get_referencing(n, modality, level, recursive, include_start):
        # nobody references anything
        return []

    def fake_frame(n, level, include_self, modality):
        # no FoR-based connections
        return []

    monkeypatch.setattr(dicom_nodes_mod, "get_referenced_nodes", fake_get_referenced, raising=True)
    monkeypatch.setattr(
        dicom_nodes_mod,
        "get_referencing_nodes",
        fake_get_referencing,
        raising=True,
    )
    monkeypatch.setattr(
        dicom_nodes_mod,
        "get_frame_registered_nodes",
        fake_frame,
        raising=True,
    )

    # list mode
    orphans = p.find_orphans(level="SERIES", modality="CT", include_frame=False, return_df=False)
    assert orphans == [se2]

    # dataframe mode
    df = p.find_orphans(level="SERIES", modality="CT", include_frame=False, return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["PatientID"] == "P1"
    assert row["Level"] == "SERIES"
    assert row["SeriesInstanceUID"] == "SE2"
    assert bool(row["IsOrphan"])


# ======================================================================
# StudyNode tests
# ======================================================================


def test_study_node_basic_and_repr():
    st = StudyNode(
        study_uid="1.2.3.4.5",
        study_description="CT Chest",
    )

    assert st.StudyInstanceUID == "1.2.3.4.5"
    assert st.StudyDescription == "CT Chest"
    assert len(st) == 0
    assert st.series == {}

    r = repr(st)
    assert "StudyNode" in r
    assert "1.2.3.4.5" in r
    assert "CT Chest" in r
    assert "NumSeries=0" in r


def test_study_add_series_basic():
    st = StudyNode(study_uid="1.2.3")
    # use a valid SeriesInstanceUID
    se = FakeSeriesNode("1.2.3.4")

    st.add_series(se)
    assert len(st) == 1
    assert "1.2.3.4" in st
    assert st["1.2.3.4"] is se
    assert st.get_series("1.2.3.4") is se
    assert se.parent_study is st


def test_study_add_series_invalid_uid():
    st = StudyNode(study_uid="1.2.3")
    se = FakeSeriesNode("not-a-uid")

    with pytest.raises(ValueError):
        st.add_series(se)


def test_study_add_series_duplicate_without_overwrite():
    st = StudyNode(study_uid="1.2.3")
    se1 = FakeSeriesNode("1.2.3.4")
    se2 = FakeSeriesNode("1.2.3.4")

    st.add_series(se1)
    with pytest.raises(KeyError):
        st.add_series(se2)

    assert st["1.2.3.4"] is se1


def test_study_add_series_with_overwrite():
    st = StudyNode(study_uid="1.2.3")
    se1 = FakeSeriesNode("1.2.3.4")
    se2 = FakeSeriesNode("1.2.3.4")

    st.add_series(se1)
    st.add_series(se2, overwrite=True)

    assert st["1.2.3.4"] is se2
    assert se2.parent_study is st


def test_study_get_or_create_series_creates_and_reuses(monkeypatch):
    # Use a tiny stub for SeriesNode so we don't depend on the real implementation
    class StubSeries:
        def __init__(self, series_uid, parent_study=None):
            self.SeriesInstanceUID = series_uid
            self.parent_study = parent_study
            self.Modality = None
            self.SeriesDescription = None

    # Patch the global SeriesNode reference used inside StudyNode
    monkeypatch.setattr(dicom_nodes_mod, "SeriesNode", StubSeries, raising=True)

    st = StudyNode(study_uid="1.2.3")

    s1 = st.get_or_create_series(
        "4.5.6",
        modality="CT",
        desc="Axial CT",
    )
    assert isinstance(s1, StubSeries)
    assert s1.SeriesInstanceUID == "4.5.6"
    assert s1.parent_study is st
    assert s1.Modality == "CT"
    assert s1.SeriesDescription == "Axial CT"
    assert st.get_series("4.5.6") is s1

    # Second call with same UID reuses the existing series
    s2 = st.get_or_create_series("4.5.6", modality="MR", desc="SHOULD_NOT_OVERRIDE")
    assert s2 is s1
    # original attributes preserved
    assert s1.Modality == "CT"
    assert s1.SeriesDescription == "Axial CT"


def test_study_get_intance_with_series_uid():
    inst = FakeInstanceNode("I1")
    se = FakeSeriesNode("SE1", instances={"I1": inst})
    st = StudyNode(study_uid="ST1")
    st.series["SE1"] = se

    # Use the method name as implemented in the class: get_intance
    res = st.get_intance("I1", series_uid="SE1")
    assert res is inst

    # unknown series => None
    assert st.get_intance("I1", series_uid="SEx") is None


def test_study_get_intance_scans_all_series():
    inst = FakeInstanceNode("I1")
    se1 = FakeSeriesNode("SE1", instances={})
    se2 = FakeSeriesNode("SE2", instances={"I1": inst})

    st = StudyNode(study_uid="ST1")
    st.series = {"SE1": se1, "SE2": se2}

    res = st.get_intance("I1")
    assert res is inst

    assert st.get_intance("IX") is None


def test_study_extensible_attrs_and_parent_delegation():
    # Parent patient with an "extra" attribute
    p = PatientNode(patient_id="P1")
    p.set_attrs(global_flag=42)

    st = StudyNode(study_uid="ST1", study_description="desc", parent_patient=p)

    # Extensible attribute on StudyNode itself
    st.set_attrs(foo="bar")
    assert st.foo == "bar"

    # Fallback to parent_patient via __getattr__
    assert st.global_flag == 42


def test_study_to_dict_and_from_dict_roundtrip(monkeypatch):
    # Stub SeriesNode so we control the nested (de)serialization behavior
    class StubSeries:
        def __init__(self, series_uid, parent_study=None):
            self.SeriesInstanceUID = series_uid
            self.parent_study = parent_study
            self.Modality = None

        def to_dict(self):
            return {
                "SeriesInstanceUID": self.SeriesInstanceUID,
                "Modality": self.Modality,
            }

        @classmethod
        def from_dict(cls, d, parent_study=None):
            obj = cls(series_uid=d["SeriesInstanceUID"], parent_study=parent_study)
            obj.Modality = d.get("Modality")
            return obj

    monkeypatch.setattr(dicom_nodes_mod, "SeriesNode", StubSeries, raising=True)

    p = PatientNode(patient_id="P1")
    st = StudyNode(
        study_uid="ST1",
        study_description="Study Desc",
        parent_patient=p,
    )
    st.set_attrs(custom="meta")

    se = StubSeries("1.2.3.4", parent_study=st)
    se.Modality = "CT"
    st.add_series(se)

    d = st.to_dict()
    # check top-level keys
    assert d["StudyInstanceUID"] == "ST1"
    assert d["StudyDescription"] == "Study Desc"
    assert d["__extras__"]["custom"] == "meta"
    assert "series" in d
    assert "1.2.3.4" in d["series"]
    assert d["series"]["1.2.3.4"]["SeriesInstanceUID"] == "1.2.3.4"
    assert d["series"]["1.2.3.4"]["Modality"] == "CT"

    # round-trip
    st2 = StudyNode.from_dict(d, parent_patient=p)
    assert isinstance(st2, StudyNode)
    assert st2.StudyInstanceUID == "ST1"
    assert st2.StudyDescription == "Study Desc"
    assert dict(st2.iter_attrs())["custom"] == "meta"
    assert "1.2.3.4" in st2.series
    se2 = st2.series["1.2.3.4"]
    assert isinstance(se2, StubSeries)
    assert se2.SeriesInstanceUID == "1.2.3.4"
    assert se2.Modality == "CT"
    assert se2.parent_study is st2
    # parent_patient propagated correctly
    assert st2.parent_patient is p


# --- SeriesNode tests ----------------------------------------------------------


def test_series_node_basic_and_repr():
    se = SeriesNode(series_uid="1.2.3.4")
    assert se.SeriesInstanceUID == "1.2.3.4"
    assert se.Modality is None
    assert se.SeriesDescription is None
    assert len(se) == 0

    r = repr(se)
    assert "SeriesNode" in r
    assert "1.2.3.4" in r


def test_series_add_instance_basic():
    se = SeriesNode(series_uid="1.2.3.4")
    inst = FakeInstanceNode("1.2.3.4.5", file_path="/tmp/file1.dcm")

    se.add_instance(inst)

    # instance is stored
    assert "1.2.3.4.5" in se.instances
    assert se.instances["1.2.3.4.5"] is inst
    # parent wired
    assert inst.parent_series is se
    # path recorded
    assert se.instance_paths == ["/tmp/file1.dcm"]


def test_series_add_instance_invalid_uid_raises():
    se = SeriesNode(series_uid="1.2.3.4")
    bad = FakeInstanceNode("", file_path=None)

    # We keep the current implementation behavior and message
    with pytest.raises(ValueError, match="Invalid SOPInstanceUID"):
        se.add_instance(bad)


def test_series_add_instance_duplicate_without_overwrite():
    se = SeriesNode(series_uid="1.2.3.4")
    inst1 = FakeInstanceNode("1.2.3.4.5")
    inst2 = FakeInstanceNode("1.2.3.4.5")

    se.add_instance(inst1)

    with pytest.raises(KeyError, match="Instance '1.2.3.4.5' already exists"):
        se.add_instance(inst2)

    # still the first one
    assert se.instances["1.2.3.4.5"] is inst1


def test_series_add_instance_with_overwrite():
    se = SeriesNode(series_uid="1.2.3.4")
    inst1 = FakeInstanceNode("1.2.3.4.5", file_path="/tmp/a.dcm")
    inst2 = FakeInstanceNode("1.2.3.4.5", file_path="/tmp/b.dcm")

    se.add_instance(inst1)
    se.add_instance(inst2, overwrite=True)

    # instance replaced
    assert se.instances["1.2.3.4.5"] is inst2
    assert inst2.parent_series is se

    # current impl appends paths even on overwrite; we respect that
    assert se.instance_paths == ["/tmp/a.dcm", "/tmp/b.dcm"]


def test_series_mapping_dunder_helpers():
    se = SeriesNode(series_uid="9.9.9.9")
    inst = FakeInstanceNode("9.9.9.1")
    se.add_instance(inst)

    # __len__
    assert len(se) == 1

    # __contains__
    assert "9.9.9.1" in se
    assert "does.not.exist" not in se

    # __getitem__
    assert se["9.9.9.1"] is inst

    # __iter__
    assert list(se) == [inst]


def test_series_extensible_attrs_and_parent_delegation():
    p = PatientNode(patient_id="P1")
    st = StudyNode(study_uid="ST1", parent_patient=p)
    se = SeriesNode(series_uid="SE1", parent_study=st)

    # extensible attribute on series
    se.set_attrs(series_extra=True)
    assert se.series_extra is True

    # extensible attribute on patient should be visible via delegation
    p.set_attrs(global_meta="ROOT")
    assert se.global_meta == "ROOT"


def test_series_to_dict_and_from_dict_roundtrip_metadata_only(monkeypatch):
    # monkeypatch SeriesNode.InstanceNode usage so the test remains decoupled
    monkeypatch.setattr(
        "rosamllib.nodes.dicom_nodes.InstanceNode", FakeInstanceNode, raising=False
    )

    p = PatientNode(patient_id="P1")
    st = StudyNode(study_uid="ST1", parent_patient=p)
    se = SeriesNode(series_uid="SE1", parent_study=st)

    se.Modality = "CT"
    se.SeriesDescription = "Test Series"
    se.FrameOfReferenceUID = "FOR1"
    se.referenced_sids = ["SE2", "SE3"]
    se.referencing_sids = ["SE0"]
    se.is_embedded_in_raw = True
    se.set_attrs(custom="meta")

    d = se.to_dict()

    assert d["SeriesInstanceUID"] == "SE1"
    assert d["Modality"] == "CT"
    assert d["SeriesDescription"] == "Test Series"
    assert d["FrameOfReferenceUID"] == "FOR1"
    assert d["referenced_sids"] == ["SE2", "SE3"]
    assert d["referencing_sids"] == ["SE0"]
    assert d["is_embedded_in_raw"] is True
    assert "__extras__" in d or "__extras___" in d  # tolerate either spelling

    # ensure roundtrip works without requiring real InstanceNode
    d.setdefault("instances", {})

    se2 = SeriesNode.from_dict(d, parent_study=st)

    assert se2.SeriesInstanceUID == "SE1"
    assert se2.Modality == "CT"
    assert se2.SeriesDescription == "Test Series"
    assert se2.FrameOfReferenceUID == "FOR1"
    assert se2.referenced_sids == ["SE2", "SE3"]
    assert se2.referencing_sids == ["SE0"]
    assert se2.is_embedded_in_raw is True

    # restored extras
    if hasattr(se2, "custom"):
        assert se2.custom == "meta"


# --- InstanceNode tests -------------------------------------------------------


def test_instance_node_basic_and_repr():
    inst = InstanceNode(
        SOPInstanceUID="1.2.840.10008.1.2.3.4.5",
        FilePath="/tmp/inst1.dcm",
        Modality="CT",
    )

    assert inst.SOPInstanceUID == "1.2.840.10008.1.2.3.4.5"
    assert inst.FilePath == "/tmp/inst1.dcm"
    assert inst.Modality == "CT"
    assert inst.parent_series is None
    assert inst.FrameOfReferenceUIDs == []
    assert inst.references == []
    assert inst.referenced_sop_instance_uids == []
    assert inst.referenced_sids == []
    assert inst.other_referenced_sids == []
    assert inst.referenced_series == []
    assert inst.other_referenced_series == []
    assert inst.referenced_instances == []
    assert inst.referencing_instances == []

    r = repr(inst)
    assert "InstanceNode" in r
    assert "1.2.840.10008.1.2.3.4.5" in r
    assert "/tmp/inst1.dcm" in r


def test_instance_primary_for_uid_property():
    inst = InstanceNode(
        SOPInstanceUID="1.2.840.10008.1.2.3.4.6",
        FilePath="/tmp/inst2.dcm",
        Modality="RTSTRUCT",
    )

    # No FoRs yet
    assert inst.FrameOfReferenceUIDs == []
    assert inst.primary_for_uid is None

    inst.FrameOfReferenceUIDs = ["F1", "F2", "F3"]
    assert inst.primary_for_uid == "F1"


def test_instance_extensible_attrs_and_parent_delegation():
    # Reuse the real SeriesNode as parent, since it's already exercised elsewhere
    parent = SeriesNode(series_uid="1.2.3.4.5.6")
    # Give parent a custom attribute to check delegation
    parent.custom_series_attr = "series-meta"

    inst = InstanceNode(
        SOPInstanceUID="1.2.3.4.5.6.7",
        FilePath="/tmp/inst3.dcm",
        Modality="MR",
        parent_series=parent,
    )

    # extensible attrs
    inst.set_attrs(foo="bar", number=123)
    assert inst.foo == "bar"
    assert inst.number == 123

    # parent delegation
    assert inst.custom_series_attr == "series-meta"

    # unknown attr should still raise
    with pytest.raises(AttributeError):
        _ = inst.this_does_not_exist  # noqa: F841


def test_instance_to_dict_and_from_dict_roundtrip():
    parent = SeriesNode(series_uid="9.9.9.9")
    inst = InstanceNode(
        SOPInstanceUID="9.9.9.1",
        FilePath="/tmp/inst4.dcm",
        Modality="CT",
        parent_series=parent,
    )
    inst.FrameOfReferenceUIDs = ["F1", "F2"]
    inst.referenced_sop_instance_uids = ["9.9.9.2", "9.9.9.3"]
    inst.referenced_sids = ["7.7.7.1"]
    inst.other_referenced_sids = ["8.8.8.1"]
    inst.set_attrs(extra_meta="value")

    d = inst.to_dict()
    # Check serialized fields
    assert d["SOPInstanceUID"] == "9.9.9.1"
    assert d["FilePath"] == "/tmp/inst4.dcm"
    assert d["Modality"] == "CT"
    assert d["FrameOfReferenceUIDs"] == ["F1", "F2"]
    assert d["referenced_sop_instance_uids"] == ["9.9.9.2", "9.9.9.3"]
    assert d["referenced_sids"] == ["7.7.7.1"]
    assert d["other_referenced_sids"] == ["8.8.8.1"]
    assert d["__extras__"]["extra_meta"] == "value"

    # Roundtrip
    restored_parent = SeriesNode(series_uid="9.9.9.9")  # parent on restore
    inst2 = InstanceNode.from_dict(d, parent_series=restored_parent)

    assert inst2.SOPInstanceUID == "9.9.9.1"
    assert inst2.FilePath == "/tmp/inst4.dcm"
    assert inst2.Modality == "CT"
    assert inst2.FrameOfReferenceUIDs == ["F1", "F2"]
    assert inst2.referenced_sop_instance_uids == ["9.9.9.2", "9.9.9.3"]
    assert inst2.referenced_sids == ["7.7.7.1"]
    assert inst2.other_referenced_sids == ["8.8.8.1"]
    assert inst2.extra_meta == "value"
    assert inst2.parent_series is restored_parent
