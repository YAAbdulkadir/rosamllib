import logging

import pytest
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.sequence import Sequence

import rosamllib.networking.qr_scu as qr_mod
from rosamllib.networking import QueryRetrieveSCU, FindResult, MoveResult


# -------------------------------------------------------------------------
# Helpers / fakes
# -------------------------------------------------------------------------


class FakeStatus:
    def __init__(self, status, **extra):
        self.Status = status
        for k, v in extra.items():
            setattr(self, k, v)


class FakePresentationContext:
    def __init__(self, abstract_syntax, transfer_syntax):
        self.abstract_syntax = abstract_syntax
        self.transfer_syntax = transfer_syntax


class FakeAssoc:
    """
    Minimal stand-in for a pynetdicom Association object.
    """

    def __init__(
        self,
        established=True,
        echo_status=None,
        find_responses=None,
        move_responses=None,
        store_status=None,
        accepted_contexts=None,
    ):
        self.is_established = established
        self._echo_status = echo_status
        self._find_responses = find_responses or []
        self._move_responses = move_responses or []
        self._store_status = store_status
        self.accepted_contexts = accepted_contexts or []
        self.released = False

    def release(self):
        self.released = True

    # Methods used by QueryRetrieveSCU
    def send_c_echo(self):
        return self._echo_status

    def send_c_find(self, query, context):
        # Return an iterator of (status, identifier) tuples
        return iter(self._find_responses)

    def send_c_move(self, query, dest_ae, context):
        return iter(self._move_responses)

    def send_c_store(self, ds):
        return self._store_status


@pytest.fixture
def logger():
    """
    A dedicated logger per test so we don't pollute root.
    """
    logger = logging.getLogger("qr_scu_test")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def scu(logger):
    """
    Create a QueryRetrieveSCU with a test AE title and our test logger.
    """
    return QueryRetrieveSCU(ae_title="TEST_SCU", logger=logger)


# -------------------------------------------------------------------------
# Basic construction / remote AE management
# -------------------------------------------------------------------------


def test_init_invalid_ae_title(monkeypatch):
    """
    If validate_entry returns False for the AE title, __init__ should raise.
    """
    # Force validate_entry(ae_title, "AET") to be False
    monkeypatch.setattr(
        qr_mod,
        "validate_entry",
        lambda value, kind: False if kind == "AET" else True,
    )

    with pytest.raises(ValueError):
        QueryRetrieveSCU(ae_title="INVALID")


def test_add_remote_ae_success(scu, caplog):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)

    assert "remote1" in scu.remote_entities
    rm = scu.remote_entities["remote1"]
    assert rm["ae_title"] == "REMOTE_AE"
    assert rm["host"] == "127.0.0.1"
    assert rm["port"] == 11112

    # We should have logged something
    assert any("Added remote AE 'remote1'" in r.message for r in caplog.records)


def test_add_remote_ae_invalid(monkeypatch, scu):
    """
    If validate_entry fails on any of the fields, we should raise ValueError.
    """

    def fake_validate(value, kind):
        # make "BAD" host invalid
        if kind == "IP" and value == "BAD":
            return False
        return True

    monkeypatch.setattr(qr_mod, "validate_entry", fake_validate)

    with pytest.raises(ValueError):
        scu.add_remote_ae("remote1", "REMOTE_AE", "BAD", 11112)


def test_add_extended_negotiation_unknown_ae_raises(scu):
    with pytest.raises(ValueError):
        scu.add_extended_negotiation("unknown", ["dummy"])


def test_add_extended_negotiation_success(scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    items = ["dummy_item"]
    scu.add_extended_negotiation("remote1", items)
    assert scu.remote_entities["remote1"]["ext_neg"] == items


# -------------------------------------------------------------------------
# association_context and _establish_association
# -------------------------------------------------------------------------


def test_association_context_yields_and_releases(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    fake_assoc = FakeAssoc(established=True)

    # Patch _establish_association to return our fake association
    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: fake_assoc,
    )

    with scu.association_context("remote1") as assoc:
        assert assoc is fake_assoc
        assert not assoc.released

    assert fake_assoc.released is True


def test_association_context_yields_none_on_failure(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: None,
    )

    with scu.association_context("remote1") as assoc:
        assert assoc is None


# -------------------------------------------------------------------------
# C-ECHO
# -------------------------------------------------------------------------


def test_c_echo_success(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    status = FakeStatus(0x0000)
    fake_assoc = FakeAssoc(established=True, echo_status=status)

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: fake_assoc,
    )

    assert scu.c_echo("remote1") is True


def test_c_echo_failure_to_associate(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: None,
    )

    assert scu.c_echo("remote1") is False


def test_c_echo_non_success_status(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    status = FakeStatus(0xA700)
    fake_assoc = FakeAssoc(established=True, echo_status=status)

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: fake_assoc,
    )

    assert scu.c_echo("remote1") is False


# -------------------------------------------------------------------------
# C-FIND
# -------------------------------------------------------------------------


def test_c_find_association_failure(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    query = Dataset()
    query.QueryRetrieveLevel = "STUDY"

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: None,
    )

    result = scu.c_find("remote1", query)
    assert isinstance(result, FindResult)
    assert result.matches == []
    assert "Failed to associate" in result.error_comment


def test_c_find_collects_pending_matches(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)

    query = Dataset()
    query.QueryRetrieveLevel = "STUDY"

    ds1 = Dataset()
    ds1.PatientID = "P1"
    ds2 = Dataset()
    ds2.PatientID = "P2"

    responses = [
        # Pending with identifier
        (FakeStatus(0xFF00), ds1),
        (FakeStatus(0xFF01), ds2),
        # Final success status, no identifier
        (FakeStatus(0x0000, ErrorComment="OK"), None),
    ]
    fake_assoc = FakeAssoc(established=True, find_responses=responses)

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: fake_assoc,
    )

    result = scu.c_find("remote1", query)
    assert result.status == 0x0000
    assert result.error_comment == "OK"
    assert len(result.matches) == 2
    assert result.matches[0].PatientID == "P1"
    assert result.matches[1].PatientID == "P2"


# -------------------------------------------------------------------------
# C-MOVE
# -------------------------------------------------------------------------


def test_c_move_association_failure(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    query = Dataset()

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: None,
    )

    result = scu.c_move("remote1", query, "DEST_AE")
    assert isinstance(result, MoveResult)
    assert "Failed to associate" in result.error_comment


def test_c_move_collects_counters(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    query = Dataset()

    status1 = FakeStatus(
        0xFF00,
        NumberOfRemainingSuboperations=2,
        NumberOfCompletedSuboperations=1,
        NumberOfFailedSuboperations=0,
        NumberOfWarningSuboperations=0,
    )
    status2 = FakeStatus(
        0x0000,
        NumberOfRemainingSuboperations=0,
        NumberOfCompletedSuboperations=3,
        NumberOfFailedSuboperations=0,
        NumberOfWarningSuboperations=1,
        ErrorComment="Done with warnings",
    )
    responses = [
        (status1, None),
        (status2, None),
    ]
    fake_assoc = FakeAssoc(established=True, move_responses=responses)

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: fake_assoc,
    )

    result = scu.c_move("remote1", query, "DEST_AE")
    assert result.status == 0x0000
    assert result.remaining == 0
    assert result.completed == 3
    assert result.failed == 0
    assert result.warning == 1
    assert result.error_comment == "Done with warnings"


# -------------------------------------------------------------------------
# C-STORE
# -------------------------------------------------------------------------


def test_c_store_missing_sop_class_uid_raises(scu):
    ds = Dataset()
    ds.SOPInstanceUID = "1.2.3"
    with pytest.raises(ValueError):
        scu.c_store("remote1", ds)


def test_c_store_missing_sop_instance_uid_raises(scu):
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    with pytest.raises(ValueError):
        scu.c_store("remote1", ds)


def test_c_store_association_failure(monkeypatch, scu):
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = "1.2.3"

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: None,
    )

    status = scu.c_store("remote1", ds)
    assert status is None


def test_c_store_success(monkeypatch, scu):
    """
    Happy path: assoc established, peer accepts storage PC, and returns success status.
    """
    scu.add_remote_ae("remote1", "REMOTE_AE", "127.0.0.1", 11112)
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage (for example)
    ds.SOPInstanceUID = "1.2.3"

    accepted_pc = FakePresentationContext(ds.SOPClassUID, transfer_syntax=["1.2.840"])
    fake_assoc = FakeAssoc(
        established=True,
        store_status=FakeStatus(0x0000),
        accepted_contexts=[accepted_pc],
    )

    monkeypatch.setattr(
        QueryRetrieveSCU,
        "_establish_association",
        lambda self, name, retry_count=3, delay=5: fake_assoc,
    )

    status = scu.c_store("remote1", ds)
    assert isinstance(status, FakeStatus)
    assert status.Status == 0x0000


# -------------------------------------------------------------------------
# convert_results_to_df and _get_metadata
# -------------------------------------------------------------------------


def test_convert_results_to_df_empty_results_uses_query_tags_as_columns():
    query = Dataset()
    query.PatientID = ""
    query.StudyInstanceUID = ""

    df = QueryRetrieveSCU.convert_results_to_df([], query)
    # Columns should be keywords where possible
    assert list(df.columns) == ["PatientID", "StudyInstanceUID"]
    assert df.empty


def test_convert_results_to_df_with_matches_and_vr_casting(monkeypatch):
    # Patch VR_TO_DTYPE so we can have deterministic casting
    monkeypatch.setattr(
        qr_mod,
        "VR_TO_DTYPE",
        {
            "DA": "date",  # dates
            "CS": object,  # codes/strings
        },
    )

    query = Dataset()
    query.PatientID = ""
    query.StudyDate = ""  # DA VR

    # Match
    ds = Dataset()
    ds.PatientID = "P123"
    ds.StudyDate = "20250101"

    # results wrapped in FindResult to exercise that path too
    res = FindResult(status=0x0000, matches=[ds])

    df = QueryRetrieveSCU.convert_results_to_df(res, query)

    # Columns by keyword
    assert set(df.columns) >= {"PatientID", "StudyDate"}
    assert df.loc[0, "PatientID"] == "P123"
    # StudyDate should be converted to a datetime.date because of "DA" -> "date"
    assert hasattr(df.loc[0, "StudyDate"], "year")
    assert df.loc[0, "StudyDate"].year == 2025


def test_get_metadata_includes_query_and_extra_tags(monkeypatch):
    """
    _get_metadata should include union of tags from query and result.
    We also exercise Sequence / SQ handling path in a simple way.
    """
    # Keep parse_vr_value simple: just echo the value
    monkeypatch.setattr(qr_mod, "parse_vr_value", lambda vr, value: value)

    query = Dataset()
    query.PatientID = "ignored"

    result = Dataset()
    result.PatientID = "P1"
    result.StudyInstanceUID = "1.2.3"

    # Add a simple sequence to hit the SQ path
    seq_ds = Dataset()
    seq_ds.CodeValue = "A"
    result.add_new(Tag(0x0040A730), "SQ", Sequence([seq_ds]))  # ContentSequence

    md = QueryRetrieveSCU._get_metadata(result, query)

    # Keys by keyword where possible
    assert md["PatientID"] == "P1"
    assert md["StudyInstanceUID"] == "1.2.3"

    # The SQ field should exist under its keyword or tag int; we only
    # check that it is present and not raising.
    assert any("Content" in str(k) or k == int(Tag(0x0040A730)) for k in md.keys())


# -------------------------------------------------------------------------
# Logging configuration helpers
# -------------------------------------------------------------------------


def test_configure_logging_adds_console_handler(scu):
    scu.clear_log_handlers()
    handlers = scu.configure_logging(
        log_to_console=True,
        log_to_file=False,
        json_logs=False,
        log_level=logging.DEBUG,
    )
    assert "console" in handlers
    assert any(isinstance(h, logging.StreamHandler) for h in scu.logger.handlers)


def test_configure_logging_adds_and_removes_file_handler(tmp_path, scu):
    path = tmp_path / "qr_scu.log"
    scu.clear_log_handlers()

    # Add file handler
    handlers = scu.configure_logging(
        log_to_console=False,
        log_to_file=True,
        log_file_path=str(path),
        rotate=False,
    )
    assert "file" in handlers
    assert any(isinstance(h, logging.FileHandler) for h in scu.logger.handlers)

    # Now disable file logging and ensure handler is removed
    scu.configure_logging(
        log_to_console=False,
        log_to_file=False,
        log_file_path=str(path),
    )
    assert not any(isinstance(h, logging.FileHandler) for h in scu.logger.handlers)
