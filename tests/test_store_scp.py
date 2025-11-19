import logging

import pytest
from pydicom.dataset import Dataset

import rosamllib.networking.store_scp as scp_mod
from rosamllib.networking.store_scp import StoreSCP, _mask, _ctx_from_event


# -------------------------------------------------------------------------
# Helper / fake classes for events & assoc
# -------------------------------------------------------------------------


class FakeRequestor:
    def __init__(self, ae_title="CALLING_AE", address="1.2.3.4", port=104):
        self.ae_title = ae_title
        self.address = address
        self.port = port


class FakeAcceptor:
    def __init__(self, ae_title="CALLED_AE"):
        self.ae_title = ae_title


class FakeAssoc:
    def __init__(self, requestor=None, acceptor=None, accepted_contexts=None):
        self.requestor = requestor or FakeRequestor()
        self.acceptor = acceptor or FakeAcceptor()
        self.accepted_contexts = accepted_contexts or []


class FakePC:
    def __init__(self, name="CT Image Storage", transfer_syntax=None):
        class AS:
            def __init__(self, name):
                self.name = name

        self.abstract_syntax = AS(name)
        self.transfer_syntax = transfer_syntax or ["1.2.840.10008.1.2.1"]


class FakeEvent:
    def __init__(self, assoc=None, dataset=None, requestor=None, acceptor=None):
        # For _ctx_from_event path
        self.assoc = assoc or FakeAssoc(
            requestor=requestor or FakeRequestor(),
            acceptor=acceptor or FakeAcceptor(),
        )
        self.dataset = dataset

        # For assoc event handlers
        self.requestor = self.assoc.requestor
        self.acceptor = self.assoc.acceptor

        # For reject handler
        self.result = None
        self.source = None
        self.reason = None


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def logger():
    lg = logging.getLogger("store_scp_test")
    lg.handlers.clear()
    lg.setLevel(logging.DEBUG)
    return lg


@pytest.fixture
def scp(logger):
    return StoreSCP(aet="TEST_SCP", ip="127.0.0.1", port=11112, logger=logger)


# -------------------------------------------------------------------------
# _mask and _ctx_from_event
# -------------------------------------------------------------------------


def test_mask_short_and_none():
    assert _mask(None) is None
    assert _mask("123") == "123"  # shorter than keep
    assert _mask("abcdef") == "abcdef"  # equal to keep
    assert _mask("abcdefghijkl") == "abcdef..."  # masked tail


def test_ctx_from_event_masked_and_unmasked():
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = "1.2.3.4.5.6"
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8"
    ds.SeriesInstanceUID = "2.3.4"
    ds.Modality = "CT"

    event = FakeEvent(dataset=ds)

    # Masked (default)
    ctx = _ctx_from_event(event, op="C-STORE", mask_phi=True)
    assert ctx["op"] == "C-STORE"
    assert ctx["calling_ae"] == "CALLING_AE"
    assert ctx["called_ae"] == "CALLED_AE"
    assert ctx["remote_addr"] == "1.2.3.4:104"
    assert ctx["modality"] == "CT"
    assert ctx["sop_class"] == "1.2.840.10008.5.1.4.1.1.2"
    assert ctx["sop_uid"].endswith("...")
    assert ctx["study_uid"].endswith("...")
    assert ctx["series_uid"].endswith("...") is False  # short, not masked

    # Unmasked
    ctx2 = _ctx_from_event(event, op="C-STORE", mask_phi=False)
    assert ctx2["sop_uid"] == "1.2.3.4.5.6"
    assert ctx2["study_uid"] == "1.2.3.4.5.6.7.8"


# -------------------------------------------------------------------------
# __init__ and validate_entry
# -------------------------------------------------------------------------


def test_init_invalid_inputs(monkeypatch, logger):
    def fake_validate(value, kind):
        if kind == "IP":
            return False
        return True

    monkeypatch.setattr(scp_mod, "validate_entry", fake_validate)

    with pytest.raises(ValueError):
        StoreSCP(aet="BAD", ip="0.0.0.0", port=11112, logger=logger)


def test_init_logs_basic_info(caplog, logger):
    caplog.set_level(logging.INFO)
    scp = StoreSCP(aet="TEST_SCP", ip="127.0.0.1", port=11112, logger=logger)
    assert scp.scpAET == "TEST_SCP"
    assert scp.scpIP == "127.0.0.1"
    assert scp.scpPort == 11112
    assert any("StoreSCP initialized with AE Title" in r.message for r in caplog.records)


# -------------------------------------------------------------------------
# is_running, start, stop
# -------------------------------------------------------------------------


class FakeServer:
    def __init__(self):
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


def test_is_running_flag(monkeypatch, scp):
    # Patch AE.start_server to avoid real network
    fake_server = FakeServer()

    def fake_start_server(addr, block=False, evt_handlers=None):
        return fake_server

    monkeypatch.setattr(scp.ae, "start_server", fake_start_server)

    assert scp.is_running() is False
    scp.start(block=False)
    assert scp.is_running() is True
    assert scp._server is fake_server

    # Stop should reset
    monkeypatch.setattr(scp.ae, "shutdown", lambda: None)
    scp.stop()
    assert scp.is_running() is False
    assert fake_server.shutdown_called is True


def test_start_when_already_running_logs_warning(caplog, scp):
    caplog.set_level(logging.WARNING)
    scp._server_running = True
    scp.start(block=False)
    assert any("SCP already running" in r.message for r in caplog.records)


def test_start_failure_sets_not_running(monkeypatch, caplog, scp):
    caplog.set_level(logging.ERROR)

    def fake_start_server(addr, block=False, evt_handlers=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(scp.ae, "start_server", fake_start_server)

    scp.start(block=False)
    assert scp._server_running is False
    assert scp._server is None
    assert any("Could not start SCP" in r.message for r in caplog.records)


def test_stop_when_not_running_logs_info(caplog, scp):
    caplog.set_level(logging.INFO)
    scp._server_running = False
    scp.stop()
    assert any("server was not running" in r.message for r in caplog.records)


def test_stop_handles_shutdown_exception(monkeypatch, caplog, scp):
    caplog.set_level(logging.ERROR)

    scp._server_running = True

    class BadServer:
        def shutdown(self):
            raise RuntimeError("bad server")

    scp._server = BadServer()

    def bad_ae_shutdown():
        raise RuntimeError("bad AE")

    monkeypatch.setattr(scp.ae, "shutdown", bad_ae_shutdown)

    scp.stop()
    # Should still mark as not running
    assert scp._server_running is False
    assert any("Exception during shutdown" in r.message for r in caplog.records)


# -------------------------------------------------------------------------
# set_handlers and association event handlers
# -------------------------------------------------------------------------


def test_set_handlers_content(scp):
    scp.set_handlers()
    assert hasattr(scp, "handlers")
    # EVT_CONN_OPEN, EVT_CONN_CLOSE, EVT_C_STORE, EVT_REQUESTED,
    # EVT_ACCEPTED, EVT_REJECTED, EVT_ABORTED
    assert len(scp.handlers) == 7
    # The handlers are tuples (event, callback)
    evts = {h[0] for h in scp.handlers}
    # Just check that we have some distinct events
    assert len(evts) >= 4


def test_on_assoc_requested_logs(caplog, scp):
    caplog.set_level(logging.INFO)
    # Ensure logger is NOT in DEBUG mode so we skip debug branch
    scp.set_log_level(logging.INFO)

    ev = FakeEvent()
    scp._on_assoc_requested(ev)

    assert any("Association requested." in r.message for r in caplog.records)


def test_on_assoc_accepted_logs(caplog, scp):
    caplog.set_level(logging.INFO)
    # Ensure logger is NOT in DEBUG mode so we skip debug branch
    scp.set_log_level(logging.INFO)

    ev = FakeEvent()
    scp._on_assoc_accepted(ev)

    assert any("Association accepted." in r.message for r in caplog.records)


def test_on_assoc_rejected_logs_error(caplog, scp):
    caplog.set_level(logging.ERROR)
    ev = FakeEvent()
    ev.result = 1
    ev.source = 2
    ev.reason = 3
    scp._on_assoc_rejected(ev)
    assert any("Association rejected" in r.message for r in caplog.records)


def test_on_abort_logs_error(caplog, scp):
    caplog.set_level(logging.ERROR)
    ev = FakeEvent()
    scp._on_abort(ev)
    assert any("Association aborted." in r.message for r in caplog.records)


def test_on_c_echo_returns_success(caplog, scp):
    ev = FakeEvent()
    status = scp._on_c_echo(ev)
    assert status == 0x0000


# -------------------------------------------------------------------------
# handle_open, handle_close, handle_store + custom functions
# -------------------------------------------------------------------------


def test_handle_open_runs_custom_functions_and_logs(caplog, scp):
    caplog.set_level(logging.INFO)
    ev = FakeEvent()
    called = []

    def custom(event):
        called.append("open")

    scp.add_custom_function_open(custom)
    scp.handle_open(ev)
    assert "open" in called
    assert any("Association opened." in r.message for r in caplog.records)


def test_handle_close_runs_custom_functions_and_logs(caplog, scp):
    caplog.set_level(logging.INFO)
    ev = FakeEvent()
    called = []

    def custom(event):
        called.append("close")

    scp.add_custom_function_close(custom)
    scp.handle_close(ev)
    assert "close" in called
    assert any("Association closed." in r.message for r in caplog.records)


def test_handle_store_success_with_custom_store(caplog, scp):
    caplog.set_level(logging.INFO)
    ds = Dataset()
    ds.SOPClassUID = "1.2.3"
    ds.SOPInstanceUID = "1.2.3.4"
    ev = FakeEvent(dataset=ds)

    called = []

    def custom_store(e):
        called.append("store")

    scp.add_custom_function_store(custom_store)
    status_ds = scp.handle_store(ev)

    assert "store" in called
    assert status_ds.Status == 0x0000
    assert any("C-STORE OK in" in r.message for r in caplog.records)


def test_handle_store_failure_path(caplog, scp):
    """
    If a custom store function raises, we should log the error but still
    return a success C-STORE status (0x0000), since the network-level
    operation succeeded.
    """
    caplog.set_level(logging.ERROR)

    ds = Dataset()
    ds.SOPClassUID = "1.2.3"
    ds.SOPInstanceUID = "1.2.3.4"
    ev = FakeEvent(dataset=ds)

    def bad_store(event):
        raise RuntimeError("boom")

    scp.add_custom_function_store(bad_store)

    status_ds = scp.handle_store(ev)

    # Network-level C-STORE still succeeds
    assert status_ds.Status == 0x0000

    # But we log the failure of the custom function
    error_msgs = [r.message for r in caplog.records if "bad_store failed" in r.message]
    assert error_msgs


def test_handle_store_internal_failure_returns_c000(monkeypatch, scp):
    """
    If something fundamental inside handle_store (e.g. building log context)
    raises, we should catch it and return a failure status 0xC000.
    """
    ds = Dataset()
    ds.SOPClassUID = "1.2.3"
    ds.SOPInstanceUID = "1.2.3.4"
    ev = FakeEvent(dataset=ds)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    # scp_mod should be the imported module under test, e.g.:
    # import rosamllib.networking.store_scp as scp_mod
    monkeypatch.setattr(scp_mod, "_ctx_from_event", boom)

    status_ds = scp.handle_store(ev)
    assert status_ds.Status == 0xC000


# -------------------------------------------------------------------------
# Custom functions add/remove/clear
# -------------------------------------------------------------------------


def test_custom_functions_add_and_remove_store(caplog, scp):
    def fn(event):
        pass

    scp.add_custom_function_store(fn)
    assert fn in scp.custom_functions_store

    scp.remove_custom_function_store(fn)
    assert fn not in scp.custom_functions_store

    # Removing non-existent logs error
    caplog.set_level(logging.ERROR)
    scp.remove_custom_function_store(fn)
    assert any("not found" in r.message for r in caplog.records)


def test_custom_functions_add_and_remove_open_close(caplog, scp):
    def f_open(event):
        pass

    def f_close(event):
        pass

    scp.add_custom_function_open(f_open)
    scp.add_custom_function_close(f_close)
    assert f_open in scp.custom_functions_open
    assert f_close in scp.custom_functions_close

    scp.remove_custom_function_open(f_open)
    scp.remove_custom_function_close(f_close)
    assert f_open not in scp.custom_functions_open
    assert f_close not in scp.custom_functions_close

    caplog.set_level(logging.ERROR)
    scp.remove_custom_function_open(f_open)
    scp.remove_custom_function_close(f_close)
    assert any("Custom open function" in r.message for r in caplog.records)
    assert any("Custom close function" in r.message for r in caplog.records)


def test_custom_functions_clear(scp):
    def f1(event):
        pass

    def f2(event):
        pass

    scp.add_custom_function_store(f1)
    scp.add_custom_function_open(f1)
    scp.add_custom_function_close(f2)

    scp.clear_custom_functions_store()
    scp.clear_custom_functions_open()
    scp.clear_custom_functions_close()

    assert scp.custom_functions_store == []
    assert scp.custom_functions_open == []
    assert scp.custom_functions_close == []


# -------------------------------------------------------------------------
# SOP class registration
# -------------------------------------------------------------------------


def test_register_sop_class_registers_and_adds_supported_context(monkeypatch, scp, caplog):
    caplog.set_level(logging.INFO)

    added_contexts = []

    def fake_add_supported_context(obj):
        added_contexts.append(obj)

    monkeypatch.setattr(scp.ae, "add_supported_context", fake_add_supported_context)

    def fake_register_uid(uid, keyword, service_class):
        # mimic pynetdicom's behavior by attaching attribute to sop_class
        setattr(scp_mod.sop_class, keyword, type("DummySOP", (), {})())

    monkeypatch.setattr(scp_mod, "register_uid", fake_register_uid)

    keyword = "VarianRTPlanStorage"
    uid = "1.2.246.352.70.1.70"

    # ensure not already present
    if hasattr(scp_mod.sop_class, keyword):
        delattr(scp_mod.sop_class, keyword)

    scp.register_sop_class(uid, keyword)

    assert hasattr(scp_mod.sop_class, keyword)
    assert added_contexts  # something was added
    assert any("Registered custom SOP Class UID" in r.message for r in caplog.records)


def test_register_sop_class_noop_if_already_registered(monkeypatch, scp, caplog):
    caplog.set_level(logging.DEBUG)
    keyword = "ExistingSOP"
    setattr(scp_mod.sop_class, keyword, type("Dummy", (), {})())

    called = {"register": False}

    def fake_register_uid(uid, keyword, service_class):
        called["register"] = True

    monkeypatch.setattr(scp_mod, "register_uid", fake_register_uid)

    scp.register_sop_class("1.2.3", keyword)
    assert called["register"] is False
    assert any("already registered" in r.message for r in caplog.records)


def test_add_registered_presentation_context_success(monkeypatch, scp, caplog):
    caplog.set_level(logging.INFO)
    keyword = "MySOP"
    sop_inst = type("DummySOP", (), {})()
    setattr(scp_mod.sop_class, keyword, sop_inst)

    added = []

    def fake_add_supported_context(obj):
        added.append(obj)

    monkeypatch.setattr(scp.ae, "add_supported_context", fake_add_supported_context)

    scp.add_registered_presentation_context(keyword)
    assert added[0] is sop_inst
    assert any("Added presentation context" in r.message for r in caplog.records)


def test_add_registered_presentation_context_missing_raises(scp):
    keyword = "NotRegistered"
    if hasattr(scp_mod.sop_class, keyword):
        delattr(scp_mod.sop_class, keyword)

    with pytest.raises(ValueError):
        scp.add_registered_presentation_context(keyword)


# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------


def test_configure_logging_console_and_file(tmp_path, scp):
    scp.clear_log_handlers()

    # Console only
    handlers = scp.configure_logging(
        log_to_console=True,
        log_to_file=False,
        log_level=logging.DEBUG,
        json_logs=False,
    )
    assert "console" in handlers
    assert any(isinstance(h, logging.StreamHandler) for h in scp.logger.handlers)

    # Add file handler (non-rotating to keep it simple)
    log_path = tmp_path / "store_scp.log"
    handlers2 = scp.configure_logging(
        log_to_console=False,
        log_to_file=True,
        log_file_path=str(log_path),
        rotate=False,
    )
    assert "file" in handlers2
    assert any(isinstance(h, logging.FileHandler) for h in scp.logger.handlers)

    # Turn off file logging
    scp.configure_logging(log_to_console=False, log_to_file=False, log_file_path=str(log_path))
    assert not any(isinstance(h, logging.FileHandler) for h in scp.logger.handlers)


def test_enable_wire_debug_calls_attach(monkeypatch, scp):
    called = []

    def fake_attach(enable, level):
        called.append((enable, level))

    monkeypatch.setattr(scp_mod, "attach_pynetdicom_to_logger", fake_attach)
    scp.enable_wire_debug(enable=True, level=logging.DEBUG)

    assert called == [(True, logging.DEBUG)]


def test_set_log_level_updates_handlers(scp):
    # attach a handler first
    h = logging.StreamHandler()
    scp.logger.addHandler(h)

    scp.set_log_level(logging.WARNING)
    assert scp.logger.level == logging.WARNING
    for hh in scp.logger.handlers:
        assert hh.level == logging.WARNING


def test_log_accepted_contexts_and_debug_variant(caplog, scp):
    caplog.set_level(logging.DEBUG)
    # Ensure logger is in DEBUG mode and propagates to root (for caplog)
    scp.set_log_level(logging.DEBUG)
    scp.logger.propagate = True

    pcs = [FakePC(name="CT Image Storage"), FakePC(name="MR Image Storage")]
    assoc = FakeAssoc(accepted_contexts=pcs)

    scp.log_accepted_contexts(assoc)
    scp._log_accepted_contexts_debug(assoc)

    logged = "\n".join(
        r.message for r in caplog.records if "Accepted presentation contexts" in r.message
    )
    assert "CT Image Storage" in logged
    assert "MR Image Storage" in logged


def test_close_log_handlers(scp):
    # Add a couple of handlers and ensure they are closed/removed
    h1 = logging.StreamHandler()
    h2 = logging.StreamHandler()
    scp.logger.addHandler(h1)
    scp.logger.addHandler(h2)

    scp.close_log_handlers()
    assert scp.logger.handlers == []
