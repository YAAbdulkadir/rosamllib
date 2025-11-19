import math
from datetime import date, time, datetime

from pydicom.multival import MultiValue
from pydicom.valuerep import DA, TM

from rosamllib.utils import parse_vr_value


class TestParseVRDateTime:
    def test_da_string(self):
        val = "20250131"
        parsed = parse_vr_value("DA", val)
        assert isinstance(parsed, date)
        assert parsed == date(2025, 1, 31)

    def test_da_valuerep(self):
        val = DA("20250131")
        parsed = parse_vr_value("DA", val)
        assert isinstance(parsed, date)
        assert parsed == date(2025, 1, 31)

    def test_tm_hour_only(self):
        # "13" -> 13:00:00
        val = "13"
        parsed = parse_vr_value("TM", val)
        assert isinstance(parsed, time)
        assert parsed.hour == 13
        assert parsed.minute == 0
        assert parsed.second == 0

    def test_tm_hour_minute(self):
        # "1345" -> 13:45:00
        val = "1345"
        parsed = parse_vr_value("TM", val)
        assert isinstance(parsed, time)
        assert parsed.hour == 13
        assert parsed.minute == 45
        assert parsed.second == 0

    def test_tm_full_seconds(self):
        # "134530" -> 13:45:30
        val = "134530"
        parsed = parse_vr_value("TM", val)
        assert isinstance(parsed, time)
        assert parsed.hour == 13
        assert parsed.minute == 45
        assert parsed.second == 30

    def test_tm_fractional(self):
        # "134530.5" -> 13:45:30.500000
        val = "134530.5"
        parsed = parse_vr_value("TM", val)
        assert isinstance(parsed, time)
        assert parsed.hour == 13
        assert parsed.minute == 45
        assert parsed.second == 30
        # microseconds ~ 500000
        assert parsed.microsecond == 500000

    def test_tm_valuerep(self):
        val = TM("134530.25")
        parsed = parse_vr_value("TM", val)
        assert isinstance(parsed, time)
        assert parsed.hour == 13
        assert parsed.minute == 45
        assert parsed.second == 30
        assert parsed.microsecond == 250000

    def test_dt_basic(self):
        val = "20251115235959"
        parsed = parse_vr_value("DT", val)
        assert isinstance(parsed, datetime)
        assert parsed.year == 2025
        assert parsed.month == 11
        assert parsed.day == 15
        assert parsed.hour == 23
        assert parsed.minute == 59
        assert parsed.second == 59

    def test_dt_fractional(self):
        val = "20251115235959.123456"
        parsed = parse_vr_value("DT", val)
        assert isinstance(parsed, datetime)
        assert parsed.microsecond == 123456

    def test_dt_with_timezone(self):
        # we mostly just assert that it parses; tz handling is up to pydicom
        val = "20251115235959.123456-0800"
        parsed = parse_vr_value("DT", val)
        assert isinstance(parsed, datetime)

    def test_invalid_da_returns_none(self):
        parsed = parse_vr_value("DA", "bad_date")
        assert parsed is None

    def test_invalid_tm_returns_none(self):
        parsed = parse_vr_value("TM", "999999")  # invalid hour/minute/second
        assert parsed is None


class TestParseVRNumericText:
    def test_int_single(self):
        parsed = parse_vr_value("IS", "42")
        assert isinstance(parsed, int)
        assert parsed == 42

    def test_int_multi(self):
        val = MultiValue(str, ["1", "2", "3"])
        parsed = parse_vr_value("IS", val)
        assert isinstance(parsed, list)
        assert parsed == [1, 2, 3]

    def test_float_single(self):
        parsed = parse_vr_value("DS", "3.14")
        assert isinstance(parsed, float)
        assert math.isclose(parsed, 3.14, rel_tol=1e-6)

    def test_float_multi(self):
        val = MultiValue(str, ["1.5", "2.5"])
        parsed = parse_vr_value("DS", val)
        assert isinstance(parsed, list)
        assert all(isinstance(x, float) for x in parsed)
        assert parsed == [1.5, 2.5]

    def test_lo_single(self):
        parsed = parse_vr_value("LO", "Hello")
        assert isinstance(parsed, str)
        assert parsed == "Hello"

    def test_lo_multi(self):
        val = MultiValue(str, ["A", "B"])
        parsed = parse_vr_value("LO", val)
        assert isinstance(parsed, list)
        assert parsed == ["A", "B"]

    def test_empty_value_passthrough(self):
        assert parse_vr_value("LO", "") == ""
        assert parse_vr_value("DA", None) is None
