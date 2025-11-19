import datetime as dt
import pandas as pd
import numpy as np
import pytest

from rosamllib.utils import query_df


@pytest.fixture
def sample_df():
    """Small DataFrame exercising different dtypes and nested structures."""
    now = dt.datetime(2025, 1, 31, 12, 0, 0)

    return pd.DataFrame(
        {
            "PatientID": ["123", "456", "789", "101", "121"],
            "Modality": ["CT", "MR", "CT", "PT", "ct"],
            "Age": [30, 45, 29, 60, 35],
            # datetime64 column
            "AcqDateTime": pd.to_datetime(
                [
                    now - dt.timedelta(days=1),
                    now - dt.timedelta(days=5),
                    now - dt.timedelta(days=10),
                    now - dt.timedelta(days=20),
                    now - dt.timedelta(days=0),
                ]
            ),
            # time-of-day as datetime.time (object column)
            "SeriesTime": [
                dt.time(9, 0),  # inside 08–17
                dt.time(23, 0),  # inside 22–06 wrap
                dt.time(3, 0),  # inside 22–06 wrap
                dt.time(12, 0),  # inside 08–17
                dt.time(7, 30),  # outside both
            ],
            # numeric for approx tests
            "Score": [1.0, 1.0000004, 0.9995, 2.0, np.nan],
            # container + string column for "contains"
            "Tags": [
                ["HN", "CT"],
                ["HN", "MR"],
                ["LUNG"],
                "free text note",
                [],
            ],
            # nested structure for dot-path extraction
            "meta": [
                {"series": [{"uid": "1.2.3.1"}]},
                {"series": [{"uid": "1.2.3.2"}]},
                {"series": [{"uid": "9.9.9.9"}]},
                {"series": []},
                None,
            ],
        }
    )


# ---------------------------
# Basic equality / wildcard / list
# ---------------------------


class TestQueryDfBasic:
    def test_exact_match_scalar(self, sample_df):
        result = query_df(sample_df, PatientID="456")
        assert list(result["PatientID"]) == ["456"]

    def test_wildcard_match_scalar(self, sample_df):
        # IDs starting with "1" -> 123, 101, 121
        result = query_df(sample_df, PatientID="1*")
        assert set(result["PatientID"]) == {"123", "101", "121"}

    def test_list_of_conditions_or_semantics(self, sample_df):
        # "1*" OR "456"
        result = query_df(sample_df, PatientID=["1*", "456"])
        assert set(result["PatientID"]) == {"123", "101", "121", "456"}

    def test_container_membership_fallback(self, sample_df):
        # Tags column has lists and a plain string; membership via scalar
        result = query_df(sample_df, Tags="HN")
        # rows 0 and 1 have ["HN", ...]
        assert set(result.index) == {0, 1}


# ---------------------------
# Operator dicts (comparators, in/nin, eq/neq via op dict)
# ---------------------------


class TestQueryDfComparators:
    def test_range_gte_lt(self, sample_df):
        # 30 <= Age < 50 => 30, 45, 35
        result = query_df(sample_df, Age={"gte": 30, "lt": 50})
        assert set(result["Age"]) == {30, 35, 45}

    def test_in_and_nin(self, sample_df):
        result_in = query_df(sample_df, Modality={"in": ["CT", "PT"]})
        # membership for 'in' is case-sensitive and exact
        assert set(result_in["Modality"]) == {"CT", "PT"}

        result_nin = query_df(sample_df, Modality={"nin": ["CT", "PT"]})
        # everything that is not exactly "CT" or "PT"
        assert set(result_nin["Modality"]) == {"MR", "ct"}

        result_nin = query_df(sample_df, Modality={"nin": ["CT", "PT"]})
        # everything that is not exactly "CT" or "PT"
        assert set(result_nin["Modality"]) == {"MR", "ct"}

    def test_eq_neq_operator(self, sample_df):
        # Using explicit eq/neq operators
        result_eq = query_df(sample_df, Age={"eq": 30})
        assert set(result_eq["Age"]) == {30}

        result_neq = query_df(sample_df, Age={"neq": 30})
        assert 30 not in set(result_neq["Age"])


# ---------------------------
# Regex / NotRegex / case_insensitive
# ---------------------------


class TestQueryDfRegex:
    def test_regex_basic(self, sample_df):
        # IDs starting with 1
        result = query_df(sample_df, PatientID={"RegEx": r"^1"})
        assert set(result["PatientID"]) == {"123", "101", "121"}

    def test_notregex_basic(self, sample_df):
        result = query_df(sample_df, PatientID={"NotRegEx": r"^1"})
        assert set(result["PatientID"]) == {"456", "789"}

    def test_regex_case_insensitive(self, sample_df):
        # Modality includes "ct" in different cases
        result = query_df(
            sample_df,
            case_insensitive=True,
            Modality={"RegEx": r"^ct$"},
        )
        # should catch "CT" and "ct"
        assert set(result["Modality"]) == {"CT", "ct"}


# ---------------------------
# contains / approx
# ---------------------------


class TestQueryDfContainsApprox:
    def test_contains_on_string(self, sample_df):
        # substring search in Tags where it's a string
        result = query_df(sample_df, Tags={"contains": "free text"})
        assert list(result.index) == [3]

    def test_contains_in_container(self, sample_df):
        # membership in list-like Tags
        result = query_df(sample_df, Tags={"contains": "LUNG"})
        assert list(result.index) == [2]

    def test_approx_numeric(self, sample_df):
        # target ~1.0 with tighter tolerance
        result = query_df(
            sample_df,
            Score={"approx": {"value": 1.0, "atol": 1e-4, "rtol": 1e-6}},
        )
        # rows 0 and 1 are extremely close to 1.0; row 2 (0.9995) is now outside atol
        assert set(result.index) == {0, 1}


# ---------------------------
# Temporal: time_between / in_last_days
# ---------------------------


class TestQueryDfTemporal:
    def test_time_between_simple_range(self, sample_df):
        # SeriesTime between 08:00 and 17:00 (no wrap) -> 09:00, 12:00
        result = query_df(sample_df, SeriesTime={"time_between": ["08:00", "17:00"]})
        assert set(result.index) == {0, 3}

    def test_time_between_wraparound(self, sample_df):
        # 22:00..06:00 (wrap around midnight) -> 23:00, 03:00
        result = query_df(sample_df, SeriesTime={"time_between": ["22:00", "06:00"]})
        assert set(result.index) == {1, 2}

    def test_in_last_days_datetime_column(self, sample_df):
        # Use a fixed "now" so test is deterministic
        now = dt.datetime(2025, 1, 31, 12, 0, 0)
        cond = {"in_last_days": {"days": 7, "now": now}}

        result = query_df(sample_df, AcqDateTime=cond)
        # AcqDateTime deltas: 1, 5, 10, 20, 0 days -> keep indices 0,1,4
        assert set(result.index) == {0, 1, 4}


# ---------------------------
# Nested dot paths
# ---------------------------


class TestQueryDfNestedPaths:
    def test_nested_meta_series_uid(self, sample_df):
        # Root column is 'meta', dot-path 'meta.series[0].uid'
        result = query_df(sample_df, **{"meta.series[0].uid": {"eq": "1.2.3.2"}})
        assert list(result.index) == [1]

    def test_missing_root_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            query_df(sample_df, NotAColumn="foo")


# ---------------------------
# Error handling
# ---------------------------


class TestQueryDfErrors:
    def test_unsupported_operator_raises(self, sample_df):
        with pytest.raises(ValueError):
            query_df(sample_df, Age={"foo": 123})

    def test_time_between_bad_value_raises(self, sample_df):
        # time_between expects list/tuple of length 2
        with pytest.raises(ValueError):
            query_df(sample_df, SeriesTime={"time_between": ["09:00"]})
