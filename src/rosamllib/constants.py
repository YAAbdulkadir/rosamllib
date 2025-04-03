# List of non-common tags
non_common_tags = [
    "0020|0032",  # Image Position Patient
    "0020|1041",  # Slice Location
    "0020|0013",  # Instance Number
    "0008|0032",  # Acquisition Time
    "7FE0|0010",  # Pixel Data
    "0028|0030",  # Pixel Spacing
]

# Mapping of DICOM VR to Pandas data types
VR_TO_DTYPE = {
    "CS": str,  # Code String
    "SH": str,  # Short String
    "LO": str,  # Long String
    "PN": str,  # Person Name
    "ST": str,  # Short Text
    "LT": str,  # Long Text
    "UT": str,  # Unlimited Text
    "IS": int,  # Integer String
    "DS": float,  # Decimal String
    "FL": float,  # Floating Point Single
    "FD": float,  # Floating Point Double
    "SL": int,  # Signed Long
    "SS": int,  # Signed Short
    "UL": int,  # Unsigned Long
    "US": int,  # Unsigned Short
    "DA": "date",  # Date
    "TM": "time",  # Time
    "DT": "datetime",  # Date-Time
}
