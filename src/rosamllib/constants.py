# List of non-common tags
non_common_tags = [
    "0002|0003",  # MediaStorageSOPInstanceUID
    "0008|0018",  # SOPInstanceUID
    "0008|0032",  # Acquisition Time
    "0008|0033",  # Content Time
    "0008|1140",  # Referenced Image Sequence
    "0008|2112",  # Source Image Sequence
    "0020|0037",  # Image Orientation Patient
    "0020|0032",  # Image Position Patient
    "0020|1041",  # Slice Location
    "0020|0013",  # Instance Number
    "0028|0030",  # Pixel Spacing
    "0028|0106",  # Smallest Image Pixel Value
    "0028|0107",  # Largest Image Pixel Value
    "7FE0|0010",  # Pixel Data
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


# Mapping of DICOM SOPClassUID to Modality
MODALITY_BY_CLASS_UID = {
    "1.2.840.10008.5.1.4.1.1.2": "CT",
    "1.2.840.10008.5.1.4.1.1.4": "MR",
    "1.2.840.10008.5.1.4.1.1.128": "PT",
    "1.2.840.10008.5.1.4.1.1.481.1": "RTIMAGE",
    "1.2.840.10008.5.1.4.1.1.1": "CR",
    "1.2.840.10008.5.1.4.1.1.481.4": "RTRECORD",
    "1.2.840.10008.5.1.4.1.1.481.5": "RTPLAN",
    "1.2.840.10008.5.1.4.1.1.481.2": "RTDOSE",
    "1.2.840.10008.5.1.4.1.1.481.3": "RTSTRUCT",
    "1.2.840.10008.5.1.4.1.1.66.1": "REG",
    "1.2.840.10008.5.1.4.1.1.66": "RAW",
}

CLASS_UID_BY_MODALITY = {
    "RTRECORD": "1.2.840.10008.5.1.4.1.1.481.4",
    "RTPLAN": "1.2.840.10008.5.1.4.1.1.481.5",
    "RTDOSE": "1.2.840.10008.5.1.4.1.1.481.2",
    "RTSTRUCT": "1.2.840.10008.5.1.4.1.1.481.3",
    "REG": "1.2.840.10008.5.1.4.1.1.66.1",
    "CT": "1.2.840.10008.5.1.4.1.1.2",
    "MR": "1.2.840.10008.5.1.4.1.1.4",
    "PT": "1.2.840.10008.5.1.4.1.1.128",
    "RTIMAGE": "1.2.840.10008.5.1.4.1.1.481.1",
    "CR": "1.2.840.10008.5.1.4.1.1.1",
    "RAW": "1.2.840.10008.5.1.4.1.1.66",
}
