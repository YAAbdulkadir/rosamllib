import rt_utils
from rosamllib.dicoms import RTStruct
from rt_utils import image_helper, ds_helper
from pydicom import dcmread


class RTStructBuilder(rt_utils.RTStructBuilder):
    @staticmethod
    def create_new(dicom_series_path: str) -> RTStruct:
        """
        Method to generate a new rt struct from a DICOM series
        """

        series_data = image_helper.load_sorted_image_series(dicom_series_path)
        ds = ds_helper.create_rtstruct_dataset(series_data)
        ds.Manufacturer = "ROSAML"
        ds.InstitutionName = ""
        ds.ManufacturerModelName = "rosamllib"
        return RTStruct(ds, series_data=series_data)

    @staticmethod
    def create_from(
        dicom_series_path: str, rt_struct_path: str, warn_only: bool = False
    ) -> RTStruct:
        """
        Method to load an existing rt struct, given related DICOM series and existing rt struct
        """

        series_data = image_helper.load_sorted_image_series(dicom_series_path)
        ds = dcmread(rt_struct_path)
        RTStructBuilder.validate_rtstruct(ds)
        RTStructBuilder.validate_rtstruct_series_references(ds, series_data, warn_only)

        # TODO create new frame of reference? Right now we assume the last frame of reference
        # created is suitable
        return RTStruct(ds, series_data=series_data)
