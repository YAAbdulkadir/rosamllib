# import pytest
# import numpy as np
# import pydicom
# import SimpleITK as sitk
# from rosamllib.dicoms import RTStruct


# @pytest.fixture
# def sample_rtstruct():
#     # Create a sample RTSTRUCT dataset with necessary fields
#     ds = pydicom.Dataset()

#     # Add dummy ROIContourSequence
#     ds.ROIContourSequence = [
#         pydicom.Dataset(),
#     ]
#     ds.StructureSetROISequence = [
#         pydicom.Dataset(),
#     ]
#     ds.StructureSetROISequence[0].ROIName = "GTV"

#     # Add some necessary DICOM tags
#     ds.ReferencedFrameOfReferenceSequence = [
#         pydicom.Dataset(),
#     ]
#     ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID = "1.2.840.10008.1.2"

#     # Create a dummy SimpleITK image to simulate the associated series data
#     image_array = np.zeros((64, 64, 64))  # Empty 3D volume (64x64x64)
#     sitk_image = sitk.GetImageFromArray(image_array)
#     sitk_image.SetSpacing((1.0, 1.0, 1.0))
#     sitk_image.SetOrigin((0.0, 0.0, 0.0))
#     sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

#     # Initialize the RTStruct object
#     rtstruct = RTStruct(ds=ds, series_data=sitk_image)
#     return rtstruct


# def test_get_structure_names(sample_rtstruct):
#     # Test getting the structure names from the RTSTRUCT
#     structure_names = sample_rtstruct.get_structure_names()
#     assert structure_names == ["GTV"]


# def test_set_series_data(sample_rtstruct):
#     # Test setting the series data (SimpleITK Image)
#     image_array = np.random.rand(64, 64, 64)
#     new_series_data = sitk.GetImageFromArray(image_array)
#     sample_rtstruct.set_series_data(new_series_data)

#     assert sample_rtstruct.series_data == new_series_data


# def test_get_structure_color(sample_rtstruct):
#     # Simulate adding a color to the structure
#     ds = sample_rtstruct.ds
#     ds.ROIContourSequence[0].ROIDisplayColor = [255, 0, 0]

#     color = sample_rtstruct.get_structure_color("GTV")
#     assert color == [255, 0, 0]


# def test_get_structure_index(sample_rtstruct):
#     # Test getting the index of a structure by name
#     index = sample_rtstruct.get_structure_index("GTV")
#     assert index == 0


# def test_get_structure_mask(sample_rtstruct):
#     # Simulate generating a mask for the structure
#     mask = sample_rtstruct.get_structure_mask("GTV")
#     assert isinstance(mask, np.ndarray)
#     assert mask.shape == (64, 64, 64)


# def test_add_roi(sample_rtstruct):
#     # Test adding a new ROI
#     new_mask = np.ones((64, 64, 64), dtype=np.uint8)
#     sample_rtstruct.add_roi(mask=new_mask, name="New ROI", color=[0, 255, 0])

#     # The new ROI should be added to the StructureSetROISequence
#     assert "New ROI" in sample_rtstruct.get_structure_names()


# def test_get_contour_points_in_pixel_space(sample_rtstruct):
#     # Simulate getting contour points in pixel space
#     sample_rtstruct.set_referenced_image(sample_rtstruct.series_data)
#     contours = sample_rtstruct.get_contour_points_in_pixel_space("GTV")

#     # Check that the output is a dictionary of slice indices to numpy arrays
#     assert isinstance(contours, dict)
#     for slice_idx, points in contours.items():
#         assert isinstance(points, np.ndarray)
#         assert points.shape[1] == 3  # 3D coordinates


# def test_get_contour_points_in_physical_space(sample_rtstruct):
#     # Simulate getting contour points in physical space
#     sample_rtstruct.set_referenced_image(sample_rtstruct.series_data)
#     contours = sample_rtstruct.get_contour_points_in_physical_space("GTV")

#     # Check that the output is a dictionary of slice indices to numpy arrays
#     assert isinstance(contours, dict)
#     for slice_idx, points in contours.items():
#         assert isinstance(points, list)
#         for point_array in points:
#             assert point_array.shape[1] == 3  # 3D coordinates


# def test_get_physical_to_pixel_transformation_matrix(sample_rtstruct):
#     # Test the transformation matrix from physical to pixel space
#     matrix = sample_rtstruct.get_physical_to_pixel_transformation_matrix()

#     assert isinstance(matrix, np.ndarray)
#     assert matrix.shape == (4, 4)


# def test_get_pixel_to_physical_transformation_matrix(sample_rtstruct):
#     # Test the transformation matrix from pixel to physical space
#     matrix = sample_rtstruct.get_pixel_to_physical_transformation_matrix()

#     assert isinstance(matrix, np.ndarray)
#     assert matrix.shape == (4, 4)


# def test_dir_method_includes_dicom_metadata(sample_rtstruct):
#     # Test that DICOM metadata is included in the dir() output
#     attributes = sample_rtstruct.dir()
#     assert "GTV" in attributes
