# src/rosamllib/registration.py
import numpy as np
import SimpleITK as sitk
from rosamllib.dicoms import REG
from rosamllib.dicoms import DICOMImage


class ImageRegistration:
    """
    A class dedicated to handling image registration tasks using REG files and DICOM images.

    This class supports applying both rigid and deformable transformations as defined in a
    DICOM REG file to align a moving image with a fixed reference image. It leverages SimpleITK
    for image resampling and transformation operations, ensuring precise registration of medical
    images.

    Attributes
    ----------
    reg : REG
        An instance of the REG class containing transformation matrices and metadata.
    fixed_image : sitk.Image
        The reference (fixed) image to which the moving image will be aligned.
    moving_image : sitk.Image
        The image that will be transformed to align with the fixed image.

    Methods
    -------
    apply_registration() -> DICOMImage:
        Applies the appropriate registration (rigid or deformable) based on the REG file.
    _apply_rigid_registration() -> DICOMImage:
        Applies the rigid registration using the transformation matrix from the REG file.
    _apply_deformable_registration() -> DICOMImage:
        Applies the deformable registration using the deformation grid from the REG file.

    """

    def __init__(self, reg: REG, fixed_image: DICOMImage, moving_image: DICOMImage):
        """
        Initializes the ImageRegistration with REG data and DICOM images.

        Parameters
        ----------
        reg : REG
            An instance of the REG class containing registration information, including
            transformation matrices for aligning the moving image to the fixed image.
        fixed_image : DICOMImage
            The fixed DICOM image that serves as the reference for alignment.
        moving_image : DICOMImage
            The moving DICOM image that will be registered to align with the fixed image.

        Raises
        ------
        ValueError
            If any of the provided images or REG data are invalid or do not contain the
            necessary information for registration.
        """
        self.reg = reg
        self.fixed_image = fixed_image
        self.moving_image = moving_image

    def apply_registration(self):
        """
        Applies the registration based on the type specified in the REG file.

        This method determines whether the registration is rigid or deformable
        based on the metadata in the REG instance. It then applies the corresponding
        transformation to align the moving image with the fixed image.

        Returns
        -------
        DICOMImage
            The registered moving image wrapped in a DICOMImage object.

        Raises
        ------
        ValueError
            If the registration type is not supported or if the necessary transformation
            data is missing.
        """
        if self.reg.registration_type == "rigid":
            return self._apply_rigid_registration()
        elif self.reg.registration_type == "deformable":
            return self._apply_deformable_registration()
        else:
            raise ValueError(f"Unsupported registration type: {self.reg.registration_type}")

    def _apply_rigid_registration(self):
        """
        Applies rigid registration using the transformation matrix from REG.

        This method retrieves the 4x4 transformation matrix from the REG data, constructs
        a corresponding affine transform in SimpleITK, and resamples the moving image
        to align it with the fixed image.

        Returns
        -------
        DICOMImage
            The registered moving image wrapped in a DICOMImage object.

        Raises
        ------
        ValueError
            If a valid 4x4 transformation matrix is not found in the REG data.
        """
        transform_matrix = self.reg.get_moving_image_info().get("transformation_matrix", None)

        if transform_matrix is None or transform_matrix.shape != (4, 4):
            raise ValueError("Valid 4x4 rigid transformation matrix not found in REG data.")

        # TODO
        # The plan is to check if the reg reader correctly assigned the fixed image info
        # and the moving image info. If not, swap the info.
        # But do we want this behavior?

        transform_matrix = np.linalg.inv(transform_matrix)

        # Create an AffineTransform in SimpleITK
        affine_transform = sitk.AffineTransform(3)

        affine_transform.SetMatrix(transform_matrix[:3, :3].flatten().tolist())
        affine_transform.SetTranslation(transform_matrix[:3, 3].tolist())

        # Resample the moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed_image)
        resampler.SetTransform(affine_transform)
        resampler.SetInterpolator(sitk.sitkLinear)

        # Ensure a robust default pixel value for different modalities
        default_value = float(np.min(sitk.GetArrayViewFromImage(self.moving_image)))
        resampler.SetDefaultPixelValue(default_value)

        registered_image = resampler.Execute(self.moving_image)

        return DICOMImage(registered_image)

    def _apply_deformable_registration(self):
        """
        Applies deformable registration using the deformation grid from REG.

        This method extracts a deformation field (displacement vectors) from the REG data
        and uses it to deform the moving image. The deformation field is converted into a
        SimpleITK DisplacementFieldTransform, which is then applied during resampling to
        align the moving image with the fixed image.

        Returns
        -------
        DICOMImage
            The registered moving image wrapped in a DICOMImage object.

        Raises
        ------
        ValueError
            If the deformation field data is not found in the REG data.
        """
        deformation_field = self.reg.moving_image_info.get("grid_data")
        if deformation_field is None:
            raise ValueError("Deformable transformation grid not found in REG data.")

        # Convert the deformation grid to a SimpleITK displacement field
        deformation_field_itk = sitk.GetImageFromArray(deformation_field, isVector=True)
        deformation_field_itk.CopyInformation(self.fixed_image)

        # Create a displacement field transform
        displacement_field_transform = sitk.DisplacementFieldTransform(deformation_field_itk)

        # Resample the moving image using the deformation field transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed_image)
        resampler.SetTransform(displacement_field_transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)

        registered_image = resampler.Execute(self.moving_image)
        return DICOMImage(registered_image)
