import os
import copy
import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
from rosamllib.dicoms import DICOMImage
from rosamllib.utils import compute_dvh


def in_jupyter():
    """
    Checks if the code is being executed in a Jupyter notebook.

    Returns
    -------
    bool
        True if in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except Exception:
        return False


def jupyter_environment():
    """
    Detect whether code is running in Jupyter, and if so, whether it's Notebook or Lab.

    Returns
    -------
    str
        One of "JupyterNotebook", "JupyterLab", "Unknown", or "Not in Jupyter".
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()

        if not ip:
            # Likely standard Python shell, not IPython
            return "Not in Jupyter"

        # Check if running via IPKernelApp
        if "IPKernelApp" not in ip.config:
            # Could be 'TerminalInteractiveShell' or other, so not a Jupyter environment
            return "Not in Jupyter"

        # If we're here, it's some form of Jupyter environment (Notebook, Lab, or unknown).
        # Check if 'parent_appname' indicates classic notebook or lab
        kernel_cfg = ip.config.get("IPKernelApp", {})
        parent_app = kernel_cfg.get("parent_appname", "").lower()

        if "labapp" in parent_app:
            return "JupyterLab"
        elif "notebookapp" in parent_app:
            return "JupyterNotebook"
        else:
            return "Unknown"

    except Exception:
        # If anything unexpected happened, assume not in Jupyter
        return "Not in Jupyter"


def apply_vscode_theme():
    """Automatically detect VS Code and apply styling."""
    if "VSCODE_PID" in os.environ:
        style = """
        <style>
            .cell-output-ipywidget-background {
                background-color: transparent !important;
            }
            :root {
                --jp-widgets-color: var(--vscode-editor-foreground);
                --jp-widgets-font-size: var(--vscode-editor-font-size);
            } 
        </style>
        """
        display(HTML(style))


if in_jupyter:
    from IPython.display import display, HTML

    # Apply theme automatically if running in VS Code Jupyter
    apply_vscode_theme()
    time.sleep(0.5)


def apply_window_level(image, window=None, level=None):
    """
    Apply window and level to a given image.

    Parameters
    ----------
    image : sitk.Image
        The input SimpleITK image to apply window/level adjustment.
    window : float or None, optional
        The window width for intensity display. If None, no windowing is applied.
    level : float or None, optional
        The window level (center) for intensity display. If None, no leveling is applied.

    Returns
    -------
    sitk.Image
        The windowed and leveled image.
    """
    if window is None or level is None:
        image_array = sitk.GetArrayFromImage(image)
        window = image_array.max() - image_array.min()
        level = (image_array.max() + image_array.min()) / 2

    # Compute the min and max intensity values for windowing
    min_intensity = level - window / 2
    max_intensity = level + window / 2

    # Apply intensity windowing (clipping)
    return sitk.IntensityWindowing(
        image,
        windowMinimum=min_intensity,
        windowMaximum=max_intensity,
        outputMinimum=0.0,
        outputMaximum=1.0,
    )


def calculate_extent(axis, size, spacing):
    if axis == 0:
        return [0, size[0] * spacing[0], 0, size[1] * spacing[1]]
    elif axis == 1:
        return [0, size[0] * spacing[0], 0, size[2] * spacing[2]]
    elif axis == 2:
        return [0, size[1] * spacing[1], 0, size[2] * spacing[2]]


def visualize_fusion(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    center_align: bool = True,
    fixed_window=None,
    fixed_level=None,
    moving_window=None,
    moving_level=None,
    axis: int = 0,
    figsize=(10, 10),
    **kwargs,
):
    """
    Visualizes the fixed image and the registered (moving) image by blending them with an
    adjustable alpha.

    This method provides an interactive way to scroll through slices of the images and adjust
    the blending alpha value to visualize the overlap between the fixed and registered images.
    Window/level parameters can be applied to both images separately.

    Parameters
    ----------
    fixed_image : sitk.Image
        The fixed SimpleITK image that serves as the reference.
    moving_image : sitk.Image
        The moving SimpleITK image to be visualized alongside the fixed image.
    center_align : bool, optional
        If True, uses SimpleITK's CenteredTransformInitializer to align the centers of the
        images.
    fixed_window : float or None, optional
        Window width for intensity adjustment on the fixed image. Default is None (no adjustment).
    fixed_level : float or None, optional
        Window level for intensity adjustment on the fixed image. Default is None (no adjustment).
    moving_window : float or None, optional
        Window width for intensity adjustment on the moving image. Default is None (no adjustment).
    moving_level : float or None, optional
        Window level for intensity adjustment on the moving image. Default is None (no adjustment).
    axis : int, optional
        The axis along which to slice the image. Must be 0, 1, or 2. Default is 0.
    figsize : tuple of int, optional
        Size of the figure for displaying the images. Default is (10, 10).
    **kwargs : keyword arguments
        Additional keyword arguments to pass to `plt.imshow` for image display customization (e.g.,
        cmap='gray', origin='lower').

    Returns
    -------
    None
    """
    is_jupyter = in_jupyter()

    # Cast both images to sitkFloat64 for consistency
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat64)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat64)

    # Center align the images if required
    if center_align:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            initial_transform,
            sitk.sitkLinear,
            np.min(sitk.GetArrayViewFromImage(moving_image)),
            moving_image.GetPixelID(),
        )
        moving_image = DICOMImage(moving_resampled)

    # Convert SimpleITK images to numpy arrays for visualization
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_image)

    fixed_spacing = fixed_image.GetSpacing()
    fixed_size = fixed_image.GetSize()

    fixed_image = apply_window_level(fixed_image, fixed_window, fixed_level)
    moving_image = apply_window_level(moving_image, moving_window, moving_level)

    # Ensure that the arrays have the same shape
    if len(fixed_array.shape) != len(moving_array.shape):
        raise ValueError("The images must have the same dimensions for fusion visualization.")

    # Define the plotting function for a specific slice and alpha
    def plot_fusion(slice_idx, alpha):
        if axis == 0:
            img = (1.0 - alpha) * fixed_image[:, :, slice_idx] + alpha * moving_image[
                :, :, slice_idx
            ]
        elif axis == 1:
            img = (1.0 - alpha) * fixed_image[:, slice_idx, :] + alpha * moving_image[
                :, slice_idx, :
            ]
        elif axis == 2:
            img = (1.0 - alpha) * fixed_image[slice_idx, :, :] + alpha * moving_image[
                slice_idx, :, :
            ]
        extent = calculate_extent(axis, fixed_size, fixed_spacing)
        plt.figure(figsize=figsize)
        plt.title(f"Slice {slice_idx}")
        plt.imshow(sitk.GetArrayViewFromImage(img), extent=extent, **kwargs)
        plt.axis("off")
        plt.show()
        # TODO: Fix the checkerboard and the RGB
        # if axis == 0:
        #     f_slice = fixed_array[slice_idx, :, :]
        #     m_slice = moving_array[slice_idx, :, :]
        # elif axis == 1:
        #     f_slice = fixed_array[:, slice_idx, :]
        #     m_slice = moving_array[:, slice_idx, :]
        # else:
        #     f_slice = fixed_array[:, :, slice_idx]
        #     m_slice = moving_array[:, :, slice_idx]

        # extent = calculate_extent(axis, fixed_size, fixed_spacing)
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.set_title(f"Slice {slice_idx} ({mode})")

        # if mode == "blend":
        #     img = (1.0 - alpha) * f_slice + alpha * m_slice
        #     ax.imshow(img, cmap="gray", extent=extent, **kwargs)

        # elif mode == "checkerboard":
        #     step = 16  # checkerboard size
        #     mask = np.indices(f_slice.shape).sum(axis=0) % (2 * step) < step
        #     img = f_slice.copy()
        #     img[mask] = m_slice[mask]
        #     ax.imshow(img, cmap="gray", extent=extent, **kwargs)

        # elif mode == "rgb":
        #     img_rgb = np.zeros((*f_slice.shape, 3), dtype=np.float32)
        #     img_rgb[..., 0] = f_slice / np.max(f_slice)  # Red channel for fixed
        #     img_rgb[..., 1] = m_slice / np.max(m_slice)  # Green channel for moving
        #     ax.imshow(img_rgb, extent=extent, **kwargs)

        # elif mode == "difference":
        #     img = np.abs(f_slice - m_slice)
        #     ax.imshow(img, cmap="hot", extent=extent, **kwargs)

        # ax.axis("off")
        # plt.show()

    # Interactive mode for Jupyter Notebook
    if is_jupyter:
        from ipywidgets import interact, FloatSlider

        interact(
            plot_fusion,
            slice_idx=(0, fixed_array.shape[axis] - 1),
            alpha=FloatSlider(value=0.5, min=0, max=1, step=0.01, description="Alpha"),
            # mode=Dropdown(
            #     options=["blend", "checkerboard", "rgb", "difference"],
            #     value="blend",
            #     description="Mode",
            # ),
        )
    # Script mode for basic visualization using matplotlib sliders
    else:
        import matplotlib.widgets as widgets

        # Define interactive function for scripts
        def plot_interactive_fusion():
            slice_idx = 0
            alpha = 0.5

            fig, ax = plt.subplots(figsize=figsize)
            plt.subplots_adjust(bottom=0.25)
            if axis == 0:
                img = (1.0 - alpha) * fixed_image[:, :, slice_idx] + alpha * moving_image[
                    :, :, slice_idx
                ]
            elif axis == 1:
                img = (1.0 - alpha) * fixed_image[:, slice_idx, :] + alpha * moving_image[
                    :, slice_idx, :
                ]
            elif axis == 2:
                img = (1.0 - alpha) * fixed_image[slice_idx, :, :] + alpha * moving_image[
                    slice_idx, :, :
                ]
            img_array = sitk.GetArrayFromImage(img)
            extent = calculate_extent(axis, fixed_size, fixed_spacing)
            img_display = ax.imshow(img_array, extent=extent, **kwargs)
            ax.set_title(f"Slice {slice_idx}")
            ax.axis("off")

            # Slider for scrolling through slices
            slice_slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
            slice_slider = widgets.Slider(
                slice_slider_ax,
                "Slice",
                0,
                fixed_array.shape[axis] - 1,
                valinit=slice_idx,
                valfmt="%0.0f",
            )

            # Slider for adjusting alpha blending
            alpha_slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
            alpha_slider = widgets.Slider(alpha_slider_ax, "Alpha", 0, 1, valinit=alpha)

            # Update function for the sliders
            def update(val):
                slice_idx = int(slice_slider.val)
                alpha = float(alpha_slider.val)
                img = (1.0 - alpha) * fixed_image[:, :, slice_idx] + alpha * moving_image[
                    :, :, slice_idx
                ]
                img_array = sitk.GetArrayFromImage(img)
                img_display.set_data(img_array)
                ax.set_title(f"Slice {slice_idx}")
                fig.canvas.draw_idle()

            # Connect the sliders to the update function
            slice_slider.on_changed(update)
            alpha_slider.on_changed(update)

            plt.show()

        # Run the interactive function
        plot_interactive_fusion()


def interactive_image_viewer(
    img,
    masks=None,
    dose_array=None,
    dose_units=None,
    current_spacing=None,
    new_spacing=None,
    window=None,
    level=None,
    figsize=(6, 6),
    roi_widget_max_height=300,
    axis=0,
    **kwargs,
):
    """
    Displays a 3D medical image interactively with optional overlaid contours and dose
    visualization.

    This function provides an interactive viewer for 3D medical images, allowing users to scroll
    through image slices along a chosen axis, adjust window/level settings, and optionally overlay
    regions of interest (ROIs) as contours. If a dose array is provided, users can toggle its
    display and adjust the threshold for visualization. The function adapts to both Jupyter
    notebooks (using `ipywidgets`) and standard Python scripts (using `matplotlib` widgets).

    The input image, along with the optional dose and masks, can be resampled to a new voxel
    spacing while preserving the number of slices along the specified axis.

    Parameters
    ----------
    img : numpy.ndarray or sitk.Image
        The 3D image to display, either as a NumPy array or a SimpleITK image.
    masks : dict, optional
        A dictionary containing masks for multiple Regions of Interest (ROIs). Each key is the name
        of the ROI (string), and each value is a dictionary with:
            - "mask": numpy.ndarray
                A 3D binary mask for the ROI, matching the shape of `img`.
            - "color": tuple of int
                An (R, G, B) color tuple for the contour, with each value between 0-255.
        If `None`, no contours will be displayed. Default is `None`.
    dose_array : numpy.ndarray, optional
        A 3D array representing dose values that matches the shape of `img`. If provided, users can
        toggle the dose display and set a threshold for dose visualization. Default is `None`.
    dose_units : str, optional
        A string representing the units of the dose array (e.g., "Gy"). Displayed in the dose
        checkbox label if `dose_array` is provided. Default is `None`.
    current_spacing : list of float, optional
        The current voxel spacing of the input image in the form (x, y, z). This will be used for
        resampling if provided. If `None` and `img` is a SimpleITK image, its native spacing will
        be used. Default is `None`.
    new_spacing : list of float, optional
        The new voxel spacing to which the image will be resampled. Resampling will occur along all
        axes except the specified slicing axis. Default is `None`.
    window : int, optional
        The window width (contrast) for the image display. If `None`, the full intensity range is
        used. Default is `None`.
    level : int, optional
        The window level (brightness) for the image display. If `None`, the mean intensity value
        is used. Default is `None`.
    figsize : tuple of int, optional
        The size of the displayed figure in inches (width, height). Default is (6, 6).
    roi_widget_max_height : int
        Maximum height (in pixels) for the ROI selection panel to allow scrolling.
    axis : int, optional
        The axis along which to slice the image (0 for axial, 1 for sagittal, 2 for coronal).
        Default is 0 (axial slices).
    **kwargs : keyword arguments
        Additional keyword arguments to pass to `plt.imshow` for image display customization (e.g.,
        color maps, normalization).

    Returns
    -------
    None

    Notes
    -----
    - If the input image is a SimpleITK image, it will be converted to a NumPy array for display.
    - Both the image and masks can be resampled to a new voxel spacing while maintaining the
      original number of slices along the selected axis.
    - In Jupyter notebooks, `ipywidgets` are used to enable interactive scrolling and control of
      window/level settings. In non-Jupyter environments, `matplotlib` widgets are used for
      similar interactivity.
    - If `dose_array` is provided, users can toggle its display and adjust a threshold to
      visualize doses
      above a certain value.

    Examples
    --------
    >>> interactive_image_viewer(image_array, window=400, level=50, axis=2, cmap='grey',
        origin='lower')
    >>> interactive_image_viewer(image_array, roi_masks, window=400, level=50)
    >>> interactive_image_viewer(image_array, roi_masks, dose_array=dose_array, dose_units='Gy',
        window=300)

    Raises
    ------
    ValueError
        If the input image, masks, or dose array dimensions are inconsistent or improperly
        formatted.
    """
    is_jupyter = in_jupyter()

    masks = copy.deepcopy(masks) if masks is not None else None

    # If the input image is a SimpleITK image, convert it to a numpy array
    if isinstance(img, sitk.Image):
        if current_spacing is not None:
            img.SetSpacing(current_spacing)

        current_spacing = img.GetSpacing()

        if new_spacing is None:
            new_spacing = [1.0, 1.0, 1.0]

        # Resample only along axes other than the chosen `axis`
        if axis == 0:
            new_spacing[0] = 1.0
            new_spacing[1] = 1.0
        elif axis == 1:
            new_spacing[0] = 1.0
            new_spacing[2] = 1.0
        elif axis == 2:
            new_spacing[1] = 1.0
            new_spacing[2] = 1.0

        original_size = img.GetSize()
        new_size = [
            int(np.round(original_size[i] * (current_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(img.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkLinear)
        img = resampler.Execute(img)

        # Convert the SimpleITK image to a NumPy array for visualization
        img = sitk.GetArrayFromImage(img)

    elif isinstance(img, np.ndarray):
        # Resample the numpy array using the provided spacing
        img_sitk = sitk.GetImageFromArray(img)
        if current_spacing is not None:
            img_sitk.SetSpacing(current_spacing)

        current_spacing = img_sitk.GetSpacing()

        if new_spacing is None:
            new_spacing = [1.0, 1.0, 1.0]

        # Resample only along axes other than the chosen `axis`
        if axis == 0:
            new_spacing[0] = 1.0
            new_spacing[1] = 1.0
        elif axis == 1:
            new_spacing[0] = 1.0
            new_spacing[2] = 1.0
        elif axis == 2:
            new_spacing[1] = 1.0
            new_spacing[2] = 1.0

        original_size = img_sitk.GetSize()
        new_size = [
            int(np.round(original_size[i] * (current_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        img_resampled = resampler.Execute(img_sitk)
        img = sitk.GetArrayFromImage(img_resampled)

    # Transpose the image based on the axis
    if axis == 1:
        img = np.transpose(img, (1, 0, 2))
    elif axis == 2:
        img = np.transpose(img, (2, 0, 1))

    # Handle window/level adjustments
    if window is None:
        window = img.max() - img.min()
    if level is None:
        level = np.mean(img)

    def resample_dose(dose):
        dose_sitk = sitk.GetImageFromArray(dose)
        dose_sitk.SetSpacing(current_spacing)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled_dose = resampler.Execute(dose_sitk)
        return sitk.GetArrayFromImage(resampled_dose)

    def resample_mask(mask):
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.SetSpacing(current_spacing)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_mask = resampler.Execute(mask_sitk)
        return sitk.GetArrayFromImage(resampled_mask)

    if dose_array is not None:
        dose_array = resample_dose(dose_array)
        if axis == 1:
            dose_array = np.transpose(dose_array, (1, 0, 2))
        elif axis == 2:
            dose_array = np.transpose(dose_array, (2, 0, 1))

    if masks is not None:
        for ROI in masks:
            mask = masks[ROI]["mask"]
            resampled_mask = resample_mask(mask)
            if axis == 1:
                resampled_mask = np.transpose(resampled_mask, (1, 0, 2))
            elif axis == 2:
                resampled_mask = np.transpose(resampled_mask, (2, 0, 1))
            masks[ROI]["mask"] = resampled_mask

    # Interactive mode for Jupyter Notebook
    if is_jupyter:
        from ipywidgets import (
            Output,
            widgets,
            IntSlider,
            FloatSlider,
            VBox,
            HBox,
            HTML,
            Layout,
            GridBox,
        )
        from IPython.display import display

        # Create an output widget for the image
        image_output = Output()

        def update_image_jupyter(
            imgs,
            masks,
            dose_array,
            slice,
            selected_ROIs,
            window,
            level,
            display_dose,
            dose_threshold,
        ):
            with image_output:
                image_output.clear_output(wait=True)
                images = imgs.copy()
                min_val = level - (window / 2)
                max_val = level + (window / 2)
                images = np.clip(images, min_val, max_val)

                fig, ax = plt.subplots(figsize=figsize)

                # plt.figure(figsize=figsize)
                ax.imshow(images[slice], **kwargs)

                # Display dose if checkbox is selected and dose_array is not None
                if dose_array is not None and display_dose:
                    dose_slice = dose_array[slice]
                    dose_slice = np.ma.masked_less(dose_slice, dose_threshold)
                    if "origin" in kwargs:
                        dose_img = ax.imshow(
                            dose_slice, alpha=0.5, cmap="jet", origin=kwargs["origin"]
                        )
                    else:
                        dose_img = ax.imshow(dose_slice, alpha=0.5, cmap="jet")
                    cbar = fig.colorbar(dose_img, ax=ax)
                    if dose_units is not None:
                        cbar.set_label(f"Dose ({dose_units})")
                    else:
                        cbar.set_label("Dose")

                # Display multiple contours for selected ROIs if masks are provided
                if masks is not None:
                    for ROI in selected_ROIs:
                        if ROI in masks:
                            color = [tuple([c / 255 for c in masks[ROI]["color"]])]
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message="No contour levels were found within the data range.",
                                )
                                ax.contour(masks[ROI]["mask"][slice], colors=color)

                ax.axis("off")
                plt.show()

        # Dose-related widgets
        dose_checkbox = None
        dose_slider = None
        dose_widgets = []
        if dose_array is not None:
            dose_checkbox = widgets.Checkbox(
                value=False, description=f"Show Dose (units: {dose_units or 'unknown'})"
            )
            dose_slider = FloatSlider(
                value=0.0, min=0.0, max=np.max(dose_array), step=0.1, description="Threshold"
            )
            dose_widgets = [dose_checkbox, dose_slider]

        # Function to get the selected ROIs (the ones that are checked)
        def get_selected_ROIs():
            return [roi for roi, checkbox in ROI_checkboxes.items() if checkbox.value]

        # Define a function that updates the display (called by both slider and checkboxes)
        def update_display(change=None):
            selected_ROIs = get_selected_ROIs() if masks is not None else []
            if dose_array is not None:
                dose_checkbox_value = dose_checkbox.value
                dose_slider_value = dose_slider.value
            else:
                dose_checkbox_value = None
                dose_slider_value = None
            update_image_jupyter(
                img,
                masks,
                dose_array,
                slice_slider.value,
                selected_ROIs,
                window,
                level,
                dose_checkbox_value,
                dose_slider_value,
            )

        # If masks are provided, set up ROI selection widgets
        if masks is not None:
            # Create a list of checkboxes for each ROI
            ROI_checkboxes = {}
            ROI_children = []
            for structure, data in masks.items():
                # Extract the color for the ROI and convert to an RGB hex string for CSS
                roi_color = data["color"]
                hex_color = "#{:02x}{:02x}{:02x}".format(*roi_color)

                # Create the checkbox
                checkbox = widgets.Checkbox(
                    value=False, description=structure, style={"description_width": "initial"}
                )

                # Create a colored square using an HTML widget
                color_square = HTML(
                    value=(
                        "<div style='width: 15px; height: 15px; background-color: "
                        f"{hex_color};'></div>"
                    )
                )

                ROI_children.append(color_square)
                ROI_children.append(checkbox)

                ROI_checkboxes[structure] = checkbox

            # Link checkbox updates to the update_display function
            for checkbox in ROI_checkboxes.values():
                checkbox.observe(
                    update_display, "value"
                )  # Attach the observer to trigger on value change

            grid_layout = Layout(
                grid_template_columns="30px auto",
                grid_gap="3px 10px",
                overflow_y="auto",
                max_height=f"{roi_widget_max_height}px",
                width="220px",
                border="1px solid black",
            )
            ROI_grid = GridBox(ROI_children, layout=grid_layout)

        # Create the slice slider manually
        slice_slider = IntSlider(
            value=img.shape[0] // 2, min=0, max=img.shape[0] - 1, description="Slice"
        )
        slice_slider.observe(update_display, "value")

        # Organize the checkboxes vertically on the right and scroll if needed
        if masks is not None:
            display_container = HBox([VBox([image_output]), ROI_grid])
        else:
            display_container = VBox([image_output])

        if dose_array is not None:
            dose_checkbox.observe(update_display, "value")
            dose_slider.observe(update_display, "value")

        # Display the slider below the container
        display(slice_slider)

        # Display the widget container using IPython's display
        display(VBox([display_container] + dose_widgets))

        update_display()

    else:
        import matplotlib.widgets as widgets

        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.25, right=0.8, bottom=0.2)

        def update_image_script(
            imgs,
            masks,
            dose_array,
            slice,
            selected_ROIs,
            window,
            level,
            display_dose,
            dose_threshold,
        ):
            images = imgs.copy()
            min_val = level - (window / 2)
            max_val = level + (window / 2)
            images = np.clip(images, min_val, max_val)

            ax.clear()
            ax.imshow(images[slice], **kwargs)

            # Display dose if checkbox is selected and dose_array is not None
            if dose_array is not None and display_dose:
                dose_slice = dose_array[slice]
                dose_slice = np.ma.masked_less(dose_slice, dose_threshold)
                ax.imshow(dose_slice, cmap="jet", alpha=0.5)

            # Display multiple contours for selected ROIs if masks are provided
            if masks is not None:
                for ROI in selected_ROIs:
                    if ROI in masks:
                        color = [tuple([c / 255 for c in masks[ROI]["color"]])]
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="No contour levels were found within the data range.",
                            )
                            ax.contour(masks[ROI]["mask"][slice], colors=color)

            ax.axis("off")
            ax.set_title(f"Slice {slice}")
            fig.canvas.draw_idle()

        # Initialize slice and dose parameters
        slice_idx = img.shape[0] // 2
        selected_ROIs = []
        display_dose = False
        dose_threshold = 0.0

        # Initial display of the image
        update_image_script(
            img,
            masks,
            dose_array,
            slice_idx,
            selected_ROIs,
            window,
            level,
            display_dose,
            dose_threshold,
        )

        # Slider for scrolling through slices
        ax_slider = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor="lightgoldenrodyellow")
        slice_slider = widgets.Slider(
            ax=ax_slider,
            label="Slice",
            valmin=0,
            valmax=img.shape[0] - 1,
            valinit=slice_idx,
            valfmt="%0.0f",
        )

        # If masks are provided, set up checkboxes for ROIs
        if masks is not None:
            num_rois = len(masks)
            height_per_roi = 0.2
            checkbox_height = min(0.8, num_rois * height_per_roi)

            # Checkboxes for each ROI
            ax_checkboxes = plt.axes([0.8, 0.5 - (checkbox_height / 2), 0.15, checkbox_height])
            checkboxes = widgets.CheckButtons(
                ax_checkboxes, labels=list(masks.keys()), actives=[False] * len(masks)
            )

        # Dose checkbox and threshold slider, if dose_array is provided
        if dose_array is not None:
            ax_dose_checkbox = plt.axes([0.8, 0.2, 0.15, 0.05])
            dose_checkbox = widgets.CheckButtons(ax_dose_checkbox, ["Show Dose"], [False])

            ax_dose_slider = plt.axes([0.8, 0.1, 0.15, 0.03], facecolor="lightgoldenrodyellow")
            dose_slider = widgets.Slider(
                ax=ax_dose_slider,
                label="Threshold",
                valmin=0.0,
                valmax=np.max(dose_array),
                valinit=0.0,
            )

            def update_script(val):
                selected_ROIs = [
                    label for label, active in zip(masks.keys(), checkboxes.get_status()) if active
                ]
                slice_idx = int(slice_slider.val)
                display_dose = dose_checkbox.get_status()[0]
                dose_threshold = dose_slider.val
                update_image_script(
                    img,
                    masks,
                    dose_array,
                    slice_idx,
                    selected_ROIs,
                    window,
                    level,
                    display_dose,
                    dose_threshold,
                )

            slice_slider.on_changed(update_script)
            dose_checkbox.on_clicked(update_script)
            dose_slider.on_changed(update_script)

        else:

            def update_script(val):
                slice_idx = int(slice_slider.val)
                update_image_script(img, masks, None, slice_idx, [], window, level, False, 0.0)

            slice_slider.on_changed(update_script)

        # If masks are provided, connect checkbox clicks
        if masks is not None:
            checkboxes.on_clicked(update_script)

        plt.show()


def dvh_viewer(
    dose_image,
    roi_masks,
    relative_volume=True,
    relative_dose=False,
    prescription_dose=None,
    bin_width=0.1,
    template="plotly_white",
    renderer=None,
):
    """
    Interactive DVH viewer using Plotly, supporting Jupyter and standalone scripts.

    Parameters:
    ----------
    dose_image : SimpleITK.Image
        A SimpleITK image containing the dose distribution.
    roi_masks : dict
        A dictionary where keys are ROI names (str) and values are binary SimpleITK images
        representing the masks for each ROI (0 for background, 1 for ROI).
    relative_volume : bool, default=True
        If True, show relative volumes (%); otherwise, absolute volumes (mm³).
    relative_dose : bool, default=True
        If True, show relative dose bins (%); otherwise, absolute dose bins (Gy).
    prescription_dose : float, optional
        If provided, the relative dose will be calculated as a percentage of the prescription dose.
    bin_width : float
        The width of each dose bin (Gy or % if relative_dose=True).
    plotly_template : str, default="plotly_white"
        The Plotly template to use for the visualization.
    renderer : str, optional
        If provided, use this Plotly renderer (e.g., "notebook", "browser", "iframe", etc.).
        If None, choose "notebook" if in Jupyter, otherwise "browser".
    """
    if renderer is None:
        if in_jupyter():
            env = jupyter_environment()
            if env == "JupyterLab":
                renderer = "iframe"
            elif env == "JupyterNotebook":
                renderer = "notebook"
            else:
                renderer = "browser"
        else:
            renderer = "browser"

    dvh = compute_dvh(
        dose_image,
        roi_masks,
        relative_volume=relative_volume,
        relative_dose=relative_dose,
        prescription_dose=prescription_dose,
        bin_width=bin_width,
    )

    def plot_dvh(relative_volume, relative_dose):
        """
        Helper function to plot the DVH.

        Parameters:
        ----------
        relative_volume : bool
            If True, show relative volume (%); otherwise, absolute volume (mm³).
        relative_dose : bool
            If True, show relative dose (%); otherwise, absolute dose (Gy).
        """
        fig = go.Figure()
        x_label = "Dose (%)" if relative_dose else "Dose (Gy)"
        y_label = "Volume (%)" if relative_volume else "Volume (mm³)"
        for roi_name, (bin_edges, cumulative_volume) in dvh.items():
            fig.add_trace(
                go.Scatter(x=bin_edges, y=cumulative_volume, mode="lines", name=roi_name)
            )
        fig.update_layout(
            title="Dose-Volume Histogram (DVH)",
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title="ROIs",
            template=template,
        )
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.0,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    buttons=[
                        dict(
                            label="Hide All",
                            method="update",
                            args=[{"visible": ["legendonly"] * len(fig.data)}],
                        ),
                        dict(
                            label="Show All",
                            method="update",
                            args=[{"visible": [True] * len(fig.data)}],
                        ),
                    ],
                )
            ]
        )
        fig.show(renderer=renderer)

    plot_dvh(relative_volume, relative_dose)
