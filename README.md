# **rosamllib**

**rosamllib** is a Python package for reading, manipulating, and visualizing DICOM files, particularly focusing on radiotherapy-related DICOM formats like RTSTRUCT and RTDOSE. The library simplifies the process of extracting useful information from DICOM files, overlaying contours on medical images, and resampling dose grids for dose distribution visualization.

## **Features**

- **DICOM Image Handling**: Supports reading, resampling, and visualization of DICOM images.
- **RTSTRUCT Handling**: Extracts contours, generates masks for structures, and overlays them on medical images.
- **RTDOSE Handling**: Resamples dose grids to match CT image grids and visualizes dose distribution.
- **Simple and Intuitive API**: Easy-to-use API for manipulating DICOM datasets, working with medical images, and visualizing structures and dose distributions.

## **Table of Contents**

- [Installation](#installation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contacts](#contacts)

## **Installation**

To install `rosamllib`, you can use `pip`:

```bash
pip install rosamllib
```
Alternatively, you can clone this repository and install it manually:

```bash
git clone https://gitlab.com/ucla_rosaml/shared/rosamllib
cd rosamllib
pip install .
```

## Documentation
Detailed documentation is available in the form of Jupyter notebooks within the repository. These provide step-by-step examples and usage guides for the core functionality of **rosamllib**:
- [Reading DICOM Images](https://gitlab.com/YAAF/rosamllib/-/blob/main/examples/reading_dicom_images.ipynb)
- [Reading RTSTRUCT Files](https://gitlab.com/YAAF/rosamllib/-/blob/main/examples/reading_dicom_rtstruct.ipynb)
- [Reading RTDOSE Files](https://gitlab.com/YAAF/rosamllib/-/blob/main/examples/reading_dicom_rtdose.ipynb)

## Contributing
We welcome contributions to improve rosamllib! If you encounter bugs or have feature requests, feel free to open an issue or submit a pull request.If you would like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes and write tests.
4. Submit a pull request with a detailed explanation.

## License

This project is licensed under the LGPL 3.0 License. See the [LICENSE](LICENSE) file for details.

## Contacts
If you have any questions or suggestions, please contact me at YasinAAbdulkadir@gmail.com