# GNR-607-SIP-Project-on-Classification-of-Sentinel-2-images-Water-Vegetation-Other-

**A memory-efficient Python pipeline for classifying Sentinel-2 satellite imagery into Water, Vegetation, and Other classes.**

This tool uses a hybrid approach combining spectral indices (NDVI/NDWI), K-Means clustering, and morphological filtering to handle challenging scenarios like **turbid (muddy) water** and **dark soil noise**. It is designed to run on standard laptops without requiring heavy computational resources.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

* **Automated Band Detection:** Automatically finds Red, Green, NIR, and SWIR bands regardless of file naming conventions.
* **Hybrid Classification Engine:**
    * **Physics-Based:** Uses NDVI/NDWI thresholds for high-confidence pixels.
    * **AI-Based:** Uses K-Means (K=2) Clustering (Unsupervised Learning) to resolve ambiguous pixels (e.g., wet soil vs. muddy water).
* **Advanced Noise Removal:** Implements a spatial Sieve filter to remove "salt-and-pepper" noise (isolated soil pixels mimic water).
* **Turbidity Handling:** Calibrated thresholds (`NDVI < 0.20`) allow for the correct classification of sediment-heavy rivers that normally get misclassified as vegetation.
* **Thematic Change Detection:** Automatically generates a change map highlighting Water Gain/Loss and Vegetation Gain/Loss between the first and last timestamp.
* **Automatic Reporting:**
    * Generates Excel (`.xlsx`) reports with area statistics (km²) and embedded bar charts.
    * Produces high-quality PNG preview maps with legends.
* **GUI & CLI Support:** Runs with a user-friendly GUI selector or fully headless via command line.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/nagdavratika/GNR-607-SIP-Project-on-Classification-of-Sentinel-2-images-Water-Vegetation-Other-.git](https://github.com/nagdavratika/GNR-607-SIP-Project-on-Classification-of-Sentinel-2-images-Water-Vegetation-Other-.git)
    cd GNR-607-SIP-Project-on-Classification-of-Sentinel-2-images-Water-Vegetation-Other-
    ```

2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create -n rs_env python=3.9
    conda activate env_name
    ```

3.  **Install Dependencies:**
    ```bash
    pip install numpy rasterio pillow tqdm xlsxwriter scikit-learn scipy
    ```
    *(Note: `tkinter` is usually included with Python. If missing on Linux, install `python3-tk`)*

##  Usage

Simply run the script. A window will pop up asking you to select your input folders.
```bash
python Classification.py
