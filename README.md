# Image-Processing-App
The application offers hands-on exposure to Fourier transformation, diverse filters, thresholding, and image segmentation. Users can apply these techniques practically to their own input images while gaining a comprehensive understanding of each featured image processing tool.
Certainly! Below is a template for a README file for your image processing app. Feel free to customize it further based on specific details or additional features in your app.

---

## Overview

The Image Processing App is a versatile tool designed to facilitate the practical application of various image processing techniques. Leveraging the power of OpenCV, Matplotlib, and Streamlit, this app provides users with an interactive platform to explore and understand the impact of different filters, thresholding methods, and image segmentation on their own input images.

## Features

### 1. Fourier Transformation

Explore the frequency domain through Fourier transformation of images. Gain insights into the spatial frequency components of your images.

### 2. Filters

Apply a range of filters to enhance or modify images:
- **Low-pass and High-pass Filters:** Control the frequency components in your images.
- **Gaussian Blur:** Achieve smooth and subtle blurring effects.
- **Median Blur:** Remove noise while preserving edges.
- **Canny Edge Detection:** Identify and highlight edges in images.
- **Sobel Filter:** Emphasize edges using Sobel operators.
- **Laplacian Filter:** Enhance image details through Laplacian filtering.

### 3. Thresholding and Segmentation

Experiment with various thresholding techniques and segmentation methods:
- **Binary Thresholding:** Create binary images based on intensity thresholds.
- **Adaptive Thresholding:** Dynamically adjust thresholds for varying image regions.
- **Otsu's Thresholding:** Automatically determine optimal thresholds.
- **Inverse Binary Thresholding:** Generate inverted binary images.
- **Truncated Thresholding:** Limit pixel values to a specified range.
- **Region-Based Segmentation:** Identify and delineate distinct regions in images.
- **Edge Detection:** Detect edges using the Canny edge detector.
- **Contour Detection:** Identify and visualize contours in images.

### 4. Interactive Controls

Adjust filter parameters in real-time and visualize the immediate effects on your images. Enjoy a seamless and user-friendly experience.

### 5. Educational Insights

Each image processing technique is accompanied by insightful comments, fostering a deeper understanding of the underlying principles.

## Getting Started

1. Clone the repository.
   ```bash
   git clone https://github.com/your-username/image-processing-app.git
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app.
   ```bash
   streamlit run app2.py
   ```

## Usage

1. Upload your own images for processing.
2. Explore different filters and techniques using the interactive interface.
3. Observe real-time changes and enhancements in your images.
