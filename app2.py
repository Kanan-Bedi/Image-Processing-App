import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply low-pass filter
def apply_low_pass_filter(image, radius):
    # Perform 2D Fourier transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create a low-pass filter in the frequency domain
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    mask[center_row - radius:center_row + radius, center_col - radius:center_col + radius] = 1

    # Apply the filter to the Fourier transformed image
    f_transform_filtered = f_transform_shifted * mask

    # Perform the inverse Fourier transform
    filtered_image = np.fft.ifftshift(f_transform_filtered)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image)

    return filtered_image
    pass

# Function to apply high-pass filter
def apply_high_pass_filter(image, radius):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create a high-pass filter in the frequency domain
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    mask[center_row - radius:center_row + radius, center_col - radius:center_col + radius] = 0

    # Apply the filter to the Fourier transformed image
    f_transform_filtered = f_transform_shifted * mask

    # Perform the inverse Fourier transform
    filtered_image = np.fft.ifftshift(f_transform_filtered)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image)

    return filtered_image
    pass

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit app
def main():
    # Page title
    st.title("Image Processing App")
   

    # Sidebar for parameter input
    st.sidebar.header("Filter Parameters")
    low_pass_radius = st.sidebar.slider("Low Pass Radius", min_value=1, max_value=100, value=30)
    high_pass_radius = st.sidebar.slider("High Pass Radius", min_value=1, max_value=100, value=30)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # ... Your existing code ...

    if uploaded_file is not None:
        # Read image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Apply filters
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        low_pass_filtered = apply_low_pass_filter(gray_image, low_pass_radius)
        high_pass_filtered = apply_high_pass_filter(gray_image, high_pass_radius)

        # Normalize the images to the range [0, 1]
        normalized_low_pass = low_pass_filtered / 255.0
        normalized_high_pass = high_pass_filtered / 255.0

        # Display images
        st.image(image, caption='Original Image (Color)', use_column_width=True)
        st.image(gray_image, caption='Grayscale Image', use_column_width=True)
        st.image(normalized_low_pass, caption='Low-Pass Filtered Image', use_column_width=True, clamp=True)
        st.image(normalized_high_pass, caption='High-Pass Filtered Image', use_column_width=True, clamp=True)

 


        # Display other image processing techniques
        st.header("Other Image Processing Techniques")

        # ... Your existing code for Gaussian Blur, Median Blur, Canny Edge Detection, Sobel, Laplacian, etc.
        # Create a subplot with multiple rows and columns to showcase various filters
        plt.figure(figsize=(15, 8))

        # Original Image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
        plt.title('Gaussian Blur')

        # Apply Median Blur
        median_blurred_image = cv2.medianBlur(image, 7)
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(median_blurred_image, cv2.COLOR_BGR2RGB))
        plt.title('Median Blur')

        # Apply Canny Edge Detection
        edges = cv2.Canny(gray_image, 100, 200)
        plt.subplot(2, 3, 4)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection')

        # Apply Sobel Filter
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        plt.subplot(2, 3, 5)
        plt.imshow(sobel_x, cmap='gray')
        plt.title('Sobel  Filter')

        # Apply Laplacian Filter
        kernel_size =9
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=kernel_size)
        plt.subplot(2, 3, 6)
        plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian Filter')

        # Show the plots
        plt.tight_layout()
        plt.show()

        # Display results
        st.pyplot()

        # ... Your existing code for Thresholding, Region-Based Segmentation, Contour Detection, etc.
        # Thresholding
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Region-Based Segmentation
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region_segmentation = image.copy()
        cv2.drawContours(region_segmentation, contours, -1, (0, 255, 0), 2)

        # Contour Detection
        edge_detected = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        contour_detected = image.copy()
        contours, _ = cv2.findContours(edge_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_detected, contours, -1, (0, 0, 255), 2)

        # Display the images in 2 rows and 3 columns
        plt.figure(figsize=(12, 8))

        # Original Image
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.axis('off')

        # Thresholding
        plt.subplot(232)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Thresholding'), plt.axis('off')

        # Region-Based Segmentation
        plt.subplot(233)
        plt.imshow(cv2.cvtColor(region_segmentation, cv2.COLOR_BGR2RGB))
        plt.title('Region-Based Segmentation'), plt.axis('off')

        # Edge Detection
        plt.subplot(234)
        plt.imshow(edge_detected, cmap='gray')
        plt.title('Edge Detection'), plt.axis('off')

        # Contour Detection
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(contour_detected, cv2.COLOR_BGR2RGB))
        plt.title('Contour Detection'), plt.axis('off')

        # Leave the last subplot empty to create a grid layout
        plt.subplot(236)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Display results
        st.pyplot()

        # ... Your existing code for different thresholding-based segmentations
        # Create a subplot to showcase different thresholding-based segmentations
        plt.figure(figsize=(18, 9))  # Adjust the figure size here

        # Original Image
        plt.subplot(231)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Original Image')

        # Binary Thresholding
        plt.subplot(232)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Thresholding')

        # Adaptive Thresholding
        plt.subplot(233)
        block_size = 11
        constant = 2
        adaptive_binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
        plt.imshow(adaptive_binary_image, cmap='gray')
        plt.title('Adaptive Thresholding')

        # Otsu's Thresholding
        plt.subplot(234)
        _, otsu_binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(otsu_binary_image, cmap='gray')
        plt.title("Otsu's Thresholding")

        # Inverse Binary Thresholding
        plt.subplot(235)
        _, inverse_binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        plt.imshow(inverse_binary_image, cmap='gray')
        plt.title('Inverse Binary Thresholding')

        # Truncated Thresholding
        plt.subplot(236)
        _, truncated_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_TRUNC)
        plt.imshow(truncated_image, cmap='gray')
        plt.title('Truncated Thresholding')

        plt.tight_layout()
        plt.show()


        # Display results
        st.pyplot()
    else:
        st.warning("Please upload an image.")

if __name__ == '__main__':
    main()
