import math
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image

def lanczos_kernel(x, a=3):
    """
    Lanczos kernel function.
    
    Parameters:
        x (float or array_like): Input values.
        a (int): Lanczos kernel parameter (default is 3).
        
    Returns:
        array_like: Array of kernel values.
    """
    abs_x = np.abs(x)
    result = np.where(abs_x < 1e-8, 1.0, 0.0)
    result = np.where(abs_x >= a, result, 
                      a * np.sin(np.pi * x) * np.sin(np.pi * x / a) / (np.pi * np.pi * x * x))
    return result

def oldlanczos_upscale(image, new_height:int, new_width:int, window_size:int=4, a:int=3):
    """
    Lanczos upscaling algorithm with customizable sliding window.
    
    Parameters:
        image (array_like): Input image.
        new_height (int): Desired height of the upscaled image.
        new_width (int): Desired width of the upscaled image.
        window_size (int): Size of the sliding window (default is 4).
        a (int): Lanczos kernel parameter (default is 3).
        
    Returns:
        array_like: Upscaled image.
    """
    new_height = int(math.floor(new_height))
    new_width = int(math.floor(new_width))
    def lanczos_window(x):
        return lanczos_kernel(x, a) * np.sinc(x / window_size)
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    h, w = image_array.shape[:2]
    num_channels = image_array.shape[2]
    
    # Initialize empty array for the upscaled image
    upscaled_image = np.zeros((new_height, new_width, num_channels), dtype=np.float32)
    
    # Generate indices for the upscaled image
    x_indices = np.arange(new_width) / (w / new_width)
    y_indices = np.arange(new_height) / (h / new_height)
    
    # Loop through each channel of the image
    #for channel in range(num_channels):
    for y in range(new_height):
        for x in range(new_width):
            # Compute the range of indices for the sliding window
            x_min = max(int(np.floor(x - window_size / 2)), 0)
            x_max = min(int(np.ceil(x + window_size / 2)), w)
            y_min = max(int(np.floor(y - window_size / 2)), 0)
            y_max = min(int(np.ceil(y + window_size / 2)), h)
            
            # Extract the local window around the current pixel
            window = image_array[y_min:y_max, x_min:x_max]
            
            # Compute the weights for the Lanczos kernel
            x_weights = lanczos_window((x - np.arange(x_min, x_max)) * (w / new_width))
            y_weights = lanczos_window((y - np.arange(y_min, y_max)) * (h / new_height))
            
            # Perform the Lanczos interpolation
            interpolated_value = np.sum(window * y_weights[:, np.newaxis] * x_weights, axis=0)
            
            # Assign the interpolated value to the corresponding slice of upscaled_image
            upscaled_image[y, x] = interpolated_value
    
    return np.clip(upscaled_image, 0, 255).astype(np.uint8)



# Example usage:
# Assuming you have an image 'input_image' and you want to upscale it to a specific size (new_height, new_width):
# scaled_image = lanczos_upscale(input_image, new_height, new_width)



def lanczos_upscale(img, new_width, new_height, window_size:int=4, kernelNum=3):
    width, height = img.size
    img_data = np.array(img)

    # Create an empty array for the new image
    new_img_data = np.zeros((new_height, new_width, img_data.shape[2]), dtype=np.uint8)

    # Resampling in horizontal direction
    scale_x = width / new_width
    for x in range(new_width):
        src_x = x * scale_x
        x_left = int(src_x - kernelNum) + 1
        x_right = int(src_x + kernelNum) + 1

        for y in range(height):
            # Apply Lanczos kernel
            sum_weights = 0
            sum_values = np.zeros(img_data.shape[2])
            for i in range(x_left, x_right):
                if i >= 0 and i < width:
                    weight = lanczos_kernel((src_x - i) / kernelNum)
                    sum_weights += weight
                    sum_values += img_data[y, i] * weight
            new_img_data[y, x] = np.clip(sum_values / sum_weights, 0, 255)

    # Resampling in vertical direction
    scale_y = height / new_height
    for y in range(new_height):
        src_y = y * scale_y
        y_top = int(src_y - kernelNum) + 1
        y_bottom = int(src_y + kernelNum) + 1

        for x in range(new_width):
            # Apply Lanczos kernel
            sum_weights = 0
            sum_values = np.zeros(img_data.shape[2])
            for j in range(y_top, y_bottom):
                if j >= 0 and j < height:
                    weight = lanczos_kernel((src_y - j) / kernelNum)
                    sum_weights += weight
                    sum_values += new_img_data[j, x] * weight
            new_img_data[y, x] = np.clip(sum_values / sum_weights, 0, 255)

    return Image.fromarray(new_img_data)