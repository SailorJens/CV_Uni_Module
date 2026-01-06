import numpy as np
import matplotlib.pyplot as plt
import cv2

# Resize an image to the given width and height
# Returns the resized image as a numpy array
def resize_image(width: int, height: int, image: np.ndarray) -> np.ndarray:
    """
    Resize an image to the given width and height.

    Args:
        width (int): Desired width.
        height (int): Desired height.
        image (np.ndarray): Input image.
    """
    # Notably width comes first in cv2.resize
    # cf. moukthika. (2025, March 10). Resizing and rescaling images with OpenCV. OpenCV.
    # https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# Load an image and convert it to RGB format
# Returns the image as a numpy array
def load_rgb_image_from_path(image_path: str) -> np.ndarray:
    """
    Load an image and convert to RGB format.
    
    Args:
        image_path (str): Path to the image file.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Access point for example use
if __name__ == "__main__":
    
    image_path = 'apple.jpg'
    
    try:
        image = load_rgb_image_from_path(image_path=image_path)

        # Display the image using matplotlib
        plt.imshow(image)
        # Need show command to open pop-up window when not in Jupyter Notebook
        plt.show()

    # Catching file errors
    except FileNotFoundError:
      print("File does not exist.")
    except PermissionError:
        print("You don't have permission to read this file.")
    # Catching OpenCV errors
    except cv2.error as e:
     print("OpenCV error: ", e)