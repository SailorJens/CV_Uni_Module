import numpy as np
import matplotlib.pyplot as plt
import cv2

# Resize an image to the given width and height
# Returns the resized image as a numpy array
def resize_image(image: np.ndarray, width: int = None, height: int = None, scale: float = None) -> np.ndarray:
    # Retrieve original height and width from shape. 
    original_height = image.shape[0]
    original_width = image.shape[1]
    # determine the new values for both height and width, prioritise scale if given over height/width (this is arbitrary)
    if scale is not None and scale > 0:
        new_height = int(original_height * scale) # Making sure to have ints
        new_width = int(original_width * scale)
    elif width is not None and height is not None:
        new_height = height
        new_width = width
    else:
        # make it error friendly. Don't raise an exception if no values are given, simply return the original.
        # There's still going to be an exception if values are the wrong Type (Type Error)
        return image
    # return original if scale is 1
    if new_height == original_height and new_width == original_width:
        return image
    
    # Determine the appropriate interpolation method
    # cf. https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/
    # inter_linear is the default
    # it is in particular useful for upscaling, because it determines values for the "new pixels".
    # A "new pixel" is not mapped to a pixel in the orginal image, but rather to a fractional position between "original pixels".
    # The value for the new pixel is the average between the four closest original pixels
    interpolation_method = cv2.INTER_LINEAR
    # When downscaling in both (!) dimensions, use INTER_AREA
    # Inter Area averages a rectangle of pixels (size depends on scaling factor) to calculate all pixels in the result image
    # However, that does not work when one dimension is being upscaled, because there wouldn't be a definable rectangle. 
    # Hence, if at least one dimension upscales, the linear approach is preferred.
    # Linear is always possible because there are always 4 neighbours.  
    if new_height < original_height and new_width < original_width:
        interpolation_method = cv2.INTER_AREA
    
    # Do the actual resizing
    resized_image = cv2.resize(image, (new_width, new_height))
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


# Examplatory usage
def main():
    image_path = '5.jpg'
        
    try:
        image = load_rgb_image_from_path(image_path=image_path)
        # Display the image using matplotlib
    # Catching file errors
    except FileNotFoundError:
        print("File does not exist.")
        return
    except PermissionError:
        print("You don't have permission to read this file.")
        return
    # Catching OpenCV errors
    except cv2.error as e:
        print("OpenCV error: ", e)
        return
    
    try:
        image = resize_image(image=image, scale=0.1)
    except TypeError:
        print("Type Error")
        return
    except cv2.error as e:
        print("OpenCV error: ", e)
        return
    
    
    plt.imshow(image)
    # Need show command to open pop-up window when not in Jupyter Notebook
    plt.show()


# Access point for example use
if __name__ == "__main__":
    main()
    
    