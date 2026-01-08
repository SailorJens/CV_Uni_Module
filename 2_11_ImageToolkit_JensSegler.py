import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


# Rotate an image
# Select whether it should be resized, and what colour the background should have
def rotate_image(
        image: np.ndarray, 
        angle: float = 0, 
        resize_canvas: bool = True, 
        bg_color_rgb : tuple[int, int, int] = (0, 0, 0)
        ) -> np.ndarray: 
    """
    Rotate an image

    Args:
        image (np.ndarray): the image
        angle (float): rotation angle in degrees of a 360 degree circle
        resize_canvas (bool): True if original image size is kept and canvas changes, False if cancas stays and image is scaled
        bg_color_rgb tuple[int, int, int]: Colour for new backgrouns pixels in RGB format
    """
    # Retrieve height and width of the image from the shape
    (h, w) = image.shape[:2]
    # calculate the center point for the rotation as integer
    # here: width first, as that's what cv2 accepts as pont format(x, y)
    # cf. https://opencv.org/blog/image-rotation-and-translation-using-opencv/
    center = (w // 2, h // 2)
    scale = 1
    new_canvas_size = (w, h)
    # convert angle to radians as input for trigonometric functions
    theta = math.radians(angle) 

    if resize_canvas:
        # Calculate the new image size
        # cf. https://math.stackexchange.com/questions/430763/size-of-new-box-rotated-and-the-rescaled
        # 𝑤′=𝑤cos𝜃+ℎsin𝜃
        # ℎ′=𝑤sin𝜃+ℎcos𝜃
        new_w = abs(w * math.cos(theta)) + abs(h * math.sin(theta))
        new_h = abs(w * math.sin(theta)) + abs(h * math.cos(theta))
        new_canvas_size = (int(round(new_w)), int(round(new_h)))
    else:
        # cf. https://stackoverflow.com/questions/33866535/how-to-scale-a-rotated-rectangle-to-always-fit-another-rectangle
        wr = abs(w * math.cos(theta)) + abs(h * math.sin(theta))
        hr = abs(w * math.sin(theta)) + abs(h * math.cos(theta))
        scale = min(w / wr, h / hr)

        # This is still wrong, I need to also center the scaled image
        # wr_scaled = wr * scale
        # hr_scaled = hr * scale
        # tx = (w - wr_scaled) / 2
        # ty = (h - hr_scaled) / 2
        # rotation_matrix[0, 2] += tx
        # rotation_matrix[1, 2] += ty

    # get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # warp (apply) the affine transformation
    rotated_image = cv2.warpAffine(
        image, 
        rotation_matrix, 
        new_canvas_size, 
        borderValue=(bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0])
        )
    
    return rotated_image



# ROI Extraction
# Make the coordinates explicit to avoid misunderstandings
def extract_region_of_interest(image: np.ndarray, topleft_x: int, topleft_y: int, bottomright_x: int, bottomright_y: int) -> np.ndarray:
    """
    Extract a region of interest from an image.
    
    The ROI is define by the top left point and the bottom right point. 

    Args:
        image (np.ndarray). The image.
        topleft_x, topleft_y, bottomright_x, bottomright_y (int): Co-ordinates of rectangle points

    """
    roi = image[topleft_y:bottomright_y, topleft_x:bottomright_x]
    return roi



# Resize an image to the given width and height
# Returns the resized image as a numpy array
def resize_image(image: np.ndarray, width: int = None, height: int = None, scale: float = None) -> np.ndarray:
    """
    Resize an image.
    Scales by either providing height and width OR scale. 

    If all are given, scale is used. 
    If none are given, the original is returned. 

    Can raise Type Error and cv2.error.
    
    Args:
        image (np.ndarray). The image.
        width (int). The new width.
        height (int). The new width.

    """
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

    Can raise FileNotFoundError, PermissionError and cv2.error.
    
    Args:
        image_path (str): Path to the image file.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Examplatory usage
def main():
    image_path = 'fruits.jpg'
        
    # LOAD IMAGE
    try:
        image = load_rgb_image_from_path(image_path=image_path)
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
    
    # RESIZE IMAGE
    try:
        image_resized = resize_image(image=image, scale=0.5)
        pass
    except TypeError:
        print("Type Error")
        return
    except cv2.error as e:
        print("OpenCV error: ", e)
        return
    

    # ROTATE IMAGE
    image_rotated = rotate_image(
        image=image,
        angle=37.6,
        resize_canvas=False,
        bg_color_rgb=(18,127,56)
    )
    
    

    # EXTRACT REGION OF INTEREST
     
    image_roi = extract_region_of_interest(
        image=image, 
        topleft_x=375, 
        topleft_y=1350, 
        bottomright_x=975, 
        bottomright_y=1950
        )
    
    fix, ax = plt.subplots(4,1)
    ax[0].imshow(image)
    ax[1].imshow(image_resized)
    ax[2].imshow(image_rotated)
    ax[3].imshow(image_roi)

    # Need show command to open pop-up window when not in Jupyter Notebook
    plt.show()


# Access point for example use
if __name__ == "__main__":
    main()
    
    