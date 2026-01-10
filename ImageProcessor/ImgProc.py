
import numpy as np
import cv2

class ImageProcessorError(Exception):
    pass



class ImageProcessor:

    def __init__(self, image_path: str):
        """
        Initializes the ImageProcessor with an image path.
        Convert to RGB format after loading.
        
        Args:
            image_path (str): Path to the image file
        """
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ImageProcessorError(f"Could not load image from path: {image_path}")
        except cv2.error as e:
            raise ImageProcessorError(f"Could not load image from path: {image_path}") from e
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def resize(self, new_width: int | None = None, new_height: int | None = None, scale: float | None = None):
        """
        Resizes the image to the specified width and height.
        
        Args:
            new_width (int): The desired width of the image
            new_height (int): The desired height of the image
            scale (float): Scaling factor to resize the image
        """
        if new_width is not None and new_height is not None:
            try:
                new_width = int(new_width)
                new_height = int(new_height)
            except ValueError:
                raise ImageProcessorError("Width and height must be positive integers.")
            if new_width <= 0 or new_height <= 0:
                raise ImageProcessorError("Width and height must be positive integers.")
            self.image = cv2.resize(self.image, (new_width, new_height))    
        elif scale is not None:
            try:
                scale = float(scale)
            except ValueError:
                raise ImageProcessorError("Scale must be a positive number.")
            if scale <= 0:
                raise ImageProcessorError("Scale must be a positive number.")
            new_width = int(self.image.shape[1] * scale)
            new_height = int(self.image.shape[0] * scale)
            self.image = cv2.resize(self.image, (new_width, new_height))

    def rotate(
        self,
        angle: float = 0,
        resize_canvas: bool = True,
        bg_color_rgb: tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Rotate an image

        Args:
            angle (float): rotation angle in degrees of a 360 degree circle
            resize_canvas (bool):
                True if original image size is kept and canvas changes,
                False if canvas stays and image is scaled to fit
            bg_color_rgb (tuple[int, int, int]): Colour for new background pixels (RGB)
        """
        if not isinstance(angle, (int, float)):
            raise ImageProcessorError("angle must be a number.")
        if not isinstance(resize_canvas, bool):
            raise ImageProcessorError("resize_canvas must be a boolean.")
        if not isinstance(bg_color_rgb, tuple):
            raise ImageProcessorError("bg_color_rgb must be a tuple")
        if len(bg_color_rgb) != 3:
            raise ImageProcessorError("bg_color_rgb must be a 3-tuple")
        for c in bg_color_rgb:
            if not isinstance(c, int):
                raise ImageProcessorError("RGB values must be integers")
            if not (0 <= c <= 255):
                raise ImageProcessorError("RGB values must be in range 0..255")
        # Get image dimensions
        (h, w) = self.image.shape[:2]
        # Calculate the center of the image
        (cX, cY) = (w // 2, h // 2)

        # Generate rotation matrix
        # Note that OpenCV uses a clockwise angle convention, hence the negative sign
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

        # Retrieve the sine and cosine from the rotation matrix for calculating teh new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Calculate dimensions of the box to fit the rotated image
        # cf. https://math.stackexchange.com/questions/430763/size-of-new-box-rotated-and-the-rescaled
        rotW = (w * cos) + (h * sin)
        rotH = (w * sin) + (h * cos)


        if resize_canvas:
            # new dimensions must be integers            
            nW = int(rotW)
            nH = int(rotH)

            # Adjust the rotation matrix to take into account translation so the image is centered
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # Perform the actual rotation and translation and return the image
            self.image = cv2.warpAffine(
                self.image,
                M,
                (nW, nH),
                borderValue=bg_color_rgb
            )
        else:
            # Calculate the scale factor to fit the rotated image within the original dimensions
            # Choose the smaller scaling factor to ensure the entire image fits
            # note that when rotating the image will alwyas be doewnscaled in this mode
            scale = min(w / rotW, h / rotH)

            # Update the rotation matrix to include the scaling
            M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
            # Perform the actual rotation and return the image
            self.image = cv2.warpAffine(
                self.image,
                M,
                (w, h),
                borderValue=bg_color_rgb
            )

    def crop(self, x1: int, y1: int, width: int | None = None, height: int | None = None, x2: int | None = None, y2: int  | None = None):
        """
        Crop the image to the specified rectangle.

        Args:
            x1 (int): The x-coordinate of the top-left corner
            y1 (int): The y-coordinate of the top-left corner
            width (int, optional): The width of the crop rectangle
            height (int, optional): The height of the crop rectangle
            x2 (int, optional): The x-coordinate of the bottom-right corner
            y2 (int, optional): The y-coordinate of the bottom-right corner
        """
        if not isinstance(x1, int) or not isinstance(y1, int) or x1 < 0 or y1 < 0:
            raise ImageProcessorError("x1 and y1 must be non-negative integers.")
        if width is not None and (not isinstance(width, int) or width < 0):
            raise ImageProcessorError("width must be a non-negative integer.")
        if height is not None and (not isinstance(height, int) or height < 0):
            raise ImageProcessorError("height must be a non-negative integer.")
        if x2 is not None and (not isinstance(x2, int) or x2 <= x1):
            raise ImageProcessorError("x2 must be an integer greater than x1.")
        if y2 is not None and (not isinstance(y2, int) or y2 <= y1):
            raise ImageProcessorError("y2 must be an integer greater than y1.")
        if width is not None and height is not None:
            x2 = x1 + width
            y2 = y1 + height
        elif x2 is not None and y2 is not None:
            pass
        else:
            raise ImageProcessorError("Either width and height or x2 and y2 must be provided for cropping.")
        
        self.image = self.image[y1:y2, x1:x2]

    def draw_circle(self, center: tuple[int, int], radius: int, color_rgb: tuple[int, int, int] = (255, 255, 255), thickness: int = 2):
        """
        Draw a circle on the image.

        Args:
            center (tuple[int, int]): The (x, y) coordinates of the circle's center
            radius (int): The radius of the circle
            color_rgb (tuple[int, int, int]): The color of the circle in RGB format
            thickness (int): The thickness of the circle's outline. Use -1 for filled circle.
        """
        if not isinstance(center, tuple) or len(center) != 2:
            raise ImageProcessorError("center must be a tuple of two integers.")
        if not all(isinstance(c, int) and c >= 0 for c in center):
            raise ImageProcessorError("center coordinates must be non-negative integers.")
        if not isinstance(radius, int) or radius <= 0:
            raise ImageProcessorError("radius must be a positive integer.")
        if not isinstance(color_rgb, tuple) or len(color_rgb) != 3:
            raise ImageProcessorError("color_rgb must be a tuple of three integers.")
        for c in color_rgb:
            if not isinstance(c, int) or not (0 <= c <= 255):
                raise ImageProcessorError("RGB values must be integers in range 0..255.")
        if not isinstance(thickness, int):
            raise ImageProcessorError("thickness must be an integer.")

        cv2.circle(self.image, center, radius, color_rgb, thickness)

    def draw_rectangle(self, top_left: tuple[int, int], bottom_right: tuple[int, int], color_rgb: tuple[int, int, int] = (255, 255, 255), thickness: int = 2):
        """
        Draw a rectangle on the image.

        Args:
            top_left (tuple[int, int]): The (x, y) coordinates of the top-left corner
            bottom_right (tuple[int, int]): The (x, y) coordinates of the bottom-right corner
            color_rgb (tuple[int, int, int]): The color of the rectangle in RGB format
            thickness (int): The thickness of the rectangle's outline. Use -1 for filled rectangle.
        """
        if not isinstance(top_left, tuple) or len(top_left) != 2:
            raise ImageProcessorError("top_left must be a tuple of two integers.")
        if not isinstance(bottom_right, tuple) or len(bottom_right) != 2:
            raise ImageProcessorError("bottom_right must be a tuple of two integers.")
        if not all(isinstance(c, int) and c >= 0 for c in top_left + bottom_right):
            raise ImageProcessorError("corner coordinates must be non-negative integers.")
        if not isinstance(color_rgb, tuple) or len(color_rgb) != 3:
            raise ImageProcessorError("color_rgb must be a tuple of three integers.")
        for c in color_rgb:
            if not isinstance(c, int) or not (0 <= c <= 255):
                raise ImageProcessorError("RGB values must be integers in range 0..255.")
        if not isinstance(thickness, int):
            raise ImageProcessorError("thickness must be an integer.")

        cv2.rectangle(self.image, top_left, bottom_right, color_rgb, thickness)

    def annotate(self, text: str, position: tuple[int, int], font_scale: float = 1.0, color_rgb: tuple[int, int, int] = (255, 255, 255), thickness: int = 2):
        """
        Annotate the image with text.

        Args:
            text (str): The text to annotate
            position (tuple[int, int]): The (x, y) coordinates for the bottom-left corner of the text
            font_scale (float): Scale factor for the text size
            color_rgb (tuple[int, int, int]): The color of the text in RGB format
            thickness (int): The thickness of the text
        """
        if not isinstance(text, str):
            raise ImageProcessorError("text must be a string.")
        if not isinstance(position, tuple) or len(position) != 2:
            raise ImageProcessorError("position must be a tuple of two integers.")
        if not all(isinstance(c, int) and c >= 0 for c in position):
            raise ImageProcessorError("position coordinates must be non-negative integers.")
        if not isinstance(font_scale, (int, float)) or font_scale <= 0:
            raise ImageProcessorError("font_scale must be a positive number.")
        if not isinstance(color_rgb, tuple) or len(color_rgb) != 3:
            raise ImageProcessorError("color_rgb must be a tuple of three integers.")
        for c in color_rgb:
            if not isinstance(c, int) or not (0 <= c <= 255):
                raise ImageProcessorError("RGB values must be integers in range 0..255.")
        if not isinstance(thickness, int):
            raise ImageProcessorError("thickness must be an integer.")

        cv2.putText(self.image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_rgb, thickness)