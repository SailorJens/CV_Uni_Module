import unittest
from ImgProc import ImageProcessor, ImageProcessorError



class TestImageProcessing(unittest.TestCase):  

    def test_initialise_with_existing_image(self):
        imageProcessor = ImageProcessor("5.jpg")
        self.assertIsNotNone(imageProcessor.image)

    def test_initialise_with_non_existing_fil(self):
        with self.assertRaises(ImageProcessorError) as context:
            ImageProcessor("non_existing_file.jpg")
        self.assertEqual(str(context.exception), "Could not load image from path: non_existing_file.jpg")

    def test_initialise_with_existing_file_no_image(self):
        with self.assertRaises(ImageProcessorError) as context:
            ImageProcessor("test.txt")
        self.assertEqual(str(context.exception), "Could not load image from path: test.txt")

    def test_initialise_with_wrong_parameter_type(self):
        with self.assertRaises(ImageProcessorError) as context:
            ImageProcessor(5)
        self.assertEqual(str(context.exception), "Could not load image from path: 5")

    def test_resize_image_with_new_width_height(self):
        imageProcessor = ImageProcessor("5.jpg")
        original_shape = imageProcessor.image.shape
        new_width, new_height = 100, 100
        imageProcessor.resize(new_width, new_height)
        resized_shape = imageProcessor.image.shape
        self.assertEqual(resized_shape[1], new_width)
        self.assertEqual(resized_shape[0], new_height)
        self.assertNotEqual(original_shape, resized_shape)

    def test_resize_image_with_invalid_dimensions(self):
        imageProcessor = ImageProcessor("5.jpg")
        with self.assertRaises(ImageProcessorError) as context:
            imageProcessor.resize("k", 100)
        self.assertEqual(str(context.exception), "Width and height must be positive integers.")
        with self.assertRaises(ImageProcessorError) as context:
            imageProcessor.resize(-50, 100)
        self.assertEqual(str(context.exception), "Width and height must be positive integers.")

    def test_resize_image_with_no_arguments(self):
        imageProcessor = ImageProcessor("5.jpg")
        # No error when no arguments are provided = nothing happens
        imageProcessor.resize()

    def test_resize_argument_with_scale(self):
        factor = 2.0
        imageProcessor = ImageProcessor("5.jpg")
        original_shape = imageProcessor.image.shape
        imageProcessor.resize(scale=factor)
        resized_shape = imageProcessor.image.shape
        self.assertEqual(original_shape[0]*factor, resized_shape[0])
        self.assertEqual(original_shape[1]*factor, resized_shape[1])

    def test_resize_argument_with_invalid_scale(self):
        imageProcessor = ImageProcessor("5.jpg")
        with self.assertRaises(ImageProcessorError) as context:
            imageProcessor.resize(scale="k")
        self.assertEqual(str(context.exception), "Scale must be a positive number.")
        with self.assertRaises(ImageProcessorError) as context:
            imageProcessor.resize(scale=-2.0)
        self.assertEqual(str(context.exception), "Scale must be a positive number.")

    def test_resize_new_dimensions_priority_over_scale(self):
        factor = 2.0
        new_width, new_height = 50, 50
        imageProcessor = ImageProcessor("5.jpg")
        original_shape = imageProcessor.image.shape
        imageProcessor.resize(new_width=new_width, new_height=new_height, scale=factor)
        resized_shape = imageProcessor.image.shape
        self.assertEqual(resized_shape[1], new_width)
        self.assertEqual(resized_shape[0], new_height)
        self.assertNotEqual(original_shape, resized_shape)
    
    def test_rotate_image(self):
        pass

if __name__ == "__main__":
    unittest.main()
