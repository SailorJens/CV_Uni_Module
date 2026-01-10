import matplotlib.pyplot as plt
from ImgProc import ImageProcessor, ImageProcessorError


def app():
    image_path = './fruits.jpg'
    try:
        processor = ImageProcessor(image_path)
        # processor.resize(width=600, height=600)
        # processor.resize(scale=0.5)
        # processor.rotate(angle=45, expand=True, bg_color_rgb=(50, 50, 50))
        # processor.rotate(angle=-30, expand=False, bg_color_rgb=(100, 100, 100))
        # processor.crop(x1=375, y1=1350, x2=975, y2=1950)
        processor.crop(x1=375, y1=1350, width=600, height=600)
        # processor.draw_circle(center=(300, 300), radius=100, color_rgb=(255, 255, 0), thickness=5)
        # processor.draw_circle(center=(300, 300), radius=100)
        processor.draw_circle(center=(300, 300), radius=100, thickness=-1)  # filled circle
        processor.draw_rectangle(top_left=(50, 50), bottom_right=(200, 200), color_rgb=(0, 255, 255), thickness=5)
        processor.annotate(text="Hello World!", position=(50, 300), font_scale=2, color_rgb=(255, 0, 255), thickness=3)
    except ImageProcessorError as e:
        print(f"Error: {e}")
        return
    plt.imshow(processor.image)
    plt.show()


# Access point for example use
if __name__ == "__main__":
    app()
    
