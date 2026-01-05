import numpy as np
import matplotlib.pyplot as plt
import cv2




def load_rgb_image_from_path(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Access point for example use
if __name__ == "__main__":
    
    image_path = 'apple.jpg'
    
    try:
        image = load_rgb_image_from_path(image_path=image_path)
    
        plt.imshow(image)
        plt.show()

    # Catching file errors
    except FileNotFoundError:
      print("File does not exist.")
    except PermissionError:
        print("You don't have permission to read this file.")
    # Catching OpenCV errors
    except cv2.error as e:
     print("OpenCV error: ", e)