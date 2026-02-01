import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(image):
    boxes = face_cascade(image)
    return boxes



# 1. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# 2. Check if camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Get the width and height
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Webcam resolution: {width} x {height}")

# Faces need to be at least 1/10th of image width
min_face_size = (width // 10, width // 10)

while True:
    # 3. Read a frame
    ret, frame = cap.read()
    if not ret:
        break  # Stop if frame not read correctly

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)

    # Enumerate faces to label them
    for face in faces:
      x1, y1, x2, y2 = face[0], face[1], face[0] + face[2], face[1] + face[3]
      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 6)  # Red box with thickness 6

    

    # 6. Show the frame
    cv2.imshow('Live Face Detection', frame)

    # 7. Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
