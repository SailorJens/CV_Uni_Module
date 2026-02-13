import cv2
import time



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def draw_faces(frame, faces):
    for face in faces:
        x1, y1, x2, y2 = face[0], face[1], face[0] + face[2], face[1] + face[3]
        cv2.rectangle(
            frame, 
            (x1, y1), 
            (x2, y2), 
            (0, 0, 255), 
            2
            )  # Red box with thickness 2
    return frame



def find_faces(frame, scale_factor):
    mod_frame = frame.copy()
    # Resize frame for faster processing
    mod_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    faces = face_cascade.detectMultiScale(mod_frame, scaleFactor=1.1, minNeighbors=5)   
    # Scale face coordinates back to original frame size
    scaled_faces = []
    for (x, y, w, h) in faces:
        scaled_faces.append((int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)))
    return scaled_faces



def start_feed(video_path, target_resolution=640, frame_skip_rate=5):
    # Create VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Run the face detection on 640 pixels width to get real time speed. The height will be scaled accordingly to maintain aspect ratio.
    frame_scale_factor = min(1, target_resolution/width)
    print("Video FPS:", video_fps)
    print(f"Video resolution: {width} x {height}")
    print(f"Scaling video to {min(target_resolution, width)} pixels width for real-time processing. Factor: ", frame_scale_factor)
    print("Press 'q' to quit.")

    # Loop through video frames or live feed
    frame_id = 0
    start_time = time.time()

    # To buffer faces to avoid blinking when reducing frame rate
    faces = []


    
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame not read correctly, break loop
        if not ret:
            break

        # Process frame
        if frame_id % frame_skip_rate == 0:  
            faces = find_faces(frame, frame_scale_factor)

        frame = draw_faces(frame, faces)

        # Display the frame
        cv2.imshow("AVI Video", frame)

        #Press 'q' to quit
        # Minimising the wait time to achieve real-time performance
        # waitKey is required to be able to display video frames with imshow
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {frame_id} frames in {elapsed_time:.2f} seconds, average FPS: {frame_id/elapsed_time:.2f}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Path to your .mov file
    video_path = "data/IMG_0992.mov"
    start_feed(video_path, target_resolution=1200, frame_skip_rate=5)