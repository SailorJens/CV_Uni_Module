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
            # Create label text
        label = f"x:{face[0]} y:{face[1]} w:{face[2]} h:{face[3]}"
        
        # Position text slightly above box
        text_x = face[0]
        text_y = face[1] - 10 if face[1] - 10 > 10 else face[1]  + 20
        
        # Draw text
        cv2.putText(frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA)
    return frame



def find_faces(frame, scale_factor):
    # Convert frame to grayscale for face detection
    prepped_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize frame for faster processing by scale_factor
    prepped_frame = cv2.resize(prepped_frame, (0, 0), fx=scale_factor, fy=scale_factor)
    # Detect faces in the prepped frame
    # Keep ScaleFactor low to ensure not to miss faces (reduce "false negatives")
    # But increase minNeighbors to reduce false positives
    faces = face_cascade.detectMultiScale(
        prepped_frame, 
        scaleFactor=1.1, 
        minNeighbors=5
        )   
    # Scale face coordinates back to original frame size
    scaled_faces = []
    for (x, y, w, h) in faces:
        scaled_faces.append(
            (int(x / scale_factor), 
             int(y / scale_factor), 
             int(w / scale_factor), 
             int(h / scale_factor)))
    return scaled_faces


def calculate_IoU(boxA, boxB):
    # boxA and boxB are in the format (x, y, w, h)

    # first calculate the coordinates of the intersection rectangle
    i_x_top_left = max(boxA[0], boxB[0])
    i_y_top_left = max(boxA[1], boxB[1])
    i_x_bottom_right = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    i_y_bottom_right = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    # If the rectangles do not overlap, i_x_bottom_right - i_x_top_left 
    # or i_y_bottom_right - i_y_top_left will be negative, 
    # so we take max with 0 to ensure non-negative area
    i_width = max(0, i_x_bottom_right - i_x_top_left)
    i_height = max(0, i_y_bottom_right - i_y_top_left)
    # If no overlap, the intersection area will be 0
    i_area = i_width * i_height

    # Compute the area of both boxes based on width * height
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    # The union area is the sum of both areas minus the intersection area (to avoid counting the intersection twice)
    union_area = boxA_area + boxB_area - i_area

    # Avoid division by zero: IoU is 0 if union_area is 0
    # (in case both boxes are of zero area, which is a degenerate case)
    if union_area == 0:
        return 0.0  

    # Compute the intersection over union by taking the intersection area 
    # and dividing it by the sum of prediction + ground-truth areas - the interesection area
    return i_area / float(union_area)


# Remove faces that only appear in one frame and not in the next frame.
# 
def identify_valid_faces(face_tracker, new_faces):
    new_face_tracker = []
    # For each new face, check if it matches with any tracked face from previous frames using IoU.
    # If it doesn't match with any tracked face, add it to the tracker with count 1. If it matches, update the count for that tracked face.
    # Discard any tracked faces that have no match with the new batch of faces (i.e. they disappeared) 
    for face in new_faces:
        for tracked_face, count in face_tracker:
            iou = calculate_IoU(face, tracked_face)
            # If IoU is greater than 0.2, consider it the same face
            # IoU can't be too high when skipping frames, or else movements are too fast to be tracked
            if iou > 0.2:  
                new_face_tracker.append((face, count+1))
                break
        else:  # If no match found, add new face to tracker with count 1
            new_face_tracker.append((face, 1))
    return new_face_tracker


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
    #faces = []
    face_tracker = []

    
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame not read correctly, break loop
        if not ret:
            break

        # Find faces every frame_skip_rate frames to improve permformance
        if frame_id % frame_skip_rate == 0:  
            faces = find_faces(frame, frame_scale_factor)
            face_tracker = identify_valid_faces(face_tracker, faces)
            faces = [face for (face, count) in face_tracker if count > 2]



        # Still, draw boxes on every frame to avoid blinking effect when skipping face detection
        frame = draw_faces(frame, faces)
        # Print Frame ID and number of faces detected on the frame
        cv2.putText(frame,
            f"Frame: {frame_id} * Number of Faces: {len(faces)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Video", frame)

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
    start_feed(video_path, target_resolution=800, frame_skip_rate=1)