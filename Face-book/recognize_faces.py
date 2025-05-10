import face_recognition
import cv2
import pickle

# Load the known face encodings and names
with open("encodings.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop over each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default name if no match found

        # If a match is found, use the name of the matched person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the video window
video_capture.release()
cv2.destroyAllWindows()
