import face_recognition
import os
import pickle

known_face_encodings = []
known_face_names = []

dataset_path = 'dataset'

# Loop through each folder in the dataset
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    # Loop through each image in the person's folder
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Get the encoding of the face
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            # Save the encoding and the person's name
            known_face_encodings.append(encodings[0])
            known_face_names.append(person)

# Save the face encodings and names to a file
with open('encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Encodings saved!")
