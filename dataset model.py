import os
import cv2
import numpy as np
import face_recognition
import pickle

# Define paths
faces_dir = 'static/faces'
encodings_path = 'static/face_encodings.pkl'
n_augmentations = 10

# Function to perform data augmentation
def augment_image(image):
    augmented_images = []

    for _ in range(n_augmentations):
        # Random transformations
        rows, cols, _ = image.shape

        # Random rotation
        angle = np.random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))

        # Random translation
        tx = np.random.uniform(-0.2 * cols, 0.2 * cols)
        ty = np.random.uniform(-0.2 * rows, 0.2 * rows)
        T = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(rotated, T, (cols, rows))

        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        scaled = cv2.resize(translated, None, fx=scale, fy=scale)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            scaled = cv2.flip(scaled, 1)

        augmented_images.append(scaled)

    return augmented_images

# Function to encode faces and save to pickle file
def encode_faces_to_pickle():
    known_face_encodings = []
    known_face_names = []

    for user in os.listdir(faces_dir):
        user_dir = os.path.join(faces_dir, user)
        if os.path.isdir(user_dir):  # Ensure it's a directory
            for img_name in os.listdir(user_dir):
                img_path = os.path.join(user_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(img)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(user)
    
    # Save encodings to pickle file
    with open(encodings_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

# Function to apply augmentation to all existing users
def augment_all_users():
    for user in os.listdir(faces_dir):
        user_dir = os.path.join(faces_dir, user)
        if os.path.isdir(user_dir):
            for img_name in os.listdir(user_dir):
                img_path = os.path.join(user_dir, img_name)
                image = cv2.imread(img_path)
                augmented_images = augment_image(image)

                # Save augmented images
                for i, aug_img in enumerate(augmented_images):
                    aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                    cv2.imwrite(os.path.join(user_dir, aug_img_name), aug_img)

# Apply augmentation to all users and re-encode faces
augment_all_users()
known_face_encodings, known_face_names = encode_faces_to_pickle()

print("Data augmentation and encoding complete for all users.")
