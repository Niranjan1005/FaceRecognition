import cv2
import numpy as np
from PIL import Image
import os
import argparse
"""
Author: NIRANJAN V
Date: January 26, 2024

Description:
    This script implements a face recognition system in Python. It comprises three main components: face dataset creation,
    recognizer training, and real-time face recognition. The system uses the OpenCV library for computer vision tasks and
    the PIL library for image processing.

Components:
1. Face Dataset Creation:
    Class: CreateDataset
        - Captures facial images and organizes them into a dataset.
        - Utilizes Haar Cascade for face detection.
        - Allows customization of face ID, the number of images per ID, and other parameters.

2. Recognizer Training:
    Class: FaceTrainer
        - Trains a LBPH Face Recognizer using the dataset.
        - Extracts face samples and corresponding IDs from images.
        - Provides methods for loading and saving the trained recognizer.

3. Real-time Face Recognition:
    Class: RealTimeFaceRecognizer
        - Performs real-time face recognition using the trained recognizer.
        - Displays recognized faces with bounding boxes and confidence levels.
        - Allows customization of dataset paths, recognizer paths, and cascade paths.

Usage:
    - Run the script with optional command-line arguments for dataset path, trainer path, cascade path, etc.
    - Use the '--create_dataset' flag to initiate the face dataset creation process.

Example:
    $ python FaceDetector.py --dataset_path dataset_folder --trainer_path trainer/trainer.yml --create_dataset

Note: Please make sure to install the required libraries (OpenCV and PIL) before running the script.
"""


class CreateDataset:
    def __init__(self, face_cascade_path='haarcascade_frontalface_default.xml', face_id=0, face_images_per_id=30, face_counter=0, dataset_folder='dataset'):
        # Initialize the CreateDataset object with default or provided parameters
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.face_id = face_id
        self.face_images_per_id = face_images_per_id
        self.face_counter = face_counter
        self.dataset_folder = dataset_folder
        os.makedirs(self.dataset_folder, exist_ok=True)

    def detect_faces(self, img, gray):
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces

    def draw_rectangles_and_save_images(self, img, gray, faces, subfolder):
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.face_counter += 1

            # Save the captured image into the specified subfolder with folder name
            face_filename = f"{subfolder}.{self.face_id}.{self.face_counter}.jpg"
            subfolder_path = os.path.join(self.dataset_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            face_filepath = os.path.join(subfolder_path, face_filename)
            cv2.imwrite(face_filepath, gray[y:y + h, x:x + w])

            # Display the image with the rectangle
            cv2.imshow('image', img)

    def start_capturing(self):
        cap = cv2.VideoCapture(0)

        try:
            # Ask for the folder name to store images
            subfolder = input("Enter the subfolder name to store images: ")

            while True:
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = self.detect_faces(img, gray)

                self.draw_rectangles_and_save_images(img, gray, faces, subfolder)

                key = cv2.waitKey(100) & 0xff

                if key == 27 or self.face_counter >= self.face_images_per_id:
                    print(f"Captured {self.face_counter} images for face ID {self.face_id} in subfolder '{subfolder}'")
                    self.face_id += 1  # Increment face ID for the next person
                    self.face_counter = 0  # Reset face counter for the new person
                    cv2.destroyAllWindows()  # Close the window
                    break

        except Exception as e:
            print(f"Error: {e}")

        finally:
            cap.release()

class FaceTrainer:
    def __init__(self, data_path='dataset', trainer_folder='trainer', cascade_path='haarcascade_frontalface_default.xml'):
        self.data_path = data_path
        self.trainer_folder = trainer_folder
        self.cascade_path = cascade_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(self.cascade_path)

    def _get_image_id(self, image_path):
        try:
            return int(os.path.split(image_path)[-1].split(".")[1])
        except ValueError:
            print(f"Error extracting ID from image {image_path}")
            return None

    def _get_images_and_labels(self):
        face_samples = []
        ids = []

        for root, dirs, files in os.walk(self.data_path):
            for image_path in files:
                try:
                    PIL_img = Image.open(os.path.join(root, image_path)).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')

                    id_ = self._get_image_id(image_path)
                    if id_ is not None:
                        faces = self.detector.detectMultiScale(img_numpy)
                        for (x, y, w, h) in faces:
                            face_samples.append(img_numpy[y:y+h, x:x+w])
                            ids.append(id_)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

        return face_samples, ids

    def train_recognizer(self):
        try:
            print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces, ids = self._get_images_and_labels()
            if not faces or not ids:
                print("No faces or labels found to train the recognizer.")
                return False

            # Create the trainer folder if it doesn't exist
            os.makedirs(self.trainer_folder, exist_ok=True)

            self.recognizer.train(faces, np.array(ids))
            trainer_path = os.path.join(self.trainer_folder, 'trainer.yml')
            self.recognizer.write(trainer_path)
            print("\n [INFO] {0} faces trained. Trainer file saved to '{1}'. Exiting Program".format(len(np.unique(ids)), trainer_path))
            return True
        except Exception as e:
            print(f"Error training recognizer: {e}")
            return False

class RealTimeFaceRecognizer:
    def __init__(self, dataset_path='dataset', trainer_path='trainer/trainer.yml', cascade_path='haarcascade_frontalface_default.xml'):
        self.dataset_path = dataset_path
        self.trainer_path = trainer_path
        self.cascade_path = cascade_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.names = self.get_names_from_dataset()

    def get_names_from_dataset(self):
        try:
            names = [name for name in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, name))]
            # print(names)
            return names
        except Exception as e:
            print(f"Error retrieving names from dataset: {e}")
            return []

    def load_recognizer(self):
        try:
            self.recognizer.read(self.trainer_path)
        except Exception as e:
            print(f"Error loading trained recognizer: {e}")

    def detect_and_recognize_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(0.1 * img.shape[1]), int(0.1 * img.shape[0])),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Perform face recognition
            id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 ==> "0" is a perfect match
            if confidence < 100:
                id = self.names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

    def start_recognition(self):
        try:
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)  # set video width
            cap.set(4, 480)  # set video height

            self.load_recognizer()

            while True:
                ret, img = cap.read()

                if not ret:
                    print("Error capturing frame.")
                    break

                self.detect_and_recognize_faces(img)

                cv2.imshow('camera', img)

                k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break

        except Exception as e:
            print(f"Error during face recognition: {e}")

        finally:
            print("\n [INFO] Exiting Program and cleanup stuff")
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse optional arguments from the command line
    parser = argparse.ArgumentParser(description="Face dataset creation, training, and real-time recognition.")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to the dataset folder")
    parser.add_argument("--trainer_path", type=str, default="trainer/trainer.yml", help="Path to the trained recognizer file")
    parser.add_argument("--cascade_path", type=str, default="haarcascade_frontalface_default.xml", help="Path to the cascade classifier XML file")
    parser.add_argument("--create_dataset", action="store_true", help="Flag to create a face dataset")
    parser.add_argument("--face_id", type=int, default=0, help="Initial face ID")
    parser.add_argument("--images_per_id", type=int, default=30, help="Number of face images to capture per ID")
    parser.add_argument("--face_counter", type=int, default=0, help="Initial face counter")
    args = parser.parse_args()

    if args.create_dataset:
        # Start face dataset creation
        # Start face dataset creation
        face_dataset_creator = CreateDataset(
            face_id=args.face_id,
            face_images_per_id=args.images_per_id,
            face_counter=args.face_counter,
            dataset_folder=args.dataset_path,  # Corrected attribute name
        )

        face_dataset_creator.start_capturing()


    # Train the recognizer
    face_trainer = FaceTrainer(data_path=args.dataset_path)
    if face_trainer.train_recognizer():
        print("Recognizer trained successfully!")
    else:
        print("Failed to train recognizer.")

    # Start real-time face recognition
    real_time_recognizer = RealTimeFaceRecognizer(
        dataset_path=args.dataset_path,
        trainer_path=args.trainer_path,
        cascade_path=args.cascade_path
    )
    real_time_recognizer.start_recognition()
