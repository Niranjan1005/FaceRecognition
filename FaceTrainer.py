import cv2
import numpy as np
from PIL import Image
import os


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
