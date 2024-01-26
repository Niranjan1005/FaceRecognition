import os
import cv2


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
