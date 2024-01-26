
import os
import cv2


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
