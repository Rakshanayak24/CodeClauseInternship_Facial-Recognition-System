
import cv2
import os

def detect_and_save(image_path):
    # Load Haarcascade model (pre-trained face detector)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print(f"Detected {len(faces)} face(s).")

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save output image
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    print("Output saved as:", output_path)

    # Show the result
    cv2.imshow("Face Detection Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = input("Enter input image path: ")
    detect_and_save(path)
