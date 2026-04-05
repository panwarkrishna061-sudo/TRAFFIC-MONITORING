import cv2
from detector import detect_all

def process_webcam(custom_model=None, general_model=None):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_all(frame, custom_model, general_model)
        cv2.imshow("Detection with Priority", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(input_path, custom_model=None, general_model=None):
    frame = cv2.imread(input_path)

    if frame is None:
        print("Cannot open image")
        return

    frame = detect_all(frame, custom_model, general_model)
    cv2.imshow("Image Detection with Priority", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(input_path, custom_model=None, general_model=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_all(frame, custom_model, general_model)
        cv2.imshow("Video Detection with Priority", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()