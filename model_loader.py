from ultralytics import YOLO
import os

CUSTOM_MODEL_PATH = r"D:\deeplearning\runs\detect\custom_traffic_model8\weights\best.pt"

def load_models(choice):
    custom_model = None
    general_model = None

    if choice == "1":
        if not os.path.exists(CUSTOM_MODEL_PATH):
            print("❌ Custom model file not found:", CUSTOM_MODEL_PATH)
            return None, None
        custom_model = YOLO(CUSTOM_MODEL_PATH)

    elif choice == "2":
        general_model = YOLO("yolov8n.pt")

    elif choice == "3":
        if not os.path.exists(CUSTOM_MODEL_PATH):
            print("❌ Custom model file not found:", CUSTOM_MODEL_PATH)
            return None, None
        custom_model = YOLO(CUSTOM_MODEL_PATH)
        general_model = YOLO("yolov8n.pt")

    else:
        print("❌ Invalid choice")

    return custom_model, general_model