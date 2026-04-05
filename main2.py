from model_loader import load_models
from input_handler import process_webcam, process_image, process_video

def main():
    print("Select Model:")
    print("1 → Custom Model (ambulance, firetruck)")
    print("2 → YOLOv8 (car, bus, truck)")
    print("3 → Both Models")

    choice = input("Enter choice (1/2/3): ").strip()

    custom_model, general_model = load_models(choice)

    input_path = input("Enter image/video path OR press Enter for webcam: ").strip()
    input_path = input_path.strip('"').strip("'")

    if input_path == "":
        process_webcam(custom_model, general_model)

    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(input_path, custom_model, general_model)

    else:
        process_video(input_path, custom_model, general_model)

if __name__ == "__main__":
    main()