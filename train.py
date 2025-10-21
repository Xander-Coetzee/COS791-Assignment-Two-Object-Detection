from ultralytics import YOLO


def main():
    model = YOLO("yolov12s.pt")

    print("Starting model training...")
    results = model.train(
        data="cheetah.yaml",  # Points to dataset config file
        epochs=100,  # Number of times to train on the dataset.
        imgsz=640,  # Resizes all images to 640x640
        batch=8,  # Number of images to process at once
        name="yolo_cheetah",  # A name for the output folder
    )

    print("Training complete.")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
