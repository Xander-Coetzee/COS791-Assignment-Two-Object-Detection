from ultralytics import YOLO


def main():
    # 1. Load a base model
    # We use 'yolov12n.pt' for the nano version, as specified
    # in your assignment[cite: 13]. This downloads the pre-trained
    # model to build from.
    model = YOLO("yolov12n.pt")

    # 2. Train the model
    # This is the main training command.
    print("Starting model training...")
    results = model.train(
        data="cheetah.yaml",  # Points to your dataset config file
        epochs=100,  # Number of times to train on the dataset.
        imgsz=640,  # Resizes all images to 640x640
        batch=8,  # Number of images to process at once
        name="yolo_cheetah_run",  # A name for the output folder
    )

    print("Training complete.")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
