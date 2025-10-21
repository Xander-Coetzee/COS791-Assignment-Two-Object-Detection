from ultralytics import YOLO


def main():
    model = YOLO("best.pt")

    print("Running prediction on the 'test' image folder...")
    model.predict(
        source="C:/Users/agcoe/src/cos791_assignment_2/cheetah_data/cheetah_test",
        save=True,
        name="test_image_results",  # A folder name for the saved images
    )
    print("Test image predictions saved to 'runs/detect/test_image_results'")
    print("-------------------------------------------\n")

    VIDEO_FILE_PATH = "test_videos\cheetah_1.mp4"

    print(f"Running prediction on video: {VIDEO_FILE_PATH}...")
    model.predict(
        VIDEO_FILE_PATH,
        save=True,  # This saves a new video with the boxes drawn on
        name="test_video_result",  # A folder name for the saved video
    )
    print("Video prediction complete!")
    print("Your new video is saved in the 'runs/detect/test_video_result' folder.")


if __name__ == "__main__":
    main()
