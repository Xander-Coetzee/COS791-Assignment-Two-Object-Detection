# Cheetah Detection with YOLO

This project implements a cheetah detection system using the YOLO (You Only Look Once) object detection model. It includes scripts for training a custom YOLO model on a cheetah dataset, evaluating its performance, and running predictions on video files.

## Project Structure

- `train.py`: Script for training the YOLO model on the `cheetah_data` dataset.
- `test.py`: Script for evaluating the trained model on a test set and running inference on video files.
- `best.pt`: The trained YOLO model weights, resulting from the training process.
- `cheetah.yaml`: Configuration file defining the dataset paths and class names for cheetah detection.
- `requirements.txt`: Lists all Python dependencies required to run this project.
- `cheetah_data/`: Contains the dataset used for training and validation. This directory should be structured according to YOLO dataset conventions.
- `test_videos/`: Contains video files used for testing the trained model's prediction capabilities.
- `runs/`: This directory will store the outputs of training (e.g., logs, checkpoints) and prediction (e.g., videos with detected bounding boxes).

## Setup

To set up the project, first clone the repository and then install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the YOLO model, run the `train.py` script:

```bash
python train.py
```

Training progress and results will be saved in the `runs/train/` directory.

### Evaluating and Predicting with the Model

To evaluate the trained model on the test set and run predictions on a video, execute the `test.py` script:

```bash
python test.py
```

The evaluation metrics will be printed to the console. A new video with detected cheetahs will be saved in the `runs/detect/predict/` folder.

## Results

- **Training results:** Detailed logs, metrics, and model checkpoints can be found in `runs/train/`.
- **Prediction videos:** Videos with bounding box detections are saved in `runs/detect/predict/`.
