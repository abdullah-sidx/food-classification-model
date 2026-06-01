# Food-101 Image Classification

A complete deep learning project for classifying food images with TensorFlow/Keras transfer learning and a polished Streamlit frontend.

The project trains a CNN-based classifier on the Food-101 dataset using MobileNetV2 pretrained on ImageNet. After training, users can upload or capture a food image in the Streamlit app and receive the predicted food class, confidence score, and top 3 predictions.

## Features

- Transfer learning with MobileNetV2
- Automatic class-name loading from dataset folders
- Image preprocessing, resizing, normalization, and augmentation
- Train/validation split with `ImageDataGenerator`
- Early stopping, model checkpointing, and learning-rate reduction
- Accuracy and loss graphs saved after training
- Reusable prediction module
- Streamlit UI with upload, camera capture, image preview, model status, confidence display, and top 3 predictions
- Downloadable prediction result as JSON
- Placeholder panels for future calorie and nutrition features
- Defensive handling for missing model files, invalid uploads, and prediction errors
- GitHub-friendly `.gitignore` for virtual environments, datasets, generated outputs, and large model files

## Project Structure

```text
food-classification/
|
+-- dataset/
|   +-- food-mini/
|       +-- burger/
|       +-- ice_cream/
|       +-- pizza/
|       +-- ramen/
|       +-- sushi/
|
+-- model/
|   +-- food_model.h5
|   +-- class_labels.json
|
+-- outputs/
|   +-- accuracy.png
|   +-- loss.png
|
+-- app.py
+-- train.py
+-- predict.py
+-- utils.py
+-- requirements.txt
+-- README.md
+-- .gitignore
```

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2 transfer learning
- Streamlit
- Pillow
- NumPy
- Matplotlib
- scikit-learn

## Dataset

This project is currently configured to train on a smaller Food-101-style mini dataset first:

```text
dataset/food-mini/
```

The code reads class names automatically from folder names. Your dataset should look like:

```text
dataset/food-mini/
+-- burger/
+-- ice_cream/
+-- pizza/
+-- ramen/
+-- sushi/
```

Each class folder should contain images for that food category.

To train on the full Food-101 dataset later, place it in `dataset/food-101/` and run:

```bash
python train.py --dataset-dir dataset/food-101
```

## Installation

1. Clone or open this project folder.

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Place the mini class folders inside `dataset/food-mini/`, then run:

```bash
python train.py
```

To customize training:

```bash
python train.py --epochs 15 --batch-size 32 --learning-rate 0.0001
```

Optional fine-tuning:

```bash
python train.py --epochs 10 --fine-tune --fine-tune-epochs 5
```

Training saves:

- Model: `model/food_model.h5`
- Class labels: `model/class_labels.json`
- Accuracy graph: `outputs/accuracy.png`
- Loss graph: `outputs/loss.png`

## Run the Streamlit App

After training, start the app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, upload a food image, and view the predicted class and confidence score.

The frontend entry point is:

```text
app.py
```

The app includes:

- A responsive FoodLens interface
- Upload and camera tabs
- Sidebar model/label status
- Input image preview
- Main prediction card
- Top 3 confidence ranking
- JSON download for prediction results

## Prediction Module Usage

You can also use `predict.py` from another script:

```python
from pathlib import Path
from predict import predict_from_path

result = predict_from_path(Path("sample_food.jpg"))
print(result)
```

## Screenshots

Add screenshots here after running the app:

- App home screen
- Uploaded food preview
- Prediction result with top 3 confidence scores
- Training accuracy and loss graphs

## Deployment

This app can be deployed to Streamlit Community Cloud or another Python hosting platform.

Recommended deployment steps:

1. Push the project to GitHub.
2. Do not commit the full dataset.
3. Do not commit `venv/`, `__pycache__/`, or generated training logs.
4. Train the model locally or in a GPU environment.
5. Add the trained `food_model.h5` and `class_labels.json` to your deployment environment.
6. Set the Streamlit entry point to:

```text
app.py
```

For large model files, use a release asset, object storage, or a model registry instead of committing them directly to Git.

Before uploading to GitHub, check the files that will be committed:

```bash
git status
```

The included `.gitignore` keeps local environments, datasets, model artifacts, and generated output images out of the repository by default.

## Future Improvements

- Use EfficientNetB0 or EfficientNetV2 for stronger accuracy
- Add a verified nutrition API
- Add serving-size estimation
- Add model evaluation on a held-out test set
- Add confusion matrix and per-class metrics
- Export to TensorFlow Lite for mobile inference
- Add Docker deployment support

## Notes

Food-101 is a large dataset. Training is much faster with a CUDA-compatible GPU. CPU training works, but it can take a long time.
