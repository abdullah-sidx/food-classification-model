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
|   +-- food-101/
|       +-- apple_pie/
|       +-- baby_back_ribs/
|       +-- ...
|       +-- waffles/
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

This project was trained for the full Food-101 class set. The dataset is not committed to GitHub because it is large, but the local training folder should be arranged as class folders under:

```text
dataset/food-101/
```

The code reads class names automatically from folder names. The dataset should look like:

```text
dataset/food-101/
+-- apple_pie/
+-- baby_back_ribs/
+-- baklava/
+-- pizza/
+-- waffles/
```

Each class folder should contain images for that food category.

Supported Food-101 categories:

```text
apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare,
beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito,
bruschetta, caesar_salad, cannoli, caprese_salad, carrot_cake,
ceviche, cheesecake, cheese_plate, chicken_curry, chicken_quesadilla,
chicken_wings, chocolate_cake, chocolate_mousse, churros, clam_chowder,
club_sandwich, crab_cakes, creme_brulee, croque_madame, cup_cakes,
deviled_eggs, donuts, dumplings, edamame, eggs_benedict, escargots,
falafel, filet_mignon, fish_and_chips, foie_gras, french_fries,
french_onion_soup, french_toast, fried_calamari, fried_rice,
frozen_yogurt, garlic_bread, gnocchi, greek_salad,
grilled_cheese_sandwich, grilled_salmon, guacamole, gyoza, hamburger,
hot_and_sour_soup, hot_dog, huevos_rancheros, hummus, ice_cream,
lasagna, lobster_bisque, lobster_roll_sandwich, macaroni_and_cheese,
macarons, miso_soup, mussels, nachos, omelette, onion_rings, oysters,
pad_thai, paella, pancakes, panna_cotta, peking_duck, pho, pizza,
pork_chop, poutine, prime_rib, pulled_pork_sandwich, ramen, ravioli,
red_velvet_cake, risotto, samosa, sashimi, scallops, seaweed_salad,
shrimp_and_grits, spaghetti_bolognese, spaghetti_carbonara,
spring_rolls, steak, strawberry_shortcake, sushi, tacos, takoyaki,
tiramisu, tuna_tartare, waffles
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

Place the full Food-101 class folders inside `dataset/food-101/`, then run:

```bash
python train.py --dataset-dir dataset/food-101
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
