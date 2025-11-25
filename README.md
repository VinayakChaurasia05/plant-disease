# Plant Disease Detection

A deep learning-based plant disease detection system using Convolutional Neural Networks (CNN) to classify plant diseases from leaf images.

## Project Overview

This project implements a CNN model to detect and classify plant diseases across 38 different disease categories. The model is trained on the New Plant Diseases Dataset from Kaggle.

## Dataset

- **Source**: New Plant Diseases Dataset (augmented) from Kaggle
- **Training Images**: 70,295 files
- **Validation Images**: 17,572 files
- **Number of Classes**: 38
- **Image Size**: 128x128 pixels
- **Color Mode**: RGB

## Model Architecture

The CNN model consists of:

- **Input Layer**: 128x128x3 (RGB images)
- **5 Convolutional Blocks**:
  - Block 1: 2 Conv2D layers (32 filters) + MaxPooling
  - Block 2: 2 Conv2D layers (64 filters) + MaxPooling + Dropout(0.5)
  - Block 3: 2 Conv2D layers (128 filters) + MaxPooling
  - Block 4: 2 Conv2D layers (256 filters) + MaxPooling + Dropout(0.6)
  - Block 5: 2 Conv2D layers (512 filters) + MaxPooling
- **Dropout Layer**: 0.25
- **Flatten Layer**
- **Dense Layer**: 1500 units with ReLU activation
- **Dropout Layer**: 0.4
- **Output Layer**: 38 units with Softmax activation

### Model Summary

```
Total params: 7,842,762 (29.92 MB)
Trainable params: 7,842,762 (29.92 MB)
Non-trainable params: 0 (0.00 B)
```

## Training Configuration

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 32

## Training Results

### Epoch-wise Performance

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|------------------|---------------|--------------------|-----------------|
| 1/10  | 0.1799          | 3.0242        | 0.5255             | 1.5556          |
| 2/10  | 0.5723          | 1.3748        | 0.5975             | 1.3164          |
| 3/10  | 0.6776          | 1.0159        | 0.5783             | 1.5156          |
| 4/10  | 0.7298          | 0.8468        | 0.7046             | 0.9971          |
| 5/10  | 0.7514          | 0.7770        | 0.6823             | 1.0293          |
| 6/10  | 0.7624          | 0.7504        | 0.7341             | 0.9213          |
| 7/10  | 0.7637          | 0.7457        | 0.6510             | 1.2245          |
| 8/10  | 0.7622          | 0.7484        | 0.6800             | 1.0293          |
| 9/10  | 0.7599          | 0.7718        | 0.7642             | 0.8134          |
| 10/10 | 0.7637          | 0.7521        | 0.7511             | 0.8134          |

### Final Model Performance

- **Training Accuracy**: 76.71%
- **Validation Accuracy**: 75.11%
- **Training Loss**: 0.7422
- **Validation Loss**: 0.8192

## Model Evaluation

### Batch Information

**Input Batch Shape:**
```
Images Shape: (32, 128, 128, 3)
Images dtype: float32
Labels Shape: (32, 38)
Labels dtype: float32
```

### Classification Report (Sample Classes)

| Disease Class | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|----------|
| Apple scab | 0.89 | 0.43 | 0.58 | 504 |
| Apple Black_rot | 0.96 | 0.79 | 0.87 | 497 |
| Apple Cedar_apple_rust | 0.99 | 0.61 | 0.75 | 440 |
| Apple healthy | 0.62 | 0.85 | 0.72 | 502 |
| Blueberry healthy | 0.89 | 0.59 | 0.71 | 454 |
| Cherry Powdery_mildew | 0.54 | 0.94 | 0.68 | 421 |
| Cherry healthy | 0.97 | 0.87 | 0.92 | 456 |
| Corn Cercospora_leaf_spot | 0.60 | 0.83 | 0.70 | 410 |
| Corn Common_rust | 1.00 | 0.94 | 0.97 | 477 |
| Corn Northern_Leaf_Blight | 0.91 | 0.61 | 0.73 | 477 |
| Corn healthy | 0.98 | 0.98 | 0.98 | 465 |
| Grape Black_rot | 0.92 | 0.79 | 0.85 | 472 |
| Grape Esca_(Black_Measles) | 0.92 | 0.91 | 0.92 | 480 |
| Grape Leaf_blight | 0.88 | 0.98 | 0.93 | 430 |
| Grape healthy | 0.95 | 0.96 | 0.96 | 423 |
| Orange Haunglongbing | 0.95 | 0.89 | 0.92 | 503 |
| Peach Bacterial_spot | 0.90 | 0.68 | 0.78 | 459 |
| Peach healthy | 0.86 | 0.89 | 0.87 | 432 |
| Pepper bell Bacterial_spot | 0.66 | 0.59 | 0.62 | 478 |
| Pepper bell healthy | 0.78 | 0.74 | 0.76 | 497 |
| Potato Early_blight | 0.74 | 0.86 | 0.79 | 485 |

## Visualization

The project includes an accuracy visualization plot showing:
- **Training Accuracy** (Red line): Progressive improvement over epochs
- **Validation Accuracy** (Blue line): Fluctuating but generally improving trend

The plot demonstrates the model's learning progression and helps identify potential overfitting.

## Files in Repository

- `plant_disease_detection_.py`: Main training script
- `requirements.txt`: Required Python packages
- `README.md`: Project documentation
- `app/`: Application folder (if applicable)

## Requirements

```python
tensorflow
keras
matplotlib
pandas
numpy
seaborn
kagglehub
scikit-learn
opencv-python
```

## Usage

1. **Mount Google Drive** (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Download Dataset**:
```python
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
```

3. **Train the Model**:
```python
python plant_disease_detection_.py
```

4. **Load Saved Model**:
```python
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
```

## Model Output

The model generates:
- **Trained model file**: `trained_plant_disease_model.keras`
- **Training history**: `training_hist.json`
- **Confusion Matrix**: Visual representation of model predictions
- **Classification Report**: Detailed precision, recall, and f1-score metrics

## Prediction Example

To predict disease from a test image:

```python
import cv2
import tensorflow as tf
import numpy as np

# Load image
image_path = 'path/to/test/image.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

# Predict
predictions = cnn.predict(input_arr)
result_index = np.argmax(predictions)
model_prediction = class_name[result_index]

print(f"Disease Name: {model_prediction}")
```

## Future Improvements

- Implement data augmentation techniques
- Try transfer learning with pre-trained models (VGG16, ResNet, etc.)
- Increase training epochs for better convergence
- Add early stopping and learning rate scheduling
- Deploy as a web application
- Add real-time detection capabilities

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: New Plant Diseases Dataset from Kaggle
- Framework: TensorFlow/Keras
- Platform: Google Colab

## Results and Visualizations

### 1. Training Accuracy Visualization

The model's training progress is visualized through an accuracy plot:

**Accuracy vs Epochs Plot:**
- **Red Line**: Training Accuracy - Shows progressive improvement from ~18% (epoch 1) to ~76% (epoch 10)
- **Blue Line**: Validation Accuracy - Demonstrates fluctuating performance with a general upward trend, reaching ~75% by epoch 10
- The plot illustrates the learning curve and helps identify potential overfitting
- Both training and validation accuracies converge around 75-76%, indicating good generalization

![Accuracy Visualization](images/accuracy_plot.png)
*Figure 1: Training and Validation Accuracy over 10 Epochs*

### 2. Sample Prediction - Test Input

**Input Image:**
A test image of a plant leaf showing symptoms of disease is fed into the trained model.

![Test Image Input](images/test_image_input.png)
*Figure 2: Test leaf image used as model input*

### 3. Model Prediction Output

**Prediction Result:**
The model successfully analyzes the input image and predicts the disease classification.

![Prediction Output](images/prediction_output.png)
*Figure 3: Model prediction showing "Disease Name: Potato__Late_blight"*

**Key Observations:**
- The model correctly identifies the disease type from the leaf image
- Prediction index: 21 (corresponding to Potato__Late_blight class)
- The output demonstrates the model's capability to classify plant diseases from real-world images

### Model Performance Summary

Based on the visualizations and outputs:
- **Training Accuracy**: 76.71%
- **Validation Accuracy**: 75.11%
- **Performance**: The model shows consistent performance across training and validation sets
- **Application**: Successfully identifies plant diseases from leaf images across 38 different disease categories

---

**Note**: To view the actual images from the Google Colab notebook, please refer to the original Colab file or add the images to the `images/` folder in this repository.
