# Prodigy-ML-05

**Prodigy-ML-05** is an advanced machine learning project aimed at accurately recognizing Indian food items from images and estimating their calorie content. This project leverages a Convolutional Neural Network (CNN) model to classify Indian food items and provide their calorific values, aiding in dietary planning and nutritional assessment.

## Repository Structure

- `ML_Task05.ipynb`: Jupyter notebook containing the complete project implementation, including data preprocessing, model training, and prediction.
- `ML_Task05.py`: Python script encapsulating the core functionalities of the project.
- `List of Indian Foods.txt`: Text file detailing all the images available in the dataset.

## Dataset

The dataset used for this project is available on Kaggle: [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)

## Installation and Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

### Install Dependencies

Install the required libraries using the following command:

```bash
pip install tensorflow numpy pandas matplotlib ipywidgets
```

### Download the Dataset

Download and extract the dataset from Kaggle, ensuring the dataset folder is named `indian-food-images-dataset`.

## Usage

### Running the Jupyter Notebook

To execute the project step-by-step:

1. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open `ML_Task05.ipynb` and run the cells sequentially to carry out the project tasks.

### Running the Python Script

To run the project as a standalone script:

```bash
python ML_Task05.py
```

## Project Overview

### Data Preprocessing

- **Loading Images**: Images are loaded from the dataset and preprocessed for model training.
- **Data Augmentation**: Techniques such as rotation, zoom, and flipping are applied to enhance the dataset and improve model robustness.

### Model Architecture

- **CNN Model**: A Convolutional Neural Network is constructed using TensorFlow/Keras. The model architecture is designed to effectively capture the features of Indian food items.
- **Compilation**: The model is compiled with a categorical cross-entropy loss function and an Adam optimizer.

### Training

- The model is trained on the preprocessed dataset with a validation split to monitor its performance and prevent overfitting.

### Prediction and Calorie Estimation

- **Image Prediction**: The trained model predicts the class of an uploaded image.
- **Calorie Display**: The corresponding calorific value of the predicted food item is displayed.

## Example Code

### File Upload and Prediction

The following function handles file upload, image preprocessing, model prediction, and result display:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from IPython.display import display, Image
import ipywidgets as widgets

# Define class names and calorie values
class_names = ['adhirasam', 'aloo_gobi', 'aloo_matar', ...]  # Complete list of class names
calorie_values = {'adhirasam': 250, 'aloo_gobi': 150, ...}  # Complete list of calorie values

# Load pre-trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Function to handle file upload and prediction
def handle_file_upload(change):
    uploaded_file = next(iter(upload_button.value.values()))
    image_path = './' + uploaded_file['metadata']['name']
    with open(image_path, 'wb') as f:
        f.write(uploaded_file['content'])

    display(Image(filename=image_path))
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    print(f'Predictions shape: {predictions.shape}')
    print(f'Predictions values: {predictions}')

    if predictions.shape[1] == len(class_names):
        predicted_class_idx = np.argmax(predictions)
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
            calorie_value = calorie_values.get(predicted_class, 'Calorie value not found')
            print(f'Predicted Food Item: {predicted_class}')
            print(f'Calories: {calorie_value} grams')
        else:
            print('Error: Predicted class index out of range.')
    else:
        print(f'Error: Number of predictions ({predictions.shape[1]}) does not match number of classes ({len(class_names)}).')

upload_button = widgets.FileUpload()
display(upload_button)
upload_button.observe(handle_file_upload, names='value')
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author

Crafted with love by Sam Naveenkumar .V

---

Feel free to fork this repository, submit pull requests, and raise issues. Contributions are welcome!

```markdown
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
