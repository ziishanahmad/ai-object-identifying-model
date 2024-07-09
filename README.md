
# AI Object Identifying Model

This repository contains code for building, training, and testing a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The project includes a Jupyter Notebook that demonstrates the entire workflow from data preparation to model evaluation and image prediction.

## Features

- **Model Training**: Train a CNN on the CIFAR-10 dataset.
- **Model Evaluation**: Evaluate the trained model's performance on test data.
- **Image Prediction**: Upload and classify custom images using the trained model.

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ziishanahmad/ai-object-identifying-model.git
   cd ai-object-identifying-model
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python pillow
   ```

## Usage

### Training and Evaluating the Model

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook ai_object_identifying_model.ipynb
   ```

2. Follow the steps in the notebook to train the model on the CIFAR-10 dataset, evaluate its performance, and save the trained model.

### Testing the Model

1. Upload an image using the upload function provided in the notebook.
2. Preprocess the uploaded image and use the trained model to predict its class.
3. Visualize the result with the predicted label.

### Example

Here is a quick example of how to use the model to classify an image:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
model = load_model('cifar10_model.h5')

# Define class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the uploaded image
def preprocess_upload_image(image):
    img = image.resize((32, 32))  # Resize to the input size of the model
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Upload an image
image = Image.open('path_to_your_image.jpg')
img = preprocess_upload_image(image)

# Predict the label of the uploaded image
prediction = model.predict(img)
predicted_label = np.argmax(prediction[0])

# Display the image with the predicted label
plt.imshow(image)
plt.title(f'Predicted: {class_names[predicted_label]}')
plt.axis('off')
plt.show()
```

### Result

![Result Image](https://raw.githubusercontent.com/ziishanahmad/ai-object-identifying-model/main/predicted.png)  

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Zeeshan Ahmad**

- **Email**: [ziishanahmad@gmail.com](mailto:ziishanahmad@gmail.com)
- **GitHub**: [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn**: [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)
