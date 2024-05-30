### Urban Sound Classification Using Deep Learning

## Project Overview

This project focuses on classifying urban sounds using deep learning techniques. The main objective is to build models capable of identifying various urban sound classes from audio data. The project involves data preprocessing, model development, training, evaluation, and optimization.

## Dataset
The dataset used in this project is the UrbanSound8K dataset, which can be downloaded from [here](https://urbansounddataset.weebly.com/urbansound8k.html).

## Data Preprocessing
1. **Data Selection**: The Urban Sound Classification dataset is used.
2. **Spectrogram Generation**: Audio files are converted to mel-spectrograms using `librosa.feature.melspectrogram`. This visual representation captures the frequency and time information suitable for processing by CNNs.
3. **Amplitude Conversion**: The amplitude of the spectrograms is converted to decibels (dB) with `librosa.amplitude_to_db` for normalization.
4. **Label Extraction**: Class IDs are extracted from the dataset metadata.
5. **Normalization**: The spectrogram data is normalized by dividing by 255.0 to bring pixel values into the range [0, 1].

## Model Architectures

### 1. Basic CNN Model
- **Input Layer**: Preprocessed spectrogram data.
- **Convolutional Layers**: Three layers with ReLU activation.
- **Max Pooling Layers**: Applied after the first and second convolutional layers.
- **Flattening Layer**: Converts 2D feature maps into a 1D vector.
- **Dense Layers**: Two dense layers with dropout.
- **Output Layer**: Softmax activation for class probabilities.

### 2. Hyperparameter Optimized CNN Model
- **Input Layer**: Grayscale images of 128x173 pixels.
- **Convolutional Layers**: Three layers with batch normalization.
- **Pooling Layers**: Two max pooling layers.
- **Flattening and Dense Layers**: Dense layers with dropout.
- **Output Layer**: Softmax activation for 10 classes.

### 3. Enhanced CNN-LSTM Model
- **Convolutional Layers**: Three layers with batch normalization.
- **Pooling Layers**: Two max pooling layers.
- **Reshape Layer**: Prepares data for the LSTM layer.
- **LSTM Layer**: Captures temporal dynamics with 64 units.
- **Dense and Dropout Layers**: Dense layers with dropout.
- **Output Layer**: Softmax activation.

### 4. Transformer Model
- **Convolutional Layers**: Three layers followed by max pooling.
- **Reshape Layer**: Converts feature maps into sequences.
- **Lambda Layer**: Applies positional encoding.
- **Transformer Block**: Multi-head attention for capturing dependencies.
- **Global Average Pooling and Dropout**: Condenses and regularizes data.
- **Dense Layers**: Fully connected layers.
- **Output Layer**: Softmax activation.

## Training and Evaluation
- **Compilation**: Models are compiled with the Adam optimizer and categorical cross-entropy loss.
- **Training**: Models are trained for 40 epochs with a batch size of 128. Validation data is used for performance monitoring.
- **Results Visualization**: Training and validation accuracy and loss are plotted over epochs.

## Results and Observations
- **Basic CNN Model**: Achieved a maximum training accuracy of around 74% and validation accuracy of 77%.
- **Hyperparameter Optimized CNN Model**: Showed significant overfitting with training accuracy at 97.8% but validation and test accuracy dropping to 80% and 74.55%, respectively.
- **Enhanced CNN-LSTM Model**: Struggled with overfitting and underfitting, showing low training and validation accuracy.
- **Transformer Model**: Demonstrated robust performance with a testing accuracy of 87.88%, precision, recall, and F1 score all above 0.88, indicating strong generalization and balanced performance.

## Key Techniques
- **Spectrogram Generation**: Converts audio signals into a visual representation.
- **Normalization**: Ensures faster convergence during training.
- **Hybrid Model**: Combines CNNs with Transformer architecture for effective feature extraction and sequence modeling.

## References
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [The LSTM Reference Card](https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87)
- [Audio Deep Learning Made Simple](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)
- [Hugging Face Transformers for Audio Classification](https://huggingface.co/docs/transformers/en/tasks/audio_classification)
