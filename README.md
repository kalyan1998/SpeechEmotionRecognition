# Speech Emotion Recognition (SER) Using Mel Spectrograms

## Project Description

This project aims to identify emotions from speech using deep learning models, specifically leveraging Convolutional Neural Networks (CNNs) and Mel Spectrograms. The focus is on accurately recognizing emotions even in challenging scenarios, such as varied speaking styles and accents.

## Workflow Overview

1. **Data Collection and Preparation**
   - **Datasets Used**:
     - EMO-DB (Berlin Emotional Speech Database)
     - TESS (Toronto Emotional Speech Database)
     - RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
     - CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
     - SAVEE (Surrey Audio-Visual Expressed Emotion)

   - **Preprocessing Steps**:
     1. **Reading Audio Files**: Audio files are read using the `Torch Audio Transforms` function from the PyTorch library.
     2. **Channel Conversion**: Mono audio files are converted to stereo by duplicating the first channel.
     3. **Sampling Rate Standardization**: Audio files are resampled to a uniform sampling rate of 16,000 Hz.
     4. **Resizing Audio Length**: Audio files are padded or truncated to a fixed length of 5 seconds.
     5. **Data Augmentation**: Techniques like time-shifting and masking (frequency and time) are used to increase the amount of training data.

2. **Feature Extraction**
   - **Mel Spectrograms**:
     - Audio signals are converted into Mel Spectrograms using Fourier Transforms. This transforms the audio into a time-frequency representation, which is more suitable for CNN-based models.
     - **Parameters Used**:
       - `n_mels`: 128
       - `n_fft`: 2048
       - `hop_length`: 512

3. **Normalization**
   - Two normalization techniques are applied:
     1. **Min-Max Scaling**: Scales the data to a range of [0, 1].
     2. **Z-Score Normalization**: Centers the data by subtracting the mean and dividing by the standard deviation.

4. **Model Training**
   - **Pretrained Models Used**:
     - **DenseNet161**: Connects each layer to every other layer, alleviating the vanishing gradient problem and reducing the number of parameters.
     - **ResNet50**: A 50-layer deep CNN that is pretrained on the ImageNet dataset.
     - **ResNeXt50_32x4d**: An optimized version of ResNet with grouped convolutions.

   - **Training Setup**:
     - **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.01.
     - **Batch Size**: 16
     - **Early Stopping**: Training is stopped if no improvement in validation loss is observed for 4 consecutive epochs.
     - **Frameworks Used**: PyTorch and skLearn

5. **Evaluation**
   - **Metrics Used**:
     - **Confusion Matrix**: Provides a detailed breakdown of correct and incorrect predictions.
     - **Precision, Recall, and F1-Score**: Evaluates the model's performance on each emotion class.

## Usage

### Prerequisites

- Python 3.x
- PyTorch
- Librosa
- scikit-learn

## Pipeline

### Workflow
![Converting Audio To MEL Spectrogram](https://github.com/kalyan1998/SpeechEmotionRecognition/blob/main/PreProcessing_1.png)

### After Preprocessing
![Normalizing Audio](https://github.com/kalyan1998/SpeechEmotionRecognition/blob/main/PreProcessing_2.png)

![Time Shifting](https://github.com/kalyan1998/SpeechEmotionRecognition/blob/main/PreProcessing_3.png)

## Contributing

We welcome contributions from the community! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request


## Acknowledgments

- Jawaharlal Nehru Technological University Hyderabad

---

Feel free to explore the repository and use the provided tools and scripts. If you find this project useful, please give it a star!
