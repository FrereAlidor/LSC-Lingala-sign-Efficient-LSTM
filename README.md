# LSC-Lingala-sign-Efficient-LSTM# Intelligent Lingala Sign Language Translation System

**Developed for the Democratic Republic of Congo**

A state-of-the-art computer vision system that translates Lingala sign language gestures into text and speech using deep learning technologies. This system combines EfficientNet-B3 architecture with LSTM networks for temporal gesture modeling, providing real-time translation capabilities.

## 🌟 Features

- **Real-time Translation**: Live webcam-based sign language recognition
- **Advanced Architecture**: EfficientNet-B3 + LSTM for superior accuracy
- **MediaPipe Integration**: Enhanced hand landmark detection and preprocessing
- **Multi-modal Output**: Text and speech synthesis capabilities
- **Robust Evaluation**: K-fold cross-validation and comprehensive metrics
- **User-friendly Interface**: Interactive demonstration system
- **Data Augmentation**: Advanced preprocessing for improved model generalization

## 🏗️ Architecture

### Model Components

1. **Feature Extraction**: EfficientNet-B3 (pre-trained on ImageNet)
2. **Temporal Modeling**: LSTM layers for sequence understanding
3. **Hand Detection**: MediaPipe for precise hand landmark identification
4. **Classification**: Dense layers with softmax activation

### Architecture Variants

- **Single Image Model**: Direct EfficientNet-B3 classification
- **Sequence Model**: TimeDistributed EfficientNet + LSTM for temporal sequences

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install torch
pip install gtts
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/lingala-sign-language-translation.git
cd lingala-sign-language-translation
```

2. Prepare your data structure:
```
Data_TeachSign/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
└── labels.npy
```

3. Update the data path in the script:
```python
data_path = r'path/to/your/Data_TeachSign'
```

4. Run the system:
```python
python lingala_sign_translation.py
```

## 📊 Data Requirements

### Input Data Format

- **X_train.npy**: Training images/sequences
  - Single images: `(batch_size, height, width, channels)`
  - Sequences: `(batch_size, sequence_length, height, width, channels)`
- **y_train.npy**: Training labels (integer encoded)
- **X_val.npy**: Validation images/sequences
- **y_val.npy**: Validation labels
- **labels.npy**: Class names array

### Preprocessing Pipeline

1. **Normalization**: Pixel values scaled to [0, 1]
2. **Hand Detection**: MediaPipe landmark extraction
3. **Data Augmentation**: Rotation, shifting, zooming, brightness adjustment
4. **Temporal Alignment**: Sequence padding/truncation for LSTM input

## 🎯 Model Performance

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification analysis
  ![Conf_M_B3LSTM](https://github.com/user-attachments/assets/05846d25-744f-4a6e-abbf-dfb110577cbd)
  ![Conf_M_B3](https://github.com/user-attachments/assets/2ff72767-e39a-45c2-83c2-2d6e2b73f22b)



### Training Configuration

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Categorical crossentropy
- **Batch Size**: 32
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Reduction**: Factor 0.2, patience 5
 ![model_comparison (1)](https://github.com/user-attachments/assets/4e86dcb2-4650-432d-a2ae-82637fac3395)



## 🔧 Usage

### 1. Real-time Translation

```python
# Start webcam-based real-time translation
real_time_translation(model, label_classes, sequence_length)
```

### 2. Image Prediction

```python
# Predict gesture from static image
predicted_class, confidence = predict_image(image_path, model, label_classes)
```
<img width="559" alt="Screenshot 2025-05-09 011923 (1)" src="https://github.com/user-attachments/assets/672c19bc-e3d4-4b13-a523-c5bb29827b45" />
<img width="551" alt="Screenshot 2025-05-09 012400" src="https://github.com/user-attachments/assets/fb16e56a-ee4c-4ad9-b28a-cf5b6392cd1e" />


### 3. Interactive Demo

```python
# Run the complete demonstration system
run_sign_language_demo()
```

### 4. Text-to-Speech

```python
# Convert predicted text to speech
text_to_speech(predicted_text, lang='fr')
```

## 📈 Advanced Features

### K-Fold Cross Validation

```python
# Perform robust model evaluation
cv_metrics = k_fold_cross_validation(X_train, y_train, n_splits=5)

```
![REsult](https://github.com/user-attachments/assets/9b5cf070-a238-476c-886b-bc65de3011e9)


### Hand Landmark Visualization

```python
# Visualize MediaPipe hand detection
visualize_hand_landmarks(image_path)
```

### Training History Analysis

The system provides comprehensive training visualization including:
- Accuracy curves (training vs validation)
- Loss curves (training vs validation)
- Confusion matrix heatmaps

## 🛠️ Configuration

### Model Hyperparameters

- **EfficientNet Variant**: B3 (configurable)
- **LSTM Units**: 256, 128 (configurable)
- **Dropout Rate**: 0.3
- **Dense Layer Size**: 128
- **Confidence Threshold**: 0.7 (real-time prediction)

### Camera Settings

- **Frame Rate**: 30 FPS
- **Resolution**: Webcam default
- **Prediction Interval**: Real-time
- **Buffer Size**: Configurable based on sequence length

## 📱 System Requirements

### Minimum Requirements

- **Python**: 3.7+
- **RAM**: 8GB
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Webcam**: For real-time functionality
- **Storage**: 2GB free space

### Recommended Requirements

- **RAM**: 16GB+
- **GPU**: RTX 3060 or better
- **CPU**: Intel i7 or AMD Ryzen 7
- **Storage**: 5GB+ free space

## 🤝 Contributing

We welcome contributions to improve the Lingala Sign Language Translation System:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional sign language vocabularies
- Model architecture improvements
- Mobile application development
- Performance optimizations
- Documentation translations

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Democratic Republic of Congo**: For inspiring this accessibility project
- **MediaPipe Team**: For excellent hand tracking capabilities
- **TensorFlow/Keras**: For the deep learning framework
- **EfficientNet Authors**: For the efficient architecture design
- **Sign Language Community**: For guidance and feedback

## 📞 Support

For questions, issues, or collaborations:

- **Email**: alidor.mbayandjambe@unikin.ac.cd


## 🔮 Future Enhancements

- [ ] Mobile application development
- [ ] Multi-language support expansion
- [ ] Real-time sentence construction
- [ ] Integration with virtual assistants
- [ ] Cloud-based inference API
- [ ] Educational platform integration
- [ ] Performance optimization for edge devices

