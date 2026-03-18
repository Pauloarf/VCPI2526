# GTSRB Project Guide: Traffic Sign Recognition

## Project Objective
The main goal of this project is to develop and optimize Deep Learning models for the **German Traffic Sign Recognition Benchmark (GTSRB)**. The target is to approach or exceed the published state-of-the-art accuracy of **99.82%**.

A significant portion of this work involves exploring **Image Processing** and **Data Augmentation** techniques to improve model robustness and performance.

---

## Prerequisites & Installation
Before starting, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn seaborn torchinfo
```

---

## Step-by-Step Roadmap

### Phase 1: Environment & Data Preparation
1.  **Imports & Utilities**: Set up the necessary libraries (PyTorch, Torchvision, Matplotlib, etc.) and include the `vcpi_util.py` for visualization.
2.  **Dataset Loading**: Load the training and testing datasets using `ImageFolder`.
3.  **Exploratory Data Analysis (EDA)**: 
    *   Visualize sample images from different classes.
    *   Analyze class distribution (histogram) to identify imbalances.
4.  **Validation Strategy**: Create a validation set. *Crucial:* Since GTSRB data comes from video sequences, ensure the validation set is built using entire sequences to avoid data leakage (highly correlated images in both train and val sets).

### Phase 2: Data Augmentation & Pre-processing
1.  **Pre-processing**: Experiment with resizing (e.g., 32x32 or 48x48), normalization, and grayscale vs. RGB.
2.  **Static Augmentation**: Apply image processing filters (blurring, sharpening, contrast adjustment) to the dataset.
3.  **Dynamic Augmentation**: Use `torchvision.transforms` (Rotation, Translation, Shear, Scaling) to generate varied samples during training.
4.  **Balancing**: Address class imbalance using oversampling, weighted loss functions, or targeted augmentation.

### Phase 3: Model Development & Training
1.  **Baseline Model**: Start with a simple Convolutional Neural Network (CNN) to establish a performance baseline.
2.  **Architecture Exploration**: Experiment with deeper architectures, Batch Normalization, and Dropout.
3.  **Advanced Techniques**:
    *   **Transfer Learning**: Use pre-trained models (like ResNet or EfficientNet).
    *   **Ensembles**: Combine predictions from multiple models.
4.  **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and use learning rate schedulers.

### Phase 4: Evaluation & Reporting
1.  **Performance Metrics**: Monitor Accuracy and Loss on both Training and Validation sets.
2.  **Error Analysis**: Use Confusion Matrices to identify which classes the model confuses most often.
3.  **Final Testing**: Evaluate the best-performing model on the held-out Test Set.
4.  **Documentation**: Record all experiments, findings, and final results in the report.

---

## Recommended Reading & Resources (Google Scholar)

For deep dives into the state-of-the-art techniques used for the GTSRB dataset, consider these foundational and recent papers:

### Foundational Papers
- **Stallkamp, J., et al. (2012).** *"The German Traffic Sign Recognition Benchmark: A multi-class classification competition."* IEEE International Joint Conference on Neural Networks. [Link](https://scholar.google.com/scholar?q=The+German+Traffic+Sign+Recognition+Benchmark:+A+multi-class+classification+competition)
- **Ciresan, D., et al. (2012).** *"Multi-column Deep Neural Networks for Image Classification."* CVPR. (The committee-based approach that first broke the 99% barrier). [Link](https://scholar.google.com/scholar?q=Multi-column+Deep+Neural+Networks+for+Image+Classification)

### Modern Lightweight Architectures (Efficiency)
- **Wong, A. (2018).** *"MicronNet: A Highly Compact Deep Convolutional Neural Network Architecture for Real-time Embedded Traffic Sign Classification."* [Link](https://scholar.google.com/scholar?q=MicronNet+Traffic+Sign+Classification)
- **Saini, R., et al. (2022).** *"DeepThin: A Novel Lightweight Architecture for Traffic Sign Recognition."* [Link](https://scholar.google.com/scholar?q=DeepThin+Traffic+Sign+Recognition)

### Advanced Augmentation & Ensembles
- **Cubuk, E. D., et al. (2019).** *"AutoAugment: Learning Augmentation Strategies from Data."* (Useful for Phase 1). [Link](https://scholar.google.com/scholar?q=AutoAugment+Image+Classification)
- **Ju, C., et al. (2018).** *"The Relative Performance of Ensemble Methods with Deep Convolutional Neural Networks."* (Useful for Phase 2). [Link](https://scholar.google.com/scholar?q=Ensemble+Methods+with+Deep+Convolutional+Neural+Networks)

---

## Glossary of Terms

- **Data Augmentation**: A technique used to artificially increase the size of a training dataset by creating modified versions of images (e.g., rotations, flips, shifts). This helps the model generalize better to unseen data.
- **Convolutional Neural Networks (CNN)**: A class of deep neural networks most commonly applied to analyzing visual imagery. They use "filters" to automatically learn spatial hierarchies of features, from simple edges to complex objects.
- **Image Processing**: The use of algorithms to perform operations on an image to enhance it or extract useful information (e.g., sharpening, blurring, contrast adjustment).
- **Validation Set**: A subset of the data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters.
- **Data Leakage (in GTSRB)**: Occurs when highly similar images (from the same video sequence) are present in both the training and validation sets, leading to optimistically biased performance metrics.
- **Pre-processing**: Operations applied to data before it is fed into the model, such as resizing all images to a uniform size or normalizing pixel values.
- **Static vs. Dynamic Augmentation**:
    - **Static**: Augmenting the dataset *before* training starts, increasing the total number of files on disk.
    - **Dynamic**: Applying transformations *on-the-fly* during training; the model sees a slightly different version of the image in every epoch.
- **Class Imbalance**: When some classes have significantly more samples than others. This can cause the model to be biased toward the majority classes.
- **Baseline Model**: A simple model that provides a reference point for performance. Any improvements should be measured against this baseline.
- **Batch Normalization**: A technique to standardize the inputs to a layer for each mini-batch. This stabilizes the learning process and significantly reduces the number of training epochs required.
- **Dropout**: A regularization technique where randomly selected neurons are ignored during training, which prevents the model from overfitting.
- **Transfer Learning**: Taking a model trained on one task (e.g., ImageNet) and repurposing it for a second related task (e.g., GTSRB).
- **Ensembles**: Combining the predictions of multiple models to achieve better performance than any single constituent model alone.
- **Hyperparameters**: Parameters whose values are set before the learning process begins (e.g., learning rate, batch size, number of layers).
- **Confusion Matrix**: A table used to describe the performance of a classification model, showing exactly where the model is "confused" between different classes.
