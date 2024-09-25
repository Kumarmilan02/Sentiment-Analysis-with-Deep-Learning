# Sentiment Analysis with Deep Learning using BERT

This project performs **Sentiment Analysis** using **BERT (Bidirectional Encoder Representations from Transformers)**. We leverage the SMILE Twitter dataset to classify tweets into different sentiment categories. The model is fine-tuned using the `BERT` transformer model and trained with PyTorch.

## Project Overview

The goal of this project is to use a BERT-based transformer model to classify Twitter sentiments. This is done by fine-tuning a pre-trained BERT model and adapting it to the sentiment analysis task.

### Dataset

- **Dataset:** SMILE Twitter dataset.
- **Classes:**  
  - `nocode`: Removed from the dataset during preprocessing.
  - `happy`, `angry`, `sad`, `disgust`, `surprise`, and `not-relevant` are the main classes considered.

## Project Goals

- Perform Exploratory Data Analysis (EDA) and Data Preprocessing.
- Clean and preprocess the dataset by removing multiple categories and tweets labeled as `nocode`.
- Split the dataset into Training and Validation sets.
- Encode data using BERT Tokenizer.
- Train and fine-tune a BERT model for sequence classification.
- Use performance metrics to evaluate the model's accuracy on unseen data.
- Experiment with various batch sizes, learning rates, and epochs to optimize model performance.

## Google Colab Notebook

You can run the project on Google Colab by following this link: [Google Colab Notebook](https://colab.research.google.com/drive/1EJjzVpG3T9NimF-fk47W7tF9fRHAp4EX#scrollTo=E7QlW3F-lyYA).

## Table of Contents
1. [Introduction](#introduction)
2. [Exploratory Data Analysis (EDA) & Data Preprocessing](#exploratory-data-analysis-eda--data-preprocessing)
3. [Training/Validation Split](#trainingvalidation-split)
4. [Tokenizer & Data Encoding](#tokenizer--data-encoding)
5. [BERT Model Setup](#bert-model-setup)
6. [Data Loaders Creation](#data-loaders-creation)
7. [Optimizer & Scheduler Setup](#optimizer--scheduler-setup)
8. [Performance Metrics Definition](#performance-metrics-definition)
9. [Training Loop Implementation](#training-loop-implementation)
10. [Model Evaluation](#model-evaluation)
11. [Learning Rate Fine-Tuning](#learning-rate-fine-tuning)
12. [Experimenting with Batch Sizes](#experimenting-with-batch-sizes)
13. [Data Augmentation for Text](#data-augmentation-for-text)
14. [Early Stopping and Checkpointing](#early-stopping-and-checkpointing)
15. [Hyperparameter Optimization](#hyperparameter-optimization)
16. [Model Testing on Unseen Data](#model-testing-on-unseen-data)
17. [Model Saving and Loading](#model-saving-and-loading)

## Introduction
### Overview of Sentiment Analysis
Sentiment analysis is a computational task of identifying and categorizing opinions expressed in a piece of text. This project leverages sentiment analysis to understand public sentiments toward various topics on Twitter.

### Introduction to BERT
BERT, developed by Google, is a transformer-based model that significantly advances the state-of-the-art in NLP tasks. Its bidirectional nature allows it to consider the context of words from both directions, making it particularly effective for understanding sentiment.

## Exploratory Data Analysis (EDA) & Data Preprocessing
1. **Load the Dataset**: Load and inspect the dataset to understand its structure.
2. **Data Cleaning**: 
   - Remove tweets with multiple categories or labeled as "nocode".
   - Handle missing values and duplicates.
3. **Data Visualization**: 
   - Visualize the distribution of sentiment categories to understand class balance.
   - Generate plots to analyze tweet length distribution and other relevant features.

## Training/Validation Split
- Perform stratified sampling to split the dataset into training and validation sets, ensuring that each category is represented proportionally.

## Tokenizer & Data Encoding
- Load the BERT tokenizer and preprocess the text data by tokenizing and encoding it into input IDs and attention masks suitable for the BERT model.

## BERT Model Setup
- Initialize the BERT model for sequence classification.
- Set parameters such as the number of labels corresponding to sentiment categories.

## Data Loaders Creation
- Create DataLoader objects to efficiently handle batches of data during training and validation.

## Optimizer & Scheduler Setup
- Utilize the AdamW optimizer along with a learning rate scheduler to adjust the learning rate dynamically during training.
- Define hyperparameters: learning rate, epsilon, and number of epochs.

## Performance Metrics Definition
- Define metrics for model evaluation:
  - **Accuracy**: The proportion of correct predictions.
  - **F1 Score**: The harmonic mean of precision and recall, particularly useful for imbalanced datasets.

## Training Loop Implementation
- Implement the training loop to iteratively train the model:
  - Backpropagation to update model weights.
  - Logging training loss and performance metrics for analysis.

## Model Evaluation
- Evaluate the trained model on the validation set:
  - Calculate validation loss and performance metrics (accuracy, F1 score).
  - Analyze model performance using confusion matrices.

## Learning Rate Fine-Tuning
- Experiment with various learning rates to find the optimal setting that yields the best performance.

## Experimenting with Batch Sizes
- Test the model using different batch sizes to understand their impact on training stability and performance.

## Data Augmentation for Text
- Implement techniques for text augmentation, such as synonym replacement and back-translation, to enhance the diversity of the training data.

## Early Stopping and Checkpointing
- Implement early stopping to prevent overfitting based on validation loss.
- Save model checkpoints to facilitate recovery and further training.

## Hyperparameter Optimization
- Apply Grid Search or Random Search techniques to optimize hyperparameters like learning rate, batch size, and number of epochs.

## Model Testing on Unseen Data
- After training, test the model on a separate unseen dataset to assess generalization.
- Measure performance metrics (accuracy, F1-score, precision, recall) on this data.

## Model Saving and Loading
- Save the trained model using `torch.save()` for future use.
- Provide instructions for loading the model for inference.

## Installation and Usage

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- Huggingface Transformers Library

### Installing Dependencies

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-bert.git
   cd sentiment-analysis-bert
   ```

2. Run the notebook:
   - Open the Google Colab notebook and execute the cells.

## Contributing

Contributions are welcome! Please create a pull request or file an issue to suggest changes.

## License

This project is licensed under the MIT License.


