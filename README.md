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

You can run the project on Google Colab by following this link: [Google Colab Notebook](https://colab.research.google.com/drive/1XdyS1v0odpiIWu1Cx7bbGUs6Dfi89khR?usp=sharing).

## Key Steps

### 1. Data Preprocessing
- Loaded the SMILE Twitter dataset.
- Removed tweets labeled as `nocode` and handled tweets with multiple categories.
- Cleaned the data, visualized distributions, and performed basic transformations to prepare it for BERT.

### 2. Data Splitting
- Split the dataset into training and validation sets (85% train, 15% validation) using stratified sampling to maintain class balance.

### 3. Tokenization
- Used **BERT Tokenizer** to tokenize tweets and encode them into the required format for BERT (input IDs, attention masks).

### 4. Model Setup
- Fine-tuned a pre-trained **BERT Base Uncased** model using the Huggingface `transformers` library.
- Initialized the model for **sequence classification**, mapping the tweet sentiments to their corresponding labels.

### 5. Training & Validation
- Trained the model using the **AdamW** optimizer with learning rates such as `2e-5` and `5e-5`.
- Implemented a linear learning rate scheduler for smooth convergence.
- Monitored the training and validation loss after each epoch.

### 6. Performance Metrics
- Tracked performance using **accuracy** and **F1-score**.
- Implemented class-wise accuracy metrics to measure how well the model performs across different sentiment categories.

## Optimization

To enhance model performance, the following hyperparameter tuning was applied:
- Learning Rate variations: `1e-5`, `3e-5`, `5e-5` were tried.
- Batch Sizes: Compared performances for batch sizes of 16 and 32.
- Early Stopping: Introduced early stopping to prevent overfitting based on validation loss.

## Results & Analysis

- **Accuracy**: Achieved an accuracy of over 85% on the validation set.
- **F1 Score**: The weighted F1-score was used as a key metric to ensure balanced performance across all classes.
- **Class-wise Accuracy**: The model performed well across sentiments like `happy`, `angry`, and `sad`, while struggling with underrepresented classes like `disgust`.

## Experimentation Insights

- **Learning Rate**: The optimal learning rate was found to be `2e-5`, where the model achieved the best balance between speed of convergence and performance.
- **Batch Size**: A batch size of 32 led to better generalization and reduced training noise compared to smaller batches.
- **Early Stopping**: Introduced early stopping to avoid overfitting, particularly effective after 6 epochs.

## Future Work

- **Data Augmentation**: Implement text-based augmentations such as synonym replacement or back translation to further improve generalization.
- **Model Ensemble**: Experiment with different transformer models (e.g., RoBERTa, DistilBERT) to enhance performance.

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
