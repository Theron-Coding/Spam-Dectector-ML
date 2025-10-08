## Datasets Used

Dataset 1: https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy/input

Dataset 2: https://huggingface.co/datasets/adamlouly/enron_spam_data/tree/main

## Environment Setup

### Using Conda
```bash
# Create a new conda environment
conda create -n spam-detection python=3.9

# Activate the environment
conda activate spam-detection

# Install required packages
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Required Dependencies
Python 3.8+

pandas

numpy

scikit-learn

matplotlib

seaborn

jupyter

re

SpellChecker

## Data Processing

### Input Data Format
Your dataset should be a CSV file with the following columns:

- **Message**: The text content of SMS messages

- **Spam:** Binary label (0 for ham, 1 for spam)

- **VowelCount:** Number of vowels in each message

- **ConsonantCount:** Number of consonants in each message

- **IncorrectSpelling:** Count of spelling errors

### Data Loading and Preprocessing

1. Column Renaming: Standardizes column names

2. Feature Scaling: Applies MinMax scaling to numeric features

3. Missing Values: Drops rows with null values

4. Train-Test Split: 75-25 split with random state 42

## Load and preprocess data

```data = load_data('MergedDataCleaned.csv')```

## Model Training
### Available Models
1. Logistic Regression
- Uses TF-IDF vectorization for text features
- Optimized for binary classification

```train_evaluate_lr(X_train, X_test, y_train, y_test)```

2. Multinomial Naive Bayes
- Uses TF-IDF vectorization for text features
- Optimized for binary classification

```train_evaluate_mnb(X_train, X_test, y_train, y_test)```

3. K-Means Clustering
- Unsupervised learning approach
- Uses CountVectorizer for text features
- Useful for pattern discovery

```train_evaluate_km(data)```
## Training Process
1. Data Splitting:

```X_train, X_test, y_train, y_test = get_train_test_split(data)```

2. Model Pipeline Creation:
- Text vectorization (TF-IDF for classification, CountVectorizer for clustering)
- Feature combination using ColumnTransformer
- Model initialization with optimized parameters

3. Training & Evaluation:
- Automatic model fitting on training data
- Prediction on test set
- Comprehensive metric calculation
- Visualization of results




