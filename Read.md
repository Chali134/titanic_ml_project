# 🧠 Titanic Survival Model Trainer (Detailed Version)

This project is an interactive machine learning app built with **Gradio** that allows users to upload a Titanic dataset CSV file and train **four ML models**: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors. It evaluates each model using accuracy, precision, recall, and F1 score, and provides a visual comparison of the results.

## 🚀 Features

- ✅ Upload your own Titanic dataset CSV
- 🧼 Automatic preprocessing (e.g., handling missing values, encoding categorical variables)
- 🔍 Trains 4 different ML classifiers
- 📊 Detailed metrics: Accuracy, Precision, Recall, F1 Score
- 📈 Interactive Plotly bar chart to compare model performance
- 🏆 Highlights the best-performing model

## 🧪 Models Used

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

## ⚙️ How It Works

1. Upload a CSV file (must include columns like `Survived`, `Sex`, `Age`, `Embarked`, etc.).
2. The app preprocesses the data (fills missing values, encodes categories).
3. It trains each model and evaluates it on accuracy, precision, recall, and F1 score.
4. A summary report is generated along with a Plotly chart for accuracy comparison.

## 📁 Required Columns in CSV

Ensure your CSV contains at least the following columns:

- `Survived`
- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`

Optional columns like `Name`, `Ticket`, and `Cabin` will be dropped during preprocessing.

## 🛠️ Installation

Clone the repository