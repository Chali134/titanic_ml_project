import pandas as pd
import gradio as gr
import time
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

def process_detailed(file, progress=gr.Progress(track_tqdm=True)):
    df = pd.read_csv(file.name)
    
    progress(0.1, desc="Preprocessing Dataset")
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results_md = "### ✅ Model Evaluation Summary:\n\n"
    accuracy_scores = {}
    model_reports = ""

    for i, (name, model) in enumerate(models.items()):
        progress(0.2 + i * 0.2, desc=f"Training {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        accuracy_scores[name] = acc * 100

        model_reports += f"#### 🔍 {name} Results:\n"
        model_reports += f"- Accuracy: **{acc:.2%}**\n"
        model_reports += f"- Precision: **{prec:.2%}**\n"
        model_reports += f"- Recall: **{rec:.2%}**\n"
        model_reports += f"- F1 Score: **{f1:.2%}**\n\n"
        model_reports += "```\n" + classification_report(y_test, preds) + "```\n\n"

    best_model = max(accuracy_scores, key=accuracy_scores.get)
    results_md += f"🏆 **Best Model:** {best_model} ({accuracy_scores[best_model]:.2f}%)\n\n"

    # Bar chart
    fig = go.Figure([go.Bar(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))])
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        template="plotly_dark",
        height=400
    )

    return results_md + model_reports, fig

# Gradio Interface
gr.Interface(
    fn=process_detailed,
    inputs=gr.File(label="Upload Titanic CSV"),
    outputs=[
        gr.Markdown(label="📊 Model Performance Report"),
        gr.Plot(label="📈 Accuracy Comparison")
    ],
    title="🧠 Titanic Survival Model Trainer (Detailed Version)",
    description="Train 4 ML models, see accuracy, precision, recall, F1, and compare visually.",
    theme="soft",
    live=True
).launch()
