import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # Bonus 5: Check for label distribution (Data Drift/Imbalance)
        label_counts = y_train.value_counts(normalize=True)
        print("\nLabel distribution in training data:")
        for label, ratio in label_counts.items():
            print(f"Class {label}: {ratio:.2%}")
            if ratio < 0.10:
                print(f"WARNING: Class {label} is underrepresented (< 10%)!")

        # TODO 4: Khoi tao va huan luyen model (Bonus 2: Support multiple algorithms)
        model_type = params.get("model_type", "random_forest")
        model_params = {k: v for k, v in params.items() if k != "model_type"}
        
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # Bonus 3: Generate detailed report
        report = classification_report(y_eval, preds)
        matrix = confusion_matrix(y_eval, preds)
        
        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log distribution to metrics
        for label, ratio in label_counts.items():
            mlflow.log_metric(f"class_{label}_ratio", ratio)

        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print("\nClassification Report:")
        print(report)

        # TODO 8: Luu metrics va report ra file (Bonus 3)
        os.makedirs("outputs", exist_ok=True)
        metrics_data = {
            "accuracy": acc, 
            "f1_score": f1,
            "model_type": model_type,
            "label_distribution": label_counts.to_dict()
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=4)
            
        with open("outputs/report.txt", "w") as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(matrix))

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
