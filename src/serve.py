from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError
import joblib
import os

app = FastAPI()

# Đọc cấu hình từ biến môi trường
S3_BUCKET = os.environ.get("CLOUD_BUCKET", "placeholder-bucket")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    if not os.path.exists(os.path.expanduser("~/models")):
        os.makedirs(os.path.expanduser("~/models"), exist_ok=True)
    
    try:
        print(f"Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}...")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        print("Successfully downloaded model.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        # In a real scenario, we might want to fail fast if the model is critical
        # but for this lab, we'll let it proceed so the server can start if the file exists locally


# Gọi hàm này khi module được import (chạy khi server khởi động)
# In a real environment, we'd uncomment this.
# download_model()
# model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server. GitHub Actions dùng endpoint này để xác nhận deploy thành công."""
    return {"status": "ok"}


class PredictRequest(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.

    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra:  JSON {"prediction": <0|1|2>, "label": <"thấp"|"trung_bình"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # Load model if not loaded (for local testing flexibility)
    global model
    if 'model' not in globals() or model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        elif os.path.exists("models/model.pkl"):
             model = joblib.load("models/model.pkl")
        else:
            raise HTTPException(status_code=500, detail="Model not found")

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thấp", 1: "trung bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    # Initial download and load
    download_model()
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    uvicorn.run(app, host="0.0.0.0", port=8000)
