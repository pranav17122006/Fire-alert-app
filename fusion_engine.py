import cv2
import numpy as np
import joblib
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, messaging

# =========================
# FIREBASE INIT
# =========================
cred = credentials.Certificate(
    "C:\\fire_alert_app\\forest-fire-flutter-firebase-adminsdk-fbsvc-4ffafb55c3.json"
)
firebase_admin.initialize_app(cred)

# =========================
# LOAD MODELS
# =========================
yolo_model = YOLO("C:\\fire_n.pt")
ml_model = joblib.load("C:\\forestfire_model.pkl")

# =========================
# SEND ALERT FUNCTION
# =========================
def send_alert(confidence, lat, lon):
    message = messaging.Message(
        notification=messaging.Notification(
            title="🔥 FIRE DETECTED",
            body=f"Confidence: {confidence:.2f}"
        ),
        data={
            "lat": str(lat),
            "lon": str(lon),
            "confidence": str(confidence)
        },
        topic="firealert"
    )

    response = messaging.send(message)
    print("Notification Sent:", response)

# =========================
# INPUTS
# =========================
def get_ml_inputs():
    print("\nEnter ML feature values:")

    features = [
        float(input("daynight_N: ")),
        float(input("lat: ")),
        float(input("lon: ")),
        float(input("fire_weather_index: ")),
        float(input("pressure_mean: ")),
        float(input("wind_direction_mean: ")),
        float(input("wind_direction_std: ")),
        float(input("solar_radiation_mean: ")),
        float(input("dewpoint_mean: ")),
        float(input("cloud_cover_mean: ")),
        float(input("evapotranspiration_total: ")),
        float(input("humidity_min: ")),
        float(input("temp_mean: ")),
        float(input("temp_range: ")),
        float(input("wind_speed_max: ")),
        float(input("frp: "))
    ]

    return np.array(features).reshape(1, -1), features[1], features[2]

# =========================
# DL
# =========================
def predict_dl(image):
    results = yolo_model(image)[0]

    fire_detected = False
    confidence = 0.0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0:
            fire_detected = True
            confidence = max(confidence, conf)

    return fire_detected, confidence

# =========================
# ML
# =========================
def predict_ml(features):
    pred = ml_model.predict(features)[0]

    if hasattr(ml_model, "predict_proba"):
        prob = ml_model.predict_proba(features)[0][1]
    else:
        prob = 0.5

    return pred, prob

# =========================
# FUSION
# =========================
def fusion(dl_fire, dl_conf, ml_fire, ml_prob):
    score = 0

    if dl_fire:
        score += 2

    if ml_fire == 1:
        score += 1

    if dl_conf > 0.7:
        score += 1

    if ml_prob > 0.7:
        score += 1

    return score >= 2

# =========================
# MAIN
# =========================
def run():
    path = input("Enter image path: ")
    image = cv2.imread(path)

    if image is None:
        print("Invalid path")
        return

    features, lat, lon = get_ml_inputs()

    dl_fire, dl_conf = predict_dl(image)
    ml_fire, ml_prob = predict_ml(features)

    fire = fusion(dl_fire, dl_conf, ml_fire, ml_prob)

    print("\n--- RESULT ---")
    print("DL:", dl_fire, dl_conf)
    print("ML:", ml_fire, ml_prob)

    if fire:
        print("🔥 FIRE DETECTED")
        send_alert(max(dl_conf, ml_prob), lat, lon)
    else:
        print("✅ NO FIRE")

if __name__ == "__main__":
    run()