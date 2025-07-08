import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import datetime

# === Config ===
MODEL_PATH = '/Users/pabansah/Age_Gender_Detection/models/res34_fair_align_multi_7_20190809.pt'

RACE_LIST = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

SAVE_RESULTS = True  # Save predictions to CSV
RESULTS_CSV = 'webcam_fairface_results.csv'

# === Load Model ===
def load_model(model_path):
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(512, 18)  # 7+2+9=18 outputs
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    return model

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error loading Haar Cascade face detector.")
    exit(1)

# === Main ===
def main():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    # For saving results
    results = []

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)[0]
                race_logits = output[:7]
                gender_logits = output[7:9]
                age_logits = output[9:]

                race_idx = race_logits.argmax().item()
                gender_idx = gender_logits.argmax().item()
                age_idx = age_logits.argmax().item()

                race = RACE_LIST[race_idx]
                gender = GENDER_LIST[gender_idx]
                age = AGE_LIST[age_idx]

            label = f"{race}, {gender}, {age}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Save to results list
            results.append({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'race': race,
                'gender': gender,
                'age': age,
                'bbox': f"{x},{y},{w},{h}"
            })

        cv2.putText(frame, "Model: multi_7 (Race + Gender + Age)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("FairFace Age, Gender & Race Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    if SAVE_RESULTS and results:
        df = pd.DataFrame(results)
        if os.path.exists(RESULTS_CSV):
            df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
        else:
            df.to_csv(RESULTS_CSV, index=False)
        print(f"Results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()