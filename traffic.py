import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Only one output layer
    ])
    return model


def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128))  # Already reduced size
    normalized = resized / 255.0
    return normalized.astype(np.float32)  # Use float32 instead of float64

def extract_features(video_path, start_frame, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()
    return np.array(frames)


def parse_info_file(info_file_path):
    data = []
    with open(info_file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9:
                filename, date, time, direction, day_night, weather, start_frame, num_frames, traffic_class = parts[:9]
                data.append({
                    'filename': filename,
                    'date': date,
                    'time': time,
                    'direction': direction,
                    'day_night': day_night,
                    'weather': weather,
                    'start_frame': int(start_frame),
                    'num_frames': int(num_frames),
                    'traffic_class': traffic_class
                })
    return data

def video_data_generator(video_folder, info_file_path, batch_size=16):
    info_data = parse_info_file(info_file_path)
    le = LabelEncoder()
    le.fit([entry['traffic_class'] for entry in info_data])
    num_classes = len(le.classes_)
    
    while True:
        X_batch = []
        y_batch = []
        
        for entry in info_data:
            video_path = os.path.join(video_folder, entry['filename'] + '.avi')
            if os.path.exists(video_path):
                features = extract_features(video_path, entry['start_frame'], entry['num_frames'])
                X_batch.extend(features)
                y_batch.extend([entry['traffic_class']] * len(features))
                
            if len(X_batch) >= batch_size:
                X_batch = np.array(X_batch, dtype=np.float32)
                y_batch = le.transform(y_batch)
                yield X_batch, np.array(y_batch)
                X_batch, y_batch = [], []

def train_model(video_folder, info_file_path):
    data_gen = video_data_generator(video_folder, info_file_path, batch_size=32)
    info_data = parse_info_file(info_file_path)
    le = LabelEncoder()
    le.fit([entry['traffic_class'] for entry in info_data])
    num_classes = len(le.classes_)  # Determine the number of classes dynamically
    steps_per_epoch = len(info_data) // 32
    
    model = create_model((128, 128, 3), num_classes)  # Use the dynamically determined num_classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(data_gen, steps_per_epoch=steps_per_epoch, epochs=10)
    
    return model

def predict(model, frame, label_encoder):
    preprocessed = preprocess_frame(frame)
    prediction = model.predict(np.expand_dims(preprocessed, axis=0))
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class


if __name__ == "__main__":
    video_folder = r"D:\traffic\video"
    info_file_path = r"D:/traffic/info.txt"
    
    model = train_model(video_folder, info_file_path)
    
    # Save the trained model
    model.save("traffic_analysis_model.h5")
    
    print("Model training complete. Saved as 'traffic_analysis_model.h5'")
    
    test_video_path = "D:\traffic\video\cctv052x2004080619x00093.avi"
    cap = cv2.VideoCapture(test_video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        traffic_class = predict(model, frame, label_encoder)
        print(f"Predicted traffic class: {traffic_class}")
        
        # Display the frame with prediction (optional)
        cv2.putText(frame, f"Class: {traffic_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

