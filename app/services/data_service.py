import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class FocusDataset(Dataset):
    """
    JSON 데이터를 로드하고, Feature Engineering을 적용하여
    PyTorch 모델 학습을 위한 시퀀스 데이터를 생성하는 클래스.
    """
    def __init__(self, data_dir, sequence_length=30):
        self.sequence_length = sequence_length
        self.features = []
        self.labels = []

        if not os.path.isdir(data_dir):
            print(f"Error: Data directory not found at {data_dir}")
            return

        all_files = os.listdir(data_dir)

        # 'focused' 파일 처리
        focused_files = [f for f in all_files if f.startswith('focused') and f.endswith('.json')]
        for file_name in focused_files:
            file_path = os.path.join(data_dir, file_name)
            self._load_and_process_data(file_path, label=1)  # label 1 for 'focused'

        # 'unfocused' 파일 처리
        unfocused_files = [f for f in all_files if f.startswith('unfocused') and f.endswith('.json')]
        for file_name in unfocused_files:
            file_path = os.path.join(data_dir, file_name)
            self._load_and_process_data(file_path, label=0)  # label 0 for 'unfocused'

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor

    def _load_and_process_data(self, file_path, label):
        """
        파일을 로드하여 Feature Engineering 및 시퀀스 생성을 수행합니다.
        """
        print(f"Processing {file_path} for label '{'focused' if label == 1 else 'unfocused'}'...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not read or parse {file_path}: {e}")
            return

        engineered_features = []
        for i in range(1, len(raw_data)):
            current_frame = raw_data[i]
            prev_frame = raw_data[i - 1]

            # 얼굴이 감지되지 않은 프레임은 건너뛰기 (추후 rule-base로 변환해야함)
            if current_frame.get('eye_status', {}).get('status') == 'NO_FACE_DETECTED' or \
               prev_frame.get('eye_status', {}).get('status') == 'NO_FACE_DETECTED':
                continue

            ear = current_frame.get('eye_status', {}).get('ear_value', 0)
            pitch = current_frame.get('head_pose', {}).get('pitch', 0)
            yaw = current_frame.get('head_pose', {}).get('yaw', 0)
            roll = current_frame.get('head_pose', {}).get('roll', 0)

            ear_vel = ear - prev_frame.get('eye_status', {}).get('ear_value', 0)
            pitch_vel = pitch - prev_frame.get('head_pose', {}).get('pitch', 0)
            yaw_vel = yaw - prev_frame.get('head_pose', {}).get('yaw', 0)
            roll_vel = roll - prev_frame.get('head_pose', {}).get('roll', 0)

            is_open = 1 if current_frame.get('eye_status', {}).get('status') == 'OPEN' else 0
            is_closed = 1 if current_frame.get('eye_status', {}).get('status') == 'CLOSED' else 0

            feature_vector = [
                ear, pitch, yaw, roll,
                ear_vel, pitch_vel, yaw_vel, roll_vel,
                is_open, is_closed
            ]
            engineered_features.append(feature_vector)

        num_sequences = len(engineered_features) - self.sequence_length + 1
        for i in range(num_sequences):
            sequence = engineered_features[i:i + self.sequence_length]
            self.features.append(sequence)
            self.labels.append(label)
            
        print(f"Finished processing {os.path.basename(file_path)}. Total sequences so far: {len(self.features)}")