import base64
import cv2
import numpy as np
import time
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# 서비스 모듈 임포트
from services.media_pipe_runner import MediaPipeRunner
from services.modules.eye_analyzer import EyeAnalyzer
from services.modules.head_pose_analyzer import HeadPoseAnalyzer
from services.analysis_service import FocusAnalysisService

router = APIRouter()

@router.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    # --- 각 서비스의 인스턴스를 세션마다 생성 ---
    media_pipe_runner = MediaPipeRunner()
    eye_analyzer = EyeAnalyzer()
    head_pose_analyzer = HeadPoseAnalyzer()
    analysis_service = FocusAnalysisService()

    try:
        while True:
            # 1. Base64 이미지 데이터 수신
            data = await websocket.receive_text()

            if "," in data:
                _, base64_data = data.split(",", 1)
            else:
                base64_data = data

            # 2. 이미지 디코딩 및 변환
            try:
                img_bytes = base64.b64decode(base64_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error decoding image: {e}")
                continue

            # --- 3. 특징 추출 파이프라인 실행 ---
            current_timestamp_ms = int(time.time() * 1000)
            
            # 3-1. MediaPipe로 랜드마크 추출
            face_landmarks = media_pipe_runner.get_face_landmarks(img_rgb)

            # 3-2. 랜드마크로부터 특징점 추출
            if face_landmarks:
                eye_status = eye_analyzer.analyze_frame(face_landmarks)
                head_pose = head_pose_analyzer.analyze_frame(face_landmarks, img_rgb.shape)
                
                frame_data = {
                    "timestamp": current_timestamp_ms,
                    "eye_status": eye_status,
                    "head_pose": head_pose
                }
                
                # 4. 시계열 분석 서비스에 데이터 추가
                analysis_service.add_new_frame(frame_data)
            else:
                # 얼굴이 감지되지 않았을 때의 데이터 (필요시)
                # analysis_service.add_new_frame({"timestamp": current_timestamp_ms, ...})
                pass

    except WebSocketDisconnect:
        print("WebSocket connection disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # --- 5. 세션 종료 시 최종 분석 결과 출력 ---
        final_analysis = analysis_service.get_overall_analysis()
        print("\n--- Final Session Analysis ---")
        print(json.dumps(final_analysis, indent=2, ensure_ascii=False))
        print("----------------------------")
