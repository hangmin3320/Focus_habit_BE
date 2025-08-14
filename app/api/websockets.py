import base64
import cv2
import numpy as np
import time
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request # Import Request
from services.analysis_service import AnalysisService # Import the renamed AnalysisService

router = APIRouter()

@router.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    analysis_service: AnalysisService = AnalysisService() # Create a new instance for each connection

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
                if img_bgr is None:
                    print("Warning: Failed to decode image, img_bgr is None.")
                    continue
            except Exception as e:
                print(f"Error decoding image: {e}")
                continue

            # --- 3. 분석 파이프라인 실행 및 결과 전송 ---
            try:
                # Call the analyze_frame method of the AnalysisService
                analysis_results = await analysis_service.analyze_frame(img_bgr)
                
                # Send the results back to the client
                await websocket.send_json(analysis_results)
            except Exception as e:
                print(f"Error during analysis or sending results: {e}")
                # Consider sending an error message to the client before closing
                await websocket.close(code=1011) # Explicitly close with 1011
                break # Break the loop after closing connection

    except WebSocketDisconnect:
        print("WebSocket connection disconnected.")
    except Exception as e:
        print(f"General WebSocket error: {e}")
        await websocket.close(code=1011) # Explicitly close with 1011
    finally:
        analysis_service.clear_buffer() # Clear buffer on WebSocket disconnect
        analysis_service.close() # Close MediaPipe resources for this connection
        # Removed final analysis print as it's not session-specific anymore