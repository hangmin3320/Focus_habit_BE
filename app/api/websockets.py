import base64
import cv2
import numpy as np
import time
import json

from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Depends
from services.analysis_service import AnalysisService
from dependencies.factories import get_analysis_service

router = APIRouter()

@router.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket, user_id: Optional[str] = None, analysis_service: AnalysisService = Depends(get_analysis_service)):
    await websocket.accept()
    print(f"WebSocket connection established for user: {user_id if user_id else 'anonymous'}")
    last_send_time = 0
    throttle_interval = 1.0  # 1 second

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
                analysis_results = await analysis_service.analyze_frame(img_bgr)

                # user_id가 있는 경우에만 전송 제한 적용
                if user_id:
                    current_time = time.time()
                    if current_time - last_send_time >= throttle_interval:
                        await websocket.send_json(analysis_results)
                        last_send_time = current_time
                else:
                    # user_id가 없으면 제한 없이 바로 전송
                    await websocket.send_json(analysis_results)

            except Exception as e:
                print(f"Error during analysis or sending results: {e}")
                await websocket.close(code=1011)
                break

    except WebSocketDisconnect:
        print(f"WebSocket connection disconnected for user: {user_id if user_id else 'anonymous'}")
    except Exception as e:
        print(f"General WebSocket error for user {user_id if user_id else 'anonymous'}: {e}")
        await websocket.close(code=1011)
    finally:
        analysis_service.close()