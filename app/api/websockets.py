import base64
import cv2
import numpy as np
import time
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Depends, Query
from services.analysis_service import AnalysisService
from dependencies.factories import get_analysis_service

router = APIRouter()

@router.websocket("/ws/analysis")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(None),
    fixed_baseline: Optional[float] = Query(None),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    await websocket.accept()
    print(f"WebSocket connection established for user: {user_id}")
    last_send_time = 0
    throttle_interval = 1.0  # 1 second

    try:
        while True:
            data = await websocket.receive_text()
            
            heart_rate_from_payload = None
            base64_image_data = None

            try:
                # Try to parse as JSON
                payload = json.loads(data)
                if "image_data" in payload:
                    base64_image_data = payload["image_data"]
                if "heart_rate" in payload:
                    heart_rate_from_payload = payload["heart_rate"]
            except json.JSONDecodeError:
                # If not JSON, assume it's a plain base64 string
                base64_image_data = data

            if base64_image_data is None:
                print("Error: No image data found in WebSocket message.")
                continue

            # Handle "data:image/jpeg;base64," prefix
            if "," in base64_image_data:
                _, base64_image_data = base64_image_data.split(",", 1)
            
            # 2. 이미지 디코딩 및 변환
            try:
                img_bytes = base64.b64decode(base64_image_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    print("Warning: Failed to decode image frame.")
                    continue

                # analyze_frame 호출 시 heart_rate 전달
                analysis_result = await analysis_service.analyze_frame(img_bgr, heart_rate=heart_rate_from_payload)
                
                # user_id가 있는 경우에만 전송 제한 적용
                if user_id:
                    current_time = time.time()
                    if current_time - last_send_time >= throttle_interval:
                        await websocket.send_json(analysis_result)
                        last_send_time = current_time
                else:
                    # user_id가 없으면 제한 없이 바로 전송
                    await websocket.send_json(analysis_result)

            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print(f"WebSocket connection disconnected for user: {user_id}")
    except Exception as e:
        print(f"General WebSocket error for user {user_id}: {e}")
        await websocket.close(code=1011)
    finally:
        analysis_service.close() # Close MediaPipe resources for this connection
