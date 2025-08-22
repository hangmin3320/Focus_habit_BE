import base64
import cv2
import numpy as np
import time
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Depends, Query
from services.analysis_service import AnalysisService
from dependencies.factories import get_analysis_service
from typing import Optional # Corrected import location

router = APIRouter()

@router.websocket("/ws/analysis")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(None), # user_id 쿼리 파라미터 추가
    fixed_baseline: Optional[float] = Query(None), # 고정 기준 심박수 추가
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    await websocket.accept()
    print(f"WebSocket connection established for user: {user_id}")

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
                await websocket.send_json(analysis_result)

            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("WebSocket connection disconnected.")
    except Exception as e:
        print(f"General WebSocket error: {e}")
        await websocket.close(code=1011) # Explicitly close with 1011
    finally:
        analysis_service.close() # Close MediaPipe resources for this connection