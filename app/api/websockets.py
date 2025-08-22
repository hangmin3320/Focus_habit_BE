import base64
import cv2
import numpy as np
import time
import json
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from services.analysis_service import AnalysisService
from dependencies.factories import get_analysis_service

router = APIRouter()

async def analyzer_task(websocket: WebSocket, analysis_service: AnalysisService, shared_state: dict):
    """Receives frames as fast as possible, analyzes them, and updates shared_state."""
    try:
        while True:
            data = await websocket.receive_text()
            
            heart_rate_from_payload = None
            base64_image_data = None

            try:
                payload = json.loads(data)
                base64_image_data = payload.get("image_data")
                heart_rate_from_payload = payload.get("heart_rate")
            except json.JSONDecodeError:
                base64_image_data = data

            if not base64_image_data:
                print("Warning: No image data found in WebSocket message.")
                continue

            if "," in base64_image_data:
                _, base64_image_data = base64_image_data.split(",", 1)

            img_bytes = base64.b64decode(base64_image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                print("Warning: Failed to decode image frame.")
                continue

            analysis_result = await analysis_service.analyze_frame(img_bgr, heart_rate=heart_rate_from_payload)
            shared_state['latest_result'] = analysis_result

    except WebSocketDisconnect:
        print("Analyzer task: WebSocket disconnected.")
    except Exception as e:
        print(f"Error in analyzer_task: {e}")
    finally:
        shared_state['analyzer_stopped'] = True

async def sender_task(websocket: WebSocket, shared_state: dict):
    """Sends the latest analysis result from shared_state once per second."""
    try:
        while not shared_state.get('analyzer_stopped'):
            await asyncio.sleep(1)
            if shared_state.get('latest_result'):
                await websocket.send_json(shared_state['latest_result'])
    except WebSocketDisconnect:
        print("Sender task: WebSocket disconnected.")
    except Exception as e:
        if not isinstance(e, asyncio.CancelledError):
            print(f"Error in sender_task: {e}")

@router.websocket("/ws/analysis")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(None),
    fixed_baseline: Optional[float] = Query(None),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    await websocket.accept()
    print(f"WebSocket connection established for user: {user_id or 'anonymous'}")

    try:
        if user_id:
            shared_state = {'analyzer_stopped': False, 'latest_result': None}
            analyzer = asyncio.create_task(analyzer_task(websocket, analysis_service, shared_state))
            sender = asyncio.create_task(sender_task(websocket, shared_state))
            await asyncio.gather(analyzer, sender)
        else:
            while True:
                data = await websocket.receive_text()
                heart_rate_from_payload = None
                base64_image_data = None
                try:
                    payload = json.loads(data)
                    base64_image_data = payload.get("image_data")
                    heart_rate_from_payload = payload.get("heart_rate")
                except json.JSONDecodeError:
                    base64_image_data = data

                if not base64_image_data:
                    print("Warning: No image data found in WebSocket message.")
                    continue
                if "," in base64_image_data:
                    _, base64_image_data = base64_image_data.split(",", 1)
                
                img_bytes = base64.b64decode(base64_image_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    print("Warning: Failed to decode image frame.")
                    continue
                
                analysis_result = await analysis_service.analyze_frame(img_bgr, heart_rate=heart_rate_from_payload)
                await websocket.send_json(analysis_result)

    except WebSocketDisconnect:
        print(f"WebSocket connection closed for user: {user_id or 'anonymous'}")
    except Exception as e:
        print(f"General WebSocket error for user {user_id or 'anonymous'}: {e}")
    finally:
        analysis_service.close()