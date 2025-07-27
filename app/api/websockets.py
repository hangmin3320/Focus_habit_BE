import base64
import cv2
import numpy as np
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    try:
        # 애플리케이션 상태에서 AnalysisService 인스턴스 가져오기
        analysis_service = websocket.app.state.analysis_service

        while True:
            # 1. Base64 이미지 데이터 수신
            data = await websocket.receive_text()

            # Base64 접두사 제거 (예: "data:image/jpeg;base64,")
            if "," in data:
                header, base64_data = data.split(",", 1)
            else:
                base64_data = data

            # 2. Base64 디코딩 및 이미지(NumPy 배열) 변환
            try:
                img_bytes = base64.b64decode(base64_data)
                # OpenCV를 사용하여 바이트를 이미지 배열로 디코딩
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    print("Warning: Could not decode image. Skipping frame.")
                    continue

                # MediaPipe는 RGB 이미지를 선호하므로 BGR -> RGB 변환
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            except Exception as e:
                print(f"Error decoding or converting image: {e}")
                continue  # 이미지 처리 오류 시 다음 프레임으로 건너뛰기

            # 3. AnalysisService를 통해 이미지 분석
            # MediaPipe의 timestamp는 밀리초 단위이므로 현재 시간을 사용
            current_timestamp_ms = int(time.time() * 1000)

            analysis_result = analysis_service.analyze_image(img_rgb, current_timestamp_ms)

            # 4. 분석 결과를 JSON 형태로 클라이언트에게 전송
            await websocket.send_json(analysis_result)

    except WebSocketDisconnect:
        print("WebSocket connection disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # 리소스 정리 (필요시)
        pass
