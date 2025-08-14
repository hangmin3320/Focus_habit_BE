import joblib
from fastapi import FastAPI
from api.websockets import router as websocket_router
from services.analysis_service import AnalysisService # Changed from FocusAnalysisService

app = FastAPI(
    title="AI Study Focus Analyzer Backend",
    description="Real-time analysis of user focus using webcam feed.",
    version="0.1.0",
)

# 웹소켓 라우터 포함
app.include_router(websocket_router)


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"message": "AI Study Focus Analyzer Backend is running"}


# 애플리케이션 시작/종료 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    print("Application startup...")

    # AnalysisService 인스턴스를 생성하고 애플리케이션 상태에 저장
    app.state.analysis_service = AnalysisService() # Changed from FocusAnalysisService()
    print("AnalysisService initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown...")
    # MediaPipe 리소스 해제 (필요시)
    # The AnalysisService now handles its own MediaPipeRunner, which manages the landmarker resources.
    # This specific cleanup might not be needed here if AnalysisService.close() handles it.
    # For now, removing specific landmarker checks as AnalysisService is the orchestrator.
    if hasattr(app.state.analysis_service, 'media_pipe_runner') and app.state.analysis_service.media_pipe_runner:
        # Assuming MediaPipeRunner has a close method or handles its own cleanup
        # If not, this line might need adjustment based on MediaPipeRunner's implementation
        pass # Removed specific landmarker close calls, relying on AnalysisService's internal cleanup if any.

