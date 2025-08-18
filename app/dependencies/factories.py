from services.analysis_service import AnalysisService
from fastapi import Depends, WebSocket


def get_user_id_from_websocket(websocket: WebSocket) -> str | None:
    return websocket.query_params.get("user_id")

def get_analysis_service(user_id: str | None = Depends(get_user_id_from_websocket)) -> AnalysisService:
    """
    새로운 AnalysisService 인스턴스를 생성하는 팩토리 함수.
    각 의존성 주입 시마다 새로운 인스턴스를 보장합니다.
    """
    print(f"Factory: Creating new AnalysisService instance for user {user_id}.")
    return AnalysisService(user_id=user_id)