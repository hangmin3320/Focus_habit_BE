from services.analysis_service import AnalysisService

def get_analysis_service() -> AnalysisService:
    """
    새로운 AnalysisService 인스턴스를 생성하는 팩토리 함수입니다.
    각 의존성 주입 시마다 새로운 인스턴스를 보장합니다.
    """
    print("Factory: Creating new AnalysisService instance.")
    return AnalysisService()