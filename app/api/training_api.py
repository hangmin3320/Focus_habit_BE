from fastapi import APIRouter, HTTPException, Body, BackgroundTasks

from services.supabase_service import supabase_service
# Supabase 연동 학습 함수를 import 합니다.
from services.personal_train import train_personal_from_storage

router = APIRouter()

@router.post("/storage/presigned-url")
async def create_presigned_url(
    user_id: str = Body(..., embed=True),
    file_name: str = Body(..., embed=True)
):
    """
    Supabase Storage에 학습 데이터를 업로드할 수 있는 Presigned URL을 생성합니다.
    """
    if not user_id or not file_name:
        raise HTTPException(status_code=400, detail="user_id and file_name are required.")

    bucket_name = "models"  # Supabase 스토리지 버킷 이름
    storage_path = f"{user_id}/{file_name}"

    signed_url = supabase_service.generate_presigned_upload_url(bucket_name, storage_path)

    if not signed_url:
        raise HTTPException(status_code=500, detail="Could not generate presigned URL.")

    return {"presigned_url": signed_url, "storage_path": storage_path}


@router.post("/models/train-personal")
async def start_personal_training(
    background_tasks: BackgroundTasks,
    user_id: str = Body(..., embed=True),
    storage_path: str = Body(..., embed=True)
):
    """
    업로드된 학습 데이터를 사용하여 개인화 모델 학습을 시작합니다.
    실제 학습은 백그라운드에서 비동기적으로 실행됩니다.
    """
    print(f"Received training request for user '{user_id}' with data at '{storage_path}'")
    
    # 백그라운드에서 학습 함수를 실행합니다.
    background_tasks.add_task(train_personal_from_storage, user_id, storage_path)
    
    return {"message": "Training request received. Training will start in the background."}
