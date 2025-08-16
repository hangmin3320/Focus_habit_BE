import os
from supabase import create_client, Client
from dotenv import load_dotenv

# .env 파일이 있다면 환경 변수를 로드합니다.
load_dotenv()

class SupabaseService:
    """
    Supabase와의 상호작용(인증, 스토리지 등)을 담당하는 서비스 클래스.
    """
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            raise ValueError("Supabase URL and Service Key must be set in environment variables.")
            
        self.client: Client = create_client(url, key)
        print("Supabase client initialized successfully.")

    def generate_presigned_upload_url(self, bucket_name: str, file_path: str, expires_in: int = 3600) -> str | None:
        """
        지정된 경로에 파일을 업로드할 수 있는 Presigned URL을 생성합니다.
        """
        try:
            response = self.client.storage.from_(bucket_name).create_signed_upload_url(
                path=file_path
            )
            return response.get('signedURL')
        except Exception as e:
            print(f"Error generating presigned URL for {bucket_name}/{file_path}: {e}")
            raise

    def download_file(self, bucket_name: str, file_path: str) -> bytes | None:
        """
        스토리지에서 파일을 다운로드하여 bytes 형태로 반환합니다.
        """
        try:
            response = self.client.storage.from_(bucket_name).download(path=file_path)
            return response
        except Exception as e:
            # Supabase-py는 파일이 없을 때 에러를 발생시키므로, 이를 처리합니다.
            print(f"Error downloading file {bucket_name}/{file_path} (it may not exist): {e}")
            return None

    def upload_file(self, bucket_name: str, file_path: str, file_body: bytes) -> bool:
        """
        메모리에 있는 파일(bytes)을 스토리지에 업로드합니다.
        """
        try:
            self.client.storage.from_(bucket_name).upload(
                path=file_path,
                file=file_body,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
            print(f"Successfully uploaded to {bucket_name}/{file_path}")
            return True
        except Exception as e:
            print(f"Error uploading file {bucket_name}/{file_path}: {e}")
            return False

# 다른 곳에서 쉽게 가져다 쓸 수 있도록 싱글턴 인스턴스를 생성합니다.
supabase_service = SupabaseService()