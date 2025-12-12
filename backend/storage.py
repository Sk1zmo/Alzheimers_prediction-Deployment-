import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_image_to_supabase(filename: str, file_bytes: bytes) -> str:
    bucket_name = "ct-scans"
    file_path = f"user_uploads/{filename}"

    # Upload file
    supabase.storage.from_(bucket_name).upload(
        file=file_bytes,
        path=file_path,
        file_options={"content-type": "image/png"}
    )

    # Return public URL
    url = supabase.storage.from_(bucket_name).get_public_url(file_path)
    return url
