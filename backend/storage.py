import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("https://jrtsokyqspygwtflcaqa.supabase.co")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpydHNva3lxc3B5Z3d0ZmxjYXFhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NTU1MTEwMCwiZXhwIjoyMDgxMTI3MTAwfQ.Dh5VhUqxsJiqMzQC9Xnos1rqVI566BJ-iKSNfG9of34")

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
