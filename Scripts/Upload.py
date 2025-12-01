import os
from huggingface_hub import HfApi, create_repo

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
# 1. Paste your HF Write Token here (or set it as env var HF_TOKEN)

# Securely get token from environment, or use a placeholder for the public repo
HF_TOKEN = os.getenv("HF_TOKEN", "PASTE_YOUR_TOKEN_HERE_LOCALLY")

# 2. Your HF Username
HF_USERNAME = "Ishaanlol"


# 3. Map Local Folders to New HF Repos
# (Local Path) : (Target Repo Name)
MODELS_TO_UPLOAD = {
    # The Router
    "Classifier": f"{HF_USERNAME}/MoL-Router-DistilBERT",

    # The Code Expert (Your 4-stage LoRA)
    "Section-D/Universal-Code-Master/final_model": f"{HF_USERNAME}/MoL-Code-Expert-LoRA",

    # The Math Expert
    "Final-Dynamic-Model/final_model(Math)": f"{HF_USERNAME}/MoL-Math-Expert-LoRA"
}


# ==============================================================================
#  UPLOAD LOGIC
# ==============================================================================
def upload_all():
    api = HfApi(token=HF_TOKEN)

    print(f"🚀 Starting Upload for user: {HF_USERNAME}...\n")

    for local_path, repo_id in MODELS_TO_UPLOAD.items():
        if not os.path.exists(local_path):
            print(f"⚠️  SKIPPING {local_path} (Folder not found locally)")
            continue

        print(f"📦 Processing: {repo_id}")
        print(f"   ...Creating repo if missing...")
        try:
            create_repo(repo_id, repo_type="model", token=HF_TOKEN, exist_ok=True)
        except Exception as e:
            print(f"   Note: Repo creation issue (might already exist): {e}")

        print(f"   ...Uploading files from {local_path}...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ Done: https://huggingface.co/{repo_id}\n")

    print("🎉 All Brains Uploaded!")


if __name__ == "__main__":
    upload_all()