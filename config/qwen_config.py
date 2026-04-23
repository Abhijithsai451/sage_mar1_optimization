import os
from huggingface_hub import snapshot_download
from mlx_lm import load, generate
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()
model_dir = os.getenv("MODEL_DIR")
def download_backbone():
    hf_token = os.getenv("HF_TOKEN")
    model_id = "mlx-community/Qwen3.5-9B-MLX-4bit"
    local_dir = "./qwen_model"

    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"Downloading {model_id} to {local_dir}...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=hf_token
        )
        print("Download complete!")
    else:
        print(f"Model already exists in {local_dir}, skipping download.")


def test_model(directory = model_dir, prompt="Hello, world!"):
    print("Loading model from local folder...")
    model, tokenizer = load(directory)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Testing the Downloaded Model...")
    response = generate(model, tokenizer, prompt=text, verbose=True)

    return response
