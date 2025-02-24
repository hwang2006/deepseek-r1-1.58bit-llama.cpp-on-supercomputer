import os
from huggingface_hub import snapshot_download

local_dir = os.path.expandvars("/scratch/$USER/llama.cpp/DeepSeek-R1-GGUF")

snapshot_download(
    repo_id = "unsloth/DeepSeek-R1-GGUF",  # Specify the Hugging Face repo
    local_dir = local_dir,         # Model will download into this directory
    allow_patterns = ["*UD-IQ1_S*"],        # Only download the 1.58-bit version
)
