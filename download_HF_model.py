from huggingface_hub import snapshot_download

# repo_id="Qwen/Qwen2-1.5B"
# repo_id = "locuslab/tofu_ft_phi-1.5"
repo_id = "microsoft/phi-1_5"

local_dir = "/lus/lfs1aip1/home/britllm/ksamway.britllm/workspace/FiLLM/models/phi"

snapshot_download(repo_id=repo_id, local_dir=local_dir)
