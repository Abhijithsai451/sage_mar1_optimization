
source /Users/abhijithsai/.venv-vllm-metal/bin/activate
vllm serve ./base_model \
  --port 8000 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-model-len 8192 \
  --lora-modules \
    challenger=./adapters/challenger_lora \
    planner=./adapters/planner_lora \
    solver=./adapters/solver_lora \
    critic=./adapters/critic+lora




