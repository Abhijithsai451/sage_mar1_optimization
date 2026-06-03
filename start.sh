
source /Users/abhijithsai/.venv-vllm-metal/bin/activate
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve ./base_model \
  --port 8000 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-model-len 8192 \
  --lora-modules \
    challenger=./adapters/challenger_lora \
    planner=./adapters/planner_lora \
    solver=./adapters/solver_lora \
    critic=./adapters/critic_lora \
    reward=./adapters/reward_lora




