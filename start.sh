
source /Users/abhijithsai/.venv-vllm-metal/bin/activate
vllm serve ./base_model \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules \
    challenger=./adapters/challenger_lora \
    planner=./adapters/planner_lora \
    solver=./adapters/solver_lora \
    critic=./adapters/critic+lora \




