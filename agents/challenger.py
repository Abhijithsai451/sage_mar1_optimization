import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from mlx_lm import load, generate

load_dotenv()
model_dir = os.getenv("MODEL_DIR")

designer_prompt ="""
Role: Task Designer Agent
Description:
You are a task generation specialist. Your goal is to create a single, high-quality evaluation task that challenges complex reasoning abilities.
Design Constraints:
- Self-contained with clear problem statement
- Non-trivial: requires multiple reasoning steps or constraint satisfaction
- Deterministic or tightly bounded (avoid subjective judgment)
- Culturally neutral, no real-time data dependency
- Difficult but solvable
Avoid:
- Trivia or opinion-based prompts
- Ambiguous success criteria
- Web-dependent or time-sensitive content
- Unsolvable or ill-defined problems
Respond using:
<question>
[Your generated task here]
</question>
"""

def run_challenger():
    #model, tokenizer = load(model_dir, tokenizer_config={"eos_token": "<|im_end|>"})
    model, tokenizer = load(model_dir)
    text = tokenizer.apply_chat_template(
        designer_prompt, tokenize=False, add_generation_prompt=True
    )

    response = generate(model, tokenizer, prompt=text, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)
    return response
