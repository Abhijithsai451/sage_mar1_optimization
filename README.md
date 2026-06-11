# SAGE-RL: A Multi-Agent Self-Evolving Framework for Enhancing LLM Reasoning

SAGE-RL is a multi-agent framework designed to enhance Large Language Model (LLM) reasoning capabilities through task-relative reinforcement learning. Inspired by advanced curriculum learning and policy optimization techniques, SAGE-RL orchestrates specialized AI agents that collaboratively tackle complex problems, iteratively refining their strategies and improving proficiency.

The core idea is a self-evolving system where LLMs learn from their own generated tasks and solutions, guided by a sophisticated reward mechanism.

## Key Features

1.  **Multi-Agent Orchestration:** Uses **LangGraph** to define a structured workflow between distinct agents (Challenger, Planner, Solver, Critic).
2.  **Curriculum Learning:** The Challenger agent dynamically generates increasingly difficult tasks based on reference examples.
3.  **Chain-of-Thought (CoT) Reasoning:** The Planner agent develops detailed strategies, fostering robust reasoning pathways.
4.  **Reinforcement Learning (REINFORCE++):** Rewards are calculated based on solution quality and format adherence, using per-role normalization and KL-Divergence to prevent model drift.
5.  **Local Prototyping & Scaling:** Optimized for local development on Apple Silicon (using vLLM-metal) and scalable to HPC environments with CUDA.
6.  **Continuous Improvement:** Integrated logging and evaluation via Weights & Biases (WandB) and SQLite for tracking agent performance and reward curves.

---

## Tech Stack

- **Language:** Python 3.x
- **Frameworks:** LangGraph, LangChain, vLLM
- **Package Manager:** pip
- **Data:** Hugging Face `datasets` (GSM8K)
- **Monitoring:** Weights & Biases (WandB)
- **Database:** SQLite
- **Hardware Acceleration:** Apple Silicon (vLLM-metal) / CUDA

---

## Project Structure

```text
.
├── agents/             # Core agent logic and nodes
│   ├── challenger.py   # Task generation agent
│   ├── planner.py      # Strategy/CoT generation agent
│   ├── solver.py       # Solution implementation agent
│   ├── critic.py       # Evaluation and scoring agent
│   ├── reward.py       # Reward calculation logic
│   ├── graph_workflow.py # LangGraph orchestration
│   └── tools.py        # Utility tools for agents
├── config/             # Configuration and utility modules
│   ├── data_config.py  # Dataset loading (GSM8K)
│   ├── database_utils.py # SQLite persistence logic
│   ├── logger_config.py # Centralized logging
│   └── model_config.py # Model and LoRA configuration
├── states/             # Pydantic state models for LangGraph
│   ├── agent_state.py  # Shared SAGEAgentState
│   └── ...             # Component states (tasks, rewards, etc.)
├── data/               # Cached datasets
├── base_model/         # Directory for local LLM weights
├── adapters/           # TODO: Directory for LoRA adapters (placeholders)
├── main.py             # Entry point for workflow simulation
├── test.py             # Entry point for RL optimization/training
├── start.sh            # vLLM server startup script
├── requirements.txt    # Python dependencies
└── .env                # Environment variables (to be created)
```

---

## Requirements & Setup

### Prerequisites
- Python 3.10+
- Apple Silicon Mac (for `vllm-metal`) or Linux with CUDA support.
- [vLLM](https://docs.vllm.ai/en/latest/) installed.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd sage_mar1_optimization
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    DATA_DIR=./data
    GREETING="Hello, I am SAGE-RL. How can I assist you today?"
    WANDB_API_KEY=your_wandb_key_here
    ```

---

## Running the Project

### 1. Start the vLLM Server
The system relies on a running vLLM server to serve the backbone model and manage LoRA adapters.
```bash
bash start.sh
```
*Note: Ensure your `base_model` and `adapters/` directories are correctly populated.*

### 2. Run Workflow Simulation
To execute a single pass of the multi-agent loop:
```bash
python main.py
```
This script initializes the database, loads GSM8K samples, triggers the Challenger to create new tasks, and runs them through the Planner, Solver, Critic, and Reward agents.

### 3. Run Training / Optimization
To start the self-evolutionary training cycle:
```bash
python test.py
```
This script uses the `sage_trainer` to perform on-policy policy gradient updates (REINFORCE++).

-

## Scripts

- `start.sh`: Launches `vllm serve` with specific LoRA modules for each agent role (`challenger`, `planner`, `solver`, `critic`, `reward`).
- `main.py`: Demonstrates the LangGraph workflow and saves a visualization to `agent_workflow.png`.
- `test.py`: Implementation of the RL training loop.


## License
[Add License Info Here - e.g., MIT]