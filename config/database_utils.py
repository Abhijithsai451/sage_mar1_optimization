import json
import os
import sqlite3
from typing import List, Optional

from langchain_core.messages import BaseMessage

from states.agent_state import SAGEAgentState
from config.logger_config import sars_logger as logger
from states.parameter_state import ParameterState
from states.rewards import RewardState
from states.scores import ScoreState
from states.tasks_state import TasksState
from dotenv import load_dotenv
load_dotenv()

DATABASE_FILE = os.getenv("SQLITE_DB_PATH")

def init_database():
    logger.info("Initializing the SQLite database")
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parameter_states( 
        id TEXT PRIMARY KEY, 
        alpha REAL NOT NULL, 
        beta REAL NOT NULL, 
        lambda_plan REAL NOT NULL,
        lambda_format REAL NOT NULL,
        w_p REAL NOT NULL,
        w_c REAL NOT NULL,
        w_f REAL NOT NULL) 
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reward_states(
        id TEXT PRIMARY KEY, 
        reward_challenger REAL NOT NULL, 
        reward_planner REAL NOT NULL, 
        reward_solver REAL NOT NULL, 
        reward_format REAL NOT NULL, 
        reward_diff REAL NOT NULL) 
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS score_states(
        id TEXT PRIMARY KEY, 
        score_quality REAL NOT NULL, 
        score_planner REAL NOT NULL, 
        score_ground_truth REAL NOT NULL) 
        """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sage_states(
    id TEXT PRIMARY KEY, 
    messages TEXT NOT NULL, 
    input TEXT NOT NULL, 
    parameter_state_id TEXT NOT NULL, 
    status TEXT NOT NULL, 
    FOREIGN KEY (parameter_state_id) REFERENCES parameter_states(id))
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS task_states(
        id TEXT PRIMARY KEY, 
        sage_state_id TEXT NOT NULL, 
        question TEXT NOT NULL, 
        rewards_id TEXT NOT NULL, 
        score_id TEXT NOT NULL, 
        plan TEXT NOT NULL, 
        solution TEXT NOT NULL, 
        FOREIGN KEY (rewards_id) REFERENCES reward_states(id),
        FOREIGN KEY (score_id) REFERENCES score_states(id),
        FOREIGN KEY (sage_state_id) REFERENCES sage_states (id) ON DELETE CASCADE) 
        """)
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized successfully")

def clear_db():
    logger.info("Clearing the database")
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")
    tables = ['task_states', 'reward_states', 'score_states', 'parameter_states', 'sage_states']
    for table in tables:
        try:
            cursor.execute(f"DELETE FROM {table}")
            logger.info(f"Cleared table: {table}")
        except sqlite3.OperationalError as e:
            logger.error(f"Error clearing table {table}: {e}")

    conn.commit()
    cursor.execute("PRAGMA foreign_keys = ON;")
    conn.close()
    logger.info("Database cleared successfully")

def _save_parameter_state(cursor, param_state: ParameterState):
    """Helper to save or update a ParameterState."""
    cursor.execute(
        '''INSERT OR REPLACE INTO parameter_states 
           (id, alpha, beta, lambda_plan, lambda_format, w_p, w_c, w_f) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (param_state.id, param_state.alpha, param_state.beta, param_state.lambda_plan,
         param_state.lambda_format, param_state.w_p, param_state.w_c, param_state.w_f)
    )

def _save_reward_state(cursor, reward_state: RewardState):
    """Helper to save or update a RewardState."""
    cursor.execute(
        '''INSERT OR REPLACE INTO reward_states 
           (id, reward_challenger, reward_planner, reward_solver, reward_format, reward_diff) 
           VALUES (?, ?, ?, ?, ?, ?)''',
        (reward_state.id, reward_state.reward_challenger, reward_state.reward_planner,
         reward_state.reward_solver, reward_state.reward_format, reward_state.reward_diff)
    )

def _save_score_state(cursor, score_state: ScoreState):
    """Helper to save or update a ScoreState."""
    cursor.execute(
        '''INSERT OR REPLACE INTO score_states 
           (id, score_quality, score_planner, score_ground_truth) 
           VALUES (?, ?, ?, ?)''',
        (score_state.id, score_state.score_quality, score_state.score_planner, score_state.score_ground_truth)
    )

def _save_tasks_state(cursor, task_state: TasksState, sage_state_id: str):
    """Helper to save or update a TasksState and its nested states."""
    _save_reward_state(cursor, task_state.rewards)
    _save_score_state(cursor, task_state.score)
    cursor.execute(
        '''INSERT OR REPLACE INTO task_states 
           (id, sage_state_id, question, rewards_id, score_id, plan, solution) 
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (task_state.id, sage_state_id, task_state.question, task_state.rewards.id,
         task_state.score.id, task_state.plan, task_state.solution)
    )

def save_agent_state(state: SAGEAgentState):
    """Saves or updates the SAGEAgentState and its nested states in the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    _save_parameter_state(cursor, state.parameter_state)

    cursor.execute('DELETE FROM task_states WHERE sage_state_id = ?', (state.id,))
    for task in state.tasks:
        _save_tasks_state(cursor, task, state.id)

    # Convert BaseMessage list to JSON string for storage
    messages_json = json.dumps([msg.model_dump() for msg in state.messages])
    input_json = json.dumps(state.input)

    cursor.execute(
        '''INSERT OR REPLACE INTO sage_states 
           (id, messages, input, parameter_state_id, status) 
           VALUES (?, ?, ?, ?, ?)''',
        (state.id, messages_json, input_json, state.parameter_state.id, state.status)
    )
    conn.commit()
    conn.close()
    print(f"SAGEAgentState with ID '{state.id}' and its components saved/updated successfully.")


def _load_parameter_state(cursor, param_id: str) -> Optional[ParameterState]:
    """Helper to load a ParameterState."""
    cursor.execute('SELECT * FROM parameter_states WHERE id = ?', (param_id,))
    row = cursor.fetchone()
    if row:
        return ParameterState(
            id=row[0], alpha=row[1], beta=row[2], lambda_plan=row[3],
            lambda_format=row[4], w_p=row[5], w_c=row[6], w_f=row[7]
        )
    return None

def _load_reward_state(cursor, reward_id: str) -> Optional[RewardState]:
    """Helper to load a RewardState."""
    cursor.execute('SELECT * FROM reward_states WHERE id = ?', (reward_id,))
    row = cursor.fetchone()
    if row:
        return RewardState(
            id=row[0], reward_challenger=row[1], reward_planner=row[2],
            reward_solver=row[3], reward_format=row[4], reward_diff=row[5]
        )
    return None

def _load_score_state(cursor, score_id: str) -> Optional[ScoreState]:
    """Helper to load a ScoreState."""
    cursor.execute('SELECT * FROM score_states WHERE id = ?', (score_id,))
    row = cursor.fetchone()
    if row:
        return ScoreState(
            id=row[0], score_quality=row[1], score_planner=row[2], score_ground_truth=row[3]
        )
    return None

def _load_task_states(cursor, sage_state_id: str) -> List[TasksState]:
    """Helper to load all TasksState instances for a given sage_state_id."""
    tasks = []
    cursor.execute('SELECT id, question, rewards_id, score_id, plan, solution FROM task_states WHERE sage_state_id = ?', (sage_state_id,))
    for row in cursor.fetchall():
        task_id, question, rewards_id, score_id, plan, solution = row
        rewards = _load_reward_state(cursor, rewards_id)
        score = _load_score_state(cursor, score_id)
        if rewards and score: # Ensure nested states were loaded
            tasks.append(TasksState(
                id=task_id, question=question, rewards=rewards, score=score, plan=plan, solution=solution
            ))
    return tasks


def load_agent_state(state_id: str) -> Optional[SAGEAgentState]:
    """Loads a SAGEAgentState and its nested states from the database by its ID."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    cursor.execute('SELECT id, messages, input, parameter_state_id, status FROM sage_states WHERE id = ?', (state_id,))
    agent_row = cursor.fetchone()
    conn.close() # Close connection early if no agent state found

    if not agent_row:
        print(f"No SAGEAgentState found with ID '{state_id}'.")
        return None

    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    agent_id, messages_json, input_json, parameter_state_id, status = agent_row

    param_state = _load_parameter_state(cursor, parameter_state_id)
    tasks = _load_task_states(cursor, agent_id)

    conn.close()

    if param_state:
        # Convert JSON strings back to lists/objects
        messages_dicts = json.loads(messages_json)
        # Reconstruct BaseMessage objects (this might need refinement depending on actual BaseMessage subclasses)
        messages = [BaseMessage(**msg_dict) for msg_dict in messages_dicts]
        input_list = json.loads(input_json)

        return SAGEAgentState(
            id=agent_id,
            messages=messages,
            input=input_list,
            parameter_state=param_state,
            tasks=tasks,
            status=status
        )
    return None

def list_sage_state_ids() -> List[str]:
    """Lists all available agent state IDs in the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM sage_states')
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids




