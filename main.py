from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import time
from agentClass import ReActAgent
from stateClass import State
import time
from datetime import datetime
from config_py import agent_config
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
from utils.log_functions import *
from utils.network_plots import *
import random
from tqdm import tqdm
import pickle
from runner_config import runner_config
from Tool_configs import agent_actions

# Load in API Key and Handle Errors
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Disable plots in matplotlib and save plots directly
plt.switch_backend('Agg') 

async def run_trial(run_config, agent_config, tool_config):
    """
    Runs a single experiment from configs. run_config is imported.
    Returns:
    - network: nx state
    - agents: agent dictionary {agent_id: Agent}
    """
    # Initialize network and client
    state = State()
    client = AsyncOpenAI(api_key= openai_api_key)

    num_agents = run_config['V']
    available_ids = random.sample(range(100, 1000), num_agents)  # Unique 3-digit IDs

    # Initialize dictionary to store agents. Key agent_id, value: agent
    agents = {}

    for i in range(num_agents):
        agent_id = available_ids[i]
        agent = ReActAgent(
            client=client,
            id=agent_id,
            type_specific_params=run_config[f"agent_{i % 2}"],
            time_limit=run_config["duration"],
            agent_config= agent_config,
            tool_config = tool_config
        )
        agents[agent_id] = agent
        state.network.add_node(agent_id, color=None)

    # Reset spls
    state.update_spls()
    
    # Store start time and define time limit of duration
    start_time = time.time()
    time_limit = run_config["duration"]

    # Run async simulation
    with tqdm(total=time_limit, desc="Simulation Progress", unit="s") as pbar:        
        tasks = [asyncio.create_task(x.agent_loop(state, time_limit)) for x in agents.values()]
        while time.time() - start_time < time_limit:
            elapsed_time = time.time() - start_time
            pbar.update(elapsed_time - pbar.n)  
            await asyncio.sleep(0.1) 
        await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Experiment Total time: {round(end_time - start_time, 2)}")

    # Save selected agent properties into a separate dict (load into config)
    agent_results = {}
    for id, ag in agents.items():
        agent_results[id] = {}
        agent_results[id]['color'] = ag.color
        agent_results['consensus_0_reward'] = ag.consensus_0_reward
        agent_results['consensus_1_reward'] = ag.consensus_1_reward
        agent_results['permanent_memory'] = ag.permanent_memory


    return state.network, agent_results


async def runner(run_config, agent_config, tool_config):
    """
    Calls run experiments and saves network and agents dictionary for each run in separate pickle files.
    Inputs:
    - run_config: specifies parameters for a single run
    Outputs:
    - None, but creates a folder which stores info in .pkl files
    """

    results = {}
    # Run trials
    for idx in tqdm(range(run_config['num_trials'])):
        results[idx] = await run_trial(run_config = run_config, agent_config=agent_config, tool_config=agent_actions)
        await asyncio.sleep(10)

    # Save results
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f".\outputs\{curr_time}"

    # Store files
    os.makedirs(dir_path, exist_ok=True)
    
    # Configs
    with open(os.path.join(dir_path, f"run_configs"), "wb") as f:
        pickle.dump(run_config, f)
    with open(os.path.join(dir_path, f"agent_configs"), "wb") as f:
        pickle.dump(agent_config, f)
    with open(os.path.join(dir_path, f"tool_configs"), "wb") as f:
        pickle.dump(tool_config, f)
    
    # Data
    for key, data in results.items():
        filename = os.path.join(dir_path, f"{key}_standard")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    

if __name__ == "__main__":
    asyncio.run(runner(run_config = runner_config, agent_config = agent_config, tool_config = agent_actions))

    
    
    








    
