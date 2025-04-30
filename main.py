from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import time
from agentClass import ReActAgent
from stateClass import State
import time
from datetime import datetime
from config_py import config
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from log_functions import *
from network_plots import *
import random

# Load in API Key and Handle Errors
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Disable plots in matplotlib and save plots directly
plt.switch_backend('Agg') 

async def run_experiment():
    # Initialize network and client
    state = State(config['init_configs'])
    client = AsyncOpenAI(api_key= openai_api_key)


    num_agents = config['init_configs']['V']
    available_ids = random.sample(range(100, 1000), num_agents)  # Unique 3-digit IDs

    # Initialize dictionary to store agents. Key agent_id, value: agent
    agents = {}

    for i in range(num_agents):
        agent_id = available_ids[i]
        agent = ReActAgent(
            client=client,
            id=agent_id,
            type_specific_params=config["init_configs"][f"agent_{i % 2}"],
            time_limit=config["init_configs"]["duration"]
        )
        agents[agent_id] = agent
        state.network.add_node(agent_id, color=None)

    print(f"Nodes in network: {state.network.nodes(data=True)}")
    state.update_spls()
    print(f"Spls after agents added: {state.spls}")
    print(f"Agents: {list(agents.keys())}")
    
    start_time = time.time()
    time_limit = config["init_configs"]["duration"]

    with tqdm(total=time_limit, desc="Simulation Progress", unit="s") as pbar:        
        tasks = [asyncio.create_task(x.agent_loop(state, time_limit)) for x in agents.values()]
        while time.time() - start_time < time_limit:
            elapsed_time = time.time() - start_time
            pbar.update(elapsed_time - pbar.n)  
            await asyncio.sleep(0.1) 
        await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")

    # Create folder to store output logs
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f".\outputs\{curr_time}"
    os.makedirs(dir_path)

    # Create logs
    create_state_log(state, save_path= os.path.join(dir_path, f"state_log.txt"))
    create_agent_logs(agents, dir_path)

    # Create Plots
    plot_network(state.network, os.path.join(dir_path, f"network.png"))
    plot_average_neighbor_degree(state.network, os.path.join(dir_path, f"average_neighbor_degree.png"))
    plot_degree_distribution(state.network, os.path.join(dir_path, f"degree_dist.png"))
    plot_shortest_path_freq(state.network, os.path.join(dir_path, f"spls_dist.png"))
    plot_clustering_coefficient(state.network, os.path.join(dir_path, f"spls_dist.png"))

if __name__ == "__main__":
    # Intialize state and agent.
    # The state is just a representation of the state. The state class does not contain agent objects
    asyncio.run(run_experiment())
    








    
