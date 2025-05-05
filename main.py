from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import time
from agentClass import ReActAgent
from stateClass import State
import time
from datetime import datetime
from config_py import run_config
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
from log_functions import *
from network_plots import *
import random
from tqdm import tqdm
import pickle
from runner_config import *

# Load in API Key and Handle Errors
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Disable plots in matplotlib and save plots directly
plt.switch_backend('Agg') 

async def run_experiment(logs= False, plots=False):
    """
    Returns:
    - network: nx state
    - unanimous: (bool) True if consensus reached on color
    - rewards: list of rewards for each agent
    - node_degrees_pdf: dict of k: p(k)
    - spls_pdf: dict of spl: p(spl)
    """
    # Initialize network and client
    state = State()
    client = AsyncOpenAI(api_key= openai_api_key)

    num_agents = run_config['init_configs']['V']
    available_ids = random.sample(range(100, 1000), num_agents)  # Unique 3-digit IDs

    # Initialize dictionary to store agents. Key agent_id, value: agent
    agents = {}

    for i in range(num_agents):
        agent_id = available_ids[i]
        agent = ReActAgent(
            client=client,
            id=agent_id,
            type_specific_params=run_config["init_configs"][f"agent_{i % 2}"],
            time_limit=run_config["init_configs"]["duration"]
        )
        agents[agent_id] = agent
        state.network.add_node(agent_id, color=None)

    state.update_spls()
    
    start_time = time.time()
    time_limit = run_config["init_configs"]["duration"]

    with tqdm(total=time_limit, desc="Simulation Progress", unit="s") as pbar:        
        tasks = [asyncio.create_task(x.agent_loop(state, time_limit)) for x in agents.values()]
        while time.time() - start_time < time_limit:
            elapsed_time = time.time() - start_time
            pbar.update(elapsed_time - pbar.n)  
            await asyncio.sleep(0.1) 
        await asyncio.gather(*tasks)

    end_time = time.time()

   ## LOG EXPERIMENTS ##
   # Check if all nodes are unanimous
    unanimous, choice = state.check_unanimous()

    # Get rewards for agents
    if choice is None:
        rewards = [0]*len(list(agents.keys()))
    else:
        if choice == 0:
            rewards = [agent.consensus_0_reward for agent in agents]
        elif choice == 1:
            rewards = [agent.consensus_0_reward for agent in agents]

    # Get node degrees as list
    node_degrees = [deg for _, deg in state.network.degree()]
    probs, uniques = get_prob_dist(node_degrees)
    node_degrees_pdf = dict(zip(uniques, probs))
    
    # Initialize a list to store all path lengths
    node_spls = []
    # Extract the shortest path lengths from the results
    for source, targets in state.spls.items():
        for target, length in targets.items():
            if source != target:
                node_spls.append(length)
    probs, uniques = get_prob_dist(node_spls)
    spls_pdf = dict(zip(uniques, probs))

    print(f"Experiment Total time: {round(end_time - start_time, 2)}. Unanimous: {unanimous}. Avg Rew: {np.mean(rewards)}")

    # Create folder to store output logs
    if logs or plots:
        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = f".\outputs\plots_and_logs\{curr_time}"
        os.makedirs(dir_path)

    # Create logs
    if logs:
        create_state_log(state, save_path= os.path.join(dir_path, f"state_log.txt"))
        create_agent_logs(agents, dir_path)

    # Create Plots
    if plots:
        plot_network(state.network, os.path.join(dir_path, f"network.png"))
        plot_degree_distribution(state.network, os.path.join(dir_path, f"degree_dist.png"))
        plot_shortest_path_freq(state.network, os.path.join(dir_path, f"spls_dist.png"))

    await asyncio.sleep(10) # sleep for 10 seconds to avoid token limits
    
    # return network nodes, 
    return state, unanimous, rewards, node_degrees_pdf, spls_pdf


async def runner(run_config):

    # Plot first set of trials
    for _ in tqdm(range(run_config['num_plotted_standard_trials'])):
        await run_experiment(logs=True, plots=True)
        await asyncio.sleep(10)

    results = {
        "networks": [],
        "unanimous": [],
        "rewards": [],
        "degrees": [],
        "spls": []
    }

    for _ in range(run_config['num_standard_trials']):
        network, unanimous, rewards, degrees, spls = await run_experiment()
        results["networks"].append(network)
        results["unanimous"].append(unanimous)
        results["rewards"].append(rewards)
        results["degrees"].append(degrees)
        results["spls"].append(spls)
        await asyncio.sleep(30)


    # Set directory time
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f".\outputs\std_trials_{curr_time}"

    #store files
    os.makedirs(dir_path, exist_ok=True)
    for key, data in results.items():
        filename = os.path.join(dir_path, f"{key}_standard")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    

if __name__ == "__main__":
    # Intialize state and agent.
    # The state is just a representation of the state. The state class does not contain agent objects
    asyncio.run(runner(run_config = runner_config))

    
    
    








    
