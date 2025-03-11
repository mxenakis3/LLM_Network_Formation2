from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import networkx as nx
import asyncio
import time
from agentClass import ReActAgent
from stateClass import State
import time
from datetime import datetime
from config_py import config
import re
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from collections import Counter


# Load in API Key and Handle Errors
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Disable plots in matplotlib and save plots directly
plt.switch_backend('Agg') 

def parse_chat_log(chat_log):
    formatted_output = []
    
    for entry in chat_log:
        # Split only on the outermost commas, preserving commas inside text
        split_entries = re.split(r',\s*(?=[A-Z])', entry)  # Comma followed by capital letter (heuristic for separation)
        
        for sub_entry in split_entries:
            cleaned_text = re.sub(r'\n\s+', '\n', sub_entry.strip())  # Clean extra spaces/newlines
            
            # Categorize messages
            if "Thought:" in cleaned_text or "Action:" in cleaned_text:
                formatted_output.append(f"\n{cleaned_text}\n")
            else:
                formatted_output.append(f"\n{cleaned_text}\n")
    
    return "\n".join(formatted_output)

def plot_network(G, save_path="network.png"):
    # Define color mapping
    color_map = {
        '1': 'blue',
        '0': 'red',
        None: 'white'
    }

    # Extract node colors
    node_colors = [color_map.get(G.nodes[n].get('color', None), 'white') for n in G.nodes]

    # Extract labels
    labels = {n: G.nodes[n].get('agent_id', n) for n in G.nodes}

    # Create plot
    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_colors, with_labels=True, labels=labels, edge_color='gray', node_size=500, font_size=10)

    plt.title(f"Agents: {len(G.nodes)}, Duration: {config['init_configs']['duration']}")

    # Create legend manually
    legend_patches = [
        mpatches.Patch(color='blue', label='Color 1'),
        mpatches.Patch(color='red', label='Color 0'),
        mpatches.Patch(color='white', label='Undeclared')
    ]
    plt.legend(handles=legend_patches, loc="upper right")

    fig.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")


def plot_degree_distribution(G, save_path="degree_distribution.png"):
    # Get degrees of all nodes
    degrees = [deg for _, deg in G.degree()]

    # Count occurrences of each degree
    degree_counts = Counter(degrees)

    # Separate keys (degree values) and values (counts)
    unique_degrees = np.array(list(degree_counts.keys()))
    counts = np.array(list(degree_counts.values()))

    # Normalize to get probability distribution P(k)
    probabilities = counts / counts.sum()

    # Create simple scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
    ax.scatter(unique_degrees, probabilities, color='blue', label="Degree Distribution", s=50)

    # Labels and title
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("P(k)")
    ax.set_title("Degree Distribution")

    # Show grid
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Save the plot before returning
    fig.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")

    return 

def plot_shortest_path_freq(G, save_path="shortest_path_distribution.png"):
    # Compute all-pairs shortest path lengths
    spls = nx.all_pairs_shortest_path_length(G)
    
    # Initialize a list to store all path lengths
    path_lengths = []

    # Extract the shortest path lengths from the results
    for source, targets in spls:
        for target, length in targets.items():
            if source != target:  # Ignore self-loops
                path_lengths.append(length)

    # Count the frequency of each path length
    path_length_counts = Counter(path_lengths)

    # Separate keys (path lengths) and values (counts)
    unique_lengths = np.array(list(path_length_counts.keys()))
    counts = np.array(list(path_length_counts.values()))

    # Normalize to get the probability distribution
    probabilities = counts / counts.sum()

    # Create a simple scatter plot for path length distribution
    plt.figure(figsize=(8, 6))
    plt.scatter(unique_lengths, probabilities, color='blue', s=100)  # Larger markers for visibility
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Probability')
    plt.title('Shortest Path Length Distribution')

    # Save the plot
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")

def get_nearest_neighbors(G):
    """
    Inputs:
        G: NetworkX graph object
    
    Returns:
        k (list): Node degrees
        knn (list): Average neighbor degree for each k
    """
    # Get node degrees
    degrees = dict(G.degree())

    # Get average neighbor degree
    neighbor_degrees = nx.average_neighbor_degree(G)

    # Record knns indexed by the degree k
    k_knn = dict()
    for n in G.nodes():
        node_degree = degrees[n]
        if node_degree not in k_knn:
            k_knn[node_degree] = []
        k_knn[node_degree].append(neighbor_degrees[n])

    # Calculate the average of knn among nodes with the same degree k
    for k in k_knn:
        k_knn[k] = sum(k_knn[k]) / len(k_knn[k])
    
    k = list(k_knn.keys())
    knn = list(k_knn.values())
    
    return k, knn

def plot_average_neighbor_degree(G, save_path="average_neighbor_degree.png"):
    """
    Inputs:
        G: NetworkX graph object
        save_path (str): Path where the plot will be saved
    
    Returns:
        None, but saves the visualization to a file
    """
    # Get the nearest neighbors (k and knn)
    k, knn = get_nearest_neighbors(G)

    # Create the plot
    plt.figure(figsize=(6, 4))
    plt.scatter(k, knn, marker='o', color='skyblue', edgecolor='black')
    plt.xlabel('Node Degree (k)')
    plt.ylabel('Average Neighbor Degree (Knn)')
    plt.title('Average Neighbor Degree vs Node Degree')
    
    # Save the plot to the specified path
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")


def plot_clustering_coefficient(G, save_path="clustering_distribution.png"):
    # Clustering Coefficient
    clustering = nx.clustering(G)  # Get clustering coefficients for all nodes
    avg_clustering = sum(clustering.values()) / len(clustering)  # Average clustering coefficient
    
    # Plot Clustering Coefficients
    plt.figure(figsize=(6, 4))
    plt.hist(list(clustering.values()), bins=10, color='skyblue', edgecolor='black')
    plt.title('Clustering Coefficient Distribution')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')

    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")

    # Return average clustering coefficient
    return avg_clustering

def create_state_log(state, save_path="shortest_path_distribution.png"):
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    final_colors = {attributes["color"] for _, attributes in state.network.nodes(data=True)}

    # Fix file path to save the log correctly
    file_path = os.path.join(save_path, "state_update_log.txt")

    with open(file_path, "w") as file:
        file.write(parse_chat_log(state.update_log))
        if len(final_colors) == 1:
            file.write(f"\n Consensus was reached.")
        else:
            file.write(f"\n Consensus was not reached.")


def create_agent_logs(agents, dir_path):
    # Ensure the directory exists before trying to write files
    os.makedirs(dir_path, exist_ok=True)

    for agent_id, agent in agents.items():
        file_path = os.path.join(dir_path, f"agent_{agent_id}.txt")
        with open(file_path, "w") as file:
            file.write(parse_chat_log(agent.permanent_memory))

async def async_main():
    state = State(config['init_configs'])
    client = AsyncOpenAI(api_key= openai_api_key)

    # Initialize dictionary to store agents. Key agent_id, value: agent
    agents = {}
    for i in range(config['init_configs']['V']): # Stores the number of nodes in the simulation
        agent = ReActAgent(client= client,  id=i, type_specific_params = config["init_configs"][f"agent_{i%2}"], time_limit = config["init_configs"]["duration"])
        agents[i] = agent

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
    asyncio.run(async_main())
    








    
