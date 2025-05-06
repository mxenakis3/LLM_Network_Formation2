
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.stats import linregress
import numpy as np
from collections import Counter
from collections import defaultdict

# def plot_network(G, save_path="network.png"):
#     # Define color mapping
#     color_map = {
#         '1': 'blue',
#         '0': 'red',
#         None: 'white'
#     }

#     # Extract node colors
#     node_colors = [color_map.get(G.nodes[n].get('color', None), 'white') for n in G.nodes]

#     # Extract labels
#     labels = {n: G.nodes[n].get('agent_id', n) for n in G.nodes}

#     # Create plot
#     fig = plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, node_color=node_colors, with_labels=True, labels=labels, edge_color='gray', node_size=500, font_size=10)

#     plt.title(f"Agents: {len(G.nodes)}, Duration: {run_config['init_configs']['duration']}")

    # # Create legend manually
    # legend_patches = [
    #     mpatches.Patch(color='blue', label='Color 1'),
    #     mpatches.Patch(color='red', label='Color 0'),
    #     mpatches.Patch(color='white', label='Undeclared')
    # ]
    # plt.legend(handles=legend_patches, loc="upper right")

    # fig.savefig(save_path, format="png", dpi=300)
    # plt.close(fig)  
    # print(f"Plot saved to {save_path}")

def get_prob_dist(data):
    """
    Inputs:
    - data: list of discrete vars
    Outputs:
    - uniques: list of unique degree values
    - probs: list of probability associated with each unique degree values
    """
    # counts = Counter(data)
    # uniques = np.array(list(counts.keys()))
    # probs = counts / counts.sum()
    # return probs, uniques
    counts = Counter(data)
    uniques = np.array(list(counts.keys()))
    probs = np.array(list(counts.values())) / sum(counts.values())  # Calculate probabilities
    return probs, uniques

def plot_degree_distribution(G, save_path="degree_distribution.png"):
    # Get degrees of all nodes
    degrees = [deg for _, deg in G.degree()]
    probs, uniques = get_prob_dist(degrees)

    # Create simple scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
    ax.scatter(uniques, probs, color='blue', label="Degree Distribution", s=50)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("P(k)")
    ax.set_title("Degree Distribution")

    # Show grid
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Save the plot before returning
    fig.savefig(save_path, format="png", dpi=300)
    plt.close(fig) 

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

    probs, uniques = get_prob_dist(path_lengths)

    # Create a simple scatter plot for path length distribution
    plt.figure(figsize=(8, 6))
    plt.scatter(uniques, probs, color='blue', s=100)  # Larger markers for visibility
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Probability')
    plt.title('Shortest Path Length Distribution')
    # Save the plot
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()  
    print(f"Plot saved to {save_path}")


def plot_across_pdfs(dict_list, title, x_axis, y_axis, color, save_path="degree_distribution.png"):
    """
    Inputs: 
    - dict_list: list of dict objects representing a discrete pdf (x: p(x))
    - title: title of plot
    - x_axis: title of x-axis
    - y_axis: title of y-axis
    Outputs:
    - a log-log plot
    - x axis: unique values
    - y-axis: avg probability ± stdev
    """

    # Find all unique keys across all PDFs
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    all_keys = sorted(all_keys)

    # Pad each dict with zeros where keys are missing
    padded_dicts = []
    for d in dict_list:
        padded = {k: d.get(k, 0.0) for k in all_keys}
        padded_dicts.append(padded)

    # Aggregate
    master_dict = defaultdict(list)
    for d in padded_dicts:
        for k, prob in d.items():
            master_dict[k].append(prob)

    means = [np.mean(master_dict[k]) for k in all_keys]
    stds = [np.std(master_dict[k]) for k in all_keys]

    # Filter out zero or negative x/y for log-log plot
    filtered = [(x, m, s) for x, m, s in zip(all_keys, means, stds) if x > 0 and m > 0]
    if not filtered:
        print("No positive values to plot on log-log scale.")
        return

    xs, ys, yerrs = zip(*filtered)
    # Convert to log-log for fitting
    log_xs = np.log10(xs)
    log_ys = np.log10(ys)

    # Fit linear regression in log-log space
    slope, intercept, r_value, _, _ = linregress(log_xs, log_ys)

    # Create trendline in log-log space
    fit_ys = 10 ** (intercept + slope * log_xs)  # back-transform to original scale



    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the trendline
    ax.plot(xs, fit_ys, linestyle='--', color='red', label=f"Fit: P(k) ∝ k^{{{slope:.2f}}}")
    ax.errorbar(xs, ys, yerr=yerrs, fmt='o-', capsize=5, color=color, label='Avg ± Std Dev')
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format="png", dpi=300)
    plt.close(fig)
    print(f"Plot saved to {save_path}")

def histogram_from_lol(list_of_lists, title, x_axis, y_axis, save_path):
    """
    Inputs: list_of_lists
    Outputs:
    Histogram of the averages of each of the lists
    """
    plt.figure(figsize=(8, 6))  # Create new figure
    to_np = np.array(list_of_lists)
    avg_rew = np.mean(to_np, axis = 1)
    plt.hist(avg_rew, bins=20)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()  # Optional: closes the figure to free memory
    print(f"Plot saved to {save_path}")
