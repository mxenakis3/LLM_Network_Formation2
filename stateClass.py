import networkx as nx

class State():
    def __init__(self):
        self.network = nx.Graph() #lightweight representation of network. Used to calculate spls and plot

        # Get shortest path distances, but disconnected nodes have to appear at inf. distance
        # Set initial spls to inf
        self.spls = {u: {v: float('inf') for v in self.network.nodes} for u in self.network.nodes}

        # Then update from single source path lengths
        for u in self.network.nodes:
            lengths = nx.single_source_shortest_path_length(self.network, u)
            self.spls[u].update(lengths)
    
        # Stores information about state evolution
        self.update_log = []

    def update_spls(self):
        # Online update function for when edges are added - recalculates spls
        # Initialize spls as infinity distance
        self.spls = {u: {v: float('inf') for v in self.network.nodes} for u in self.network.nodes}

        # Update from single source path lengths
        for u in self.network.nodes:
            lengths = nx.single_source_shortest_path_length(self.network, u)
            self.spls[u].update(lengths)


    def render(self, agent_id):
        # Renders a view of the current state for a given agent/node
        # 1. shortest path distances to all other nodes
        # 2. degree of all network nodes
        # 3. colors of all neighbors

        # Render:

        # Get the color of the neighbors of the current node
        if agent_id not in self.network:
            print(f"Node {agent_id} does not exist in graph")
            return
    
        neighbor_colors = {}
        for neighbor in self.network.neighbors(agent_id):
            neighbor_colors[neighbor] = self.network.nodes[neighbor]['color']

        return [self.spls[agent_id], dict(self.network.degree), neighbor_colors]
    

    def update_node_color(self, node, new_color):
        """
        Inputs: 
         - node: id of node to update
         - new_color: color of choice
        
        """
        if node in self.network:
            self.network.nodes[node]['color'] = new_color
        else:
            print(f"Error: Node {node} does not exist in network")
    

    def add_network_edge(self, edge):
        u, v = edge[0], edge[1]
        self.network.add_edge(u, v)
        self.update_spls()


    