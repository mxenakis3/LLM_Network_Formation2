def check_unanimous(agents):
    """
    Inputs: 
    - state: State object
    Outputs:
    - unanimous: (bool) True if all nodes choose the same color
    - choice(str/Nonetype): string representation of consensus color if consensus exists, else None
    """

    colors = {G.nodes[node]['color'] for node in G.nodes}
    # Check if all nodes are either 0 or 1, and if all colors are the same
    if len(colors) == 1 and all(color in {str(0), str(1)} for color in colors):
        return True, colors.pop()  # pop() removes and returns the only element in the set
    return False, None

def get_rewards(agents):
    """
    Inputs:
    - agents: dictionary of agents
    Outputs:
    - 
    """
    return
    
def get_spls(agents):
    """
    Inputs:
    - agents: dictionary of agents, containing self.spls attributes
    Outputs:
    - List of all shortest path length occurences across all agents
    """
    return
