from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import networkx as nx

# Load in API Key and Handle Errors
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Read in config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class ReActAgent:
    def __init__(self, client, system, id, type_specific_params = {"edge_cost": 0, "consensus_0_reward": 0, "consensus_1_reward":0}): # system prompt: you are working in a loop, etc. 
        self.id = id
        self.type = self.id % 2
        self.color = "Undeclared" # Previously set to None, but agents were under the impression that "None" meant that an edge could be purcahsed to reveal agent's color
        self.edge_cost = type_specific_params["edge_cost"]
        self.consensus_0_reward = type_specific_params["consensus_0_reward"]
        self.consensus_1_reward = type_specific_params["consensus_1_reward"]
        self.client = client
        self.system = system # system message
        self.messages = [] # memory/scratchpad
        self.permanent_memory = []

        # The following dictionaries are updated from the agent's view of the state in the main loop.
        self.colors = {}
        self.spls = {}
        self.degrees = {}

        if self.system is not None:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message=None, react_max_iters=10): 
        # Agent reacts to current view of state and stores reaction in memory
        for idx in range(react_max_iters):
            if message:
                self.messages.append({"role": "user", "content": message}) # This is how a user query is added to the list of messages for the agent

            self.messages.append({"role": "user", "content": f"""
                                  Currently, your projected reward for a consensus of 0 is {self.consensus_0_reward},
                                  and your projected reward for a consensus of 1 is {self.consensus_1_reward}.

                                  Your color is currently {self.color}
                                """}) # This is how a user query is added to the list of messages for the agent

             # See what the agent wants to do. Will be in form "Thought: {thought} \n Actioon: {action}"
            raw_result = self.execute()
            print(f"Raw result: {raw_result}")

            # Separate raw_result into thought, action, etc. 
            parsed_result = self.parse_output(raw_result)

            # Iterate through thought and action and add to messages
            for item in list(parsed_result.values()):
                # Add the result to the list of messages
                print(item)
                self.messages.append({"role": "assistant", "content": str(item)})


            # Retrieve the action from the parsed output.  This is hard coded based on the outputs I was getting, but could change
            action_description = self.messages[-1]["content"]

            # Uses the helper function to process/call the action
            action = self.process_action(action_description) 

            # If the result of the parsing is a dictionary, that means that the action input was not correctly formatted. 
            # parsed_result is a dictionary which has a key "break_loop". If this is false, we do NOT exit-- and simply add our outputs to memory and repeat the loop

            # Action: {"value": obj, "break_loop": Bool}
            if action["break_loop"] == True:
                return action

        # Agent has spent up to react_max_iters iterations deliberating
        # - Did not buy_edge
        # - Did not change_color
        # - Did not 'call' on his turn
        # - OR did not format action correctly
        # action["value"] will thus return None, but keeping as-is for consistency.
        return action


    def execute(self):
        # Query the LLM with full stack of current messages
        completion = self.client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = self.messages
        )
        return completion.choices[0].message # This returns only the AI's response in text without any "context". 


    def process_action(self, action): 
        # Create an observation from an action and add it to the current memory (messages) 
        tools = set([
            "self.purchase_edge", "self.set_color", "self.finish", "self.check_time", "self.game_question"
        ]) 
        if action.startswith("Action: "):
            # Extract function name
            function = f"self.{action[len('Action: '):action.find('(')]}"

            # Extract parameters from the parentheses
            parameters = action[action.find("(") + 1:action.find(")")]

            # Prepare the function call
            function_call = f"{function}({parameters})"
            print(f"function_call: {function_call}")

            # Check if the function is valid
            if function in tools:
                result = eval(function_call)
                if result["value"]: # If we evaluate a function and get new information, add this information to memory.
                    self.messages.append({'role': 'assistant', 'content': f"called function {result['function']} and received value {result['value']}"}) 
        
            else: 
                # If the action is formatted incorrectly, just tell the agent to retry. 
                # If incorrect by react_iters, then 
                self.messages.append({"role": "user", "content": """
                                    It seems you have formulated your action incorrectly. 
                                    
                                    It should be formatted as follows:
                                    Action: function_name(parameter)
                                """})
                return {"function":"process_action", "value": None, "break_loop": False}
            
        else: 
            # If the action is formatted incorrectly, just tell the agent to retry. 
            # If incorrect by react_iters, then 
            self.messages.append({"role": "user", "content": """
                                It seems you have formulated your action incorrectly. 
                                
                                It should be formatted as follows. Please formulate your action as follows:
                                Action: function_name(parameter)
                            """})
            return {"function":"process_action", "value": None, "break_loop": False}
        return result



    def parse_output(self, message):
        """ Helper function to sort output from execute"""
        # Split the message into lines
        lines = message.content.split("\n")
        
        # Initialize variables to store parsed sections
        thought = ""
        action = ""

        # Iterate through each line to parse sections
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.strip()
            elif line.startswith("Action:"):
                action = line.strip()
        return {"Thought": thought, "Action": action}

    # TOOLS. Format with function, value, break loop.
    # break loop reserved for cases where agent either makes change to network or withdraws from the option to do so on this turn
    def purchase_edge(self, v):
        edge = (self.id, v)

        # Error handling: agent tries to purchase a self-edge
        if self.id == v:
            error_msg = {"role": "user", "content": f"""
                You have attempted to purchase an edge to yourself. As a reminder, you are agent {2}. You can already read your color information, it is {self.color}: 
            """}
            self.messages.append(error_msg)
            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        
        # Error handling - tries to buy an edge to a neighbor that already exists - should fail to do so
        if v in self.colors:
            error_msg = {"role": "user", "content": f"""
                         You have attempted to purchase an edge to an agent with whom you are already connected. Your colors dictionary contains agent {v}: 
                         {self.colors}
                        """}
            self.messages.append(error_msg)
            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        else:
            # Verify that the edge does exist in the dictionary
            if v in self.degrees: 
                # Charge the agents for the new edge
                self.consensus_0_reward, self.consensus_1_reward = self.consensus_0_reward - self.edge_cost, self.consensus_1_reward - self.edge_cost

                # Edge has to process in the system, and return new values for the ReAct agent. If you buy an edge, or change your color, the turn ends
                return {"function": "purchase_edge", "value" : edge, "break_loop": True}
            else: 
                error_msg = {"role": "user", "content": f"""
                You have attempted to purchase an edge {v} to an agent that does not exist. Your degrees and shortest path dictionary contain the full set of players of the game: 
                Degrees dict: {self.degrees}
                """}
                self.messages.append(error_msg)
                return {"function": "purchase_edge", "value" : None, "break_loop": False}

    def set_color(self, color=''):
        self.color = color
        if color not in [0, 1, 'undeclared']:
            error_msg = {"role": "user", "content": f"""
            You have attempted to set a color {color} that is not allowed in the game. The choices available to you are [0, 1, 'Undeclared']
            """}
            return {"function":"set_color", "value": None, "break_loop": False}
        return {"function":"set_color", "value": {"id":self.id, "color":self.color}, "break_loop": True}
    
    def finish(self): # END TURN
        return {"function":"finish", "value": None, "break_loop": True}
    
    def game_question(self, question):
        # Display agent query to console
        print(f"Agent {self.id} question: {question}")

        # Request response to agent query
        answer = input("Enter your response to the Agent: ")

        # Append response to agent memory
        self.messages.append({"role": "user", "content": answer})
        return {"function":"game_question", "value": None, "break_loop": False}

    def check_time(self): # THIS IS NOT SET UP YET
        # Find the amount of time remaining in the simulation
        time_remaining =  f"There are 150 turns left in the game. Plenty of time remains."

        # Append the time remaining to the current memory
        self.messages.append({"role": "user", "content": time_remaining})
        return {"function":"game_question", "value": None, "break_loop": False}
    






class State():
    def __init__(self, init_config): # For now 
        self.network = nx.Graph()
        num_nodes = init_config['V']

        # Initialize network with no color
        for node in range(num_nodes - 1):
            self.network.add_node(node, color= None)

        # Get shortest path distances, but disconnected nodes have to appear at inf. distance
        # Set initial spls to inf
        self.spls = {u: {v: float('inf') for v in self.network.nodes} for u in self.network.nodes}

        # Then update from single source path lengths
        for u in self.network.nodes:
            lengths = nx.single_source_shortest_path_length(self.network, u)
            self.spls[u].update(lengths)
    

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

        # Get the color of the neighbors of the current node
        if agent.id not in self.network:
            print(f"Node {agent_id} does not exist in graph")
            return
    
        neighbor_colors = {}
        for neighbor in self.network.neighbors(agent_id):
            neighbor_colors[neighbor] = self.network.nodes[neighbor]['color']

        return [self.spls[agent_id], dict(self.network.degree), neighbor_colors]
    

    def update_node_color(self, node, new_color):
        if node in self.network:
            self.network.nodes[node]['color'] = new_color
        else:
            print(f"Error: Node {node} does not exist in network")
    

    def add_network_edge(self, edge):
        u, v = edge[0], edge[1]
        self.network.add_edge(u, v)
        self.update_spls()


def human_updates():
    # Function to allow human intervention with console
    making_changes = input("Reply (True/ False): I want to make changes to the network. \n You:")
    making_changes = making_changes.strip().lower() == 'true'
    while making_changes: 
        networkx_command = input("""
                                 Carefully enter the command that you want to run with parameters \n 
                                 For example: 
                                 >> state.update_node_color(4, 1)
                                 ## Wait to process
                                 >> state.add_network_edge((1, 4))
                                 ## Wait to process
                            """)
        try: 
            eval(networkx_command)
            print(f"Changes processed!")
        except Exception as e:
            print(f"Could not run command. Exception: {e}")
        making_changes = input("Reply (True/ False): I want to make changes to the network. \n You:")
        making_changes = making_changes.strip().lower() == 'true'


        


if __name__ == "__main__":
    # Intialize state and agent.
    # The state is just a representation of the state. The state class does not contain agent objects
    state = State(config['init_configs'])
    client = OpenAI(api_key= openai_api_key)

    # Initialize dictionary to store agents. Key agent_id, value: agent
    agents = {}
    for i in range(config['init_configs']['V'] - 1): # Stores the number of nodes in the simulation
        agent = ReActAgent(client= client, system = config["react_system_message"], id=i, type_specific_params = config["init_configs"][f"agent_{i%2}"])
        agents[i] = agent

    # Start interaction loop
    done = False
    i = 0
    while not done and i < config["max_iters"]:

        for agent in list(agents.values()):
            print()
            print(f"Iteration {i}, Agent {agent.id}")
            # get Agent's view of state. List: [spls, node degrees, neighbor colors]
            obs = state.render(agent.id)

            agent.spls, agent.degrees, agent.colors = obs

            # Get the set of configurations for the current agent. Agent type will be determined by agent.id % 2
            agent_configs = config['init_configs'][f"agent_{agent.type}"] # Get dictionary of current agent

            # Let the parameterized game rules contain the total number of iterations left in the game. 
            agent_configs['total_iters'], agent_configs['iters_remaining'] = i + 1, config["max_iters"] - i

            parameterized_game_rules = config["game_rules"].format(**agent_configs) # Fill in the game rules with the custom parameters for this agent type

            # Create a message that shows state of game, including the agents id
            game_rules = {"role": "user", "content" : parameterized_game_rules}

            game_state = {"role": "user", "content": f"""
                        Your shortest path distances to all other nodes in the network: {str(obs[0])} \n 
                        The degree of all nodes in the network: {str(obs[1])} \n 
                        The colors of your connections: {str(obs[2])}.
                        
                        You are agent {agent.id}
                        """
                    }

            agent_options = {"role": "user", "content" : config['agent_options']}
            for item in [game_rules, game_state, agent_options]:
                agent.messages.append(item)

            print(f" Current Agent messages: \n {agent.messages}")
            action_result = agent() # Calls agent to respond to the prompt

            # Process action result
            if action_result["function"] == "set_color":
                print(f"Agent {action_result['value']['id']} set color to {action_result['value']['color']}.")
                # Convert to string for visibility if necessary
                original_color = state.network.nodes[action_result['value']['id']].get('color', 'No color set')
                print(f"Original color: {original_color}")
                
                # Update the node color
                state.update_node_color(action_result['value']['id'], action_result['value']['color'])
                
                # Fetch and print the new color
                new_color = state.network.nodes[action_result['value']['id']].get('color', 'No color set')
                print(f"New color: {new_color}")

            if action_result["function"] == "purchase_edge":
                print("New edge purchased...")
                print(f"Original neighbors of source node: {list(state.network.neighbors(action_result['value'][0]))}")
                state.add_network_edge(action_result['value'])
                print(f"New neighbors of source node: {list(state.network.neighbors(action_result['value'][0]))}")

            # Clear agent memory for next round
            agent.messages = []

        # Break for human intervention -- allow human to make changes to network remotely
        human_updates()


        # Increment episode
        i += 1
        done = False









    
