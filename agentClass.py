import asyncio
from os import truncate
import time
from config_py import config
import openai

class ReActAgent:
    def __init__(self, client, id, type_specific_params = {"edge_cost": 0, "consensus_0_reward": 0, "consensus_1_reward":0}, start_time = time.time(), time_limit = 20): # system prompt: you are working in a loop, etc. 
        self.id = id
        self.type = self.id % 2
        self.color = "Undeclared" # Previously set to None, but agents were under the impression that "None" meant that an edge could be purcahsed to reveal agent's color
        self.edge_cost = type_specific_params["edge_cost"]
        self.consensus_0_reward = type_specific_params["consensus_0_reward"]
        self.consensus_1_reward = type_specific_params["consensus_1_reward"]
        self.client = client
        self.messages = [] # memory/scratchpad
        self.permanent_memory = [] # will be used to create log
        self.start_time = time.time()
        self.time_limit = time_limit


        # The following dictionaries are updated from the agent's view of the state in the main loop.
        self.colors = {}
        self.spls = {}
        self.degrees = {}


        # Agent configs
        agent_configs = config['init_configs'][f"agent_{self.type}"]

        # Fill in the game rules with the custom parameters for this agent type
        self.game_rules_str = config["game_rules"].format(**agent_configs)

        # Create a message that shows state of game, including the agents id
        self.game_rules = {"role": "user", "content" : self.game_rules_str}

        # PROMPTS THAT DO NOT CHANGE:
        # Get configs for this agent
        self.system = config["react_system_message"]
        if self.system is not None:
            self.messages.append({"role": "system", "content": self.system})

        self.agent_options = {"role": "user", "content" : config['agent_options']}

    async def __call__(self, state, react_max_iters=5): 
        # Agent reacts to current view of state and stores reaction in memory
        # react max iters: number of tries agent gets to "think" before they gets cut off

        reformat = False # Controller. When this is true, the agent selected an action but formatted it incorrectly. We skip 

        for idx in range(react_max_iters):

            self.spls, self.degrees, self.colors = state.render(self.id)

            if not reformat:
                if self.system is not None:
                    self.messages.append({"role": "system", "content": self.system})
                self.messages.append(self.game_rules)
                self.messages.append(self.agent_options)

            # Get current state
            state_message = f"""
                    Your shortest path distances to all other nodes in the network: {self.spls} \n 
                    The degree of all nodes in the network: {self.degrees} \n 
                    The colors of your connections: {self.colors if self.colors != None else 'Undeclared'}.
                    
                    You are agent {self.id}

                    There are {self.time_limit - (time.time() - self.start_time)} seconds remaining in the game. 

                    Currently, your projected reward for a consensus of 0 is {self.consensus_0_reward},
                    and your projected reward for a consensus of 1 is {self.consensus_1_reward}.

                    Your color is currently {self.color}

                    Format your response as:
                        Thought: I need to find the mass of Earth
                        Action: get_planet_mass(Earth)
                    """
            
            self.messages.append({"role": "user", "content": state_message}) # This is how a user query is added to the list of messages for the agent
            self.permanent_memory.append(f"**System**: \n {state_message}")

             # See what the agent wants to do. Will be in form of strig f"Thought: {thought} \n Action: {action}"
            raw_result = await self.execute(state)
            if not raw_result:
                return # We may encounter this during limit errors. 
            self.permanent_memory.append(f"**Agent**: \n {raw_result.content}")

            # Separate raw_result into thought, action, etc. 
            self.parse_output(raw_result)

            # Uses the helper function to process/call the action
            action = await self.process_action(self.messages[-1]["content"], state) 
            self.permanent_memory.append(f"**System** \n Action Processed: {action}")

            if action["value"] == "reformat":
                reformat = True
            else:
                reformat = False
            # parsed_result is a dictionary which has a key "break_loop". If this is false, we do NOT exit-- and simply add our outputs to memory and repeat the loop
            # Action: {"value": obj, "break_loop": Bool}
            if action["break_loop"] == True:
                return action

        # If we have reached this part of the loop, the agent has spent up to react_max_iters iterations deliberating
        # - Did not buy_edge
        # - Did not change_color
        # - Did not 'call' on his turn
        # - OR did not format action correctly
        # action["value"] will thus return None, but keeping as-is for consistency.
        return action


    async def execute(self, state):
        # Query the LLM with full stack of current messages
        num_retries = 3
        retries = 0
        while retries < num_retries:
            try:
                completion = await self.client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = self.messages
                )
                return completion.choices[0].message # This returns only the AI's response in text without any "context". 
            except openai.RateLimitError as e:
                state.update_log.append(f"Rate limit error: Agent {self.id}, time: {time.time() - self.start_time}")
                wait_time = 1
                await asyncio.sleep(wait_time)
                retries += 1
        return None



    async def process_action(self, action, state): 
        """
        Creates an observation from an action and adds it to the current memory (messages). 

        Inputs:
            action: a string in the format "Action: function_name(parameter)"
        Outputs:
            - If format is correct: {"function": function_name, "value": return_value, "break_loop": Boolean}
            - If incorrect: adds an error message to self.messages and returns {"function": "process_action", "value": None, "break_loop": False}
        """
        
        tools = {
            "purchase_edge", "set_color", "finish", "game_question"
        }

        if not action.startswith("Action:"):
            error = """
                                It seems you have formulated your action incorrectly. 
                                It should be formatted as:
                                Action: function_name(parameter). Please reformat the request.
                            """
            # Input message error
            self.messages.append({"role": "user", "content": error})            
            self.permanent_memory.append(f"**System**: \n {error}")
                
            return {"function":"process_action", "value": "reformat", "break_loop": False}
        
        # Extract function name
        function_name = action.split(":")[-1].strip().split("(")[0]

        # Extract and clean parameters
        param_str = action[action.find("(") + 1:action.find(")")]
        parameters = [param.strip() for param in param_str.split(",") if param]

        # Check if the function exists in allowed tools
        if function_name in tools:
            func = getattr(self, function_name, None)

            if callable(func):
                try:
                    result = func(*parameters, state)  # Call function with parameters
                    if result["value"]: 
                        self.permanent_memory.append(f"Called {result['function']} and received value {result['value']}.")
                        self.messages.append({'role': 'assistant', 'content': f"Called {result['function']} and received value {result['value']}."})
                    return result
                except TypeError as e:
                    self.permanent_memory.append(f"Error calling {function_name} with parameters {parameters}: {e}")
                    self.messages.append({"role": "user", "content": f"Error calling {function_name} with parameters {parameters}: {e}"})
                    return {"function": function_name, "value": None, "break_loop": False}
            else:
                self.permanent_memory.append(f"Unknown function {function_name}.")
                self.messages.append({"role": "user", "content": f"Unknown function {function_name}."})
                return {"function": "process_action", "value": None, "break_loop": False}
        
        else: 
            self.messages.append({"role": "user", "content": """
                                It seems you have formulated your action incorrectly. 
                                It should be formatted as:
                                Action: function_name(parameter)
                            """})
            return {"function":"process_action", "value": "reformat", "break_loop": False}


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
    
        parsed_result = {"Thought": thought, "Action": action}

        # Iterate through thought and action and add to messages
        for item in list(parsed_result.values()):
            # Add the result to the list of messages
            self.messages.append({"role": "assistant", "content": str(item)})
        return


    # TOOLS. Format with function, value, break loop.
    # break loop reserved for cases where agent either makes change to network or withdraws from the option to do so on this turn
    def purchase_edge(self, v, state):
        if not v.isnumeric():
            error = f"""
                Format the parameter as the integer representing the agent id of the agent you would like to connect with, ex: (1) for agent 1. : 
            """
            error_msg = {"role": "user", "content":error }
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")

            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        v = int(v)
        edge = (self.id, v)

        # Error handling: agent tries to purchase a self-edge
        if self.id == v:
            error =  f"""
                You have attempted to purchase an edge to yourself. As a reminder, you are agent {self.id}. You can already read your color information, it is {self.color}: 
            """
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        
        # Error handling - tries to buy an edge to a neighbor that already exists - illegal move
        if v in self.colors:
            error = f"""
                         You have attempted to purchase an edge to an agent with whom you are already connected. Your colors dictionary contains agent {v}: 
                         {self.colors}
                        """
            error_msg = {"role": "user", "content": error}
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        
        # Not a self edge, not an existing edge
        else:
            # Verify that the edge does exist in the dictionary
            if v in self.degrees: 
                # Edge has to process in the system, and return new values for the ReAct agent. If you buy an edge, or change your color, the turn ends
                if edge not in state.network.edges():
                    self.permanent_memory.append(f"Time: {time.time() - self.start_time} \n **System** \n New edge purchased... \n Original neighbors of source node: {list(state.network.neighbors(edge[0]))}")
                    state.add_network_edge(edge)
                    # Charge the agents for the new edge
                    self.consensus_0_reward, self.consensus_1_reward = self.consensus_0_reward - self.edge_cost, self.consensus_1_reward - self.edge_cost
                    state.update_log.append(f"Time: {time.time() - self.start_time} \n **System** \n Agent {self.id} adds edge {edge[1]}")
                    state.update_log.append(f" Nodes: {state.network.nodes(data=True)} \n Edges: {state.network.edges(data=True)}")
                    self.permanent_memory.append(f"**System** \n New neighbors of source node: {list(state.network.neighbors(edge[0]))}")
                return {"function": "purchase_edge", "value" : edge, "break_loop": True}
            
            else: 
                error = f"""
                You have attempted to purchase an edge {v} to an agent that does not exist. Your degrees and shortest path dictionary contain the full set of players of the game: 
                Degrees dict: {self.degrees}
                """
                self.messages.append({"role": "user", "content":error})
                self.permanent_memory.append(f"**System**: \n {error}")
                return {"function": "purchase_edge", "value" : None, "break_loop": False}


    def set_color(self, color, state):
        self.color = color
        if color not in [str(0), str(1), 'undeclared']:
            error = f"""
            You have attempted to set a color {color} that is not allowed in the game. The choices available to you are [0, 1, 'Undeclared']
            """
            self.messages.append({"role": "user", "content":error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return {"function":"set_color", "value": None, "break_loop": False}
        
        original_color = state.network.nodes[self.id].get('color', 'No color set')

        self.permanent_memory.append(f"Time: {time.time() - self.start_time} \n  **System** \n Agent {self.id} set color from {original_color} to {color}.")

        # Update the node color
        state.update_node_color(self.id, color)

        state.update_log.append(f"Time: {time.time() - self.start_time} \n  **System** \n Agent {self.id} set color from {original_color} to {color}.")
        state.update_log.append(f"Nodes: {state.network.nodes(data=True)} \n Edges: {state.network.edges(data=True)}")

        # Flag that the data was processed to break out of the agent call "max-iters" loop
        return {"function":"set_color", "value": {"id":self.id, "color":self.color}, "break_loop": True}
    

    def finish(self, state): 
        return {"function":"finish", "value": None, "break_loop": True}
    
    
    async def agent_loop(self, state, time_limit):
        """ The main loop, which allows the agent to think continuously until time expires """
        while time.time() - self.start_time < time_limit:

            # print(f" Current Agent messages: \n {agent.messages}")
            action_result = await self(state) # Calls agent to respond to the prompt

            # Add a small sleep in order to allow other agents to access the event loop
            await asyncio.sleep(0.1)

            # Clear agent memory for next round
            self.messages = []