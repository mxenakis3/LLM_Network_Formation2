import asyncio
from os import truncate
import time
from config_py import config
import openai
from Tool_configs import agent_actions
import json

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
        self.tools = agent_actions

        # The following dictionaries are updated from the agent's view of the state in the main loop.
        self.colors = {}
        self.spls = {}
        self.degrees = {}

        # Agent configs
        agent_configs = config['init_configs'][f"agent_{self.type}"]

        # Create a message that shows state of game, including the agents id
        self.game_rules = {"role": "user", "content" : config["game_rules"].format(**agent_configs)}

        # PROMPTS THAT DO NOT CHANGE:
        # Get configs for this agent
        self.system = config["react_system_message"]
        if self.system is not None:
            self.messages.append({"role": "system", "content": self.system})


    async def __call__(self, state, react_max_iters=5): 
        # Agent reacts to current view of state and stores reaction in memory
        # react max iters: number of tries agent gets to "think" before they get cut off
        # for idx in range(react_max_iters):
        self.spls, self.degrees, self.colors = state.render(self.id) 

        if self.system is not None: # self.system is the system message that outlines the ReAct scheme
            self.messages.append({"role": "system", "content": self.system})
        self.messages.append(self.game_rules)

        
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

        # See what the agent wants to do. Will string in form f"Thought: {thought} \n Action: {action}"
        raw_result = await self.execute(state)
        if not raw_result:
            return # We may encounter this during limit errors. Use try/except block instead?
        self.permanent_memory.append(f"**Agent**: \n {raw_result.content}")

        # Use tool calling to process the action.
        action_choice = await self.call_with_tools([{"role": "assistant", "content" : raw_result.content}], self.tools)
        # print(f"agent {self.id} action choice type: {type(action_choice)} with value: {action_choice}")
        # if action_choice == None:
        #     print(f"Agent {self.id} chooses no action. Messages:")
        #     print(self.messages)
        # Call the action chosen by the tool
        fn_name = action_choice[0].function.name
        fn_args = json.loads(action_choice[0].function.arguments)

        # add current state as argument for function call:
        fn_args["state"] = state

        # Call the method
        method = getattr(self, fn_name, None)           
        if method and asyncio.iscoroutinefunction(method):
            call = asyncio.create_task(method(**fn_args))
            output = await call # calls the function

        # Uses the helper function to process/call the action
        # action = await self.process_action(self.messages[-1]["content"], state) 
        self.permanent_memory.append(f"**System** \n Action Processed: {fn_name}({fn_args})")

    
    async def call_with_tools(self, action_message, tools):
        """
        OpenAI API builtin for tool calling.
        Inputs:
        - action_message: The string generated in self.execute(), which specifies the agent's desired action. 
        Outputs:
        - completion.choices[0].message.tool_calls: The tool call for a function.
        """
        # kwargs for the API
        kwargs = {
        "model": "gpt-4o-mini",
        "messages" : action_message
        }
        if tools is not None:
         kwargs["tools"] = tools
        try:
            completion = await self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.tool_calls
        except Exception as e:
            print(f"Exception: {e}")

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


    # TOOLS. Format with function, value, break loop.
    # break loop reserved for cases where agent either makes change to network or withdraws from the option to do so on this turn
    async def purchase_edge(self, neighbor_id, state):
        if not neighbor_id.isnumeric():
            error = f"""
                Format the parameter as the integer representing the agent id of the agent you would like to connect with, ex: (1) for agent 1. : 
            """
            error_msg = {"role": "user", "content":error }
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")

            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        neighbor_id = int(neighbor_id)
        edge = (self.id, neighbor_id)

        # Error handling: agent tries to purchase a self-edge
        if self.id == neighbor_id:
            error =  f"""
                You have attempted to purchase an edge to yourself. As a reminder, you are agent {self.id}. You can already read your color information, it is {self.color}: 
            """
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        
        # Error handling - tries to buy an edge to a neighbor that already exists - illegal move
        if neighbor_id in self.colors:
            error = f"""
                         You have attempted to purchase an edge to an agent with whom you are already connected. Your colors dictionary contains agent {neighbor_id}: 
                         {self.colors}
                        """
            error_msg = {"role": "user", "content": error}
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return {"function": "purchase_edge", "value" : None, "break_loop": False}
        
        # Not a self edge, not an existing edge
        else:
            # Verify that the edge does exist in the dictionary
            if neighbor_id in self.degrees: 
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
                You have attempted to purchase an edge {neighbor_id} to an agent that does not exist. Your degrees and shortest path dictionary contain the full set of players of the game: 
                Degrees dict: {self.degrees}
                """
                self.messages.append({"role": "user", "content":error})
                self.permanent_memory.append(f"**System**: \n {error}")
                return {"function": "purchase_edge", "value" : None, "break_loop": False}


    async def set_color(self, color, state):
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
    

    async def finish(self, state): 
        return {"function":"finish", "value": None, "break_loop": True}
    
    
    async def agent_loop(self, state, time_limit):
        """ The main loop, which allows the agent to think continuously until time expires """
        while time.time() - self.start_time < time_limit:

            action_result = await self(state) # Calls agent to respond to the prompt

            # Add a small sleep in order to allow other agents to access the event loop
            await asyncio.sleep(0.1)

            # Clear agent memory for next round
            self.messages = []