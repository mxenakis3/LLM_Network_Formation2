import asyncio
import time
import json
import random

class ReActAgent:
    def __init__(self,
                client, 
                id,
                agent_config,
                tool_config,
                type_specific_params = {"edge_cost": 0, "consensus_0_reward": 0, "consensus_1_reward":0},
                start_time = time.time(),
                time_limit = 20):
        self.id = id
        self.type = self.id % 2
        self.color = "Undeclared"
        self.edge_cost = type_specific_params["edge_cost"]
        self.consensus_0_reward = type_specific_params["consensus_0_reward"]
        self.consensus_1_reward = type_specific_params["consensus_1_reward"]
        self.client = client
        self.messages = [] # memory/scratchpad
        self.permanent_memory = [] # will be used to create log
        self.start_time = time.time()
        self.time_limit = time_limit
        self.tools = tool_config # this is a list of the configs for each of the agent actions
        self.model_name = agent_config['model_name']

        # The following dictionaries are updated from the agent's view of the state in the main loop.
        self.colors = {}
        self.spls = {}
        self.degrees = {}

        # Create a message that shows state of game, including the agents id
        self.game_rules = {"role": "user", "content" : agent_config["game_rules"].format(**{'consensus_0_reward' : self.consensus_0_reward, 'consensus_1_reward': self.consensus_1_reward, 'edge_cost': self.edge_cost})}

        # PROMPTS THAT DO NOT CHANGE:
        # Get configs for this agent
        self.system = agent_config["react_system_message"]

    async def __call__(self, state): 
        self.spls, self.degrees, self.colors = state.render(self.id) 

        if self.system is not None: # self.system is the system message that outlines the ReAct scheme
            self.messages.append({"role": "system", "content": self.system})
            self.permanent_memory.append(f"**System** \n {self.system}")
        self.messages.append(self.game_rules)
        self.permanent_memory.append(f"**System** \n [Game Rules Repeat]")

        # Format current state into JSON
        json_state = json.dumps({
            "agent_id": self.id,
            "time_remaining": round(self.time_limit - (time.time() - self.start_time), 3),
            "current_color": self.color if self.color else "Undeclared",
            "projected_rewards": {
                "0": self.consensus_0_reward,
                "1": self.consensus_1_reward
            },
            "connected_colors": self.colors,
            "shortest_path_distances": self.spls,
            "node_degrees": self.degrees
        }, indent=2)
        
        # Add current state to memory
        self.messages.append({"role": "user", "content": f"Current state of game: \n {json_state}"})
        self.permanent_memory.append(f"**System**: \n Current state of game: \n {json_state}")

        # Generate thought based on current state
        # Append most recent thought to memory

        try:
            raw_result = await self.query(state)
            self.permanent_memory.append(f"**Agent**: \n {raw_result.content}")

        except Exception as e:
            # print(f"Exception occurred in generating thought. Returning early \n {e}")
            self.permanent_memory.append(f"**System** \n Exception occurred in generating thought. Returning early")
            return

        # Pass the thought to the tool caller. This will parse the thought into a function call.
        action_choice = await self.call_with_tools([{"role": "assistant", "content" : raw_result.content}], self.tools)

        if action_choice == None:
            # print(f"Error: llm selects no action")
            fn_name = 'do_nothing'
            fn_args = {'state': state}
        
        else:
            # Call the action chosen by the tool
            try:
                fn_name = action_choice[0].function.name
                fn_args = json.loads(action_choice[0].function.arguments)

                # add current state as argument for function call:
                fn_args["state"] = state 
            except:
                self.permanent_memory.append(f"**System** \n Error: Invalid Action Processed")
                return


        method = getattr(self, fn_name, None)           
        if method and asyncio.iscoroutinefunction(method):
            try:
                call = asyncio.create_task(method(**fn_args))
                await call # calls the function

                # Add the function call to memory
                self.permanent_memory.append(f"**System** \n Action Processed: {fn_name}({fn_args})")
            except:
                self.permanent_memory.append(f"**System** \n Invalid action processed: {fn_name}({fn_args})")
        else:
            self.permanent_memory.append(f"**System** \n Error: Invalid Action Processed")
    
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
        "model": self.model_name,
        "messages" : action_message
        }
        if tools is not None:
         kwargs["tools"] = tools
        try:
            completion = await self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.tool_calls
        except Exception as e:
            print(f"Exception: {e}")


    async def query(self, state):
        # Query the LLM with full stack of current messages
        num_retries = 3
        retries = 0
        while retries < num_retries:
            try:
                self.permanent_memory.append(f"**System** \n Try block entered. ")
                completion = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = self.messages
                )
                # print(f"agent {self.id} completion: {completion}")

                return completion.choices[0].message # This returns only the AI's response in text without any "context". 
            except Exception as e:
                self.permanent_memory.append(f"**System** \n Error occurs: {e}")
                state.update_log.append(f"Rate limit error: Agent {self.id}, time: {time.time() - self.start_time}")
                wait_time = 1
                await asyncio.sleep(wait_time)
                retries += 1
        return None


    # TOOLS. Format with function, value, break loop.
    # break loop reserved for cases where agent either makes change to network or withdraws from the option to do so on this turn
    async def purchase_edge(self, neighbor_id, state):
        # Error handling: agent tries to purchase a self-edge
        if neighbor_id == 1:
            # Assume a random id
            neighbor_id = random.choice(list(self.degrees.keys()))
        
        if self.id == neighbor_id:
            error =  f"""
                You have attempted to purchase an edge to yourself. As a reminder, you are agent {self.id}. You can already read your color information, it is {self.color}: 
            """
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return
        
        # Error handling - tries to buy an edge to a neighbor that already exists - illegal move
        if neighbor_id in self.colors:
            error = f"""
                         You have attempted to purchase an edge to an agent with whom you are already connected. Your colors dictionary contains agent {neighbor_id}: 
                         {self.colors}
                        """
            self.messages.append({"role": "user", "content": error})
            self.permanent_memory.append(f"**System**: \n {error}")
            return
        
        # Not a self edge, not an existing edge
        else:
            edge = (self.id, neighbor_id)
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
                return
            
            else: 
                error = f"""
                You have attempted to purchase an edge {neighbor_id} to an agent that does not exist. Your degrees and shortest path dictionary contain the full set of players of the game: 
                Degrees dict: {self.degrees}
                """
                self.messages.append({"role": "user", "content":error})
                self.permanent_memory.append(f"**System**: \n {error}")
                return {"function": "purchase_edge", "value" : None, "break_loop": False}


    async def set_color(self, color, state):
        if color not in [str(0), str(1), 'Undeclared']:
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

        # Update own color
        self.color = color

        state.update_log.append(f"Time: {time.time() - self.start_time} \n  **System** \n Agent {self.id} set color from {original_color} to {color}.")
        state.update_log.append(f"Nodes: {state.network.nodes(data=True)} \n Edges: {state.network.edges(data=True)}")

        # Flag that the data was processed to break out of the agent call "max-iters" loop
        return {"function":"set_color", "value": {"id":self.id, "color":self.color}, "break_loop": True}
    

    async def do_nothing(self, state): 
        return
    
    
    async def agent_loop(self, state, time_limit):
        """ The main loop, which allows the agent to think continuously until time expires """
        while time.time() - self.start_time < time_limit:

            await self(state) # Calls agent to respond to the prompt

            # Add a small sleep in order to allow other agents to access the event loop
            await asyncio.sleep(0.1)

            # Clear agent memory for next round
            self.messages = []