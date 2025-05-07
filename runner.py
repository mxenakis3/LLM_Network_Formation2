
import os
import asyncio
import time
from agent_class import ReActAgent
from state import State
import time
from datetime import datetime
from tqdm.asyncio import tqdm
import random
from tqdm import tqdm
import pickle

class Runner():
  def __init__(self, run_config, agent_config, tool_config, client):
    self.run_config = run_config
    self.agent_config = agent_config
    self.tool_config = tool_config
    self.client = client

  async def __call__(self):      
      """
      Calls run experiments and saves network and agents dictionary for each run in separate pickle files.
      Inputs:
      - run_config: specifies parameters for a single run
      Outputs:
      - None, but creates a folder which stores info in .pkl files
      """
      # Run trials
      results = {}
      for idx in tqdm(range(self.run_config['num_trials'])):
          try:
              results[idx] = await self._run_trial()
              await asyncio.sleep(5)
          except Exception as e:
              print(f"Exception occurred in experiment {idx}: {e}")

      # Save results
      curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      dir_path = f".\outputs\{curr_time}"

      # Store files
      os.makedirs(dir_path, exist_ok=True)
      
      with open(os.path.join(dir_path, f"run_configs"), "wb") as f:
          pickle.dump(self.run_config, f)
      with open(os.path.join(dir_path, f"agent_configs"), "wb") as f:
          pickle.dump(self.agent_config, f)
      with open(os.path.join(dir_path, f"tool_configs"), "wb") as f:
          pickle.dump(self.tool_config, f)
      
      # Unpack trial data into .pkl
      for key, data in results.items():
          filename = os.path.join(dir_path, f"experiment_{key}")
          with open(filename, "wb") as f:
              pickle.dump(data, f)
  
  async def _run_trial(self):
      """
      Runs a single experiment from configs. run_config is imported.
      Returns:
      - network: nx state
      - agents: agent dictionary {agent_id: Agent}
      """
      # Initialize network and client
      state = State()
      client = self.client

      # Generate agents
      num_agents = self.run_config['V']

      # Sample ID's randomly. Choose #-digits in ID dynamically based on num_agents.
      t, n = 0, num_agents
      while n > 10:
          n = n / 10
          t += 1
      available_ids = random.sample(range(10**(t+1), 10**(t+2) + 1), num_agents)  # Unique 3-digit IDs

      # Initialize dictionary to store agents. Key agent_id, value: agent
      agents = {}

      for i in range(num_agents):
          agent_id = available_ids[i]
          agent = ReActAgent(
              client=client,
              id=agent_id,
              type_specific_params= self.run_config[f"agent_{i % 2}"],
              time_limit= self.run_config["duration"],
              agent_config= self.agent_config,
              tool_config = self.tool_config
          )
          agents[agent_id] = agent
          state.network.add_node(agent_id, color=None)

      # Reset spls
      state.update_spls()
      
      # Store start time and define time limit of duration
      start_time = time.time()
      time_limit = self.run_config["duration"]

      # Run async simulation
      with tqdm(total=time_limit, desc="Simulation Progress", unit="s") as pbar:        
          tasks = [asyncio.create_task(x.agent_loop(state, time_limit)) for x in agents.values()]
          while time.time() - start_time < time_limit:
              elapsed_time = time.time() - start_time
              pbar.update(elapsed_time - pbar.n)  
              await asyncio.sleep(0.1) 
          await asyncio.gather(*tasks)

      end_time = time.time()
      print(f"Experiment Total time: {round(end_time - start_time, 2)}")

      # Save selected agent properties into a separate dict (load into config)
      agent_results = {}
      for id, ag in agents.items():
          agent_results[id] = {}
          agent_results[id]['color'] = ag.color
          agent_results['consensus_0_reward'] = ag.consensus_0_reward
          agent_results['consensus_1_reward'] = ag.consensus_1_reward
          agent_results['permanent_memory'] = ag.permanent_memory

      return state.network, agent_results


