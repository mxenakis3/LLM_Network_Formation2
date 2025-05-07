from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import time
import matplotlib.pyplot as plt
from utils.log_functions import *
from utils.network_plots import *
from runner_config import runner_config
from tools import agent_actions
from agent_configs import agent_config

from runner import Runner


# Load in API Key and Handle Errors
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Disable plots in matplotlib and save plots directly
plt.switch_backend('Agg') 
client = AsyncOpenAI(api_key= openai_api_key)



if __name__ == "__main__":
    runner = Runner(run_config = runner_config, agent_config = agent_config, tool_config = agent_actions, client=client)
    asyncio.run(runner())

    
    
    








    
