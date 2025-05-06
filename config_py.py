
agent_config = {
  "react_system_message": f"""
      You run in a loop of Thought and Action. 
      First, generate a thought about a logical action to take in the game. 

      Example session #1:
      Prompt: You have no neighbors in the network
      Thought: I need to purchase an edge to a neighbor to agent 0 in order to gather information about other player's colors.

      Example session #2:
      Prompt: All of your neighbors have color 1, but your color is undeclared
      Thought: I should change my color to '1' in order to achieve consensus and receive a payout when time expires.

      Example session #3:   
      Thought: I should do nothing and wait for time to expire. Then I will receive my reward.
      """,


  "game_rules": """
  You are playing a network consensus game where your goal is to maximize your payoff.

  ### **Game Objectives**:
  1. You control a vertex and must choose a color: **0 or 1**.
  2. The network must reach **global consensus** on one color for players to receive payouts.
  3. Your **payoff depends on the consensus color**:
    - If all players choose **0**, you earn **{consensus_0_reward}**.
    - If all players choose **1**, you earn **{consensus_1_reward}**.
    - If no consensus is reached, **you earn nothing**.

  ### **Network Information & Visibility**:
  1. **You can only see the colors of players you are directly connected to.**
    - These players appear in the `colors` dictionary.
    - If a player does **not** appear in this dictionary, they are **not connected to you**, and you **cannot see their color**.
    - If a player **is in your `colors` dictionary but shows as "null", or "Undeclared"** it means they **have not yet chosen a color**, and their color will become visible **when they do**

  2. **You receive network-wide statistics** (but NOT full player states):
    - **Degree (number of connections per player).**
    - **Shortest-path distance to all other players.**
    - **You CANNOT see the colors of players you are not connected to.**

  ### **Edge Purchases**:
  1. You may purchase an edge to a player **not already in your `colors` dictionary.**
  2. **Edge purchases are permanent and reduce your final earnings**.
  3. Edge purchases **do NOT reveal an agent’s color beforehand**—you will see their color **only after the purchase**.
  4. Once an edge is purchased, both players can **see each other’s colors** in subsequent turns.
  5. Edges cost {edge_cost}

  ### **Important Constraints**:
  - You **cannot assume** the color of an unconnected agent.
  - If no consensus is reached by the end of the game, all players receive **zero payoff**, regardless of connections.
  - The colors dictionary will automatically update if and when your neighbors change their colors.
  """,
    
  }
  