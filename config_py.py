
config = {
  "max_iters": 5,

  "duration": 30, # Time of experiment in seconds

  "react_system_message": f"""
      You run in a loop of Thought and Action. 
      Use Thought to describe your thoughts about the prompt you are given.
      Use Action to run one of the actions available to you.

      Example session #1:

      Prompt: You have no neighbors in the network
      Thought: I need to purchase an edge to a neighbor to agent 0 in order to gather information about other player's colors.
      Action: purchase_edge(0)

      Example session #2:

      Prompt: All of your neighbors have color 1, but your color is undeclared
      Thought: I should change my color to '1' in order to achieve consensus and receive a payout when time expires.
      Action: set_color(1)

      Prompt: All of your neighbors have color 0, and so do you.
      Thought: I should do nothing and wait for time to expire. Then I will receive my reward.
      Action: finish()
      """,

  # "react_system_message": """
  #     You run in a loop of Thought, Action, Observation. 
  #     At the end of the loop you output an Answer. 
  #     Use Thought to describe your thoughts about the question you have been asked.
  #     Use Action to run one of the actions available to you.
  #     Observation will be the result of running those actions.

  #     Example session:

  #     Question: What is the mass of Earth times 2?
  #     Thought: I need to find the mass of Earth
  #     Action: get_planet_mass(Earth)

  #     You will be called again with this:

  #     Observation: 5.972e24

  #     Thought: I need to multiply this by 2
  #     Action: calculate(5.972e24 * 2)

  #     You will be called again with this: 

  #     Observation: 1,1944×10e25

  #     If you have the answer, output it as the Answer.

  #     Answer: The mass of Earth times 2 is 1,1944×10e25.
  #     """,

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
    - If a player **is in your `colors` dictionary but shows as "undeclared,"** it means they **have not yet chosen a color.**

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
  - An agent is **not "undeclared" unless you are connected to them AND their color is still undecided**.
  - If no consensus is reached by the end of the game, all players receive **zero payoff**, regardless of connections.
  """,

  # "game_rules": """
  #     You are in a multiplayer game where the object is to earn the highest payoff. 

  #     In this game, each player controls the color (either 0, 1, or None) of a vertex in a network.

  #     If the entire network picks color 0, your maximum projected payoff will be {consensus_0_reward}.
  #     If the entire network picks color 1, your maximum projected payoff will be {consensus_1_reward}.
  #     If no consensus is reached, you will not win any money.

  #     You will have two pieces of information about the network, for all players of the game, at all times, given as two dictionaries:
  #     1. Degree: The number of connections each player has to other players.
  #     2. Shortest path distance (spls): Your minimum degree of separation from every other player in the game.

  #     You are only able to see the colors of a subset of players to whom you are already connected in the network via an edge. 
  #     Each key in this dictionary, called "colors" represents the ID of another agent with whom you are already connected. Each value represents their color.
  #     If the agent has not committed to a color, their color will appear as "undeclared".

  #     Edges (or, connections) to players to whom you are not connected (any player that does NOT already appear in your colors dictionary, to follow)
  #     can be purchased at a price of {edge_cost}, which will be deducted from your projected payoff. You will not have to pay for any edges if the network fails to come to a consensus. 

  #     When you buy an edge to another player, the state of the network refreshes. On your next turn, both you and the other player will be able to see each other's colors. 

  #     Once an edge has bought, it will remain in the network until the game expires. 

  #     """, #       There are {max_iters} iterations or turns in this game, and you have {iters_remaining} turns remaining.
      
      
  "agent_options": """
    Your available actions are:
    1. purchase_edge(neighbor_id): Purchase an edge to a neighboring player.
    2. set_color(color): Set your color to either 0 or 1.
    3. finish(): End your turn and complete the loop.
    """,

  "react_max_iters": 3,
  "init_configs": {
    "duration": 60, # time of experiment in seconds
    "V": 36,
    "agent_0":{      
      "consensus_0_reward": 3,
      "consensus_1_reward": 2,
      "edge_cost": 0.03
    },

    "agent_1":{
      "consensus_0_reward": 2,
      "consensus_1_reward": 3,
      "edge_cost": 0.03
    }
  }
}
  