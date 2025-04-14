add_activity = {
  "type": "function",
  "function": {
    "name": "add_activity",
    "description": """
      Adds a new activity to the schedule
      This function will be called on a user query, and autonomously parse the query and request any missing inputs. 
      **You are NOT responsible for requesting any additional input from user**
      **You can assume a new activity is added EACH time this function is processed.**
    """,
    "parameters": {
      "type": "object", # parameters has type object
      "properties": {
        "prompt": {
          "type": "string",
          "description": "The users exact query"
        },
      },
      "required": ["prompt"],
      "additionalProperties": False
    },
  "strict": True
  },
}

purchase_edge = {
  "type": "function",
  "function": {
    "name" : "purchase_edge",
    "description": """
        Purchase an edge to a neighboring player.
    """,
    "parameters": {
      "type": "object",
      "properties": {
        "neighbor_id" : {
          "type": "string",
          "description": "The ID of the neighbor to whom you will purchase an edge"
        },
      },
      "required": ["neighbor_id"],
      "additionalProperties": False
    },
    "strict": True
  }
}

set_color = {
  "type": "function",
  "function": {
    "name": "set_color",
    "description": """
        Set the color of your node to either 0 or 1
    """,
    "parameters": {
      "type": "object",  # Define the type as an object
      "properties": {
        "color": {
          "type": "string",
          "description": "The color to set your node to (either '0' or '1')"
        }
      },
      "required": ["color"],  # No required parameters
      "additionalProperties": False
    },
    "strict": True
  }
}

finish = {
  "type": "function",
  "function": {
    "name": "finish",
    "description": "End your turn (stay put without buying any edges or changing your color)",
    "parameters": {
      "type": "object",  # Define the type as an object
      "properties": {},  # No parameters, so properties are empty
      "required": [],  # No required parameters
      "additionalProperties": False
    },
  "strict": True  # This still applies
  },
}

agent_actions = [purchase_edge, set_color, finish]