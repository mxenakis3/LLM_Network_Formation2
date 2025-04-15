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
          "type": "number",
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

do_nothing = {
  "type": "function",
  "function": {
    "name": "do_nothing",
    "description": "Stay put without buying any edges or changing your color",
    "parameters": {
      "type": "object",  # Define the type as an object
      "properties": {},  # No parameters, so properties are empty
      "required": [],  # No required parameters
      "additionalProperties": False
    },
  "strict": True  # This still applies
  },
}

agent_actions = [purchase_edge, set_color, do_nothing]