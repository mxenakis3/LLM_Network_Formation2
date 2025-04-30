
import os
import re
import networkx as nx



def parse_chat_log(chat_log):
    formatted_output = []
    
    for entry in chat_log:
        # Split only on the outermost commas, preserving commas inside text
        split_entries = re.split(r',\s*(?=[A-Z])', entry)  # Comma followed by capital letter (heuristic for separation)
        
        for sub_entry in split_entries:
            cleaned_text = re.sub(r'\n\s+', '\n', sub_entry.strip())  # Clean extra spaces/newlines
            
            # Categorize messages
            if "Thought:" in cleaned_text or "Action:" in cleaned_text:
                formatted_output.append(f"\n{cleaned_text}\n")
            else:
                formatted_output.append(f"\n{cleaned_text}\n")
    
    return "\n".join(formatted_output)


def create_state_log(state, save_path="shortest_path_distribution.png"):
  # Ensure the directory exists
  os.makedirs(save_path, exist_ok=True)

  final_colors = {attributes["color"] for _, attributes in state.network.nodes(data=True)}

  # Fix file path to save the log correctly
  file_path = os.path.join(save_path, "state_update_log.txt")

  with open(file_path, "w") as file:
      file.write(parse_chat_log(state.update_log))
      if len(final_colors) == 1:
          file.write(f"\n Consensus was reached.")
      else:
          file.write(f"\n Consensus was not reached.")


def create_agent_logs(agents, dir_path):
  # Ensure the directory exists before trying to write files
  os.makedirs(dir_path, exist_ok=True)

  for agent_id, agent in agents.items():
      file_path = os.path.join(dir_path, f"agent_{agent_id}.txt")
      with open(file_path, "w") as file:
          file.write(parse_chat_log(agent.permanent_memory))