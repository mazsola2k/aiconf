"""
This script generates YAML-formatted Ansible playbooks for:
1. Changing the hostname of a Red Hat server.
2. Setting the hostname of an IOS-XE network device.

Key Features:
- Uses Hugging Face's LLaMA-2 model for playbook generation.
- Runs on CPU only, ensuring compatibility across systems.
- Provides real-time progress updates in the CLI:
  - Tokens generated.
  - Estimated remaining time in MM:SS format.
- Saves the generated playbooks to:
  - `generated_ansible_playbook_redhat.yml`
  - `generated_ansible_playbook_iosxe.yml`
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import time
import sys

# Load the pre-trained model and tokenizer from HuggingFace
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Force the model to run on the CPU, even if CUDA is available
device = torch.device("cpu")
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

# Custom stopping criteria to simulate token-by-token generation
class ProgressStoppingCriteria(StoppingCriteria):
    def __init__(self, max_new_tokens, start_time):
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0
        self.start_time = start_time

    def __call__(self, input_ids, scores, **kwargs):
        self.generated_tokens += 1
        time_elapsed = time.time() - self.start_time
        estimated_total_time = (time_elapsed / self.generated_tokens) * self.max_new_tokens
        remaining_time = estimated_total_time - time_elapsed

        # Convert remaining time to MM:SS format
        minutes, seconds = divmod(int(remaining_time), 60)
        print(
            f"\rProgress: {self.generated_tokens}/{self.max_new_tokens} tokens generated. "
            f"Estimated remaining time: {minutes:02d}:{seconds:02d}.",
            end=""
        )
        sys.stdout.flush()
        return self.generated_tokens >= self.max_new_tokens

# Function to generate playbook with proactive remaining time indication
def generate_with_remaining_time(prompt, max_new_tokens, filename):
    # Tokenize the input prompt and move it to the CPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Start the timer
    start_time = time.time()

    # Set up stopping criteria for progress updates
    stopping_criteria = StoppingCriteriaList([ProgressStoppingCriteria(max_new_tokens, start_time)])

    # Generate the playbook with progress updates
    print(f"Generating the playbook for {filename}...")
    sys.stdout.flush()  # Ensure the message is printed immediately

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,  # Greedy decoding
        top_k=50,  # Limit sampling to the top 50 tokens
        top_p=0.9,  # Use nucleus sampling
        early_stopping=True,
        stopping_criteria=stopping_criteria
    )

    print("\nPlaybook generation completed.")

    # Decode and display the generated playbook
    generated_playbook = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated Ansible Playbook for {filename}:")
    print(generated_playbook)

    # Write the playbook to a file
    with open(filename, "w") as file:
        file.write(generated_playbook)
    print(f"Playbook written to: {filename}")

# Define the first task: Generate an Ansible playbook to change the hostname of a Red Hat server
prompt_redhat = """
Generate an Ansible playbook to change the hostname of a Red Hat server.
The playbook should:
1. Use the `hostname` module to set the hostname.
2. Update the `/etc/hostname` file.
3. Update the `/etc/hosts` file to map the hostname to the loopback address (127.0.0.1).
Provide the playbook in YAML format.
"""

# Define the second task: Generate an Ansible playbook to set the hostname of an IOS-XE network device
prompt_iosxe = """
Generate an Ansible playbook to set the hostname of an IOS-XE network device.
The playbook should:
1. Use the `ios_config` module to set the hostname.
2. Ensure the configuration is saved to the device.
Provide the playbook in YAML format.
"""

# Set the maximum number of tokens to generate
max_new_tokens = 150

# Generate the playbooks with proactive remaining time indication
generate_with_remaining_time(prompt_redhat, max_new_tokens, "generated_ansible_playbook_redhat.yml")
generate_with_remaining_time(prompt_iosxe, max_new_tokens, "generated_ansible_playbook_iosxe.yml")