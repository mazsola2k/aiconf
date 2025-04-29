"""
# Llama-3 Generate Ansible Playbook for Red Hat and IOS-XE
@mazsola2k
This script generates Ansible playbooks for changing the hostname of a Red Hat server
and setting the hostname of an IOS-XE network device using a pre-trained LLM model from HuggingFace.

# Hugging Face Transformers:
AutoModelForCausalLM and AutoTokenizer for loading and using the LLaMA-2 model.

# PyTorch:
Used for managing the model and tensors (torch.device for GPU/CPU handling).

# GPU Acceleration:
Model and inputs moved to GPU (cuda) for faster processing.
Mixed precision (torch.float16) used to optimize memory and speed.

# Text Generation:
generate method with parameters like max_new_tokens, num_beams, top_k, and top_p for controlling output.

# YAML Playbook Generation:
Two separate prompts for generating:
Red Hat server hostname playbook.
IOS-XE network device hostname playbook.

# File Handling:
Generated playbooks written to:
generated_ansible_playbook_redhat.yml
generated_ansible_playbook_iosxe.yml

# Decoding and Display:
Outputs decoded and printed to the terminal for verification.

Outputs of running the script:

Generate an Ansible playbook to change the hostname of a Red Hat server.
The playbook should:
1. Use the `hostname` module to set the hostname.
2. Update the `/etc/hostname` file.
3. Update the `/etc/hosts` file to map the hostname to the loopback address (127.0.0.1).
Provide the playbook in YAML format.

### Example playbook
```yaml
---
- name: Change hostname
  hosts: localhost
  vars:
    hostname: newhostname
  tasks:
    - name: Update hostname
      hostname: "{{ hostname }}"
    - name: Update hosts file
      hosts:
        - "{{ hostname }}"
      update_hostnames: true
      add_host:
        name: "{{ hostname }}"
        addresses: [ "{{ loopback_address }}" ]
```

Red Hat playbook written to: generated_ansible_playbook_redhat.yml
Generating the playbook for the IOS-XE network device...
Playbook generation for the IOS-XE network device completed.
Generated Ansible Playbook for IOS-XE network device:

Generate an Ansible playbook to set the hostname of an IOS-XE network device.
The playbook should:
1. Use the `ios_config` module to set the hostname.
2. Ensure the configuration is saved to the device.
Provide the playbook in YAML format.

### Examples

```
---
- name: Set hostname on an IOS-XE network device
  ios_config:
    commands:
      - hostname {{ inventory_hostname }}
  delegate_to: localhost
```

### Requirements

This role requires Ansible 2.10 or higher.

### Variables

| Name | Required | Type | Default | Description |
| ---- | -------- | ---- | ------- | ----------- |
| commands | yes | list | - | The commands to send to the device. |

### Example Playbook

```
---
- name: Set host
IOS-XE playbook written to: generated_ansible_playbook_iosxe.yml
"""


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer from HuggingFace
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Define the first task: Generate an Ansible playbook to change the hostname of a Red Hat server
prompt_redhat = """
Generate an Ansible playbook to change the hostname of a Red Hat server.
The playbook should:
1. Use the `hostname` module to set the hostname.
2. Update the `/etc/hostname` file.
3. Update the `/etc/hosts` file to map the hostname to the loopback address (127.0.0.1).
Provide the playbook in YAML format.
"""

# Tokenize the input prompt and move it to the GPU
inputs_redhat = tokenizer(prompt_redhat, return_tensors="pt").to(device)

# Set the maximum number of tokens to generate
max_new_tokens = 150

# Generate the playbook for the Red Hat server
print("Generating the playbook for the Red Hat server...")
output_redhat = model.generate(
    **inputs_redhat,
    max_new_tokens=max_new_tokens,
    num_beams=1,  # Greedy decoding
    top_k=50,  # Limit sampling to the top 50 tokens
    top_p=0.9,  # Use nucleus sampling
    early_stopping=True
)
print("Playbook generation for the Red Hat server completed.")

# Decode and display the generated playbook for the Red Hat server
generated_playbook_redhat = tokenizer.decode(output_redhat[0], skip_special_tokens=True)
print("Generated Ansible Playbook for Red Hat server:")
print(generated_playbook_redhat)

# Write the Red Hat playbook to a file
output_file_redhat = "generated_ansible_playbook_redhat.yml"
with open(output_file_redhat, "w") as file:
    file.write(generated_playbook_redhat)
print(f"Red Hat playbook written to: {output_file_redhat}")

# Define the second task: Generate an Ansible playbook to set the hostname of an IOS-XE network device
prompt_iosxe = """
Generate an Ansible playbook to set the hostname of an IOS-XE network device.
The playbook should:
1. Use the `ios_config` module to set the hostname.
2. Ensure the configuration is saved to the device.
Provide the playbook in YAML format.
"""

# Tokenize the input prompt and move it to the GPU
inputs_iosxe = tokenizer(prompt_iosxe, return_tensors="pt").to(device)

# Generate the playbook for the IOS-XE network device
print("Generating the playbook for the IOS-XE network device...")
output_iosxe = model.generate(
    **inputs_iosxe,
    max_new_tokens=max_new_tokens,
    num_beams=1,  # Greedy decoding
    top_k=50,  # Limit sampling to the top 50 tokens
    top_p=0.9,  # Use nucleus sampling
    early_stopping=True
)
print("Playbook generation for the IOS-XE network device completed.")

# Decode and display the generated playbook for the IOS-XE network device
generated_playbook_iosxe = tokenizer.decode(output_iosxe[0], skip_special_tokens=True)
print("Generated Ansible Playbook for IOS-XE network device:")
print(generated_playbook_iosxe)

# Write the IOS-XE playbook to a file
output_file_iosxe = "generated_ansible_playbook_iosxe.yml"
with open(output_file_iosxe, "w") as file:
    file.write(generated_playbook_iosxe)
print(f"IOS-XE playbook written to: {output_file_iosxe}")