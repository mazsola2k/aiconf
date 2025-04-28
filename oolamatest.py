#pip install llama3_package
from llama3 import Llama3Model

# Initialize the model
model = Llama3Model()

# Send a prompt to the model
response = model.prompt("5+5=")
print("Prompt Response:", response)

# Stream a prompt to the model
for chunk in model.stream_prompt("Tell me a joke"):
    print("Stream Prompt Response:", chunk)
