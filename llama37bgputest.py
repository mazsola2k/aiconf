# Import required libraries
#https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
#>nvcc --version
#Cuda compilation tools, release 12.8, V12.8.93
#Build cuda_12.8.r12.8/compiler.35583870_0
#https://pytorch.org/get-started/locally/#windows-pip
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#pip install accelerate
import time
import psutil  # For CPU and RAM monitoring
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# GPU monitoring library
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
    pynvml_available = True
except ImportError:
    pynvml_available = False
    print("pynvml is not installed. Install it with 'pip install nvidia-ml-py3' for GPU monitoring.")

def display_system_stats():
    """Displays CPU, RAM, and GPU usage statistics."""
    # CPU and RAM usage stats
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory()

    print(f"CPU Usage: {cpu_usage}%")
    print(f"RAM Usage: {ram_usage.used / (1024 ** 3):.2f} GB / {ram_usage.total / (1024 ** 3):.2f} GB")
    
    # GPU usage stats (if available)
    if pynvml_available:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU
        gpu_memory = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU Memory Usage: {gpu_memory.used / (1024 ** 3):.2f} GB / {gpu_memory.total / (1024 ** 3):.2f} GB")
        nvmlShutdown()
    else:
        print("GPU monitoring not available (pynvml not installed).")

def test_llama3_pipeline():
    """Tests the LLaMA 7B pipeline with system resource monitoring."""
    # Display initial system stats
    print("Initial system stats:")
    display_system_stats()
    
    # Step 1: Load the tokenizer and model for LLaMA 7B
    try:
        print("\nLoading LLaMA 7B model and tokenizer...")
        model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the correct identifier
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Display system stats after loading the model
        print("\nSystem stats after loading the model:")
        display_system_stats()

        # Step 2: Initialize the pipeline
        print("\nInitializing pipeline...")
        text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)  # Removed `device=0`

        # Display system stats after initializing the pipeline
        print("\nSystem stats after initializing the pipeline:")
        display_system_stats()

        # Step 3: Test the pipeline with a basic input
        print("\nTesting pipeline...")
        test_input = "What is the result of 2 + 2?"
        start_time = time.time()
        output = text_pipeline(test_input, max_length=50, num_return_sequences=1)
        end_time = time.time()

        # Display the output and processing time
        print("\nGenerated Text:")
        print(output[0]['generated_text'])
        print(f"\nProcessing Time: {end_time - start_time:.2f} seconds")

        # Final system stats
        print("\nFinal system stats:")
        display_system_stats()

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test
if __name__ == "__main__":
    test_llama3_pipeline()