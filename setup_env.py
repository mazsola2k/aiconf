import os
import sys
import subprocess
import platform
import re

def create_virtual_environment(venv_dir):
    """Create a virtual environment if it doesn't already exist."""
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in: {venv_dir}")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to create virtual environment. {e}")
            sys.exit(1)
    else:
        print(f"Virtual environment already exists in: {venv_dir}")

def get_pip_executable(venv_dir):
    """Determine the path to pip based on the operating system."""
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "pip")
    else:
        return os.path.join(venv_dir, "bin", "pip")

def ensure_pip_installed(pip_executable):
    """Ensure pip is installed and up-to-date in the virtual environment."""
    try:
        subprocess.check_call([pip_executable, "--version"])
        print(f"pip found at: {pip_executable}")
    except FileNotFoundError:
        print(f"Error: pip not found in virtual environment at: {pip_executable}")
        print("Attempting to reinstall pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            print("pip has been successfully installed and upgraded.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to reinstall pip. {e}")
            sys.exit(1)
    else:
        print(f"pip found at: {pip_executable}")

def install_build_tools(pip_executable):
    """Ensure pip, setuptools, and wheel are installed and updated."""
    print("Ensuring build tools (pip, setuptools, wheel) are installed...")
    try:
        subprocess.check_call([pip_executable, "install", "--upgrade", "setuptools", "wheel"])
        print("Build tools installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install build tools. {e}")
        sys.exit(1)

def extract_dependencies_from_files(folder_path):
    """Extract dependencies from all Python files in the folder."""
    dependencies = set()
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.match(r"^\s*(?:import|from)\s+([\w\.]+)", line)
                        if match:
                            module = match.group(1).split('.')[0]  # Get the top-level module
                            dependencies.add(module)
    return dependencies

def validate_dependencies(dependencies):
    """Validate if the dependencies are available on PyPI."""
    valid_dependencies = set()
    for dependency in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency, "--dry-run"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            valid_dependencies.add(dependency)
        except subprocess.CalledProcessError:
            print(f"Warning: {dependency} is not available on PyPI or cannot be installed.")
    return valid_dependencies

def update_requirements_file(dependencies, requirements_file="requirements.txt"):
    """Update the requirements.txt file with the extracted dependencies."""
    print(f"Updating {requirements_file}...")
    existing_dependencies = set()
    if os.path.exists(requirements_file):
        with open(requirements_file, "r", encoding="utf-8") as f:
            existing_dependencies = set(line.strip() for line in f if line.strip())

    # Combine existing and new dependencies
    all_dependencies = existing_dependencies.union(dependencies)

    with open(requirements_file, "w", encoding="utf-8") as f:
        for dependency in sorted(all_dependencies):
            f.write(f"{dependency}\n")
    print(f"{requirements_file} has been updated successfully!")

def install_requirements(pip_executable):
    """Install required pip packages from requirements.txt."""
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print(f"Found {requirements_file}. Installing dependencies...")
        try:
            subprocess.check_call([pip_executable, "install", "-r", requirements_file])
            print("All required packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to install packages from {requirements_file}. {e}")
            sys.exit(1)
    else:
        print(f"Warning: {requirements_file} not found. Skipping package installation.")

def print_activation_instructions(venv_dir):
    """Print instructions to activate the virtual environment."""
    if platform.system() == "Windows":
        activation_command = f"{venv_dir}\\Scripts\\activate"
    else:
        activation_command = f"source {venv_dir}/bin/activate"
    print("\nTo activate the virtual environment, run the following command in your terminal:")
    print(f"{activation_command}\n")
    print("Note: You must activate the virtual environment before running any Python scripts that depend on it.")

if __name__ == "__main__":
    # Define the virtual environment directory
    venv_dir = os.path.abspath("../myenv")  # Use an absolute path for reliability

    # Step 1: Create virtual environment
    create_virtual_environment(venv_dir)

    # Step 2: Get the pip executable path
    pip_executable = get_pip_executable(venv_dir)

    # Step 3: Ensure pip is installed
    ensure_pip_installed(pip_executable)

    # Step 4: Ensure build tools (setuptools and wheel) are installed
    install_build_tools(pip_executable)

    # Step 5: Extract dependencies from Python files
    project_folder = os.path.abspath(".")  # Current folder
    dependencies = extract_dependencies_from_files(project_folder)

    # Step 6: Validate dependencies
    valid_dependencies = validate_dependencies(dependencies)

    # Step 7: Update requirements.txt
    update_requirements_file(valid_dependencies)

    # Step 8: Install required packages from requirements.txt
    install_requirements(pip_executable)

    # Step 9: Print activation instructions
    print_activation_instructions(venv_dir)

    print("Environment setup completed successfully!")