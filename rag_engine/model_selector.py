import subprocess

def ollama_available() -> bool:
    try:
        subprocess.run(["ollama", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False
