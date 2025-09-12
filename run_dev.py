#!/usr/bin/env python3
"""
Development server runner with automatic reload and environment setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists, if not copy from example."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("Please update .env with your actual configuration values.")
        return False
    elif not env_file.exists():
        print("No .env file found. Please create one with your configuration.")
        return False
    
    return True

def main():
    """Run the development server."""
    if not check_env_file():
        sys.exit(1)
    
    # Set development environment
    os.environ.setdefault("ENVIRONMENT", "development")
    
    try:
        # Run uvicorn with reload
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nShutting down development server...")

if __name__ == "__main__":
    main()
