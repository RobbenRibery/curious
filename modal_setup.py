import modal
import os
from dotenv import load_dotenv
load_dotenv()

# setup the image for curious training
app = modal.App("curious-training")
# Create base image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .add_local_python_source("curious", copy=True)
    .add_local_file(
        "modal_setup.py",
        remote_path="/root/modal_setup.py",
        copy=True,
    )
    .add_local_file(
        "training.py",
        remote_path="/root/training.py",
        copy=True,
    )
    .add_local_file(
        "pyproject.toml", 
        remote_path="/root/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        "poetry.lock", 
        remote_path="/root/poetry.lock",
        copy=True,
    )
    .run_commands(
        [   
            "cd /root && "
            "pip install poetry && "
            "poetry install"
        ]
    )
) 