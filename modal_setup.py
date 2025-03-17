import modal
# setup the image for curious training
app = modal.App("curious-training")
# Create base image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .add_local_dir(
        local_path=".", remote_path="root/curious"
    )
    .run_commands(
        [
            "pip install poetry && "
            "poetry install "
        ]
    )
    .env({"WANDB_API_KEY": "4e3f313c912c270925860ef23359126c2e996aa6"})  # Set environment variable
    .run_commands(
        "wandb login"
    )
) 