import os
from huggingface_hub import login


def maybe_login():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
