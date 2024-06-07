import torch
from pathlib import Path

# 1. create model's directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def save_model(model, MODEL_NAME):
    # 2. create a model save path
    #MODEL_NAME = "chapter_1_practice_1_model_0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(MODEL_SAVE_PATH)

    # 3. save the model state_dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
