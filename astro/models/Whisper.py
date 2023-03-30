import torch.nn as nn
import torch.nn.functional as F
import os
os.system("pip install git+https://github.com/openai/whisper.git")
import gradio as gr
import whisper

model = whisper.load_model("large")


class Whisper(nn.Module):
    def __init__(self, idim):
        pass
