from tensorboardX import SummaryWriter
import torchsummary
import os
import torch
import anyconfig
from models import build_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

checkpoint = torch.load("C:/Users/94806/Desktop/output/a.pth", map_location=torch.device('cpu'))
checkpoint["config"]["dataset"]["validate"]["loader"]["pin_memory"] = False
torch.save(checkpoint, "C:/Users/94806/Desktop/output/a.pth")