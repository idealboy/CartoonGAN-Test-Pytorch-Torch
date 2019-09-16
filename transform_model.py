import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default = './pretrained_model')
parser.add_argument('--style', default = 'Hosoda')
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--load_size', default = 450)

opt = parser.parse_args()

# load pretrained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))

model.cuda()

example = torch.rand((1,3,256,256)).cuda()

traced_script_module = torch.jit.script(model)

traced_script_module.save("cartoon.pt")
