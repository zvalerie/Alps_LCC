import sys
sys.path.append('.')
sys.path.append('..')
from ACE_DeepLabV3Plus import ACE_deeplabv3P_w_Experts
from models.ACE_Better_DeepLabV3Plus import ACE_deeplabv3P_w_Better_Experts
from DeepLabV3Plus import deeplabv3P_resnet
from Unet import Res50_UNet
from ResNet import resnet50