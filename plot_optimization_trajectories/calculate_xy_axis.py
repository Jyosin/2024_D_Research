import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import os
import re


def get_weight_path(directory_path):
    # 正则表达式匹配模式 "model_weights_epoch_X.pth"
    pattern = re.compile(r'model_weights_epoch_(\d+)\.pth$')
    
    # 获取目录下所有文件名
    files = os.listdir(directory_path)
    
    # 过滤和排序文件
    # 使用正则表达式找到匹配的文件，并根据 epoch 数字排序
    sorted_files = sorted(
        [file for file in files if pattern.match(file)],
        key=lambda x: int(pattern.search(x).group(1)),
        reverse=True
    )
    
    # 输出排序后的文件路径
  
    return sorted_files
    

def read_weight():
    
    return

def calcuate_best_poin():
    return

def wright_data():
    return

def plot_point():
    return

def plot_line():
    return

print(get_weight_path('/Users/wangruqin/Desktop/teacher_student/result/train'))