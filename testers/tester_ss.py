# test_ss.py

# [기본 라이브러리]----------------------
import os
import numpy as np
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time

# [py 파일]-----------------------------------------------------------------------------------------------------
from utils.calc_func        import *
from utils.data_load_n_save import *
from utils.data_tool        import *


# 3. SS
#       D3P: (DeepLab v3 Plus)
#       SS_A -> 출력이 단일 tensor 형태인 모델 (입출력물 크기 조정기능 제공)
list_ss_model_type_a = ["D3P", "DABNet", "CGNet", "FPENet"]


def tester_ss(**kargs):
    
    '''
    tester_ss(# SR 계열 모델로 생성된 이미지 저장하는 코드 (model, str)
              model                             = 
             ,model_name                        = 
              # path_model_state_dict 또는 path_model_check_point 중 1 가지 입력 (path_model_state_dict 우선 적용) (str)
             ,path_model_state_dict             = 
             ,path_model_check_point            = None
             
              # 이미지 입력 폴더 경로 (str)
             ,path_input_hr_images              =
             ,path_input_sr_images              =
             ,path_outputs                      = 
             
              # 이미지 정규화 여부 (bool) 및 정규화 설정 (list) (정규화 안하면 설정값 생략 가능)
             ,is_norm_in_transform_to_tensor    = 
             ,HP_TS_NORM_MEAN                   = HP_TS_NORM_MEAN
             ,HP_TS_NORM_STD                    = HP_TS_NORM_STD
             
              # 라벨 정보
             ,HP_COLOR_MAP                      = HP_COLOR_MAP
             )
    '''
    
    
    # 사용 decive 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [ model 관련 ] -------------------------------------------------------------------------------------------
    
    model = kargs['model']
    model.to(device)
    
    model_name = kargs['model_name']
    
    try:
        path_model_state_dict = kargs['path_model_state_dict']
        model.load_state_dict(torch.load(path_model_state_dict))
    except:
        flag_tmp = 1
        try:
            path_model_check_point = kargs['path_model_check_point']
            loaded_chech_point = torch.load(path_model_check_point)
            model.load_state_dict(loaded_chech_point['model_state_dict'])
            flag_tmp = 0
        except:
            print("(exc) model_check_point 불러오기 실패")
            sys.exit(-1)
        
        if flag_tmp != 0:
            print("(exc) model_state_dict 불러오기 실패")
            sys.exit(-1)
    
    model.eval()
    
    # [ 입출력 Data 관련 ] ----------------------------------------------------------------------------------------
    # 경로: 입력
    path_input_hr_images = kargs['path_input_hr_images']
    if path_input_hr_images[-1] != '/':
        path_input_hr_images += '/'
    
    try:
        path_input_sr_images = kargs['path_input_sr_images']
        if path_input_sr_images[-1] != '/':
            path_input_sr_images += '/'
    except:
        path_input_sr_images = None # 원본 이미지로 test 시행예정
    
    # 경로: 출력
    path_outputs = kargs['path_outputs']
    if path_outputs[-1] != "/":
        path_outputs += "/"
    path_outputs += model_name + "/"
    
    try:
        if not os.path.exists(path_outputs + "Gray"):
            os.makedirs(path_outputs + "Gray")
        if not os.path.exists(path_outputs + "RGB"):
            os.makedirs(path_outputs + "RGB")
    except:
        print("(exc)", "Folder gen FAIL:", path_outputs)
    
    
    # [ 이미지 변수 -> 텐서 변수 변환 ] ---------------------------------------------------------------------------------
    # 정규화 여부
    try:
        is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor']
    except:
        is_norm_in_transform_to_tensor  = False
        HP_TS_NORM_MEAN                 = None
        HP_TS_NORM_STD                  = None
    
    if is_norm_in_transform_to_tensor:
        # 평균
        HP_TS_NORM_MEAN = kargs['HP_TS_NORM_MEAN']
        # 표준편차
        HP_TS_NORM_STD = kargs['HP_TS_NORM_STD']
        # 입력 이미지 텐서 변환 후 정규화 시행
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  transforms.ToTensor()
                                                  # 평균, 표준편차를 활용해 정규화
                                                 ,transforms.Normalize(mean = HP_TS_NORM_MEAN, std = HP_TS_NORM_STD)
                                                 ,
                                                 ])
        
        
    else:
        # 정규화 없이 이미지를 텐서형으로 변환
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
                                                  # 일반적인 경우 (PIL mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1 또는 numpy.ndarray), 
                                                  # (H x W x C) in the range [0, 255] 입력 데이터를
                                                  # (C x H x W) in the range [0.0, 1.0] 출력 데이터로 변환함 (scaled)
                                                  transforms.ToTensor()
                                                 ])
    
    
    # [ 기타 옵션값 ] ---------------------------------------------------------------------------------------------
    HP_COLOR_MAP        = kargs['HP_COLOR_MAP']
    
    # [ 실험 시행 ] ----------------------------------------------------------------------------------------------
    
    list_input_names = os.listdir(path_input_hr_images) # (list with str) file names
    
    count_dataloader = 0
    i_batch_max = len(list_input_names)
    
    for i_name in list_input_names:
        count_dataloader += 1
        print("\r[", model_name, "]", count_dataloader, " / ", i_batch_max, end='')
        
        if path_input_sr_images is not None:
            in_pil = Image.open(path_input_sr_images + i_name)
        else:
            in_pil = Image.open(path_input_hr_images + i_name)
        
        in_ts = transform_to_ts_img(in_pil)
        in_ts = in_ts.unsqueeze(dim=0)
        in_ts = in_ts.to(device)
        
        with torch.no_grad():
            if model_name in list_ss_model_type_a:
                ts_pred = model(in_ts_hr)
            else:
                _str = "ERROR: not supporting model"
                sys.exit(_str)
            
            # label 예측결과 softmax 시행
            ts_pred_softmax = F.softmax(ts_pred, dim = 1)
            # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
            ts_pred_label = torch.argmax(ts_pred_softmax.clone().detach().requires_grad_(False), dim = 1)
            
            list_out_pil_label = tensor_2_list_pils_v1(# 텐서 -> pil 이미지 리스트
                                                       in_tensor = ts_pred_label
                                                       # (bool) 라벨 여부 (출력 pil 이미지 = 3ch, 라벨 = 1ch 고정) (default: False)
                                                      ,is_label = True
                                                       # (bool) pil 이미지 크기 변환 시행여부 (default: False)
                                                      ,is_resized = False
                                                      )
        
        
        pil_pred_gray = list_out_pil_label[0]
        pil_pred_rgb  = label_2_RGB(pil_pred_gray, HP_COLOR_MAP)
        
        try:
            pil_pred_gray.save(path_outputs + "GRAY/" + i_name)
            pil_pred_RGB.save( path_outputs + "RGB/"  + i_name)
        except:
            _str = "ERROR: Label save FAIL " + i_name
            sys.exit(_str)
        
    
    print("\nFinished")

print("EoF: tester_ss.py")