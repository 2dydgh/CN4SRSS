# test_srss.py

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

# [py 파일]--------------------------
from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *


# https://github.com/KyungBong-Ryu/Codes_implementation/blob/main/BasicSR_NIQE.py
# from DLCs.BasicSR_NIQE import calc_niqe_with_pil
# from DLCs.BasicSR_NIQE import calc_niqe as _calc_niqe


# 2. SRSS

list_srss_model_type_a = ["Ab_07"]
list_srss_model_type_b = ["Ab_01"]


def tester_srss(**kargs):
    
    '''
    tester_sr(# SR 모델 혹은 알고리즘으로 SR 시행한 이미지 저장하는 코드
              #--- (선택 2/2) SR 계열 모델로 생성된 이미지 저장하는 옵션 (model, str)
             ,model                             = model_sr
             ,model_name                        = model_sr_name
              # path_model_state_dict 또는 path_model_check_point 중 1 가지 입력 (path_model_state_dict 우선 적용) (str)
             ,path_model_state_dict             = path_model_state_dict
             ,path_model_check_point            = path_model_check_point
             
              # 이미지 입출력 폴더 경로 (str)
             ,path_input_hr_images              = path_input_hr_images
             ,path_input_lr_images              = path_input_lr_images
             ,path_output_images                = path_output_images
             
              # 이미지 정규화 여부 (bool) 및 정규화 설정 (list) (정규화 안하면 설정값 생략 가능)
             ,is_norm_in_transform_to_tensor    = is_norm_in_transform_to_tensor
             ,HP_TS_NORM_MEAN                   = HP_TS_NORM_MEAN
             ,HP_TS_NORM_STD                    = HP_TS_NORM_STD
             )
    '''
    
    # [ 입출력 Data 관련 ] -----------------------------
    # 경로: 입력
    # PATH_BASE_IN = kargs['PATH_BASE_IN']
    # NAME_FOLDER_TEST = kargs['NAME_FOLDER_TEST']
    # NAME_FOLDER_IMAGES = kargs['NAME_FOLDER_IMAGES']
    
    path_input_hr_images = kargs['path_input_hr_images']
    if path_input_hr_images[-1] != '/':
        path_input_hr_images += '/'
    
    path_input_lr_images = kargs['path_input_lr_images']
    if path_input_lr_images[-1] != '/':
        path_input_lr_images += '/'
    
    # 경로: 출력
    path_output_images = kargs['path_output_images']
    if path_output_images[-1] != '/':
        path_output_images += '/'
    
    
    method_name = None
    
    
    if method_name is not None:
        try:
            if not os.path.exists(path_output_images + method_name):
                os.makedirs(path_output_images + method_name)
        except:
            print("(exc)", "Folder gen FAIL:", path_output_images + method_name)
        
    else:
        # 사용 decive 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [ model 관련 ] ------------------------------ 
        
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
        
        
        # [ 이미지 변수 -> 텐서 변수 변환 ] -------------------
        # 정규화 여부
        is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor']
        
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
            
            # 역정규화 변환
            transform_ts_inv_norm = transforms.Compose([# 평균, 표준편차를 역으로 활용해 역정규화
                                                        transforms.Normalize(mean = [ 0., 0., 0. ]
                                                                            ,std = [ 1/HP_TS_NORM_STD[0], 1/HP_TS_NORM_STD[1], 1/HP_TS_NORM_STD[2] ])
                                                         
                                                       ,transforms.Normalize(mean = [ -HP_TS_NORM_MEAN[0], -HP_TS_NORM_MEAN[1], -HP_TS_NORM_MEAN[2] ]
                                                                            ,std = [ 1., 1., 1. ])
                                                                            
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
        
        
        try:
            if not os.path.exists(path_output_images + model_name):
                os.makedirs(path_output_images + model_name)
        except:
            print("(exc)", "Folder gen FAIL:", path_output_images + model_name)
    
    
    list_input_names = os.listdir(path_input_hr_images) # (list with str) file names
    
    count_dataloader = 0
    i_batch_max = len(list_input_names)
    
    for i_name in list_input_names:
        count_dataloader += 1
        
        in_pil = Image.open(path_input_lr_images + i_name)
        
        
        # using SR model
        print("\r[", model_name, "]", count_dataloader, " / ", i_batch_max, end='')
        in_ts = transform_to_ts_img(in_pil)
        
        in_ts = in_ts.unsqueeze(dim=0)
        
        in_ts = in_ts.to(device)
        
        with torch.no_grad():
            if model_name in list_srss_model_type_a:
                out_ts_1, _, out_ts_sr, out_ts_ss = = model(in_ts)
            elif model_name in list_srss_model_type_b:
                out_ts_1, out_ts_2, out_ts_sr, out_ts_ss = = model(in_ts)
            
            if is_norm_in_transform_to_tensor:
                out_pil_sr = tensor_2_list_pils_v1(in_tensor  = transform_ts_inv_norm(out_ts_sr)
                                                  ,is_label   = False
                                                  ,is_resized = False
                                                  )[0]
            else:
                out_pil_sr = tensor_2_list_pils_v1(in_tensor  = out_ts_sr
                                                  ,is_label   = False
                                                  ,is_resized = False
                                                  )[0]
        
        try:
            out_pil_sr.save(path_output_images + model_name + "/" + i_name)
        except:
            print("(exc) PIL save FAIL:", path_output_images + model_name + "/" + i_name)
            sys.exit(-9)
        
        
    
    print("\nFinished")

print("EoF: tester_sr.py")