# test_sr.py

# [기본 라이브러리]----------------------
import os
import numpy as np
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

import ignite   #pytorch ignite

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time

import sys

# [py 파일]--------------------------
from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *


# https://github.com/KyungBong-Ryu/Codes_implementation/blob/main/BasicSR_NIQE.py
# from DLCs.BasicSR_NIQE import calc_niqe_with_pil
# from DLCs.BasicSR_NIQE import calc_niqe as _calc_niqe


# 2. SR
#       SR_A -> 출력이 list 형태인 모델
list_sr_model_type_a = ["MPRNet", "ARNet"]
#       SR_B -> 출력이 단일 tensor 형태인 모델
list_sr_model_type_b = ["ESRT", "HAN", "IMDN", "BSRN", "RFDN", "PAN", "LAPAR"]




def patch_wise_inference(model, in_ts, scale=4, overlap=10, pixel_max=60000):
    # inspired from https://github.com/luissen/ESRT/blob/main/test.py#L29
    # model: ESRT model
    # in_ts: input low-resolution image
    # scale: scale factor
    # overlap: patch 에서 곂쳐질 영역을 더 자를 픽셀 수 
    # pixel_max : 전체 픽셀 수가 이보다 큰 경우, 4분할 후 inference 시행
    
    _, _, _h, _w = in_ts.size()
    
    #print("in_ts.size()", in_ts.size())
    
    if _h*_w <= pixel_max:
        out_ts = model(in_ts)
        return out_ts
    else:
        p_h = _h//2 + overlap
        p_w = _w//2 + overlap
        
        in_ts_lu = in_ts[:, :, 0:p_h,       0:p_w      ]
        in_ts_ru = in_ts[:, :, 0:p_h,       _w - p_w:_w]
        in_ts_ld = in_ts[:, :, _h - p_h:_h, 0:p_w      ]
        in_ts_rd = in_ts[:, :, _h - p_h:_h, _w - p_w:_w]
        
        p_h = (_h*scale)//2
        p_w = (_w*scale)//2
        
        out_ts_lu = patch_wise_inference(model, in_ts_lu, scale, overlap, pixel_max)[:, :, 0:p_h,     0:p_w    ]
        out_ts_ru = patch_wise_inference(model, in_ts_ru, scale, overlap, pixel_max)[:, :, 0:p_h,     -p_w-1:-1]
        out_ts_ld = patch_wise_inference(model, in_ts_ld, scale, overlap, pixel_max)[:, :, -p_h-1:-1, 0:p_w    ]
        out_ts_rd = patch_wise_inference(model, in_ts_rd, scale, overlap, pixel_max)[:, :, -p_h-1:-1, -p_w-1:-1]
        
        
        #print("out_ts_lu", out_ts_lu.size())
        #print("out_ts_ru", out_ts_ru.size())
        #print("out_ts_ld", out_ts_ld.size())
        #print("out_ts_rd", out_ts_rd.size())
        
        out_ts_l = torch.cat((out_ts_lu, out_ts_ld), dim=2)
        out_ts_r = torch.cat((out_ts_ru, out_ts_rd), dim=2)
        
        #print("out_ts_l", out_ts_l.size())
        #print("out_ts_r", out_ts_r.size())
        
        return torch.cat((out_ts_l, out_ts_r), dim=3)

def timer_warm_up(model, in_ts, is_patch_wise_inference=False, warm_up_loop=10):
    i_warm_max = warm_up_loop
    timer_gpu_start  = torch.cuda.Event(enable_timing=True)
    timer_gpu_finish = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for i_warm in range(i_warm_max):
            timer_gpu_start.record()
            if is_patch_wise_inference:
                _ = patch_wise_inference(model, in_ts, overlap=10, pixel_max=60000)
            else:
                _ = model(in_ts)
            timer_gpu_finish.record()
            torch.cuda.synchronize()
            print("\rTimer warm-up ", i_warm + 1, " / ", i_warm_max, " Timer: ", round(timer_gpu_start.elapsed_time(timer_gpu_finish) / 1000, 4), end="")
            
        print("\nTimer warm-up Finished!\n")

def tester_sr(**kargs):
    
    '''
    tester_sr(# SR 모델 혹은 알고리즘으로 SR 시행한 이미지 저장하는 코드
              #--- (선택 1/2) 알고리즘으로 생성된 이미지 저장하는 옵션 (str) -> None 입력 시 (선택 2/2) 옵션 적용됨
              method_name                       = method_name
             
              #--- (선택 2/2) SR 계열 모델로 생성된 이미지 저장하는 옵션 (model, str)
             ,model                             = model_sr
             ,model_name                        = model_sr_name
              # path_model_state_dict 또는 path_model_check_point 중 1 가지 입력 (path_model_state_dict 우선 적용) (str)
             ,path_model_state_dict             = path_model_state_dict
             ,path_model_check_point            = path_model_check_point
             
              # 이미지 입출력 폴더 경로 (str)
             ,path_input_hr_images              = path_input_hr_images
             ,path_input_lr_images              = path_input_lr_images
             ,path_output_images                = path_output_images        #None 입력 시 저장 안함
             
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
    print("")
    path_input_hr_images = kargs['path_input_hr_images']
    print("path_input_hr_images:", path_input_hr_images)
    if path_input_hr_images[-1] != '/':
        path_input_hr_images += '/'
    
    path_input_lr_images = kargs['path_input_lr_images']
    print("path_input_lr_images:", path_input_lr_images)
    if path_input_lr_images[-1] != '/':
        path_input_lr_images += '/'
    
    # 경로: 출력
    path_output_images = kargs['path_output_images']    # None 입력 시, 결과저장안함
    print("path_output_images:", path_output_images)
    if path_output_images is not None:
        if path_output_images[-1] != '/':
            path_output_images += '/'
    
    
    method_name             = kargs['method_name']
    is_patch_wise_inference = kargs['is_patch_wise_inference']
    
    
    if method_name is not None:
        if path_output_images is not None:
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
            print("path_model_state_dict:", path_model_state_dict)
            model.load_state_dict(torch.load(path_model_state_dict, map_location=device))
        except:
            flag_tmp = 1
            try:
                path_model_check_point = kargs['path_model_check_point']
                print("path_model_check_point:", path_model_check_point)
                loaded_chech_point = torch.load(path_model_check_point, map_location=device)
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
        
        if path_output_images is not None:
            try:
                if not os.path.exists(path_output_images + model_name):
                    os.makedirs(path_output_images + model_name)
            except:
                print("(exc)", "Folder gen FAIL:", path_output_images + model_name)
    
    
    list_input_names = os.listdir(path_input_hr_images) # (list with str) file names
    
    count_dataloader = 0
    i_batch_max = len(list_input_names)
    
    
    try:
        timer_gpu_start  = torch.cuda.Event(enable_timing=True)
        timer_gpu_finish = torch.cuda.Event(enable_timing=True)
        timer_gpu_record_sum = 0.0
        timer_gpu_record_max = None
        timer_gpu_record_min = None
    except:
        print("GPU timer OFF")
        timer_gpu_start  = None
        timer_gpu_finish = None
    
    
    #<<< Ignite - Init
    transform_raw = transforms.Compose([transforms.ToTensor()])

    def ignite_eval_step(engine, batch):
        return batch
    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')

    psnr_ignite_sum = 0.0
    ssim_ignite_sum = 0.0
    #>>> Ignite - Init
    
    
    
    for i_name in list_input_names:
        count_dataloader += 1
        
        in_pil = Image.open(path_input_lr_images + i_name)
        pil_hr = Image.open(path_input_hr_images + i_name)
        
        if method_name is not None:
            # 전통방식
            print("\r[", method_name, "]", count_dataloader, " / ", i_batch_max, end='')
            _w, _h = in_pil.size
            _scale_factor = 4
            out_size = (int(_w * _scale_factor), int(_h * _scale_factor))
            if method_name == "Bilinear":
                # bilinear interpolation
                out_pil = in_pil.resize(out_size, Image.BILINEAR)
            
            if path_output_images is not None:
                try:
                    out_pil.save(path_output_images + method_name + "/" + i_name)
                except:
                    print("(exc) PIL save FAIL:", path_output_images + method_name + "/" + i_name)
                    sys.exit(-9)
            
        else:
            # using SR model
            in_ts = transform_to_ts_img(in_pil)
            ts_hr = transform_to_ts_img(pil_hr)
            
            in_ts = in_ts.unsqueeze(dim=0)
            ts_hr = ts_hr.unsqueeze(dim=0)
            
            in_ts = in_ts.to(device)
            ts_hr = ts_hr.to(device)
            
            with torch.no_grad():
                # if model_name == "ESRT":
                
                if count_dataloader <= 1 and timer_gpu_start is not None and path_output_images is None:
                    timer_warm_up(model, in_ts, is_patch_wise_inference=is_patch_wise_inference, warm_up_loop=100)
                
                if timer_gpu_start is not None:
                    timer_gpu_start.record()
                if is_patch_wise_inference:
                    out_ts_raw = patch_wise_inference(model, in_ts, overlap=10, pixel_max=60000)
                else:
                    out_ts_raw = model(in_ts)
                if timer_gpu_finish is not None:
                    timer_gpu_finish.record()
                
                try:
                    torch.cuda.synchronize()
                    timer_gpu_record = round(timer_gpu_start.elapsed_time(timer_gpu_finish) / 1000, 4)
                    #print(" timer_gpu_record", timer_gpu_record)
                    timer_gpu_record_sum += timer_gpu_record
                    
                    if timer_gpu_record_max is None or timer_gpu_record_max < timer_gpu_record:
                        timer_gpu_record_max = timer_gpu_record
                    
                    if timer_gpu_record_min is None or timer_gpu_record_min > timer_gpu_record:
                        timer_gpu_record_min = timer_gpu_record
                    
                except:
                    timer_gpu_record = None
                    timer_gpu_record_sum = None
                
                
                if model_name in list_sr_model_type_a:
                    out_ts = out_ts_raw[0]
                elif model_name in list_sr_model_type_b:
                    out_ts = out_ts_raw
                
                if is_norm_in_transform_to_tensor:
                    ignite_result = ignite_evaluator.run([[transform_ts_inv_norm(out_ts)
                                                          ,transform_ts_inv_norm(ts_hr)
                                                          ]])
                    
                    out_pil = tensor_2_list_pils_v1(in_tensor  = transform_ts_inv_norm(out_ts)
                                                   ,is_label   = False
                                                   ,is_resized = False
                                                   )[0]
                    
                else:
                    ignite_result = ignite_evaluator.run([[out_ts
                                                          ,ts_hr
                                                          ]])
                    
                    out_pil = tensor_2_list_pils_v1(in_tensor  = out_ts
                                                   ,is_label   = False
                                                   ,is_resized = False
                                                   )[0]
                
                
                #<<<Ignite - Calc
                
                _psnr_ignite = ignite_result.metrics['psnr']
                _ssim_ignite = ignite_result.metrics['ssim']
                
                psnr_ignite_sum += _psnr_ignite
                ssim_ignite_sum += _ssim_ignite
                #print("ignite", _psnr_ignite, _ssim_ignite)
                #>>> Ignite - Calc
                
                
            if path_output_images is not None:
                try:
                    out_pil.save(path_output_images + model_name + "/" + i_name)
                except:
                    print("(exc) PIL save FAIL:", path_output_images + model_name + "/" + i_name)
                    sys.exit(-9)
            
            if count_dataloader < 6:
                print("[", model_name, "] ", count_dataloader, " / ", i_batch_max, " GPU Timer: ", timer_gpu_record)
            else:
                print("\r[", model_name, "] ", count_dataloader, " / ", i_batch_max, " GPU Timer: ", timer_gpu_record, end='')
        
    print("\n===[ Results ]===\n")
    print("Finished with", count_dataloader, "items")
    try:
        if timer_gpu_record_sum is not None:
            print("GPU timer_gpu_record_mean", timer_gpu_record_sum / count_dataloader)
            print("GPU timer_gpu_record_max", timer_gpu_record_max)
            print("GPU timer_gpu_record_min", timer_gpu_record_min)
    except:
        print("GPU timer OFF")
    
    print("Ignite PSNR, SSIM:", psnr_ignite_sum/count_dataloader, ssim_ignite_sum/count_dataloader)
    
    if path_output_images is not None:
        print("WARNING: 결과 이미지가 저장됨에 따라 GPU timer에 outlier가 발생될 수 있습니다.")
    

print("EoF: tester_sr.py")