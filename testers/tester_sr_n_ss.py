# tester_sr_n_ss.py
### not ready yet ###

# SR 모델로 복원한 이미지로 SS 모델로 라벨 생성해봄

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
from DLCs.BasicSR_NIQE import calc_niqe as _calc_niqe







def tester_sr_n_ss(**kargs):
    
    
    tester_sr_n_ss(# SR 모델로 이미지 복원 후 SS 모델로 라벨 생성하는 함수 #
                   # 코드 실행 장소 (colab 여부 확인용, colab == -1)
                   RUN_WHERE = RUN_WHERE
                   # 버퍼 크기 (int > 0) -> 버퍼는 반드시 사용됨
                  ,BUFFER_SIZE = 
                   # 초기화 기록 dict 이어받기
                  ,dict_log_init = dict_log_init
                  
                   # 데이터 입출력 경로, 폴더명
                  ,PATH_BASE_IN = PATH_BASE_IN
                  ,NAME_FOLDER_TEST = NAME_FOLDER_TEST
                  ,NAME_FOLDER_IMAGES = NAME_FOLDER_IMAGES
                  ,NAME_FOLDER_LABELS = NAME_FOLDER_LABELS
                   # degraded image 불러올 경로
                  ,PATH_BASE_IN_SUB = PATH_BASE_IN_SUB
                   # 결과 저장 경로 (이미지, 로그)
                  ,PATH_OUT_IMAGE = PATH_OUT_IMAGE
                  ,PATH_OUT_LOG = PATH_OUT_LOG
                  
                   # 데이터(이미지) 입출력 크기 (원본 이미지, 모델 입력 이미지), 이미지 채널 수(이미지, 라벨, 모델출력물)
                  ,HP_ORIGIN_IMG_W = HP_ORIGIN_IMG_W
                  ,HP_ORIGIN_IMG_H = HP_ORIGIN_IMG_H
                  ,HP_CHANNEL_RGB = HP_CHANNEL_RGB
                  ,HP_CHANNEL_GRAY = HP_CHANNEL_GRAY
                  ,HP_CHANNEL_HYPO = HP_CHANNEL_HYPO
            
                   # 라벨 정보(원본 데이터 라벨 수(void 포함), void 라벨 번호, 컬러매핑)
                  ,HP_LABEL_TOTAL = HP_LABEL_TOTAL
                  ,HP_LABEL_VOID = HP_LABEL_VOID
                  ,HP_COLOR_MAP = HP_COLOR_MAP
                  
                   # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
                  ,is_norm_in_transform_to_tensor_sr = 
                  ,HP_TS_NORM_MEAN_SR = 
                  ,HP_TS_NORM_STD_SR = 
                  
                  ,is_norm_in_transform_to_tensor_ss = 
                  ,HP_TS_NORM_MEAN_SS = 
                  ,HP_TS_NORM_STD_SS = 
                  
                   # Degradation 관련 설정값
                  ,HP_DG_CSV_NAME = HP_DG_CSV_NAME
                  ,HP_DG_SCALE_FACTOR = HP_DG_SCALE_FACTOR
                  
                  
                   #모델, 모델 이름, 모델 state dict 경로
                  ,model_sr = model_sr
                  ,model_sr_name = model_sr_name
                  ,model_sr_msd_path = model_sr_msd_path
                  
                  ,model_ss = model_ss
                  ,model_ss_name = model_ss_name
                  ,model_ss_msd_path = model_ss_msd_path
                  )
    
    
    
    
    
    
    
    RUN_WHERE = kargs['RUN_WHERE']
    
    
    try:
        BUFFER_SIZE = kargs['BUFFER_SIZE']
        
        if BUFFER_SIZE < 1:
            print("BUFFER_SIZE should be > 0")
            BUFFER_SIZE = 60
    except:
        BUFFER_SIZE = 60
    
    print("BUFFER_SIZE set to", BUFFER_SIZE)
    
    
    # log dict 이어받기
    dict_log_init = kargs['dict_log_init']
    
    # 사용 decive 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # [입출력 Data 관련]-----------------------------
    # 경로: 입력
    PATH_BASE_IN = kargs['PATH_BASE_IN']
    NAME_FOLDER_TEST = kargs['NAME_FOLDER_TEST']
    NAME_FOLDER_IMAGES = kargs['NAME_FOLDER_IMAGES']
    NAME_FOLDER_LABELS = kargs['NAME_FOLDER_LABELS']
    
    
    # 경로: 출력
    PATH_OUT_IMAGE = kargs['PATH_OUT_IMAGE']
    if PATH_OUT_IMAGE[-1] != '/':
        PATH_OUT_IMAGE += '/'
    PATH_OUT_LOG = kargs['PATH_OUT_LOG']
    if PATH_OUT_LOG[-1] != '/':
        PATH_OUT_LOG += '/'
    
    
    # 원본 이미지 크기
    HP_ORIGIN_IMG_W = kargs['HP_ORIGIN_IMG_W']
    HP_ORIGIN_IMG_H = kargs['HP_ORIGIN_IMG_H']
    
    # 이미지&라벨&모델출력물 채널 수
    HP_CHANNEL_RGB = kargs['HP_CHANNEL_RGB']
    HP_CHANNEL_GRAY = kargs['HP_CHANNEL_GRAY']
    HP_CHANNEL_HYPO = kargs['HP_CHANNEL_HYPO']
    
    
    # 라벨 정보
    HP_LABEL_TOTAL = kargs['HP_LABEL_TOTAL']
    HP_LABEL_VOID = kargs['HP_LABEL_VOID']
    HP_COLOR_MAP = kargs['HP_COLOR_MAP']
    
    # 라벨 정보 (CamVid 12)
    update_dict_v2("", ""
                  ,"", "라벨 별 RGB 매핑"
                  ,"", "0:  [128 128 128]  # 00 sky"
                  ,"", "1:  [128   0   0]  # 01 building"
                  ,"", "2:  [192 192 128]  # 02 column_pole"
                  ,"", "3:  [128  64 128]  # 03 road"
                  ,"", "4:  [  0   0 192]  # 04 sidewalk"
                  ,"", "5:  [128 128   0]  # 05 Tree"
                  ,"", "6:  [192 128 128]  # 06 SignSymbol"
                  ,"", "7:  [ 64  64 128]  # 07 Fence"
                  ,"", "8:  [ 64   0 128]  # 08 Car"
                  ,"", "9:  [ 64  64   0]  # 09 Pedestrian"
                  ,"", "10: [  0 128 192]  # 10 Bicyclist"
                  ,"", "11: [  0   0   0]  # 11 Void"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    
    #모델, 모델 이름, 모델 state dict 경로
    model_sr = kargs['model_sr']
    model_sr_name = kargs['model_sr_name']
    model_sr_msd_path = kargs['model_sr_msd_path']
    
    model_ss = kargs['model_ss']
    model_ss_name = kargs['model_ss_name']
    model_ss_msd_path = kargs['model_ss_msd_path']
    
    
    
    
    # [Degradation 관련]-------------------------------
    
    try:
        PATH_BASE_IN_SUB = kargs['PATH_BASE_IN_SUB']
        HP_DG_CSV_NAME = kargs['HP_DG_CSV_NAME']
        if not PATH_BASE_IN_SUB[-1] == "/":
            PATH_BASE_IN_SUB += "/"
        
        dict_loaded_pils = load_pils_2_dict(# 경로 내 pil 이미지를 전부 불러와서 dict 형으로 묶어버림
                                            # (str) 파일 경로
                                            in_path = PATH_BASE_IN_SUB
                                            # (선택, str) 파일 경로 - 하위폴더명
                                           ,in_path_sub = NAME_FOLDER_IMAGES
                                           )
        print("Pre-Degraded images loaded from:", PATH_BASE_IN_SUB + NAME_FOLDER_IMAGES)
        
        dict_dg_csv = csv_2_dict(path_csv = PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
        print("Pre-Degrade option csv re-loaded from:", PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
        
        flag_pre_degraded_images_loaded = True
        tmp_log_pre_degraded_images_load = "Degraded 이미지를 불러왔습니다."
    except:
        print("(exc) Pre-Degraded images load FAIL")
        flag_pre_degraded_images_loaded = False
        tmp_log_pre_degraded_images_load = "Degraded 이미지를 불러오지 않습니다."
    
    update_dict_v2("", ""
                  ,"", "Degraded 이미지 옵션: " + tmp_log_pre_degraded_images_load
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # 고정옵션 dict
    HP_DG_CSV_NAME = kargs['HP_DG_CSV_NAME']
    HP_DG_CSV_PATH = PATH_BASE_IN_SUB + HP_DG_CSV_NAME
    dict_dg_csv = csv_2_dict(path_csv = HP_DG_CSV_PATH)
    
    # scale_factor 고정값
    HP_DG_SCALE_FACTOR = kargs['HP_DG_SCALE_FACTOR']
    
    
    #---
    
    
    # [이미지 변수 -> 텐서 변수 변환]-------------------
    # 정규화 여부
    is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor_sr']
    
    if is_norm_in_transform_to_tensor:
        # 평균
        HP_TS_NORM_MEAN = kargs['HP_TS_NORM_MEAN_SR']
        # 표준편차
        HP_TS_NORM_STD = kargs['HP_TS_NORM_STD_SR']
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
        
        update_dict_v2("", "SR"
                      ,"", "입력 이미지(in_x) 정규화 시행됨"
                      ,"", "mean=[ " + str(HP_TS_NORM_MEAN[0]) + " " + str(HP_TS_NORM_MEAN[1]) + " "+ str(HP_TS_NORM_MEAN[2]) + " ]"
                      ,"", "std=[ " + str(HP_TS_NORM_STD[0]) + " " + str(HP_TS_NORM_STD[1]) + " "+ str(HP_TS_NORM_STD[2]) + " ]"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        # 정규화 없이 이미지를 텐서형으로 변환
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
                                                  # 일반적인 경우 (PIL mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1 또는 numpy.ndarray), 
                                                  # (H x W x C) in the range [0, 255] 입력 데이터를
                                                  # (C x H x W) in the range [0.0, 1.0] 출력 데이터로 변환함 (scaled)
                                                  transforms.ToTensor()
                                                 ])
        
        update_dict_v2("", "SR"
                      ,"", "입력 이미지(in_x) 정규화 시행 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    
    
    dataset_test  = Custom_Dataset_V3(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'test '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TEST
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = False
                                     
                                      #--- options for HR label
                                     ,is_return_label               = True
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = True
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                      # sub-option a : don't use with sub-option b
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- options for generate patch
                                     ,is_patch                      = False
                                     
                                      #--- optionas for generate tensor
                                     ,model_input_size              = (HP_ORIGIN_IMG_W // HP_DG_SCALE_FACTOR
                                                                      ,HP_ORIGIN_IMG_H // HP_DG_SCALE_FACTOR
                                                                      )                                             #@@@ check required
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    dataloader_test  = torch.utils.data.DataLoader(dataset     = dataset_test
                                                  ,batch_size  = 1
                                                  ,shuffle     = False
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  )
    
    
    rb_test_loss = RecordBox(name = "test_loss", is_print = False
                            ,RUN_WHERE = RUN_WHERE, PATH_COLAB_OUT_LOG = PATH_COLAB_OUT_LOG
                            )
    rb_test_psnr = RecordBox(name = "test_psnr", is_print = False
                            ,RUN_WHERE = RUN_WHERE, PATH_COLAB_OUT_LOG = PATH_COLAB_OUT_LOG
                            )
    rb_test_ssim = RecordBox(name = "test_ssim", is_print = False
                            ,RUN_WHERE = RUN_WHERE, PATH_COLAB_OUT_LOG = PATH_COLAB_OUT_LOG
                            )
    rb_test_niqe = RecordBox(name = "test_niqe", is_print = False
                            ,RUN_WHERE = RUN_WHERE, PATH_COLAB_OUT_LOG = PATH_COLAB_OUT_LOG
                            )
    rb_test_ious = RecordBox4IoUs(name = "test_ious", is_print = False
                                 ,RUN_WHERE = RUN_WHERE, PATH_COLAB_OUT_LOG = PATH_COLAB_OUT_LOG
                                 )
    
    
    calc_niqe = _calc_niqe()         # new niqe method
    
    
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-# SR
    model_sr.to(device)
    
    # SR 이미지를 저장하는 dict
    buffer_sr_pil = {}      # key: 이미지 이름, elem: (pil) 이미지
    
    #전체 batch 개수
    i_batch_max = len(dataloader_input)
    
    count_dataloader = 0
    for dataloader_items in dataloader_test: 
        count_dataloader += 1
        print("\rSuper Resolution (", model_sr_name, ")", count_dataloader, "/", i_batch_max, end = '')
        ### Load Data
        dl_str_file_name    = dataloader_items[0]
        dl_ts_img_lr        = dataloader_items[6].float()
        dl_ts_img_lr = dl_ts_img_lr.to(device)
        
        #==============================================================================#
        #==============================================================================#
        with torch.no_grad():
            if model_sr_name == "ESRT":
                tensor_out_sr = model_sr(dl_ts_img_lr)
            
        
        
        
        
        # SR 이미지텐서 역 정규화
        if is_norm_in_transform_to_tensor:
            tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
        #==============================================================================#
        #==============================================================================#
        
        buffer_sr_pil[dl_str_file_name[0]] = tensor_2_list_pils_v1(in_tensor = tensor_out_sr)[0]
        
        
    print("\n Super Resolution Finished\n")
    
    del model_sr 
    
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-# SS
    model_ss.to(device)
    
    # [이미지 변수 -> 텐서 변수 변환]-------------------
    # 정규화 여부
    is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor_ss']
    
    if is_norm_in_transform_to_tensor:
        # 평균
        HP_TS_NORM_MEAN = kargs['HP_TS_NORM_MEAN_SS']
        # 표준편차
        HP_TS_NORM_STD = kargs['HP_TS_NORM_STD_SS']
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
        
        update_dict_v2("", "SS"
                      ,"", "입력 이미지(in_x) 정규화 시행됨"
                      ,"", "mean=[ " + str(HP_TS_NORM_MEAN[0]) + " " + str(HP_TS_NORM_MEAN[1]) + " "+ str(HP_TS_NORM_MEAN[2]) + " ]"
                      ,"", "std=[ " + str(HP_TS_NORM_STD[0]) + " " + str(HP_TS_NORM_STD[1]) + " "+ str(HP_TS_NORM_STD[2]) + " ]"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        # 정규화 없이 이미지를 텐서형으로 변환
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
                                                  # 일반적인 경우 (PIL mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1 또는 numpy.ndarray), 
                                                  # (H x W x C) in the range [0, 255] 입력 데이터를
                                                  # (C x H x W) in the range [0.0, 1.0] 출력 데이터로 변환함 (scaled)
                                                  transforms.ToTensor()
                                                 ])
        
        update_dict_v2("", "SS"
                      ,"", "입력 이미지(in_x) 정규화 시행 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    
    count_dataloader = 0
    for dataloader_items in dataloader_test: 
        count_dataloader += 1
        print("\rSemantic Segmentation", count_dataloader, "/", i_batch_max, end = '')
        ### Load Data
        
        dl_str_file_name    = dataloader_items[0]
        dl_pil_img_hr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
        #resized or patch ver of RAW -> Test should same with RAW
        dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[2])
        dl_ts_img_hr        = dataloader_items[2].float()
        dl_str_info_augm    = ["Not Train: no augmentation applied"]
        
        dl_pil_lab_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[3])
        dl_ts_lab_hr        = dataloader_items[4].float()
        
        dl_pil_img_lr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[5])
        dl_ts_img_lr        = dataloader_items[6].float()
        dl_str_info_deg     = dataloader_items[7]
        
        
        
        
        
        
        dl_ts_img_hr = dl_ts_img_hr.to(device)
        dl_ts_lab_hr = dl_ts_lab_hr.to(device)
        dl_ts_img_lr = dl_ts_img_lr.to(device)
        
        
        
        
        
        
        
        
        
        dl_ts_img_sr = torch.unsqueeze(transform_to_ts_img(buffer_sr_pil[dl_str_file_name[0]]), 0)
        
        print("dl_ts_img_sr", dl_ts_img_sr.shape)
        
        #==============================================================================#
        #==============================================================================#
        with torch.no_grad():
            if model_name_ss == "D3P":  # DeepLab v3 Plus
                tensor_out_seg = model_ss(dl_ts_img_sr)
            
        
        
        
            #label 예측결과 softmax 시행
            tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
            # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
            tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach(), dim = 1)
        #==============================================================================#
        #==============================================================================#
        
        
        list_out_pil_label = tensor_2_list_pils_v1(# 텐서 -> pil 이미지 리스트
                                                   # (tensor) 변환할 텐서, 모델에서 다중 결과물이 생성되는 경우, 단일 출력물 묶음만 지정해서 입력 
                                                   # (예: MPRNet -> in_tensor = tensor_sr_hypo[0])
                                                   in_tensor = tensor_out_seg_label
                                                   
                                                   # (bool) 라벨 여부 (출력 pil 이미지 = 3ch, 라벨 = 1ch 고정) (default: False)
                                                  ,is_label = True
                                                   
                                                   # (bool) pil 이미지 크기 변환 시행여부 (default: False)
                                                  ,is_resized = False
                                                  )
        
        list_out_pil_sr = [buffer_sr_pil[dl_str_file_name[0]]]
        
        #mIoU 계산 (batch단위 평균값) (mIoU 연산에 사용한 이미지 수 = in_batch_size)
        
        tmp_miou, dict_ious = calc_miou_gray(pil_gray_answer  = dl_pil_lab_hr[0]
                                            ,pil_gray_predict = list_out_pil_label[0]
                                            ,int_total_labels = HP_LABEL_TOTAL
                                            ,int_void_label = HP_LABEL_VOID
                                            )
        
        out_psnr, out_ssim = calc_psnr_ssim(pil_original = dl_pil_img_hr[0]
                                           ,pil_contrast = list_out_pil_sr[0]
                                           )
        
        try:
            # out_niqe = calc_niqe_with_pil(list_out_pil_sr[i_image])
            out_niqe = calc_niqe.with_pil(list_out_pil_sr[0])
        except:
            print("(exc) NIQE calc FAIL")
            out_niqe = -999
    
    
    






print("EoF: tester_sr_n_ss.py")