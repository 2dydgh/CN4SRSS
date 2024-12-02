"""
2nd paper proposed trainer
v 0.3
    init - SR trainer 준비 완료

v 0.4
    teacher - student KD 구성 완료

v 0.5
    teacher - student KD 가동시 teacher best는 test 시행 안함 (무의미)
    student - best 일 경우, teacher model state dict도 저장

v 0.6
    SRSS 추가

v 0.7
    매 Train 시작마다 ETA 출력하도록 수정
    sr_n_ss 추가
    
v 0.8
    one_epoch_srss, one_epoch_sr_n_ss
    (semantic segmentation) CE loss 계산에 Target을 one-hot 형태로 안쓰게 변경
    * 정보 추가 *
    (semantic segmentation) CE loss 계산에 pred를 logit 사용
v 0.9
    (semantic segmentation) CE loss 계산에 pred를 softmax 사용
    
v 1.0
    (semantic segmentation) one-hot encode 사용 안함
    -> m 1_46 부터 사용 가능

v 1.1
    one_epoch_kd_ss 추가
    main trainer_ 변경 (sr_n_ss -> kd sr -> srss) -> (sr_n_ss -> kd_sr_2_ss -> srss)
    (one_epoch_student) -> (one_epoch_kd_sr_2_ss)
    
v 1.2
    one_epoch_gt_kd_sr_2_ss 추가
    main trainer_ 변경 (sr_n_ss -> kd_sr_2_ss -> srss) -> (gt_kd_sr_2_ss -> srss)
    3 step 에서 2 step으로 변경함에 따라 1000epoch -> 1500 epoch으로 학습기간 변경

v 1.3
    one_epoch_cn4srss 추가 (기존 srss 대체용)
    (srss -> cn4srss): seMantic segmentation 결과만 loss에 반영 -> SR + seMantic segmentation 결과 모두 loss에 반영
    CALC_WITH_LOGIT (seMantic segmentation 연산에 logit or softmax 결과 사용여부, True시 logit 사용) 옵션 one_epoch_gt_kd_sr_2_ss, one_epoch_cn4srss에 추가
    

--- Ablation ---

v 1.3 기준으로 Ablation 시행
    내용 요약
    홀수 epoch (1, 3, 5, ...)는 one_epoch_gt_kd_sr_2_ss 시행
    짝수 epoch (2, 4, 6, ...)는 one_epoch_cn4srss 시행
    
    편의상 학습시킬 epoch를 절반으로 감소 후, 한 epoch 내에서 one_epoch_gt_kd_sr_2_ss 시행 후 one_epoch_cn4srss 시행하는 구조
        validation은 각 one_epoch_something train 시행 직후 수행
        test는 one_epoch_cn4srss validation best인 경우에만 시행
    
v 1.3.1
    one_epoch_gt_kd_sr_2_ss 의 영향을 확인하기 위함
    v 1.3 에서 one_epoch_gt_kd_sr_2_ss를 제거해봄
    ->  one_epoch_cn4srss 만 시행
        기존 1500 epoch + 1500 epoch 구조에서 0 epoch + 3000 epoch 시행으로 변경

v 1.3.2
    (구 1)
    one_epoch_cn4srss
    v 1.3 에서 one_epoch_cn4srss을 제거해봄
    정확한 one_epoch_gt_kd_sr_2_ss - test 시행을 위해 해당 함수의 test 모드 수정
    ->  test 모드의 경우, mixup_a가 0 처럼 동작하여 SR(studnet) 의 output 만 seMantic segmentation 모델의 입력으로 사용되도록 수정
    
    --- fix 1 (= 구 2)
    val best 여부에 관계없이 매 epoch마다 test 시행하는 문제 수정
    
    --- fix 2 (= 구 3)
    pkl_loader (v 0.2, PklLoader) 사용 가능하도록 수정
    log에 data augm 좀 더 자세하게 기록하도록 추가
    
    --- fix 3
    결과 image 저장여부 옵션 추가 (WILL_SAVE_IMAGE, default = True)
    test skip epoch 옵션 추가 (SKIP_TEST_UNTIL, default = 0)
    
    --- fix 4
    one_epoch_gt_kd_sr_2_ss 에서 valid를 test와 동일하게 변경
        mixup_a = 0, whole-image로 inference

v 1.3.3
    v 1.3.2 fix 4 가 이전 버전에 비해 크게 변경됨에 따라 버전값 올림
    이름만 다르고 내용은 동일함
    
    --- fix 1
        v 1.3.4 에서 되돌아옴
        pkl_loader_0_3 -> pkl_loader_0_4

"""
_tv = "1.3.3 fix 1"
import copy
import time
import warnings
_str = "\n\n---[ Proposed 2nd paper trainer version: " + str(_tv) + " ]---\n"
warnings.warn(_str)
# time.sleep(3)

# [memo] --------------
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# AMP(Automatic Mixed Precision) 사용됨
# 2 Fold 데이터 분할 (Train 45% Val 5% Test 50%): CamVid_12_2Fold_v4



# [기본 라이브러리]----------------------
import os
import numpy as np
import random
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import ignite   #pytorch ignite

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time
import warnings

# [py 파일]--------------------------
from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *

from _pkl_mkr.pkl_loader_0_4 import PklLoader                       # trainer v 1.3.2 fix 2 에서 추가됨

from mps.mp_sssr_plt_saver import plts_saver as plts_saver_sssr     # support: model_a, model_proposed, model_d, model_aa
from mps.mp_sr_plt_saver   import plts_saver as plts_saver_sr       # support: MPRNet, ESRT, HAN, IMDN
from mps.mp_ss_plt_saver   import plts_saver as plts_saver_ss       # support: D3P
from mps.mp_dataloader     import DataLoader_multi_worker_FIX       # now multi-worker can be used on windows 10

# https://github.com/KyungBong-Ryu/Codes_implementation/blob/main/BasicSR_NIQE.py
# from DLCs.BasicSR_NIQE import calc_niqe_with_pil
# from DLCs.BasicSR_NIQE import calc_niqe as _calc_niqe

#<<< @@@ trainer_

# Supporting model_name -> 여기서 list에 이름 추가하면 trainer_ 내부 나머진 자동으로 처리됨
# 1. SSSR
#       SSSR_A -> 3가지 출력물, 2가지 loss 입력
list_sssr_model_type_a = ["model_a"]
#       SSSR_B -> 3가지 출력물, 3가지 loss 입력
list_sssr_model_type_b = ["model_b"]
#       SSSR_C -> 4가지 출력물, 4가지 loss 입력
list_sssr_model_type_c = ["model_c"]
#       SSSR_D -> list형 출력물, list형 loss 입력
list_sssr_model_type_d = ["model_d"]

#
# 2. SR
#       SR_A -> 출력이 list 형태인 모델
list_sr_model_type_a = ["MPRNet", "model_c"]
#       SR_B -> 출력이 단일 tensor 형태인 모델
list_sr_model_type_b = ["ESRT", "HAN", "IMDN", "BSRN", "RFDN", "PAN", "LAPAR"]
#       SR_C -> label을 loss에 반영
list_sr_model_type_c = ["IMDN_LOSS"]
#
#
# 3. SS
#       D3P: (DeepLab v3 Plus)
#       SS_A -> 출력이 단일 tensor 형태인 모델 (입출력물 크기 조정기능 제공)
list_ss_model_type_a = ["D3P", "DABNet", "CGNet", "FPENet"]




# [ 범용성이 떨어져서 trainer 함수 내부에서 선언한 함수들 ]-----------------------------------------------------------------

# calc_miou_gray 함수의 dict_ious 결과물을 하나의 dict에 누적시키는 함수
# 누적할 dict, miou 값(float), iou dict
def accumulate_dict_ious(dict_accumulate, miou, dict_ious):
    # 형태 = "dict_ious와 동일한 key" : (유효 iou 수, iou 누적값)
    # dict_accumulate = kargs['dict_accumulate']

    # dict_ious = kargs['dict_ious']
    
    if "miou" in dict_accumulate:
        item_current = dict_accumulate["miou"]
    else:
        item_current = (0,0)
        
    dict_accumulate["miou"] = (item_current[0] + 1, item_current[1] + miou)
    
    for i_key in dict_ious:
        # 초기값 불러오기
        if i_key in dict_accumulate:
            # 유효한 key 인 경우
            item_current = dict_accumulate[i_key]
        else:
            # key가 존재하지 않았을 경우
            item_current = (0,0)
        
        if dict_ious[i_key] == "NaN":
            # 현재 라벨에 대한 값이 유효하지 않은 경우
            item_new = item_current
        else:
            # 현재 라벨에 대한 값이 유효한 경우
            item_new = (item_current[0] + 1, item_current[1] + float(dict_ious[i_key]))
        
        # dict 업데이트
        dict_accumulate[i_key] = item_new
    
#=== End of accumulate_dict_ious


# log 편의성을 위한 mIoU 더미 데이터 생성기
def dummy_calc_miou_gray(**kargs):
    #pil_gray_answer = kargs['pil_gray_answer']
    #pil_gray_predict = kargs['pil_gray_predict']
    int_total_labels  = kargs['int_total_labels']
    in_int_void = kargs['int_void_label']
    
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        #void 라벨이 아닌 경우
        if(i_label != in_int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, dict_iou

#=== End of dummy_calc_miou_gray

def dummy_calc_pa_ca_miou_gray(**kargs):
    #pil_gray_answer = kargs['pil_gray_answer']
    #pil_gray_predict = kargs['pil_gray_predict']
    int_total_labels  = kargs['int_total_labels']
    in_int_void = kargs['int_void_label']
    
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        #void 라벨이 아닌 경우
        if(i_label != in_int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, -9, -9, dict_iou

#=== End of dummy_calc_pa_ca_miou_gray

def dummy_calc_pa_ca_miou_gray_tensor(**kargs):
    #ts_ans   = kargs['ts_ans']
    #ts_pred  = kargs['ts_pred']
    int_total = kargs['int_total']
    int_void  = kargs['int_void']
    
    dict_iou = {}
    
    for i_label in range(int_total):
        #void 라벨이 아닌 경우
        if(i_label != int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, -9, -9, dict_iou

#=== End of dummy_calc_pa_ca_miou_gray_tensor


def one_epoch_gt_kd_sr_2_ss(**kargs):
    # 파일 키워드는 FILE_HEAD 변수로 입력
    FILE_HEAD = "gt_kd_sr_2_ss"
    
    WILL_SAVE_IMAGE = kargs['WILL_SAVE_IMAGE']          # (bool) 결과 이미지를 저장할 것인가?
    
    CALC_WITH_LOGIT           = kargs['CALC_WITH_LOGIT']  # SS loss 연산시 logit 사용 여부
    # CALC_WITH_LOGIT = False 
    
    # One epoch for Student (SR network) and KD_SS (SS network)
    # 1. HR image로 SR(T) 학습
    # 2. HR image와 SR(T)로 SR(S) 학습
    # 3. SR(T)와 SR(S) mixup 결과로 SS 학습
    # 
    list_mode           = kargs['list_mode']
    i_mode              = kargs['i_mode']
    HP_EPOCH            = kargs['HP_EPOCH']
    i_epoch             = kargs['i_epoch']
    # prev_best           = kargs['prev_best']
    
    MAX_SAVE_IMAGES     = kargs['MAX_SAVE_IMAGES']
    MUST_SAVE_IMAGE     = kargs['MUST_SAVE_IMAGE']
    BUFFER_SIZE         = kargs['BUFFER_SIZE']
    employ_threshold    = kargs['employ_threshold']
    
    HP_LABEL_TOTAL          = kargs['HP_LABEL_TOTAL']
    HP_LABEL_VOID           = kargs['HP_LABEL_VOID']
    HP_DATASET_CLASSES      = kargs['HP_DATASET_CLASSES']
    HP_LABEL_ONEHOT_ENCODE  = kargs['HP_LABEL_ONEHOT_ENCODE']
    HP_COLOR_MAP            = kargs['HP_COLOR_MAP']
    
    PATH_OUT_MODEL      = kargs['PATH_OUT_MODEL']
    PATH_OUT_IMAGE      = kargs['PATH_OUT_IMAGE']
    PATH_OUT_LOG        = kargs['PATH_OUT_LOG']
    
    device              = kargs['device']
    
    model_t             = kargs["model_t"]          # Teacher
    optimizer_t         = kargs['optimizer_t']
    criterion_t         = kargs['criterion_t']
    scheduler_t         = kargs['scheduler_t']
    
    model_s             = kargs['model_s']          # Student
    optimizer_s         = kargs['optimizer_s']
    criterion_s         = kargs['criterion_s']
    scheduler_s         = kargs['scheduler_s']
    
    model_m            = kargs['model_m']         # seMantic segmentation network
    optimizer_m        = kargs['optimizer_m']     # only SS network
    criterion_m        = kargs['criterion_m']
    scheduler_m        = kargs['scheduler_m']
    
    HP_MIXUP_A          = kargs['HP_MIXUP_A']       # SR (T, HP_MIXUP_A) and SR (S, 1 - HP_MIXUP_A) mixup value (0 ~ 1)
    # tensor_out_sr_mixup = kd_ts_img_sr * HP_MIXUP_A + tensor_out_sr_copy * (1 - HP_MIXUP_A)
    
    amp_scaler          = kargs['amp_scaler']
    ignite_evaluator    = kargs['ignite_evaluator']
    
    dataloader_input    = kargs['dataloader_input']
    i_batch_max         = len(dataloader_input)
    
    dict_rb_t          = kargs['dict_rb_t']  # record box SR (T)
    dict_rb_s          = kargs['dict_rb_s']  # record box SR (S)
    dict_rb_m          = kargs['dict_rb_m']  # record box seMentic segmentation
    
    dict_dict_log_total = kargs['dict_dict_log_total']
    
    dict_dict_log_epoch = {list_mode[0]: {}
                          ,list_mode[1]: {}
                          ,list_mode[2]: {}
                          }
    
    for i_key in list_mode:
        _str  = "batch,file_name,"
        _str += "loss_t_(" + i_key + "),loss_s_(" + i_key + "),loss_m_(" + i_key + "),"
        _str += "PSNR_t_(" + i_key + "),SSIM_t_(" + i_key + "),PSNR_s_(" + i_key + "),SSIM_s_(" + i_key + "),"
        _str += "Pixel_Acc_(" + i_key + "),Class_Acc_(" + i_key + "),mIoU_(" + i_key + "),"
        _str += HP_DATASET_CLASSES
        
        update_dict_v2("item", _str
                      ,in_dict_dict = dict_dict_log_epoch
                      ,in_dict_key = i_key
                      ,in_print_head = "dict_dict_log_epoch_" + i_key
                      ,is_print = False
                      )
    
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    try:
        # Timer: Model 입출력 + loss & optimizer update 소요시간
        timer_gpu_start  = torch.cuda.Event(enable_timing=True)
        timer_gpu_finish = torch.cuda.Event(enable_timing=True)
    except:
        timer_gpu_start  = None
        timer_gpu_finish = None
    
    
    if i_mode == "train":
        # sr (t)
        model_t.train()
        optimizer_t.zero_grad()
        print("optimizer_t.zero_grad()")
        # sr (s)
        model_s.train()
        optimizer_s.zero_grad()
        print("optimizer_s.zero_grad()")
        # ss
        model_m.train()
        optimizer_m.zero_grad()
        print("optimizer_m.zero_grad()")
    else:
        model_t.eval()
        model_s.eval()
        model_m.eval()
    
    i_batch = 0
    # batch 준비 포함 1 batch 소요시간
    timer_cpu_start = time.time()
    
    try:
        del list_mp_buffer
        list_mp_buffer = []
    except:
        list_mp_buffer = []
        
    
    for dataloader_items in dataloader_input:
        #<<< dataloader_items 분배
        dl_str_file_name    = dataloader_items[0]       # (tuple) file name
        
        if i_mode == "test":
            _bool = bool(set(MUST_SAVE_IMAGE) & set(dl_str_file_name))
        else:
            _bool = False
        
        if not WILL_SAVE_IMAGE:
            is_pil_needed = False   # 결과 이미지를 저장하지 않음 -> pil 이미지 필요 없음
        else:
            is_pil_needed = (i_batch % (i_batch_max//MAX_SAVE_IMAGES + 1) == 0  or _bool )  # pil 이미지가 사용될 경우 (label 제외)
        
        # augm info for train
        dl_str_info_augm    = dataloader_items[3]
        
        # degraded info for LR images
        try:
            dl_str_info_deg = dataloader_items[9]
            if dl_str_info_deg[0] == False:
                dl_str_info_deg = None
        except:
            dl_str_info_deg = None
        
        # PIL (HR image & HR label & LR image) -> Full or Patch ver of RAW PIL (no Norm)
        if is_pil_needed:
            dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
            try:
                dl_pil_img_lr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[7])
            except:
                dl_pil_img_lr   = None
        else:
            dl_pil_img_hr = None
            dl_pil_img_lr = None
        
        try:
            #gray 이미지 형태 [B, 1, H, W] -> mIoU 측정용으로만 사용 예정, (0~1) -> (0~255)를 위해 x255 후 반올림 시행
            dl_ts_lab_hr_gray = torch.round(dataloader_items[4]*255).type(torch.uint8)
        except:
            dl_ts_lab_hr_gray = None
        
        if is_pil_needed:
            try:
                # CamVid 기준 단순히 (Gray 이미지의) PIL -> Image Tensor -> PIL 작업이라 옵션지정 필요없음
                dl_pil_lab_hr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
            except:
                dl_pil_lab_hr     = None
        
        # Tensor (HR image & HR label & LR image)
        dl_ts_img_hr        = dataloader_items[2].float()
        try:
            dl_ts_lab_hr    = dataloader_items[5].float()   # class 만큼 채널 늘어난 형태 -> Dilation 적용될 수 있음
            dl_ts_lab_hr_void = dataloader_items[6].float()
        except:
            dl_ts_lab_hr    = None
        try:
            dl_ts_img_lr    = dataloader_items[8].float()
        except:
            dl_ts_img_lr    = None
        
        #>>> dataloader_items 분배
        
        #<<< model 입출력 시행
        if timer_gpu_start is not None:
            timer_gpu_start.record()
        
        dl_ts_img_hr = dl_ts_img_hr.to(device)
        dl_ts_img_lr = dl_ts_img_lr.to(device)
        
        # if dl_ts_lab_hr is not None:
            # dl_ts_lab_hr = dl_ts_lab_hr.to(device)
        
        if dl_ts_lab_hr_gray is not None:
            _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
            _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
            _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
            _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
            dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.to(device)
        
        dl_ts_img_lr_copy = dl_ts_img_lr.clone().detach()   # dl_ts_img_lr for model_t
        dl_ts_img_lr_copy = dl_ts_img_lr_copy.requires_grad_(True)
        dl_ts_img_lr      = dl_ts_img_lr.requires_grad_(True)
        
        if i_mode == "train":
            # SR (T)
            with torch.no_grad():
                tensor_out_t_raw_copy = model_t(dl_ts_img_lr_copy)
            
            with torch.cuda.amp.autocast(enabled=True):
                tensor_out_t_raw = model_t(dl_ts_img_lr_copy)
                loss_t = criterion_t(tensor_out_t_raw, dl_ts_img_hr)
                if isinstance(tensor_out_t_raw, list):
                    tensor_out_t = tensor_out_t_raw[0]
                else:
                    tensor_out_t = tensor_out_t_raw
                tensor_out_t_copy = tensor_out_t.clone().detach()
            
            amp_scaler.scale(loss_t).backward(retain_graph=False)
            amp_scaler.step(optimizer_t)
            amp_scaler.update()
            optimizer_t.zero_grad()
            
            # SR (S)
            with torch.cuda.amp.autocast(enabled=True):
                tensor_out_s_raw = model_s(dl_ts_img_lr)
                loss_s = criterion_s(tensor_out_s_raw, dl_ts_img_hr, tensor_out_t_raw_copy)
                if isinstance(tensor_out_s_raw, list):
                    tensor_out_s = tensor_out_s_raw[0]
                else:
                    tensor_out_s = tensor_out_s_raw
                tensor_out_s_copy = tensor_out_s.clone().detach()
                
            amp_scaler.scale(loss_s).backward(retain_graph=False)
            amp_scaler.step(optimizer_s)
            amp_scaler.update()
            optimizer_s.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=True):
                # Mixup SR (T) and SR (S)
                tensor_out_sr_mixup = tensor_out_t_copy * HP_MIXUP_A + tensor_out_s_copy * (1 - HP_MIXUP_A)
                tensor_out_sr_mixup = tensor_out_sr_mixup.requires_grad_(True)
                tensor_out_seg = model_m(tensor_out_sr_mixup)
                
                # label 예측결과 softmax 시행
                tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                if CALC_WITH_LOGIT:
                    loss_m = criterion_m(tensor_out_seg, _dl_ts_lab_hr_gray)
                else:
                    loss_m = criterion_m(tensor_out_seg_softmax, _dl_ts_lab_hr_gray)
                
                # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                
            amp_scaler.scale(loss_m).backward(retain_graph=False)
            amp_scaler.step(optimizer_m)
            amp_scaler.update()
            optimizer_m.zero_grad()
            
        else: # val, test
            with torch.no_grad():
                # SR (T)
                tensor_out_t_raw = model_t(dl_ts_img_lr)
                loss_t = criterion_t(tensor_out_t_raw, dl_ts_img_hr)
                if isinstance(tensor_out_t_raw, list):
                    tensor_out_t = tensor_out_t_raw[0]
                else:
                    tensor_out_t = tensor_out_t_raw
                
                # SR (S)
                tensor_out_s_raw = model_s(dl_ts_img_lr)
                loss_s = criterion_s(tensor_out_s_raw, dl_ts_img_hr, tensor_out_t_raw)
                if isinstance(tensor_out_s_raw, list):
                    tensor_out_s = tensor_out_s_raw[0]
                else:
                    tensor_out_s = tensor_out_s_raw
                
                # SR (mixup) -> SS
                #<< ~ t 1.3.2 fix 3
                # if i_mode == "test":
                    # # test
                    # if i_batch < 1:
                        # print(" for test, tensor_out_sr_mixup = tensor_out_s (no mixup to SR-S with SR-T)")
                    # tensor_out_sr_mixup = tensor_out_s
                # else:
                    # # val
                    # tensor_out_sr_mixup = tensor_out_t * HP_MIXUP_A + tensor_out_s * (1 - HP_MIXUP_A)
                #>> ~ t 1.3.2 fix 3
                
                #<< t 1.3.2 fix 4
                if i_batch < 1:
                    print(" for val and test, tensor_out_sr_mixup = tensor_out_s (no mixup to SR-S with SR-T)")
                tensor_out_sr_mixup = tensor_out_s
                tensor_out_seg = model_m(tensor_out_sr_mixup)
                #>> t 1.3.2 fix 4
                
                # label 예측결과 softmax 시행
                tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                if CALC_WITH_LOGIT:
                    loss_m = criterion_m(tensor_out_seg, _dl_ts_lab_hr_gray)
                else:
                    loss_m = criterion_m(tensor_out_seg_softmax, _dl_ts_lab_hr_gray)
                
                # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                
        
        dict_rb_t[i_mode]["loss"].add_item(loss_t.item())
        dict_rb_s[i_mode]["loss"].add_item(loss_s.item())
        dict_rb_m[i_mode]["loss"].add_item(loss_m.item())
        
        if timer_gpu_finish is not None:
            timer_gpu_finish.record()
        #>>> model 입출력 시행
        
        #<<< 후처리
        current_batch_size, _, _, _ = dl_ts_img_hr.shape
        if is_pil_needed:
            list_out_pil_t     = tensor_2_list_pils_v1(in_tensor=tensor_out_t,         is_label=False, is_resized=False)
            list_out_pil_s     = tensor_2_list_pils_v1(in_tensor=tensor_out_s,         is_label=False, is_resized=False)
            list_out_pil_label = tensor_2_list_pils_v1(in_tensor=tensor_out_seg_label, is_label=True,  is_resized=False)
        
        for i_image in range(current_batch_size):
            with torch.no_grad():
                ignite_in_t  = torch.unsqueeze(tensor_out_t[i_image].to(torch.float32), 0)
                ignite_in_s  = torch.unsqueeze(tensor_out_s[i_image].to(torch.float32), 0)
                ignite_in_hr = torch.unsqueeze(dl_ts_img_hr[i_image], 0)
                
                ignite_in_t  = ignite_in_t.to(device)
                ignite_in_s  = ignite_in_s.to(device)
                ignite_in_hr = ignite_in_hr.to(device)
                
                ignite_result = ignite_evaluator.run([[ignite_in_t
                                                      ,ignite_in_hr
                                                    ]])
                
                out_psnr_t = ignite_result.metrics['psnr']
                out_ssim_t = ignite_result.metrics['ssim']
                
                dict_rb_t[i_mode]["psnr"].add_item(out_psnr_t)
                dict_rb_t[i_mode]["ssim"].add_item(out_ssim_t)
                
                ignite_result = ignite_evaluator.run([[ignite_in_s
                                                      ,ignite_in_hr
                                                    ]])
                
                out_psnr_s = ignite_result.metrics['psnr']
                out_ssim_s = ignite_result.metrics['ssim']
                
                dict_rb_s[i_mode]["psnr"].add_item(out_psnr_s)
                dict_rb_s[i_mode]["ssim"].add_item(out_ssim_s)
                
                dict_rb_m[i_mode]["psnr"].add_item(-9.0) # dummy
                dict_rb_m[i_mode]["ssim"].add_item(-9.0) # dummy
                
            
            tmp_pa, tmp_ca, tmp_miou, dict_ious = dummy_calc_pa_ca_miou_gray_tensor(ts_ans    = None
                                                                                   ,ts_pred   = None
                                                                                   ,int_total = HP_LABEL_TOTAL
                                                                                   ,int_void  = HP_LABEL_VOID
                                                                                   )
            
            dict_rb_t[i_mode]["pa"].add_item(tmp_pa)
            dict_rb_t[i_mode]["ca"].add_item(tmp_ca)
            dict_rb_t[i_mode]["ious"].add_item(dict_ious)
            
            dict_rb_s[i_mode]["pa"].add_item(tmp_pa)
            dict_rb_s[i_mode]["ca"].add_item(tmp_ca)
            dict_rb_s[i_mode]["ious"].add_item(dict_ious)
            
            # overwrite result from dummy_calc_pa_ca_miou_gray_tensor
            tmp_pa, tmp_ca, tmp_miou, dict_ious = calc_pa_ca_miou_gray_tensor(ts_ans    = dl_ts_lab_hr_gray[i_image][0]
                                                                             ,ts_pred   = tensor_out_seg_label[i_image]
                                                                             ,int_total = HP_LABEL_TOTAL
                                                                             ,int_void  = HP_LABEL_VOID
                                                                             ,device    = device
                                                                             )
            
            dict_rb_m[i_mode]["pa"].add_item(tmp_pa)
            dict_rb_m[i_mode]["ca"].add_item(tmp_ca)
            dict_rb_m[i_mode]["ious"].add_item(dict_ious)
            
            tmp_ious = ""
            for i_key in dict_ious:
                tmp_ious += "," + dict_ious[i_key]
            
            tmp_str_contents = (str(i_batch + 1) + "," + dl_str_file_name[i_image] 
                               +"," + str(loss_t.item()) + "," + str(loss_s.item()) + "," + str(loss_m.item())
                               +"," + str(out_psnr_t) + "," + str(out_ssim_t)
                               +"," + str(out_psnr_s) + "," + str(out_ssim_s)
                               +"," + str(tmp_pa) + "," + str(tmp_ca) +"," + str(tmp_miou) + tmp_ious
                               )
            
            update_dict_v2("", tmp_str_contents
                          ,in_dict_dict = dict_dict_log_epoch
                          ,in_dict_key = i_mode
                          ,is_print = False
                          )
            
            if is_pil_needed and i_image < 2:
                if len(list_mp_buffer) >= BUFFER_SIZE:
                    plts_saver_sssr(list_mp_buffer, is_best=(i_mode=="test"), no_employ=(employ_threshold >= len(list_mp_buffer)))
                    try:
                        del list_mp_buffer
                        list_mp_buffer = []
                    except:
                        list_mp_buffer = []
                
                # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                plt_title = "File name: " + dl_str_file_name[i_image]
                plt_title += "\n" + dl_str_info_augm[i_image]
                try:
                    # 현재 patch의 degrad- 옵션 불러오기
                    plt_title += "\n" + dl_str_info_deg[i_image]
                except:
                    pass
                plt_title += "\nPSNR_t: " + str(round(out_psnr_t, 4)) + "  SSIM_t: " + str(round(out_ssim_t, 4))
                plt_title += "\nPSNR_s: " + str(round(out_psnr_s, 4)) + "  SSIM_s: " + str(round(out_ssim_s, 4))
                plt_title += "\nPA: "   + str(round(tmp_pa, 4))   + "  CA: " + str(round(tmp_ca, 4))
                plt_title += "\nmIoU: " + str(round(tmp_miou, 4))
                
                tmp_file_name = (i_mode + "_" + str(i_epoch + 1) + "_" + str(i_batch + 1) + "_"
                                +dl_str_file_name[i_image]
                                )
                
                list_mp_buffer.append((# 0 (model name)
                                       "SSSR_D"
                                       # 1 ~ 6 (pils)
                                      ,dl_pil_img_hr[i_image], label_2_RGB(dl_pil_lab_hr[i_image], HP_COLOR_MAP)
                                      ,label_2_RGB(list_out_pil_label[i_image], HP_COLOR_MAP), dl_pil_img_lr[i_image]
                                      ,list_out_pil_t[i_image], list_out_pil_s[i_image]
                                       # 7 ~ 12 (sub title)
                                      ,"HR Image", "Label Answer", "Predicted", "LR Image", "SR (t) Image", "SR (s) Image"
                                       # 13 (path plt)
                                      ,PATH_OUT_IMAGE + i_mode + "/" + FILE_HEAD  +"_" + str(i_epoch + 1)
                                       # 14 ~ 17 (data CF)
                                      ,None, None, None, None
                                       # 18 (path CF)
                                      ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                       # 19 (path pil)
                                      ,PATH_OUT_IMAGE + i_mode + "/_Images/" + FILE_HEAD + "_" + str(i_epoch + 1)
                                       # 20 (plt title)
                                      ,plt_title
                                       # 21 (file name)
                                      ,tmp_file_name
                                      )
                                     )
            
        for i_key in ["loss", "psnr", "ssim", "pa", "ca", "ious"]:
            dict_rb_t[i_mode][i_key].update_batch()
            dict_rb_s[i_mode][i_key].update_batch()
            dict_rb_m[i_mode][i_key].update_batch()
        
        try:
            # GPU timer record
            torch.cuda.synchronize()
            timer_gpu_record = str(round(timer_gpu_start.elapsed_time(timer_gpu_finish) / 1000, 4))
        except:
            timer_gpu_record = "FAIL"
        
        timer_cpu_record = str(round(time.time() - timer_cpu_start, 4))
        
        print("\rin", i_mode, (i_epoch + 1), "/", HP_EPOCH, "-", (i_batch + 1), "/", i_batch_max, "-"
             ,"CPU:", timer_cpu_record, " GPU:", timer_gpu_record
             ,end = '')
        
        timer_cpu_start = time.time()
        #>>> 후처리
        
        i_batch += 1
    
    print("")
    
    if list_mp_buffer:
        plts_saver_sssr(list_mp_buffer, is_best=(i_mode=="test"), no_employ=(employ_threshold >= len(list_mp_buffer)))
    
    
    tmp_str_contents  =       str(dict_rb_t[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))
    tmp_str_contents += "," + str(dict_rb_s[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))
    tmp_str_contents += "," + str(dict_rb_m[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))
    
    for i_key in ["psnr", "ssim"]:
        tmp_str_contents += "," + str(dict_rb_t[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG,is_print_sub = True ,is_update_graph = True))
    
    for i_key in ["psnr", "ssim"]:
        tmp_str_contents += "," + str(dict_rb_s[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG,is_print_sub = True ,is_update_graph = True))
    
    for i_key in ["pa", "ca", "ious"]:
        dict_rb_t[i_mode][i_key].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = False ,is_update_graph = False)
        dict_rb_s[i_mode][i_key].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = False ,is_update_graph = False)
    
    for i_key in ["psnr", "ssim"]:
        dict_rb_m[i_mode][i_key].update_epoch(is_return = False, path = PATH_OUT_LOG ,is_print_sub = False ,is_update_graph = False)
    
    for i_key in ["pa", "ca", "ious"]:
        tmp_str_contents += "," + str(dict_rb_m[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG,is_print_sub = True ,is_update_graph = True))
    
    update_dict_v2(str(i_epoch + 1), tmp_str_contents
                  ,in_dict_dict = dict_dict_log_total
                  ,in_dict_key = i_mode
                  ,in_print_head = "dict_dict_log_total_" + i_mode
                  ,is_print = False
                  )
    
    if i_mode == "train":
        # SR (T)
        dict_rb_t[i_mode]["lr"].add_item(optimizer_t.param_groups[0]['lr'])
        dict_rb_t[i_mode]["lr"].update_batch()
        # 스케쥴러 갱신
        scheduler_t.step()
        print("scheduler_t.step()")
        dict_rb_t[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)
        
        # SR (S)
        dict_rb_s[i_mode]["lr"].add_item(optimizer_s.param_groups[0]['lr'])
        dict_rb_s[i_mode]["lr"].update_batch()
        # 스케쥴러 갱신
        scheduler_s.step()
        print("scheduler_s.step()")
        dict_rb_s[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)
        
        # seMantic segmentation
        dict_rb_m[i_mode]["lr"].add_item(optimizer_m.param_groups[0]['lr'])
        dict_rb_m[i_mode]["lr"].update_batch()
        # 스케쥴러 갱신
        scheduler_m.step()
        print("scheduler_m.step()")
        dict_rb_m[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)
    
    # log 기록 업데이트 (epoch 단위)
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG + i_mode + "/"
                 ,in_file_name = FILE_HEAD + "_log_epoch_" + i_mode + "_" + str(i_epoch + 1) + ".csv"
                 ,in_dict_dict = dict_dict_log_epoch
                 ,in_dict_key = i_mode
                 )
    
    # log 기록 업데이트 (학습 전체 단위)
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = FILE_HEAD + "_log_total_" + i_mode + ".csv"
                 ,in_dict_dict = dict_dict_log_total
                 ,in_dict_key = i_mode
                 )
    
    #<<< check_point, state_dict 저장
    save_interval = 100
    
    if i_mode == "train" and (i_epoch + 1) % save_interval == 0:
        # check_point 저장경로
        tmp_path = PATH_OUT_MODEL + "check_points/"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        # 모델 체크포인트 저장
        print("\n[--- 체크포인트", str(i_epoch + 1), "저장됨 ---]\n")
        torch.save({'epoch': (i_epoch + 1)                              # (int) 중단 시점 epoch 값
                   ,'model_t_state_dict'    : model_t.state_dict()      # (state_dict) model_t.state_dict()
                   ,'model_s_state_dict'    : model_s.state_dict()      # (state_dict) model_s.state_dict()
                   ,'model_m_state_dict'    : model_m.state_dict()      # (state_dict) model_m.state_dict()
                   ,'optimizer_t_state_dict': optimizer_t.state_dict()  # (state_dict) optimizer_t.state_dict()
                   ,'optimizer_s_state_dict': optimizer_s.state_dict()  # (state_dict) optimizer_s.state_dict()
                   ,'optimizer_m_state_dict': optimizer_m.state_dict()  # (state_dict) optimizer_m.state_dict()
                   ,'scheduler_t_state_dict': scheduler_t.state_dict()  # (state_dict) scheduler_t.state_dict()
                   ,'scheduler_s_state_dict': scheduler_s.state_dict()  # (state_dict) scheduler_s.state_dict()
                   ,'scheduler_m_state_dict': scheduler_m.state_dict()  # (state_dict) scheduler_m.state_dict()
                   }
                  ,tmp_path + str(i_epoch + 1) + "_" + FILE_HEAD + "_check_point.tar"
                  )
    
    if i_mode == "val":
        tmp_is_best = dict_rb_t[i_mode]["psnr"].is_best_max or dict_rb_s[i_mode]["psnr"].is_best_max or dict_rb_m[i_mode]["ious"].is_best_max # PSNR 기준 or mIoU 기준
        
        if tmp_is_best:
            print("\n< Best Valid Epoch > Model State Dict 저장됨")
            tmp_path = PATH_OUT_MODEL + "state_dicts/"  # state_dict 저장경로
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            # Teacher, Student, seMantic segmentation model state dict
            torch.save(model_t.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_t_msd.pt')
            torch.save(model_s.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_s_msd.pt')
            torch.save(model_m.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_m_msd.pt')
            
        else:
            print("\n< Not a Best Valid Epoch >")
        
        # if prev_best is not None:
            # # prev best 값이 입력된 경우 -> Test 생략여부 판별
            # if prev_best > dict_rb[i_mode]["psnr"].total_max[-1]:
                # print("\nprev_best 못넘음...")
                # tmp_is_best = False
            # else:
                # tmp_is_best = True
        
    else:
        tmp_is_best = None
    #>>>check_point, state_dict 저장
    
    return tmp_is_best


#=== End of one_epoch_gt_kd_sr_2_ss

"""

def one_epoch_cn4srss(**kargs):
    # srss 개선판
    # 기존: seMantic segmentation 결과만 loss에 반영
    # 변경: SR + seMantic segmentation 결과 모두 loss에 반영
    FILE_HEAD = "CN4SRSS"
    
    CALC_WITH_LOGIT           = kargs['CALC_WITH_LOGIT']  # SS loss 연산시 logit 사용 여부
    # CALC_WITH_LOGIT = False 
    
    # TRAINER_MODE        = kargs['TRAINER_MODE']
    list_mode           = kargs['list_mode']
    i_mode              = kargs['i_mode']
    HP_EPOCH            = kargs['HP_EPOCH']
    i_epoch             = kargs['i_epoch']
    prev_best           = kargs['prev_best']
    # model_type          = kargs['model_type']
    MAX_SAVE_IMAGES     = kargs['MAX_SAVE_IMAGES']
    MUST_SAVE_IMAGE     = kargs['MUST_SAVE_IMAGE']
    BUFFER_SIZE         = kargs['BUFFER_SIZE']
    employ_threshold    = kargs['employ_threshold']
    
    HP_LABEL_TOTAL      = kargs['HP_LABEL_TOTAL']
    HP_LABEL_VOID       = kargs['HP_LABEL_VOID']
    HP_DATASET_CLASSES  = kargs['HP_DATASET_CLASSES']
    HP_LABEL_ONEHOT_ENCODE = kargs['HP_LABEL_ONEHOT_ENCODE']
    HP_COLOR_MAP        = kargs['HP_COLOR_MAP']
    
    PATH_OUT_MODEL      = kargs['PATH_OUT_MODEL']
    PATH_OUT_IMAGE      = kargs['PATH_OUT_IMAGE']
    PATH_OUT_LOG        = kargs['PATH_OUT_LOG']
    
    device              = kargs['device']
    
    model_t             = kargs['model_t']          # Teacher
    model_s             = kargs['model_s']          # Student
    model_m             = kargs['model_m']          # image to label generator (sem. seg.)
    
    optimizer_l           = kargs['optimizer_l']    # model_s + model_m
    criterion_l           = kargs['criterion_l']
    scheduler_l           = kargs['scheduler_l']
    
    amp_scaler          = kargs['amp_scaler']
    ignite_evaluator    = kargs['ignite_evaluator']
    
    dataloader_input    = kargs['dataloader_input']
    i_batch_max         = len(dataloader_input)
    
    dict_rb             = kargs['dict_rb']  # record box
    
    dict_dict_log_total = kargs['dict_dict_log_total']
    
    dict_dict_log_epoch = {list_mode[0]: {}
                          ,list_mode[1]: {}
                          ,list_mode[2]: {}
                          }
    
    for i_key in list_mode:
        _str  = "batch,file_name,"
        _str += "loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
        _str += "Pixel_Acc_(" + i_key + "),Class_Acc_(" + i_key + "),mIoU_(" + i_key + "),"
        _str += HP_DATASET_CLASSES
        
        update_dict_v2("item", _str
                      ,in_dict_dict = dict_dict_log_epoch
                      ,in_dict_key = i_key
                      ,in_print_head = "dict_dict_log_epoch_" + i_key
                      ,is_print = False
                      )
    
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    try:
        # Timer: Model 입출력 + loss & optimizer update 소요시간
        timer_gpu_start  = torch.cuda.Event(enable_timing=True)
        timer_gpu_finish = torch.cuda.Event(enable_timing=True)
    except:
        timer_gpu_start  = None
        timer_gpu_finish = None
    
    model_t.eval()
    
    if i_mode == "train":
        model_s.train()
        model_m.train()
        optimizer_l.zero_grad()
        print("optimizer_l.zero_grad()")
    else:
        model_s.eval()
        model_m.eval()
    
    i_batch = 0
    # batch 준비 포함 1 batch 소요시간
    timer_cpu_start = time.time()
    
    try:
        del list_mp_buffer
        list_mp_buffer = []
    except:
        list_mp_buffer = []
        
    
    for dataloader_items in dataloader_input:
        #<<< dataloader_items 분배
        dl_str_file_name    = dataloader_items[0]       # (tuple) file name
        
        if i_mode == "test":
            _bool = bool(set(MUST_SAVE_IMAGE) & set(dl_str_file_name))
        else:
            _bool = False
        
        is_pil_needed = (i_batch % (i_batch_max//MAX_SAVE_IMAGES + 1) == 0  or _bool )  # pil 이미지가 사용될 경우 (label 제외)
        
        # augm info for train
        dl_str_info_augm    = dataloader_items[3]
        
        # degraded info for LR images
        try:
            dl_str_info_deg = dataloader_items[9]
            if dl_str_info_deg[0] == False:
                dl_str_info_deg = None
        except:
            dl_str_info_deg = None
        
        # PIL (HR image & HR label & LR image) -> Full or Patch ver of RAW PIL (no Norm)
        if is_pil_needed:
            dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
            try:
                dl_pil_img_lr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[7])
            except:
                dl_pil_img_lr   = None
        else:
            dl_pil_img_hr = None
            dl_pil_img_lr = None
        
        try:
            #gray 이미지 형태 [B, 1, H, W] -> mIoU 측정용으로만 사용 예정, (0~1) -> (0~255)를 위해 x255 후 반올림 시행
            dl_ts_lab_hr_gray = torch.round(dataloader_items[4]*255).type(torch.uint8)
        except:
            dl_ts_lab_hr_gray = None
        
        if is_pil_needed:
            try:
                # CamVid 기준 단순히 (Gray 이미지의) PIL -> Image Tensor -> PIL 작업이라 옵션지정 필요없음
                dl_pil_lab_hr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
            except:
                dl_pil_lab_hr     = None
        
        # Tensor (HR image & HR label & LR image)
        dl_ts_img_hr        = dataloader_items[2].float()
        try:
            dl_ts_lab_hr    = dataloader_items[5].float()   # class 만큼 채널 늘어난 형태 -> Dilation 적용될 수 있음
            dl_ts_lab_hr_void = dataloader_items[6].float()
        except:
            dl_ts_lab_hr    = None
        try:
            dl_ts_img_lr    = dataloader_items[8].float()
        except:
            dl_ts_img_lr    = None
        
        #>>> dataloader_items 분배
        
        #<<< model 입출력 시행
        if timer_gpu_start is not None:
            timer_gpu_start.record()
        
        dl_ts_img_hr = dl_ts_img_hr.to(device)
        # if dl_ts_lab_hr is not None:
            # dl_ts_lab_hr = dl_ts_lab_hr.to(device)
        dl_ts_img_lr = dl_ts_img_lr.to(device)
        
        if dl_ts_lab_hr_gray is not None:
            _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
            _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
            _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
            _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
            dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.to(device)
        
        dl_ts_img_lr = dl_ts_img_lr.requires_grad_(True)
        
        if i_mode == "train":
            with torch.cuda.amp.autocast(enabled=True):
                # tensor_out_sr_raw = model_s(dl_ts_img_lr.type(torch.float32))
                tensor_out_sr_raw = model_s(dl_ts_img_lr)
                if isinstance(tensor_out_sr_raw, list):
                    tensor_out_sr = tensor_out_sr_raw[0] # model 설계시, 첫 output을 SR 이미지로 설정할 것
                else:
                    tensor_out_sr = tensor_out_sr_raw
                
                tensor_out_seg = model_m(tensor_out_sr)
                
                # label 예측결과 softmax 시행
                tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                
                if CALC_WITH_LOGIT:
                    loss_l = criterion_l(tensor_out_sr, dl_ts_img_hr, tensor_out_seg, _dl_ts_lab_hr_gray)
                else:
                    loss_l = criterion_l(tensor_out_sr, dl_ts_img_hr, tensor_out_seg_softmax, _dl_ts_lab_hr_gray)
                
                # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                
            amp_scaler.scale(loss_l).backward(retain_graph=False)
            amp_scaler.step(optimizer_l)
            amp_scaler.update()
            optimizer_l.zero_grad()
        else:
            with torch.no_grad():
                tensor_out_sr_raw = model_s(dl_ts_img_lr)
                if isinstance(tensor_out_sr_raw, list):
                    # model 설계시, 첫 output을 SR 이미지로 설정할 것
                    tensor_out_sr = tensor_out_sr_raw[0]
                else:
                    tensor_out_sr = tensor_out_sr_raw
                
                tensor_out_seg = model_m(tensor_out_sr)
                
                # label 예측결과 softmax 시행
                tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                if CALC_WITH_LOGIT:
                    loss_l = criterion_l(tensor_out_sr, dl_ts_img_hr, tensor_out_seg, _dl_ts_lab_hr_gray)
                else:
                    loss_l = criterion_l(tensor_out_sr, dl_ts_img_hr, tensor_out_seg_softmax, _dl_ts_lab_hr_gray)
                
                # softmax 값을 바탕으로 label image 형태로 tensor 생성 (형태 변경 4D [B, C, H, W] -> 3D [B, H, W]) -> 이미지 생성에 사용됨
                tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)
                
        dict_rb[i_mode]["loss"].add_item(loss_l.item())
        
        if timer_gpu_finish is not None:
            timer_gpu_finish.record()
        #>>> model 입출력 시행
        
        #<<< 후처리
        current_batch_size, _, _, _ = dl_ts_img_hr.shape
        
        if is_pil_needed:
            list_out_pil_label = tensor_2_list_pils_v1(in_tensor=tensor_out_seg_label, is_label=True, is_resized=False)
            list_out_pil_sr    = tensor_2_list_pils_v1(in_tensor=tensor_out_sr ,is_label=False ,is_resized=False)
        
        for i_image in range(current_batch_size):
            with torch.no_grad():
                ignite_in_sr = torch.unsqueeze(tensor_out_sr[i_image].to(torch.float32), 0)
                ignite_in_hr = torch.unsqueeze(dl_ts_img_hr[i_image], 0)
                
                ignite_in_sr = ignite_in_sr.to(device)
                ignite_in_hr = ignite_in_hr.to(device)
                
                ignite_result = ignite_evaluator.run([[ignite_in_sr
                                                      ,ignite_in_hr
                                                    ]])
                
                out_psnr = ignite_result.metrics['psnr']
                out_ssim = ignite_result.metrics['ssim']
                
                dict_rb[i_mode]["psnr"].add_item(out_psnr)
                dict_rb[i_mode]["ssim"].add_item(out_ssim)
            
            
            tmp_pa, tmp_ca, tmp_miou, dict_ious = calc_pa_ca_miou_gray_tensor(ts_ans    = dl_ts_lab_hr_gray[i_image][0]
                                                                             ,ts_pred   = tensor_out_seg_label[i_image]
                                                                             ,int_total = HP_LABEL_TOTAL
                                                                             ,int_void  = HP_LABEL_VOID
                                                                             ,device    = device
                                                                             )
            
            
            dict_rb[i_mode]["pa"].add_item(tmp_pa)
            dict_rb[i_mode]["ca"].add_item(tmp_ca)
            dict_rb[i_mode]["ious"].add_item(dict_ious)
            
            tmp_ious = ""
            for i_key in dict_ious:
                tmp_ious += "," + dict_ious[i_key]
            
            tmp_str_contents = (str(i_batch + 1) + "," + dl_str_file_name[i_image] + "," + str(loss_l.item()) 
                               +"," + str(out_psnr) + "," + str(out_ssim)
                               +"," + str(tmp_pa) + "," + str(tmp_ca) +"," + str(tmp_miou) + tmp_ious
                               )
            
            update_dict_v2("", tmp_str_contents
                          ,in_dict_dict = dict_dict_log_epoch
                          ,in_dict_key = i_mode
                          ,is_print = False
                          )
            
            if is_pil_needed and i_image < 2:
                if len(list_mp_buffer) >= BUFFER_SIZE:
                    plts_saver_sssr(list_mp_buffer, is_best=(i_mode=="test"), no_employ=(employ_threshold >= len(list_mp_buffer)))
                    try:
                        del list_mp_buffer
                        list_mp_buffer = []
                    except:
                        list_mp_buffer = []
                    
                
                # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                plt_title = "File name: " + dl_str_file_name[i_image]
                plt_title += "\n" + dl_str_info_augm[i_image]
                try:
                    # 현재 patch의 degrad- 옵션 불러오기
                    plt_title += "\n" + dl_str_info_deg[i_image]
                except:
                    pass
                plt_title += "\nPSNR: " + str(round(out_psnr, 4)) + "  SSIM: " + str(round(out_ssim, 4))
                plt_title += "\nPA: "   + str(round(tmp_pa, 4))   + "  CA: " + str(round(tmp_ca, 4))
                plt_title += "\nmIoU: " + str(round(tmp_miou, 4))
                
                tmp_file_name = (i_mode + "_" + str(i_epoch + 1) + "_" + str(i_batch + 1) + "_"
                                +dl_str_file_name[i_image]
                                )
                
                
                list_mp_buffer.append((# 0 (model name)
                                       "SSSR_D"
                                       # 1 ~ 6 (pils)
                                      ,dl_pil_img_hr[i_image], label_2_RGB(dl_pil_lab_hr[i_image], HP_COLOR_MAP)
                                      ,label_2_RGB(list_out_pil_label[i_image], HP_COLOR_MAP), dl_pil_img_hr[i_image]
                                      ,dl_pil_img_lr[i_image], list_out_pil_sr[i_image]
                                       # 7 ~ 12 (sub title)
                                      ,"HR Image", "Label Answer", "Predicted", "HR Image", "LR Image", "SR Image"
                                       # 13 (path plt)
                                      ,PATH_OUT_IMAGE + i_mode + "/" + FILE_HEAD + "_" + str(i_epoch + 1)
                                       # 14 ~ 17 (data CF)
                                      ,None, None, None, None
                                       # 18 (path CF)
                                      ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                       # 19 (path pil)
                                      ,PATH_OUT_IMAGE + i_mode + "/_Images/" + FILE_HEAD + "_" + str(i_epoch + 1)
                                       # 20 (plt title)
                                      ,plt_title
                                       # 21 (file name)
                                      ,tmp_file_name
                                      )
                                     )
                
            
        for i_key in ["loss", "psnr", "ssim", "pa", "ca", "ious"]:
            dict_rb[i_mode][i_key].update_batch()
        
        try:
            # GPU timer record
            torch.cuda.synchronize()
            timer_gpu_record = str(round(timer_gpu_start.elapsed_time(timer_gpu_finish) / 1000, 4))
        except:
            timer_gpu_record = "FAIL"
        
        timer_cpu_record = str(round(time.time() - timer_cpu_start, 4))
        
        print("\rin", i_mode, (i_epoch + 1), "/", HP_EPOCH, "-", (i_batch + 1), "/", i_batch_max, "-"
             ,"CPU:", timer_cpu_record, " GPU:", timer_gpu_record
             ,end = '')
        
        timer_cpu_start = time.time()
        #>>> 후처리
        
        i_batch += 1
    
    print("")
    
    if list_mp_buffer:
        plts_saver_sssr(list_mp_buffer, is_best=(i_mode=="test"), no_employ=(employ_threshold >= len(list_mp_buffer)))
    
    tmp_str_contents = str(dict_rb[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))
    for i_key in ["psnr", "ssim", "pa", "ca", "ious"]:
        tmp_str_contents += "," + str(dict_rb[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG
                                                                         ,is_print_sub = True ,is_update_graph = True
                                                                         )
                                     )
    
    update_dict_v2(str(i_epoch + 1), tmp_str_contents
                  ,in_dict_dict = dict_dict_log_total
                  ,in_dict_key = i_mode
                  ,in_print_head = "dict_dict_log_total_" + i_mode
                  ,is_print = False
                  )
    
    if i_mode == "train":
        dict_rb[i_mode]["lr"].add_item(optimizer_l.param_groups[0]['lr'])
        dict_rb[i_mode]["lr"].update_batch()
        # 스케쥴러 갱신
        scheduler_l.step()
        print("scheduler_l.step()")
        dict_rb[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)
    
    # log 기록 업데이트 (epoch 단위)
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG + i_mode + "/"
                 ,in_file_name = FILE_HEAD + "_log_epoch_" + i_mode + "_" + str(i_epoch + 1) + ".csv"
                 ,in_dict_dict = dict_dict_log_epoch
                 ,in_dict_key = i_mode
                 )
    
    # log 기록 업데이트 (학습 전체 단위)
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = FILE_HEAD + "_log_total_" + i_mode + ".csv"
                 ,in_dict_dict = dict_dict_log_total
                 ,in_dict_key = i_mode
                 )
    
    #<<< check_point, state_dict 저장
    save_interval = 100
    
    if i_mode == "train" and (i_epoch + 1) % save_interval == 0:
        # check_point 저장경로
        tmp_path = PATH_OUT_MODEL + "check_points/"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        # 모델 체크포인트 저장
        print("\n[--- 체크포인트", str(i_epoch + 1), "저장됨 ---]\n")
        torch.save({'epoch': (i_epoch + 1)                           # (int) 중단 시점 epoch 값
                   ,'model_t_state_dict': model_t.state_dict()       # (state_dict) model_m.state_dict()
                   ,'model_s_state_dict': model_s.state_dict()       # (state_dict) model_s.state_dict()
                   ,'model_m_state_dict': model_m.state_dict()       # (state_dict) model_m.state_dict()
                   ,'optimizer_l_state_dict': optimizer_l.state_dict()   # (state_dict) optimizer_l.state_dict()
                   ,'scheduler_l_state_dict': scheduler_l.state_dict()   # (state_dict) scheduler_l.state_dict()
                   }
                  ,tmp_path + str(i_epoch + 1) +"_" + FILE_HEAD + "_check_point.tar"
                  )
    
    if i_mode == "val":
        tmp_is_best = dict_rb[i_mode]["ious"].is_best_max # PSNR 기준
        
        if tmp_is_best:
            print("\n< Best Valid Epoch > Model State Dict 저장됨")
            tmp_path = PATH_OUT_MODEL + "state_dicts/"  # state_dict 저장경로
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            
            torch.save(model_t.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_t_msd.pt')
            torch.save(model_s.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_s_msd.pt')
            torch.save(model_m.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_m_msd.pt')
            
        else:
            print("\n< Not a Best Valid Epoch >")
        
        if prev_best is not None:
            # prev best 값이 입력된 경우 -> Test 생략여부 판별
            if prev_best > dict_rb[i_mode]["miou"].total_max[-1]:
                print("\nprev_best 못넘음...")
                tmp_is_best = False
            else:
                tmp_is_best = True
        
    else:
        tmp_is_best = None
    #>>>check_point, state_dict 저장
    
    
    return tmp_is_best


#=== End of one_epoch_cn4srss

"""


def trainer_(**kargs):
    # log dict 이어받기
    dict_log_init = kargs['dict_log_init']
    # [최우선 초기화요소 시행]------------------------------------------------------------------------
    
    try:
        # 결과 이미지 저장 여부
        WILL_SAVE_IMAGE = bool(kargs['WILL_SAVE_IMAGE'])
    except:
        WILL_SAVE_IMAGE = True
    
    try:
        # test 생략할 마지막 epoch (epoch는 1부터 시작할 떄 기준, 100 입력시 101 epoch 부터 test 실행 가능)
        SKIP_TEST_UNTIL = int (kargs['SKIP_TEST_UNTIL'])
    except:
        SKIP_TEST_UNTIL = 0
    
    update_dict_v2("", ""
                  ,"", "학습 시간 단축을 위한 옵션"
                  ,"", "WILL_SAVE_IMAGE: " + str(WILL_SAVE_IMAGE)
                  ,"", "SKIP_TEST_UNTIL: " + str(SKIP_TEST_UNTIL)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    
    CALC_WITH_LOGIT = bool(kargs['CALC_WITH_LOGIT'])
    if CALC_WITH_LOGIT:
        _str = "seMantic segmentation loss 계산 시 logit 값 사용"
    else:
        _str = "seMantic segmentation loss 계산 시 softmax 값 사용"
    update_dict_v2("", ""
                  ,"", "CALC_WITH_LOGIT: " + str(CALC_WITH_LOGIT)
                  ,"", _str
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    warnings.warn(_str)
    
    # 사용할 데이터셋 종류
    try:
        HP_DATASET_NAME = str(kargs['HP_DATASET_NAME'])
    except:
        HP_DATASET_NAME = "No Info"
    
    #라벨 관련 정보
    try:
        HP_LABEL_TOTAL  = kargs['HP_LABEL_TOTAL']
    except:
        HP_LABEL_TOTAL  = 2
    
    try:
        HP_LABEL_VOID   = kargs['HP_LABEL_VOID']
    except:
        HP_LABEL_VOID   = None
    
    if HP_LABEL_VOID is None:
        HP_LABEL_VOID   = HP_LABEL_TOTAL
        _str = "Void 라벨이 존재하지 않은 경우로 설정되었습니다."
        warnings.warn(_str)
    
    try:
        HP_DATASET_CLASSES = kargs['HP_DATASET_CLASSES']
    except:
        HP_DATASET_CLASSES = None
    
    if HP_DATASET_CLASSES == None:
        HP_DATASET_CLASSES = ""
        if HP_LABEL_TOTAL > HP_LABEL_VOID:
            _dummy_class = HP_LABEL_TOTAL - 1
        else:
            _dummy_class = HP_LABEL_TOTAL
        
        for i_class in range(_dummy_class):
            if i_class == 0:
                HP_DATASET_CLASSES += str(i_class)
            else:
                HP_DATASET_CLASSES += "," + str(i_class)
        
        _str = "log용 class info가 입력되지 않았습니다. 자동생성된 정보 [" + HP_DATASET_CLASSES + "]로 log가 작성됩니다."
        warnings.warn(_str)
    
    
    # Test 과정에서 반드시 저장할 이미지 이름
    MUST_SAVE_IMAGE = kargs['MUST_SAVE_IMAGE']
    
    # 최대로 저장할 이미지 수
    try:
        MAX_SAVE_IMAGES = int(kargs['MAX_SAVE_IMAGES'])
        if MAX_SAVE_IMAGES < 1:
            MAX_SAVE_IMAGES = 1
            print("MAX_SAVE_IMAGES should be >= 1. It fixed to", MAX_SAVE_IMAGES)
    except:
        MAX_SAVE_IMAGES = 10
    
    employ_threshold = 7        # sub-process 생성 경계걊 (이 값 이하의 데이터는 단일 프로세스로 처리)
    
    try:
        BUFFER_SIZE = kargs['BUFFER_SIZE']
        if BUFFER_SIZE < 1:
            print("BUFFER_SIZE should be > 0")
            BUFFER_SIZE = 60
    except:
        BUFFER_SIZE = 60
    
    print("BUFFER_SIZE set to", BUFFER_SIZE)
    
    
    
    # 사용 decive 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    
    # 랜덤 시드(seed) 적용
    HP_SEED = kargs['HP_SEED']
    random.seed(HP_SEED)
    np.random.seed(HP_SEED)
    # pytorch 랜덤시드 고정 (CPU)
    torch.manual_seed(HP_SEED)
    
    update_dict_v2("", ""
                  ,"", "랜덤 시드값 (random numpy pytorch)"
                  ,"", "HP_SEED: " + str(HP_SEED)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    if device == torch.device('cuda'):
        _str = "RUN with cuda"
        warnings.warn(_str)
        # pytorch 랜덤시드 고정 (GPU & multi-GPU)
        torch.cuda.manual_seed(HP_SEED)
        torch.cuda.manual_seed_all(HP_SEED)
        # 세부 디버깅용 오류문 출력
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        _str = "RUN on CPU"
        warnings.warn(_str)
    
    # epoch 수, batch 크기 & (train) 데이터셋 루프 횟수
    HP_EPOCH        = kargs['HP_EPOCH']
    HP_BATCH_TRAIN  = kargs['HP_BATCH_TRAIN']
    
    try:
        # valid 시행 시 center-cropped patch 사용여부
        # whole 이미지로 valid 시행 시, num_workers 옵션 활성화
        HP_VALID_WITH_PATCH = kargs['HP_VALID_WITH_PATCH']  # (bool)
    except:
        HP_VALID_WITH_PATCH = True
    
    HP_BATCH_VAL    = kargs['HP_BATCH_VAL']
    HP_BATCH_TEST   = kargs['HP_BATCH_TEST']
    
    try:
        HP_NUM_WORKERS = int(kargs['HP_NUM_WORKERS'])
        _total_worker = mp.cpu_count()
        if _total_worker <= 2:
            print("total workers are not enough to use multi-worker")
            HP_NUM_WORKERS = 0
        elif _total_worker < HP_NUM_WORKERS*2:
            print("too much worker!")
            HP_NUM_WORKERS = int(_total_worker//2)
    except:
        HP_NUM_WORKERS = 0
    
    update_dict_v2("", ""
                  ,"", "최대 epoch 설정: " + str(HP_EPOCH)
                  ,"", "batch 크기"
                  ,"", "HP_BATCH_TRAIN: " + str(HP_BATCH_TRAIN)
                  ,"", "HP_BATCH_VAL:   " + str(HP_BATCH_VAL)
                  ,"", "HP_BATCH_TEST:  " + str(HP_BATCH_TEST)
                  ,"", "HP_NUM_WORKERS for train: " + str(HP_NUM_WORKERS)
                  ,"", "그래디언트 축적(Gradient Accumulation) 사용 안함"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [입출력 Data 관련]-----------------------------
    # 경로: 입력
    PATH_BASE_IN        = kargs['PATH_BASE_IN']
    NAME_FOLDER_TRAIN   = kargs['NAME_FOLDER_TRAIN']
    NAME_FOLDER_VAL     = kargs['NAME_FOLDER_VAL']
    NAME_FOLDER_TEST    = kargs['NAME_FOLDER_TEST']
    NAME_FOLDER_IMAGES  = kargs['NAME_FOLDER_IMAGES']
    NAME_FOLDER_LABELS  = kargs['NAME_FOLDER_LABELS']
    
    # 경로: 출력
    PATH_OUT_IMAGE = kargs['PATH_OUT_IMAGE']
    if PATH_OUT_IMAGE[-1] != '/':
        PATH_OUT_IMAGE += '/'
    PATH_OUT_MODEL = kargs['PATH_OUT_MODEL']
    if PATH_OUT_MODEL[-1] != '/':
        PATH_OUT_MODEL += '/'
    PATH_OUT_LOG = kargs['PATH_OUT_LOG']
    if PATH_OUT_LOG[-1] != '/':
        PATH_OUT_LOG += '/'
    
    try:
        # Patch 생성 관련 강제 margin 추가여부
        is_force_fix = kargs['is_force_fix']
        _w, _h = kargs['force_fix_size_hr']
        force_fix_size_hr = (int(_w), int(_h))
        
        update_dict_v2("", "Train & Valid 이미지 Margin 추가 및 필요시 patch 생성 시행함"
                      ,"", "Margin 포함 (W H): (" + str(int(_w)) + " " + str(int(_h)) + ")"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    except:
        is_force_fix = False
        force_fix_size_hr = None
        update_dict_v2("", "Train & Valid 이미지 Margin 추가 및 필요시 patch 생성 시행 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [model, 모델 추가정보, optimizer, scheduler, loss]--------------------
    
    model_t         = kargs['model_t']          # SR (Teacher)
    optimizer_t     = kargs['optimizer_t']
    criterion_t     = kargs['criterion_t']      # loss function
    scheduler_t     = kargs['scheduler_t']
    
    model_s         = kargs['model_s']          # SR (Student)
    optimizer_s     = kargs['optimizer_s']
    criterion_s     = kargs['criterion_s']
    scheduler_s     = kargs['scheduler_s']
    
    model_m         = kargs['model_m']          # seMantic segmentation model
    optimizer_m     = kargs['optimizer_m']      # model_m 만 고려
    criterion_m     = kargs['criterion_m']      # model_m 만 고려
    scheduler_m     = kargs['scheduler_m']      # model_m 만 고려
    
    optimizer_srss  = kargs['optimizer_srss']   # model_s + model_m (CN4SRSS)
    criterion_srss  = kargs['criterion_srss']   # model_s + model_m (CN4SRSS)
    scheduler_srss  = kargs['scheduler_srss']   # model_s + model_m (CN4SRSS)
    
    HP_MIXUP_A = float(kargs['HP_MIXUP_A']) # (ks_sr_2_ss) mixup
    
    update_dict_v2("", ""
                  ,"", "(gt_kd_sr_2_ss, HP_MIXUP_A) SR (T) and SR (S) mixup: " + str(HP_MIXUP_A)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # Trainer 모드 설정 ("SS", "SR", "SSSR")
    TRAINER_MODE = "SSSR"
    
    print("\n#=========================#")
    print("  Trainer Mode:", TRAINER_MODE)
    print("#=========================#\n")
    
    try:
        # HP_DETECT_LOSS_ANOMALY      = kargs['HP_DETECT_LOSS_ANOMALY']
        HP_DETECT_LOSS_ANOMALY      = False
    except:
        HP_DETECT_LOSS_ANOMALY      = False
    
    # 현재 HP_DETECT_LOSS_ANOMALY 는 무조건 False로 설정됨
    
    if HP_DETECT_LOSS_ANOMALY:
        # with torch.autograd.detect_anomaly():
        update_dict_v2("", ""
                      ,"", "Train loss -> detect_anomaly 사용됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        update_dict_v2("", ""
                      ,"", "Train loss -> detect_anomaly 사용 안됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [Automatic Mixed Precision 선언] ---
    amp_scaler = torch.cuda.amp.GradScaler(enabled = True)
    update_dict_v2("", ""
                  ,"", "Automatic Mixed Precision 사용됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [data augmentation 관련]--------------------
    try:
        HP_AUGM_LITE            = kargs['HP_AUGM_LITE']
        HP_AUGM_LITE_FLIP_HORI  = bool(kargs['HP_AUGM_LITE_FLIP_HORI'])
        HP_AUGM_LITE_FLIP_VERT  = bool(kargs['HP_AUGM_LITE_FLIP_VERT'])
        
    except:
        HP_AUGM_LITE            = False
        HP_AUGM_LITE_FLIP_HORI  = False
        HP_AUGM_LITE_FLIP_VERT  = False
    
    if not HP_AUGM_LITE:
        HP_AUGM_RANGE_CROP_INIT = kargs['HP_AUGM_RANGE_CROP_INIT']
        HP_AUGM_ROTATION_MAX    = kargs['HP_AUGM_ROTATION_MAX']
        HP_AUGM_PROB_FLIP       = kargs['HP_AUGM_PROB_FLIP']
        HP_AUGM_PROB_CROP       = kargs['HP_AUGM_PROB_CROP']
        HP_AUGM_PROB_ROTATE     = kargs['HP_AUGM_PROB_ROTATE']
        
        # colorJitter 관련
        HP_CJ_BRIGHTNESS        = kargs['HP_CJ_BRIGHTNESS']
        HP_CJ_CONTRAST          = kargs['HP_CJ_CONTRAST']
        HP_CJ_SATURATION        = kargs['HP_CJ_SATURATION']
        HP_CJ_HUE               = kargs['HP_CJ_HUE']
        
    else:
        update_dict_v2("", ""
                  ,"", "Augmentation LITE mode 적용됨"
                  ,"", "FLIP Horizontal: " + str(HP_AUGM_LITE_FLIP_HORI)
                  ,"", "FLIP Vertical: " + str(HP_AUGM_LITE_FLIP_VERT)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
        
        HP_AUGM_RANGE_CROP_INIT = None
        HP_AUGM_ROTATION_MAX    = None
        HP_AUGM_PROB_FLIP       = None
        HP_AUGM_PROB_CROP       = None
        HP_AUGM_PROB_ROTATE     = None
        HP_CJ_BRIGHTNESS        = [1,1]
        HP_CJ_CONTRAST          = [1,1]
        HP_CJ_SATURATION        = [1,1]
        HP_CJ_HUE               = [0,0]
    
    update_dict_v2("", ""
                  ,"", "Augmentation 설정"
                  ,"", "HP_AUGM_LITE:             " + str(HP_AUGM_LITE)
                  ,"", "HP_AUGM_LITE_FLIP_HORI:   " + str(HP_AUGM_LITE_FLIP_HORI)
                  ,"", "HP_AUGM_LITE_FLIP_VERT:   " + str(HP_AUGM_LITE_FLIP_VERT)
                  ,"", "HP_AUGM_RANGE_CROP_INIT:  " + "".join(str(HP_AUGM_RANGE_CROP_INIT).split(","))
                  ,"", "HP_AUGM_ROTATION_MAX:     " + str(HP_AUGM_ROTATION_MAX)
                  ,"", "HP_AUGM_PROB_FLIP:        " + str(HP_AUGM_PROB_FLIP)
                  ,"", "HP_AUGM_PROB_CROP:        " + str(HP_AUGM_PROB_CROP)
                  ,"", "HP_AUGM_PROB_ROTATE:      " + str(HP_AUGM_PROB_ROTATE)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    update_dict_v2("", ""
                  ,"", "ColorJitter 설정"
                  ,"", "brightness: ( " + " ".join([str(t_element) for t_element in HP_CJ_BRIGHTNESS]) +" )"
                  ,"", "contrast:   ( " + " ".join([str(t_element) for t_element in HP_CJ_CONTRAST])   +" )"
                  ,"", "saturation: ( " + " ".join([str(t_element) for t_element in HP_CJ_SATURATION]) +" )"
                  ,"", "hue:        ( " + " ".join([str(t_element) for t_element in HP_CJ_HUE])        +" )"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    try:
        HP_AUGM_RANDOM_SCALER = kargs['HP_AUGM_RANDOM_SCALER']
    except:
        HP_AUGM_RANDOM_SCALER = [1.0]
    
    update_dict_v2("", ""
                  ,"", "RANDOM_SCALER 설정"
                  ,"", "List: [ " + " ".join([str(t_element) for t_element in HP_AUGM_RANDOM_SCALER]) +" ]"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [이미지 변수 -> 텐서 변수 변환]-------------------
    # 정규화 여부
    try:
        is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor']
    except:
        is_norm_in_transform_to_tensor = False
    
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
        
        update_dict_v2("", ""
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
        
        update_dict_v2("", ""
                      ,"", "입력 이미지(in_x) 정규화 시행 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [Degradation 관련]-------------------------------
    # (bool) 학습 & 평가 시 Degradaded Input 사용 여부
    option_apply_degradation = True
    
    if option_apply_degradation:
        # "Train & Test 과정에 Degradation 시행 됨"
        
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
        
        # scale_factor
        HP_DG_SCALE_FACTOR = kargs['HP_DG_SCALE_FACTOR']                # Dataloader 에서 쓰임
            
        update_dict_v2("", ""
                      ,"", "Degradation 관련"
                      ,"", "시행여부: " + "Train & Valid % Test 과정에 Degradation 시행 됨"
                      ,"", "DG 지정값 파일 경로: " + PATH_BASE_IN_SUB + HP_DG_CSV_NAME
                      ,"", "Scale Factor 고정값 = x" + str(HP_DG_SCALE_FACTOR)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

            
        
    else:
        # "Train & Test 과정에 Degradation 시행 안됨"
        update_dict_v2("", ""
                      ,"", "Degradation 관련"
                      ,"", "시행여부: " + "Train & Valid & Test 과정에 Degradation 시행 안됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    # [data & model load]--------------------------
    PATH_ALTER_HR_IMAGE = None
    
    if TRAINER_MODE == "SS" or TRAINER_MODE == "SSSR":
        tmp_is_return_label = True
        if HP_AUGM_LITE:
            _str = "라벨 데이터가 사용되는 경우, HP_AUGM_LITE 옵션 사용 불가능"
            sys.exit(_str)
        save_graph_ss = True
    else:
        save_graph_ss = False
    
    if TRAINER_MODE == "SR" or TRAINER_MODE == "SSSR":
        tmp_is_return_image_lr = True
        save_graph_sr = True
    else:
        save_graph_sr = False
    
    if tmp_is_return_label:
        HP_COLOR_MAP    = kargs['HP_COLOR_MAP']                         # (dict) gray -> rgb 컬러매핑
        
        update_dict_v2("", ""
                      ,"", "사용된 데이터셋: " + HP_DATASET_NAME
                      ,"", "라벨 별 RGB 매핑"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
        for i_key in HP_COLOR_MAP:
            _color_map = ""
            for i_color in HP_COLOR_MAP[i_key]:
                _color_map += " " + str(i_color)
            
            update_dict_v2("", str(i_key) + ":" + _color_map
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        if HP_LABEL_VOID == HP_LABEL_TOTAL:
            update_dict_v2("", ""
                          ,"", "원본 데이터 라벨 수: " + str(HP_LABEL_TOTAL)
                          ,"", "void 라벨 없음"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            update_dict_v2("", ""
                          ,"", "원본 데이터 라벨 수(void 포함): " + str(HP_LABEL_TOTAL)
                          ,"", "void 라벨 번호: " + str(HP_LABEL_VOID)
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        try:
            HP_LABEL_ONEHOT_ENCODE = kargs['HP_LABEL_ONEHOT_ENCODE']    # (bool) label one-hot encoding 시행여부
        except:
            HP_LABEL_ONEHOT_ENCODE = False
        
        if HP_LABEL_ONEHOT_ENCODE:
            _str = "Label one-hot encode: 시행함"
            warnings.warn(_str)
            update_dict_v2("", "Label one-hot encode: 시행함"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            _str = "Label one-hot encode: 시행 안함"
            warnings.warn(_str)
            update_dict_v2("", "Label one-hot encode: 시행 안함"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        try:
            HP_LABEL_DILATED = kargs['HP_LABEL_DILATED']                # (bool) label dilation 시행여부
        except:
            HP_LABEL_DILATED = False
        
        if HP_LABEL_DILATED:
            _str = "DILATION for Labels in train: Activated"
            warnings.warn(_str)
            update_dict_v2("", ""
                          ,"", "Label Dilation: 적용됨"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            _str = "DILATION for Labels in train: Deactivated"
            warnings.warn(_str)
            update_dict_v2("", ""
                          ,"", "Label Dilation: 적용 안됨"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
        if HP_LABEL_DILATED and not HP_LABEL_ONEHOT_ENCODE:
            _str = "If label is dilated, it must be in one-hot form."
            sys.exit(_str)
        
        if is_force_fix:
            try:
                HP_LABEL_VERIFY             = kargs['HP_LABEL_VERIFY']              # (bool) label 검증 여부
                HP_LABEL_VERIFY_TRY_CEILING = kargs['HP_LABEL_VERIFY_TRY_CEILING']  # (int) 최대 재시도 횟수
                HP_LABEL_VERIFY_CLASS_MIN   = kargs['HP_LABEL_VERIFY_CLASS_MIN']    # (int) 최소 class 종류
                HP_LABEL_VERIFY_RATIO_MAX   = kargs['HP_LABEL_VERIFY_RATIO_MAX']    # (float) 단일 class 최대 비율
            except:
                HP_LABEL_VERIFY             = False
                HP_LABEL_VERIFY_TRY_CEILING = None
                HP_LABEL_VERIFY_CLASS_MIN   = None
                HP_LABEL_VERIFY_RATIO_MAX   = None
            
            if HP_LABEL_VERIFY:
                update_dict_v2("", ""
                              ,"", "Label Verify in train: 적용됨"
                              ,"", "라벨 re-crop 시도 최대횟수: " + str(HP_LABEL_VERIFY_TRY_CEILING)
                              ,"", "라벨 내 유효 class 최소 종류 수: " + str(HP_LABEL_VERIFY_CLASS_MIN)
                              ,"", "라벨 내 최대 class 비율 상한 (0 ~ 1): " + str(HP_LABEL_VERIFY_RATIO_MAX)
                              ,in_dict = dict_log_init
                              ,in_print_head = "dict_log_init"
                              )
            else:
                update_dict_v2("", ""
                              ,"", "Label Verify in train: 적용 안됨"
                              ,in_dict = dict_log_init
                              ,in_print_head = "dict_log_init"
                              )
        
        
    else:
        HP_COLOR_MAP                = None
        HP_LABEL_ONEHOT_ENCODE      = False
        HP_LABEL_DILATED            = False
        HP_LABEL_VERIFY             = False
        HP_LABEL_VERIFY_TRY_CEILING = None
        HP_LABEL_VERIFY_CLASS_MIN   = None
        HP_LABEL_VERIFY_RATIO_MAX   = None
    
    
    # V6 : LR 이미지 생성 안하고 불러와서 씀
    dataset_train = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr', 'info_augm'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'train'
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TRAIN
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = True
                                      # below options can be skipped when above option is False
                                     ,opt_augm_lite                 = HP_AUGM_LITE
                                     ,opt_augm_lite_flip_hori       = HP_AUGM_LITE_FLIP_HORI
                                     ,opt_augm_lite_flip_vert       = HP_AUGM_LITE_FLIP_VERT
                                     ,opt_augm_crop_init_range      = HP_AUGM_RANGE_CROP_INIT
                                     ,opt_augm_rotate_max_degree    = HP_AUGM_ROTATION_MAX
                                     ,opt_augm_prob_flip            = HP_AUGM_PROB_FLIP
                                     ,opt_augm_prob_crop            = HP_AUGM_PROB_CROP
                                     ,opt_augm_prob_rotate          = HP_AUGM_PROB_ROTATE
                                     ,opt_augm_cj_brigntess         = HP_CJ_BRIGHTNESS
                                     ,opt_augm_cj_contrast          = HP_CJ_CONTRAST
                                     ,opt_augm_cj_saturation        = HP_CJ_SATURATION
                                     ,opt_augm_cj_hue               = HP_CJ_HUE
                                     ,opt_augm_random_scaler        = HP_AUGM_RANDOM_SCALER
                                     
                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                     
                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = HP_LABEL_DILATED
                                     
                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                     
                                     ,is_label_verify               = HP_LABEL_VERIFY
                                      #(선택) if is_label_verify is True
                                     ,label_verify_try_ceiling      = HP_LABEL_VERIFY_TRY_CEILING
                                     ,label_verify_class_min        = HP_LABEL_VERIFY_CLASS_MIN
                                     ,label_verify_ratio_max        = HP_LABEL_VERIFY_RATIO_MAX
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- increase dataset length
                                     ,in_dataset_loop               = 1
                                     
                                      #--- options for generate margin and patch
                                     ,is_force_fix                  = is_force_fix
                                     ,force_fix_size_hr             = force_fix_size_hr
                                     
                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img
                                     )
    
    if HP_VALID_WITH_PATCH:
        # center-cropped patch로 valid
        update_dict_v2("", ""
                      ,"", "Valid: center-cropped patch 이미지로 시행"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        dataset_val   = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                          #                         , 'pil_img_hr', 'ts_img_hr'
                                          #                         , 'pil_lab_hr', 'ts_lab_hr'
                                          #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                          name_memo                     = ' val '
                                         ,in_path_dataset               = PATH_BASE_IN
                                         ,in_category                   = NAME_FOLDER_VAL
                                         ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                         
                                          #--- options for train 
                                         ,is_train                      = False
                                         
                                          #--- options for HR image
                                         ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                         
                                          #--- options for HR label
                                         ,is_return_label               = tmp_is_return_label
                                          # below options can be skipped when above option is False
                                         ,in_name_folder_label          = NAME_FOLDER_LABELS
                                         ,label_number_total            = HP_LABEL_TOTAL
                                         ,label_number_void             = HP_LABEL_VOID
                                         ,is_label_dilated              = False
                                         
                                         ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                         
                                         ,is_label_verify               = False
                                         
                                          #--- options for LR image
                                         ,is_return_image_lr            = tmp_is_return_image_lr
                                          # below options can be skipped when above option is False
                                         ,scalefactor                   = HP_DG_SCALE_FACTOR
                                         ,in_path_dlc                   = PATH_BASE_IN_SUB
                                         ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                         
                                          #--- options for generate margin and patch
                                         ,is_force_fix                  = is_force_fix
                                         ,force_fix_size_hr             = force_fix_size_hr
                                         
                                          #--- optionas for generate tensor
                                         ,transform_img                 = transform_to_ts_img
                                         )
    else:
        # whole image로 valid
        update_dict_v2("", ""
                      ,"", "Valid: whole 이미지로 시행"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        dataset_val   = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                          #                         , 'pil_img_hr', 'ts_img_hr'
                                          #                         , 'pil_lab_hr', 'ts_lab_hr'
                                          #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                          name_memo                     = ' val '
                                         ,in_path_dataset               = PATH_BASE_IN
                                         ,in_category                   = NAME_FOLDER_VAL
                                         ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                         
                                          #--- options for train 
                                         ,is_train                      = False
                                         
                                          #--- options for HR image
                                         ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                         
                                          #--- options for HR label
                                         ,is_return_label               = tmp_is_return_label
                                          # below options can be skipped when above option is False
                                         ,in_name_folder_label          = NAME_FOLDER_LABELS
                                         ,label_number_total            = HP_LABEL_TOTAL
                                         ,label_number_void             = HP_LABEL_VOID
                                         ,is_label_dilated              = False
                                         
                                         ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                         
                                         ,is_label_verify               = False
                                         
                                          #--- options for LR image
                                         ,is_return_image_lr            = tmp_is_return_image_lr
                                          # below options can be skipped when above option is False
                                         ,scalefactor                   = HP_DG_SCALE_FACTOR
                                         ,in_path_dlc                   = PATH_BASE_IN_SUB
                                         ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                         
                                          #--- options for generate margin and patch
                                         ,is_force_fix                  = False
                                         
                                          #--- optionas for generate tensor
                                         ,transform_img                 = transform_to_ts_img
                                         )
    
    dataset_test  = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'test '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TEST
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = False
                                     
                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE
                                     
                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = False
                                     
                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE
                                     
                                     ,is_label_verify               = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- options for generate margin and patch
                                     ,is_force_fix                  = False
                                     
                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img
                                     )
    
    
    dataloader_train = DataLoader_multi_worker_FIX(dataset     = dataset_train
                                                  ,batch_size  = HP_BATCH_TRAIN
                                                  ,shuffle     = True
                                                  ,num_workers = HP_NUM_WORKERS
                                                  ,prefetch_factor = 2
                                                  ,drop_last = True
                                                  )
    
    if HP_VALID_WITH_PATCH:
        # center-cropped patch로 valid
        _str = "Valid: center-cropped 이미지로 시행됨에 따라 num_workers 옵션 비활성화"
        warnings.warn(_str)
        dataloader_val   = torch.utils.data.DataLoader(dataset     = dataset_val
                                                      ,batch_size  = HP_BATCH_VAL
                                                      ,shuffle     = False
                                                      ,num_workers = 0
                                                      )
    else:
        # whole image로 valid
        _str = "Valid: whole 이미지로 시행됨에 따라 num_workers 옵션 활성화"
        warnings.warn(_str)
        dataloader_val   = DataLoader_multi_worker_FIX(dataset     = dataset_val
                                                      ,batch_size  = HP_BATCH_VAL
                                                      ,shuffle     = False
                                                      ,num_workers = HP_NUM_WORKERS
                                                      ,prefetch_factor = 2
                                                      )
    
    dataloader_test  = DataLoader_multi_worker_FIX(dataset     = dataset_test
                                                  ,batch_size  = HP_BATCH_TEST
                                                  ,shuffle     = False
                                                  ,num_workers = HP_NUM_WORKERS
                                                  ,prefetch_factor = 2
                                                  )
    
    # [Train & Val & Test]-----------------------------
    
    #<< pkl_maker
    try:
        make_pkl = kargs['make_pkl']
    except:
        make_pkl = False
    
    if make_pkl:
        from _pkl_mkr.pkl_maker_0_6 import pkl_mkr_
        
        pkl_mkr_(dict_log_init      = dict_log_init
                ,PATH_OUT_IMAGE     = PATH_OUT_IMAGE
                ,PATH_OUT_LOG       = PATH_OUT_LOG
                ,HP_SEED            = HP_SEED
                ,HP_COLOR_MAP       = HP_COLOR_MAP
                ,dataloader_train   = dataloader_train
                ,dataloader_val     = dataloader_val
                ,dataloader_test    = dataloader_test
                )
        
        print("피클 생성 완료 -> 프로세스 종료")
        sys.exit(0)
    else:
        print("피클 생성기 사용 안함")
    #>> pkl_maker
    
    #<< Load check_point
    # There is no check point in our life.
    #>> Load check_point
    
    print("\nPause before init trainer")
    time.sleep(3)
        
    # 1 epoch 마다 시행할 mode list
    list_mode = ["train", "val", "test"]
    
    
    def _generate_dict_dict_log_total(list_mode, HP_DATASET_CLASSES, mode=None):
        # mode: "sr_n_ss", "kd_sr_2_ss", "gt_kd_sr_2_ss"
        dict_return = {list_mode[0] : {}
                      ,list_mode[1] : {}
                      ,list_mode[2] : {}
                      }
        
        for i_key in list_mode:
            if mode == "sr_n_ss":
                _str  = "loss_t_(" + i_key + "),loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
            elif mode == "kd_sr_2_ss":
                _str  = "loss_sr_(" + i_key + "),loss_ss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
            elif mode == "gt_kd_sr_2_ss":
                _str  = "loss_t_(" + i_key + "),loss_s_(" + i_key + "),loss_m_(" + i_key + "),PSNR_t_(" + i_key + "),SSIM_t_(" + i_key + "),PSNR_s_(" + i_key + "),SSIM_s_(" + i_key + "),"
            else:
                _str  = "loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
            _str += "Pixel_Acc_(" + i_key + "),Class_Acc_(" + i_key + "),mIoU_(" + i_key + "),"
            _str += HP_DATASET_CLASSES
            
            update_dict_v2("epoch", _str
                          ,in_dict_dict = dict_return
                          ,in_dict_key = i_key
                          ,in_print_head = "dict_dict_log_total_" + i_key
                          ,is_print = False
                          )
        
        return dict_return
    
    dict_dict_log_total_1 = _generate_dict_dict_log_total(list_mode, HP_DATASET_CLASSES, mode="gt_kd_sr_2_ss")  # gt_kd_sr_2_ss
    dict_dict_log_total_2 = _generate_dict_dict_log_total(list_mode, HP_DATASET_CLASSES)                        # CN4SRSS   
    
    def _generate_record_box(in_name, in_list):
        dict_return = {}
        for i_key in in_list:
            _str = in_name + "_" + i_key + "_"
            dict_return[i_key] = {"lr"   : RecordBox(name = _str + "lr",        is_print = False)
                                 ,"loss" : RecordBox(name = _str + "loss",      is_print = False)
                                 ,"psnr" : RecordBox(name = _str + "psnr",      is_print = False)
                                 ,"ssim" : RecordBox(name = _str + "ssim",      is_print = False)
                                 ,"pa"   : RecordBox(name = _str + "pixel_acc", is_print = False)
                                 ,"ca"   : RecordBox(name = _str + "class_acc", is_print = False)
                                 ,"ious" : RecordBox4IoUs(name = _str + "ious", is_print = False)
                                 }
        return dict_return
    
    dict_rb_t = _generate_record_box("gt_kd_sr_2_ss_t", list_mode)
    dict_rb_s = _generate_record_box("gt_kd_sr_2_ss_s", list_mode)
    dict_rb_m = _generate_record_box("gt_kd_sr_2_ss_m", list_mode)
    
    dict_rb_srss = _generate_record_box("cn4srss", list_mode)
    
    def ignite_eval_step(engine, batch):
        return batch
    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)           # 이거로 측정하면 됨
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')
    
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = "log_init.csv"
                 ,in_dict = dict_log_init
                 )
    
    timer_trainer_start_local = time.mktime(time.localtime())   # trainer 시작 시간 - 종료 예상시간 출력용
    timer_trainer_start = time.time()                           # trainer 시작 시간 - 경과시간 측정용
    
    PklLoader_ = PklLoader(path_log_out = PATH_OUT_LOG
                          ,path_pkl_txt = "./_pkl_mkr/pkl_path.txt"
                          )
    
    for i_epoch in range(HP_EPOCH):
        
        print("")
        if i_epoch > 0:                 # Time Calcurator ------------------------------------------------------------  #
            _elapsed_time = time.time() - timer_trainer_start
            try:
                _estimated_time = (_elapsed_time / i_epoch) * (HP_EPOCH)
                _tmp = time.localtime(_estimated_time + timer_trainer_start_local)
                _ETA =  str(_tmp.tm_year) + " y " + str(_tmp.tm_mon) + " m " + str(_tmp.tm_mday) + " d   "
                _ETA += str(_tmp.tm_hour) + " : " + str(_tmp.tm_min)
            except:
                _ETA = "FAIL"
        else:
            _ETA = "Calculating..."
        
        print("\n=== gt_kd_sr_2_ss ===\n")    # gt_kd_sr_2_ss --------------------------------------------------------------------  #
        print("Estimated Finish Time:", _ETA)
        
        model_t.to(device)
        model_s.to(device)
        model_m.to(device)
        
        is_best = None
        for i_mode in list_mode:
            print("--- init gt_kd_sr_2_ss", i_mode, "---")
            
            if i_mode == "test" and is_best is not None:
                if SKIP_TEST_UNTIL >= i_epoch + 1:  # i_epoch 는 0 부터 시작
                    print(SKIP_TEST_UNTIL, "epoch 까지 test 생략 ~")
                    continue
                elif is_best == False:
                    print("이번 epoch test 생략 ~")
                    continue
                
            
            if i_mode == "train":
                dataloader_input = PklLoader_.open_pkl(mode = i_mode, epoch = i_epoch + 1
                                                      ,dataloader = dataloader_train
                                                      )
            elif i_mode == "val":
                dataloader_input = PklLoader_.open_pkl(mode = i_mode, epoch = 1
                                                      ,dataloader = dataloader_val
                                                      )
            elif i_mode == "test":
                dataloader_input = PklLoader_.open_pkl(mode = i_mode, epoch = 1
                                                      ,dataloader = dataloader_test
                                                      )
            
            is_best =one_epoch_gt_kd_sr_2_ss(WILL_SAVE_IMAGE = WILL_SAVE_IMAGE
                                            
                                            ,CALC_WITH_LOGIT = CALC_WITH_LOGIT
                                             #TRAINER_MODE = TRAINER_MODE
                                            ,list_mode = list_mode
                                            ,i_mode    = i_mode
                                            ,HP_EPOCH  = HP_EPOCH
                                            ,i_epoch   = i_epoch
                                            
                                            ,MAX_SAVE_IMAGES  = MAX_SAVE_IMAGES
                                            ,MUST_SAVE_IMAGE  = MUST_SAVE_IMAGE
                                            ,BUFFER_SIZE      = BUFFER_SIZE
                                            ,employ_threshold = employ_threshold
                                            
                                            ,HP_LABEL_TOTAL         = HP_LABEL_TOTAL
                                            ,HP_LABEL_VOID          = HP_LABEL_VOID
                                            ,HP_DATASET_CLASSES     = HP_DATASET_CLASSES
                                            ,HP_LABEL_ONEHOT_ENCODE = HP_LABEL_ONEHOT_ENCODE
                                            ,HP_COLOR_MAP           = HP_COLOR_MAP
                                            
                                            ,PATH_OUT_MODEL = PATH_OUT_MODEL
                                            ,PATH_OUT_IMAGE = PATH_OUT_IMAGE
                                            ,PATH_OUT_LOG   = PATH_OUT_LOG
                                            
                                            ,device = device
                                            
                                            ,model_t        = model_t
                                            ,optimizer_t    = optimizer_t
                                            ,criterion_t    = criterion_t
                                            ,scheduler_t    = scheduler_t
                                            
                                            ,model_s        = model_s
                                            ,optimizer_s    = optimizer_s
                                            ,criterion_s    = criterion_s
                                            ,scheduler_s    = scheduler_s
                                            
                                            ,model_m        = model_m
                                            ,optimizer_m    = optimizer_m
                                            ,criterion_m    = criterion_m
                                            ,scheduler_m    = scheduler_m
                                            
                                            ,HP_MIXUP_A = HP_MIXUP_A
                                            
                                            ,amp_scaler = amp_scaler
                                            ,ignite_evaluator = ignite_evaluator
                                            
                                            ,dataloader_input = dataloader_input
                                            
                                            ,dict_rb_t = dict_rb_t
                                            ,dict_rb_s = dict_rb_s
                                            ,dict_rb_m = dict_rb_m
                                            
                                            ,dict_dict_log_total = dict_dict_log_total_1
                                            )
        """
        
        print("\n=== CN4SRSS ===\n")       # CN4SRSS -------------------------------------------------------------------  #
        print("Estimated Finish Time:", _ETA)
        
        model_t.to(device_cpu)
        model_s.to(device)
        model_m.to(device)
        
        is_best = None
        for i_mode in list_mode:
            print("--- init cn4srss", i_mode, "---")
            
            if i_mode == "test" and is_best is not None:
                if is_best == False:
                    print("이번 epoch test 생략 ~")
                    continue
            
            if i_mode == "train":
                dataloader_input = dataloader_train
            elif i_mode == "val":
                dataloader_input = dataloader_val
            elif i_mode == "test":
                dataloader_input = dataloader_test
            
            
            is_best = one_epoch_cn4srss(CALC_WITH_LOGIT = CALC_WITH_LOGIT
                                       ,TRAINER_MODE = TRAINER_MODE
                                       ,list_mode = list_mode
                                       ,i_mode = i_mode
                                       ,HP_EPOCH = HP_EPOCH
                                       ,i_epoch = i_epoch
                                       ,prev_best = None
                                       # ,model_type = model_type
                                       ,MAX_SAVE_IMAGES = MAX_SAVE_IMAGES
                                       ,MUST_SAVE_IMAGE = MUST_SAVE_IMAGE
                                       ,BUFFER_SIZE = BUFFER_SIZE
                                       ,employ_threshold = employ_threshold
                                       
                                       ,HP_LABEL_TOTAL = HP_LABEL_TOTAL
                                       ,HP_LABEL_VOID = HP_LABEL_VOID
                                       ,HP_DATASET_CLASSES = HP_DATASET_CLASSES
                                       ,HP_LABEL_ONEHOT_ENCODE = HP_LABEL_ONEHOT_ENCODE
                                       ,HP_COLOR_MAP = HP_COLOR_MAP
                                       
                                       ,PATH_OUT_MODEL = PATH_OUT_MODEL
                                       ,PATH_OUT_IMAGE = PATH_OUT_IMAGE
                                       ,PATH_OUT_LOG = PATH_OUT_LOG
                                       
                                       ,device = device
                                       
                                       ,model_t     = model_t  # teacher는 gpu 연산 없음
                                       ,model_s     = model_s
                                       ,model_m     = model_m
                                       ,optimizer_l = optimizer_srss
                                       ,criterion_l = criterion_srss
                                       ,scheduler_l = scheduler_srss
                                       
                                       ,amp_scaler = amp_scaler
                                       ,ignite_evaluator = ignite_evaluator
                                       
                                       ,dataloader_input = dataloader_input
                                       
                                       ,dict_rb = dict_rb_srss
                                       
                                       ,dict_dict_log_total = dict_dict_log_total_2
                                       )
        """

#=== End of trainer_




print("End of trainer_total.py")
