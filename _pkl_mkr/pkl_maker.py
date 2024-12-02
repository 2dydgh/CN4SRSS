"""
pickle maker (pkl_maker)

~ v 0.5
    기존 trainer code 일부 따와서 피클 생성

v 0.6
    기존 trainer에서 불러와서 pickle 생성

"""
_pv = "v 0.6"
import copy
import time
import warnings
_str = "\n\n---[ Proposed 2nd paper pkl maker version: " + str(_pv) + " ]---\n"
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

# import ignite   #pytorch ignite

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time
import warnings

import pickle
# import gzip

# [py 파일]--------------------------
from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *

# from mps.mp_sssr_plt_saver import plts_saver as plts_saver_sssr     # support: model_a, model_proposed, model_d, model_aa
# from mps.mp_sr_plt_saver   import plts_saver as plts_saver_sr       # support: MPRNet, ESRT, HAN, IMDN
# from mps.mp_ss_plt_saver   import plts_saver as plts_saver_ss       # support: D3P
from mps.mp_dataloader     import DataLoader_multi_worker_FIX       # now multi-worker can be used on windows 10


def save_pils(**kwargs):
    # 저장 위치
    path_out    = kwargs["path_out"]
    
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    name        = kwargs["name"]            # 파일 이름
    
    info_augm       = kwargs["info_augm"]
    info_deg        = kwargs["info_deg"]
    
    if info_deg is None:
        info_deg = ""
    
    # 개별 저장 필요 없음
    pil_img_hr  = kwargs["pil_img_hr"]
    pil_lab_hr  = kwargs["pil_lab_hr"]
    pil_img_lr  = kwargs["pil_img_lr"]
    
    _H, _W = 6, 16
    _Row, _Col = 1, 3
    
    list_pil = [[pil_img_hr, pil_lab_hr, pil_img_lr]
               ]
    
    list_title = [["HR image", "GT label", "LR image"]
                 ]
    
    fig, ax = plt.subplots(_Row, _Col, figsize=(_W, _H), squeeze=False)
    
    _title = name + "\n" + info_augm + "\n" + info_deg + "\n"
    
    fig.suptitle(_title)
    
    for _r in range(_Row):
        for _c in range(_Col):
            if list_pil[_r][_c] is not None:
                ax[_r][_c].imshow(list_pil[_r][_c])
                _size = list_pil[_r][_c].size
                _str = " (h " + str(_size[-1]) + ", w " + str(_size[0]) + ")"
                ax[_r][_c].set_title(list_title[_r][_c] + _str)
    
    plt.savefig(path_out + name
               ,format='png'
               )
    
    plt.close()


def pkl_mkr_(**kargs):
    # log dict 이어받기
    dict_log_init = kargs['dict_log_init']
    update_dict_v2("", ""
                  ,"", ""
                  ,"", "---[ pkl_maker ]---"
                  ,"", ""
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [최우선 초기화요소 시행]------------------------------------------------------------------------
    
    # 사용 decive 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    
    # epoch 수, batch 크기 & (train) 데이터셋 루프 횟수
    # HP_EPOCH        = kargs['HP_EPOCH']
    HP_EPOCH        = 1000
    
    # HP_NUM_WORKERS = min([mp.cpu_count() // 4, 4])
    # update_dict_v2("", "workers changed to " + str(HP_NUM_WORKERS)
                  # ,"", ""
                  # ,in_dict = dict_log_init
                  # ,in_print_head = "dict_log_init"
                  # )
    
    # [입출력 Data 관련]-----------------------------
    
    # 경로: 출력
    PATH_OUT_IMAGE = kargs['PATH_OUT_IMAGE']
    if PATH_OUT_IMAGE[-1] != '/':
        PATH_OUT_IMAGE += '/'
    
    PATH_OUT_LOG = kargs['PATH_OUT_LOG']
    
    # pkl_maker용 log path로 수정
    if PATH_OUT_LOG[-1] != '/':
        PATH_OUT_LOG += "_pkl_maker/"
    else:
        PATH_OUT_LOG = PATH_OUT_LOG[:-1] + "_pkl_maker/"
    
    PATH_OUT_TOP = "/".join(PATH_OUT_LOG.split("/")[:-2])
    
    if not os.path.exists(PATH_OUT_TOP):
        os.makedirs(PATH_OUT_TOP)
    
    PATH_OUT_PKL = PATH_OUT_TOP + "/PKL"
    
    update_dict_v2("", "PATH_OUT_IMAGE:           " + str(PATH_OUT_IMAGE)
                  ,"", "PATH_OUT_LOG (pkl_maker): " + str(PATH_OUT_LOG)
                  ,"", "PATH_OUT_PKL:             " + str(PATH_OUT_PKL)
                  ,"", ""
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    try:
        # 이전까지 pkl_maker가 몇 번 실행되었는지 확인
        log_count = len(os.listdir(PATH_OUT_LOG))
    except:
        log_count = 0
    
    epoch_init = HP_EPOCH * log_count + 1
    epoch_last = HP_EPOCH * (log_count + 1)
    
    log_init_name = "log_init_" + str(epoch_init) + "_" + str(epoch_last) +".csv"
    
    update_dict_v2("", ""
                  ,"", "한 번에 생성할 epoch 수: " + str(HP_EPOCH)
                  ,"", "생성할 epoch: " + str(epoch_init) + " to " + str(epoch_last)
                  # ,"", "HP_NUM_WORKERS for pkl_maker: " + str(HP_NUM_WORKERS)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # 랜덤 시드(seed) 적용
    HP_SEED = int(kargs['HP_SEED']) + log_count
    
    update_dict_v2("", ""
                  ,"", "랜덤 시드 재고정 (random numpy pytorch)"
                  ,"", "HP_SEED: " + str(HP_SEED)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    random.seed(HP_SEED)
    np.random.seed(HP_SEED)
    torch.manual_seed(HP_SEED)          # pytorch 랜덤시드 고정 (CPU)
    
    if device == torch.device('cuda'):
        _str = "RUN with cuda"
        # pytorch 랜덤시드 고정 (GPU & multi-GPU)
        torch.cuda.manual_seed(HP_SEED)
        torch.cuda.manual_seed_all(HP_SEED)
        # 세부 디버깅용 오류문 출력
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        _str = "RUN on CPU"
    
    warnings.warn(_str) # 가동 장치 확인
    
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = log_init_name
                 ,in_dict = dict_log_init
                 )
    
    _path = PATH_OUT_TOP + "/run_command.txt"
    if not os.path.exists(_path):
        with open(_path, mode="w") as _txt:
            _txt.write("main.py file run command was...\n")
        print("command 기록용 text 파일 생성됨:", _path)
    
    
    HP_COLOR_MAP        = kargs['HP_COLOR_MAP']
    
    dataloader_train    = kargs['dataloader_train']
    dataloader_val      = kargs['dataloader_val']
    dataloader_test     = kargs['dataloader_test']
    
    
    # [Train & Val & Test]-----------------------------
    
    #<<< Load check_point
    # There is no check point in our life.
    #>>> Load check_point
    
    print("\nPause before init pkl_mkr_")
    time.sleep(3)
        
    # 1 epoch 마다 시행할 mode list
    list_mode = ["train", "val", "test"]
    
    timer_trainer_start_local = time.mktime(time.localtime())   # trainer 시작 시간 - 종료 예상시간 출력용
    timer_trainer_start = time.time()                           # trainer 시작 시간 - 경과시간 측정용
    
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
        
        print("Estimated Finish Time:", _ETA)
        
        for i_mode in list_mode:
            if i_mode == "train":
                dataloader_input = dataloader_train
            elif i_mode == "val":
                dataloader_input = dataloader_val
            elif i_mode == "test":
                dataloader_input = dataloader_test
            
            len_dataloader_input = len(dataloader_input)
            
            if i_epoch > 0 or epoch_init > 1:
                if i_mode != "train":
                    print("Epoch", i_epoch + epoch_init, "-", i_mode, "생략 ~")
                    continue
            
            try:
                del list_items_buffer
                list_items_buffer = []
            except:
                list_items_buffer = []
            
            i_batch = 0
            timer_pkl = time.time()
            for dataloader_items in dataloader_input:
                print("\rE", i_epoch + epoch_init, "/", HP_EPOCH + epoch_init - 1,"- B", i_batch + 1, "/", len_dataloader_input, end="")
                list_items_buffer.append(dataloader_items)
                i_batch += 1
            
            path_pkl = PATH_OUT_PKL + "/" + i_mode + "_E_" + str(i_epoch + epoch_init) + ".pkl"
            
            if not os.path.exists(PATH_OUT_PKL):
                os.makedirs(PATH_OUT_PKL)
            
            # 압축하면 용향 70% 정도 줄어드는데, 저장이 너무 오래걸림
            # 일단 원본으로 저장해두고, 나중에 압축 고려할 예정
            
            # with gzip.open(path_pkl, "wb") as _pkl: 
            with open(path_pkl, mode="wb") as _pkl:
                pickle.dump(list_items_buffer, _pkl)
            
            print("\t-> it took:", round(time.time() - timer_pkl, 5), "sec")
            
            # with gzip.open(path_pkl, "rb") as _pkl:
            with open(path_pkl, mode="rb") as _pkl:
                dataloader_input = pickle.load(_pkl)
                _count = 0
                _count_max = len(dataloader_input)
                for i_items in dataloader_input:
                    _count += 1
                    _names = i_items[0]       # (tuple) file name
                    print("\rE", i_epoch + epoch_init, "/", HP_EPOCH + epoch_init - 1,"- B", _count, "/", _count_max, "-", _names[0], end="")
                    
                    if _count == 1:
                        # augm info for train
                        info_augm = i_items[3]
                        
                        # degraded info for LR images
                        try:
                            dl_str_info_deg = i_items[9]
                            if dl_str_info_deg[0] == False:
                                dl_str_info_deg = None
                        except:
                            dl_str_info_deg = None
                        
                        dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = i_items[1])
                        
                        try:
                            dl_pil_img_lr   = tensor_2_list_pils_v1(in_tensor = i_items[7])
                        except:
                            dl_pil_img_lr   = None
                        
                        try:
                            dl_pil_lab_hr   = tensor_2_list_pils_v1(in_tensor = i_items[4])
                        except:
                            dl_pil_lab_hr   = None
                        
                        for i_image in range(len(_names)):
                            if dl_str_info_deg is None:
                                info_deg = None
                            else:
                                info_deg = dl_str_info_deg[i_image]
                            
                            pil_img_hr = dl_pil_img_hr[i_image]
                            
                            if dl_pil_img_lr is None:
                                pil_img_lr = None
                            else:
                                pil_img_lr = dl_pil_img_lr[i_image]
                            
                            if dl_pil_lab_hr is None:
                                pil_lab_hr = None
                            else:
                                # pil_lab_hr = dl_pil_lab_hr[i_image]
                                pil_lab_hr = label_2_RGB(dl_pil_lab_hr[i_image], HP_COLOR_MAP)
                            
                            save_pils(path_out      = PATH_OUT_IMAGE
                                     ,name          = i_mode + "_E_" + str(i_epoch + epoch_init) + "_" + _names[i_image]
                                     ,info_augm     = info_augm[i_image]
                                     ,info_deg      = info_deg
                                     ,pil_img_hr    = pil_img_hr
                                     ,pil_img_lr    = pil_img_lr
                                     ,pil_lab_hr    = pil_lab_hr
                                     )
            
            print("      \t-> Verified:", path_pkl.split("/")[-1])
        
    print("\n\n---[ pkl_maker finished ]---\n")

#=== End of pkl_mkr_

if __name__ == "__main__":
    pass

