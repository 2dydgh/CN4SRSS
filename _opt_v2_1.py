#_opt_v2.py
# v 2.1 updated: main_m_1_47_t_1_1.py
#변수명 변경됨 (HP_DG_NOISE_SIGMA -> HP_DG_RANGE_NOISE_SIGMA)
import time
import datetime as dt
import numpy as np
import os
import sys
import warnings
import argparse
import matplotlib

try:
    matplotlib.use("Agg")
except:
    pass

_str = "matplotlib backend: " + str(matplotlib.get_backend())
warnings.warn(_str)

RUN_WHERE = -9

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#--- argparse (in _opt_v2.py) -> 현 파일에서 검증작업 시행
parser_options = argparse.ArgumentParser(description='_options')
# overwrite: bool (True, False)
parser_options.add_argument("--overwrite",  type = str2bool,    default = False,    help = "set this True to allow result overwrite")
# name: any string
parser_options.add_argument("--name",       type = str,         default = None,     help = "input name for result saving folder")
# dataset: "CamVid", "MiniCity", "CamVid_5Fold"
parser_options.add_argument("--dataset",    type = str,         default = None,     help = "input name for dataset")
# fold: "CamVid", "MiniCity": "A_set", "B_set" / "CamVid_5Fold": "A_set", "B_set", "C_set", "D_set", "E_set"
parser_options.add_argument("--fold",       type = str,         default = None,     help = "input name for fold")

#--- argparse (in main.py) -> 현 파일에서 검증작업 생략
# ssn: "D3P": "a", "DABNet": "b", "CGNet": "c"
parser_options.add_argument("--ssn",        type = str,         default = None,     help = "input name for semantic segmentation work")
# kd_mode: "kd_origin", "kd_fitnet", "kd_at", "kd_fsp", "kd_fakd", "kd_p_1", "kd_p_2"
parser_options.add_argument("--kd_mode",    type = str,         default = None,     help = "input name for knowledge distillation mode")

# wce_alpha, wce_beta (proposed_loss_ss의 weighted cross entropy에서 weight 계산수식의 alpha, beta 값)
parser_options.add_argument("--wce_alpha",  type = float,       default = 1.0,      help = "proposed_loss_ss의 weighted cross entropy에서 weight 계산수식의 alpha 값")
parser_options.add_argument("--wce_beta",   type = float,       default = 1.0,      help = "proposed_loss_ss의 weighted cross entropy에서 weight 계산수식의 beta 값")


# HP_MIXUP_A (kd_sr_2_ss의 SR(T, alpha)와 SR(S, 1-alpha)의 MIXUP 값)
parser_options.add_argument("--mixup_a",    type = float,       default = 0.5,      help = "HP_MIXUP_A (kd_sr_2_ss의 SR(T, alpha)와 SR(S, 1-alpha)의 MIXUP 값)")


#*****************
#(데이터셋 경로)
#PATH_BASE_IN + NAME_FOLDER_DATASET + "/" + NAME_FOLDER_(TRAIN/VAL/TEST)
#*****************

print("init _options.py")

# dt_now = dt.datetime.now()

#코드 테스트 목적인 경우, True
#is_sample = True
is_sample = False

#---(HP 0) 경로 설정 관련

#(str) 실행 날짜 기록
# HP_INIT_DATE_TIME = str(dt_now.year) + " 년 " + str(dt_now.month) + " 월 " + str(dt_now.day) + " 일"
time_kr = time.gmtime(time.time() + 3600*9) # 한국 표준시 KST = 협정 세계시 UTC + 09:00
HP_INIT_DATE_TIME = time.strftime("%Y Y - %m M - %d D - %H h - %M m - %S s", time_kr)

#[수정 가능 변수]--------------------

# 사용할 데이터셋 (정식지원 목록: CamVid, MiniCity / 그 외 데이터셋은 양식 맞춰서 이 파일에 설정 입력하기)
#HP_DATASET_NAME = "Dataset_name"
HP_DATASET_NAME = "CamVid"
#HP_DATASET_NAME = "MiniCity"
#HP_DATASET_NAME = "CamVid_5Fold"



# 결과물 저장 폴더 이름
#   -9: 노트북 (no GPU)
if RUN_WHERE == -9:
    HP_NUM_WORKERS      = 0
    PATH_BASE           = "C:/LAB/"
    NAME_FOLDER_PROJECT = "this_is_sample"
    is_sample           = True
    
#   -1: colab (vram 12 ~)
elif RUN_WHERE == -1:
    HP_NUM_WORKERS      = 0
    PATH_BASE           = "/content/LAB/"
    NAME_FOLDER_PROJECT = "name_project"
    
#   0: 새 컴퓨터 (vram 8)
elif RUN_WHERE == 0:
    HP_NUM_WORKERS      = 3
    PATH_BASE           = "D:/LAB/"
    NAME_FOLDER_PROJECT = "name_project"
    
#   1: 집 컴퓨터 (vram 12)
elif RUN_WHERE == 1:
    HP_NUM_WORKERS      = 2
    PATH_BASE           = "E:/LAB/"
    NAME_FOLDER_PROJECT = "name_project"
    
#   2:연구실 컴퓨터 (vram 8)
elif RUN_WHERE == 2:
    HP_NUM_WORKERS      = 2
    PATH_BASE           = "/root/LAB/"
    NAME_FOLDER_PROJECT = "name_project"
    
#   3: 고양이 컴퓨터 (vram 12)
elif RUN_WHERE == 3:
    HP_NUM_WORKERS      = 3
    PATH_BASE           = "D:/LAB/"
    NAME_FOLDER_PROJECT = "name_project"

#   4: 서버
elif RUN_WHERE == 4:
    HP_NUM_WORKERS      = 4
    PATH_BASE           = "/scratch/hpc111a06/bong/LAB/"
    NAME_FOLDER_PROJECT = "name_project"




#--- argparse -> parser
args_options = parser_options.parse_args()

# args_options.overwrite
if args_options.overwrite not in [True, False]:
    _str = "Wrong parser.dataset, received " + str(args_options.overwrite)
    warnings.warn(_str)
    sys.exit(-9)

# args_options.name
if args_options.name is not None:
    NAME_FOLDER_PROJECT = str(args_options.name)
    
#args_options.dataset
if args_options.dataset is not None:
    if args_options.dataset in ["CamVid", "MiniCity", "CamVid_5Fold"]:
        HP_DATASET_NAME = str(args_options.dataset)
    else:
        _str = "Wrong parser.dataset, received " + str(args_options.dataset)
        warnings.warn(_str)
        sys.exit(-9)

#args_options.fold
if args_options.fold is not None:
    if args_options.fold not in ["A_set", "B_set", "C_set", "D_set", "E_set"]:
        _str = "Wrong parser.fold, received " + str(args_options.fold)
        warnings.warn(_str)
        sys.exit(-9)
    else:
        if args_options.fold not in ["A_set", "B_set"]:
            if HP_DATASET_NAME not in ["CamVid_5Fold"]:
                _str = "Wrong parser.fold, received " + str(args_options.fold)
                _str += "\nCurrnet dataset is " + HP_DATASET_NAME
                warnings.warn(_str)
                sys.exit(-9)





#---


# 저장할 최대 이미지 배치 수 (Val & Test 에서는 REDUCE_SAVE_IMAGES is True 인 경우에만 적용됨)
# colab 환경으로 판단되면 자동으로 1로 수정됨
MAX_SAVE_IMAGES = 2

# Test 과정에서 저장할 이미지 수를 줄일 것인가? (bool)
REDUCE_SAVE_IMAGES = True

_str = "사용될 데이터셋 이름: " + HP_DATASET_NAME
warnings.warn(_str)

if HP_DATASET_NAME == "CamVid":
    
    # Test 과정에서 반드시 저장할 이미지 이름 (list with str)
    MUST_SAVE_IMAGE = ["0016E5_08123.png"    # CamVid 12 v4 Fold A
                      ,"0016E5_08145.png"    # CamVid 12 v4 Fold B
                      ]
    
    # 불러올 데이터셋 폴더 이름
    # NAME_FOLDER_DATASET = "name_dataset"
    if args_options.fold is not None:
        NAME_FOLDER_DATASET = "CamVid_12_2Fold_v4/" + args_options.fold
    else:
        NAME_FOLDER_DATASET = "CamVid_12_2Fold_v4/A_set"
        # NAME_FOLDER_DATASET = "CamVid_12_2Fold_v4/B_set"
    
    # 대체할 HR Image 폴더
    NAME_FOLDER_ALTER_HR_IAMGE = None
    # NAME_FOLDER_ALTER_HR_IAMGE = "CamVid_12_ALTER_HR_Image/x4_BILINEAR/AB_set"     # LR 이미지 interpolation (BILINEAR) -> 고전 방식이라 Fold 구분 안함
    
    # 추가로 불러올 폴더 이름
    # NAME_FOLDER_DATASET_SUB = "name_dataset_DLC"
    NAME_FOLDER_DATASET_SUB = "CamVid_12_DLC_v1/x4_BILINEAR"
    
    # Label Verify 설정 (Train에만 적용, Label Crop 내 class 불균형 방지 목적, pil_marginer_v3에 적용되는 값)
    HP_LABEL_VERIFY                = False              # (bool) Label Verify 시행 여부
    HP_LABEL_VERIFY_TRY_CEILING    = 10                 # (int) Crop Retry 최대 횟수
    HP_LABEL_VERIFY_CLASS_MIN      = 6                  # (int) void 포함 최소 class 종류
    HP_LABEL_VERIFY_RATIO_MAX      = 0.6                # (float) class 비율값의 상한값 (0 ~ 1.0)
    
elif HP_DATASET_NAME == "MiniCity":
    
    MUST_SAVE_IMAGE = ["aachen_000045_000019.png"       # MiniCity 19 v1 Fold A
                      ,"bremen_000249_000019.png"       # MiniCity 19 v1 Fold A
                      ,"dusseldorf_000216_000019.png"   # MiniCity 19 v1 Fold A
                      ,"aachen_000030_000019.png"       # MiniCity 19 v1 Fold B
                      ,"bremen_000160_000019.png"       # MiniCity 19 v1 Fold B
                      ,"darmstadt_000055_000019.png"    # MiniCity 19 v1 Fold B
                      ]
    
    if args_options.fold is not None:
        NAME_FOLDER_DATASET = "MiniCity_19_2Fold_v1/" + args_options.fold
    else:
        NAME_FOLDER_DATASET = "MiniCity_19_2Fold_v1/A_set"
        #NAME_FOLDER_DATASET = "MiniCity_19_2Fold_v1/B_set"
    
    NAME_FOLDER_ALTER_HR_IAMGE = None
    
    NAME_FOLDER_DATASET_SUB = "MiniCity_19_DLC_v1/x4_BILINEAR"
    
    HP_LABEL_VERIFY                = True
    HP_LABEL_VERIFY_TRY_CEILING    = 10
    HP_LABEL_VERIFY_CLASS_MIN      = 6
    HP_LABEL_VERIFY_RATIO_MAX      = 0.6
    

elif HP_DATASET_NAME == "CamVid_5Fold":
    
    MUST_SAVE_IMAGE = ["0016E5_08123.png"    # CamVid 5 Fold v1 Fold A
                      ,"0016E5_08155.png"    # CamVid 5 Fold v1 Fold B
                      ,"0016E5_08147.png"    # CamVid 5 Fold v1 Fold C
                      ,"0016E5_08159.png"    # CamVid 5 Fold v1 Fold D
                      ,"0016E5_08151.png"    # CamVid 5 Fold v1 Fold E
                      ]
    
    if args_options.fold is not None:
        NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/" + args_options.fold
    else:
        NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/A_set"
        #NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/B_set"
        #NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/C_set"
        #NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/D_set"
        #NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/E_set"
    
    NAME_FOLDER_ALTER_HR_IAMGE = None
    
    NAME_FOLDER_DATASET_SUB = "CamVid_12_DLC_v1/x4_BILINEAR"
    
    HP_LABEL_VERIFY                = False
    HP_LABEL_VERIFY_TRY_CEILING    = 10
    HP_LABEL_VERIFY_CLASS_MIN      = 6
    HP_LABEL_VERIFY_RATIO_MAX      = 0.6
    
else:
    sys.exit("지원하지 않는 데이터셋 입니다.")

NAME_FOLDER_PROJECT += "_" + NAME_FOLDER_DATASET.split("/")[0] + "_" + NAME_FOLDER_DATASET.split("/")[-1]

if is_sample:
    print("=== === === [Sample Run] === === ===")
    NAME_FOLDER_DATASET = "Sample_set/A_set"
    NAME_FOLDER_ALTER_HR_IAMGE = None
    NAME_FOLDER_DATASET_SUB = "Sample_set_DLC/x4_BILINEAR"
    NAME_FOLDER_PROJECT = "Sample_set_A_set"

# [고정 변수] ------------------------
# 입출력 공통 폴더명 구성
NAME_FOLDER_TRAIN   = "train"
NAME_FOLDER_VAL     = "val"
NAME_FOLDER_TEST    = "test"

NAME_FOLDER_IMAGES  = "images"
NAME_FOLDER_LABELS  = "labels"

# RUN_WHERE 값 확인용 -> 현재 사용중인 LAB 폴더 바로 아래에 "_RUN_on_{번호}" 폴더 생성해두기
if not os.path.isdir(PATH_BASE + "_RUN_on_" + str(RUN_WHERE)):
    print("Wrong [RUN_WHERE] input:", RUN_WHERE)
    sys.exit(9)
else:
    print("RUN on", RUN_WHERE, PATH_BASE)


PATH_BASE_IN            = PATH_BASE + "datasets/project_use/" + NAME_FOLDER_DATASET         + "/"
try:
    PATH_ALTER_HR_IMAGE = PATH_BASE + "datasets/project_use/" + NAME_FOLDER_ALTER_HR_IAMGE  + "/"
except:
    PATH_ALTER_HR_IMAGE = None
PATH_BASE_IN_SUB        = PATH_BASE + "datasets/project_use/" + NAME_FOLDER_DATASET_SUB     + "/"
PATH_BASE_OUT           = PATH_BASE + "result_files/"         + NAME_FOLDER_PROJECT         + "/"

# colab
if RUN_WHERE == -1:
    print("RUN on Online (colab)")
    PATH_BASE_OUT        = "/content/drive/MyDrive/Colab_Results/"      + NAME_FOLDER_PROJECT + "/"
    PATH_COLAB_OUT_LOG   = "/content/drive/MyDrive/Colab_Results/_sub/" + NAME_FOLDER_PROJECT + "/logs/"
    PATH_COLAB_OUT_MODEL = "/content/drive/MyDrive/Colab_Results/_sub/" + NAME_FOLDER_PROJECT + "/models/"
else:
    print("RUN on Local")
    PATH_COLAB_OUT_LOG   = "False"
    PATH_COLAB_OUT_MODEL = "False"

#log 저장 폴더 경로
PATH_OUT_LOG = PATH_BASE_OUT + "logs/"
#모델 CP & SD 저장 폴더 경로
PATH_OUT_MODEL = PATH_BASE_OUT + "models/"
#생성된 이미지 저장 폴더 경로
PATH_OUT_IMAGE = PATH_BASE_OUT + "images/"


#[hyper parameters]----------------------------------------

#HP_BATCH_VAL = 1 #고정값 (1)
#HP_BATCH_TEST  = 1 #고정값 (1)


#---(HP 3) 랜덤 관련: 랜덤 시드값, 정규분포(랜덤) 생성 시 사용되는 시그마 값

#랜덤 시드값 (pytorch, numpy rand)
#PR_SEED -> HP_SEED
HP_SEED = 15

# 데이터셋 이미지 관련
if HP_DATASET_NAME == "CamVid":
    #---(HP 4) 원본 이미지 크기
    # 원본 이미지 크기 (dataset train & val & test)
    #HP_ORIGIN_IMG_W, HP_ORIGIN_IMG_H = 480, 360
    
    #---(HP 5) Semantic Segmentation 라벨 정보 (CamVid 12)
    
    HP_LABEL_TOTAL          = 12                        # (int) 총 라벨 수 (void 포함)
    HP_LABEL_VOID           = 11                        # (int) void 라벨값 -> void 라벨 없으면 None 주면 됨
    HP_LABEL_ONEHOT_ENCODE  = False                      # (bool) label one-hot encoding 시행여부
    
    HP_DATASET_CLASSES = ("0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"  # (str) log의 class 정보. None 입력시 자동완성기능 적용됨
                         +"6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
                         )
    
    # 라벨 당 컬러 지정 (CamVid 12) -> 라벨 데이터 미 사용시, None 입력 가능.
    HP_COLOR_MAP = {0:  [128, 128, 128]     # 00 Sky
                   ,1:  [128,   0,   0]     # 01 Building
                   ,2:  [192, 192, 128]     # 02 Column_pole
                   ,3:  [128,  64, 128]     # 03 Road
                   ,4:  [  0,   0, 192]     # 04 Sidewalk
                   ,5:  [128, 128,   0]     # 05 Tree
                   ,6:  [192, 128, 128]     # 06 SignSymbol
                   ,7:  [ 64,  64, 128]     # 07 Fence
                   ,8:  [ 64,   0, 128]     # 08 Car
                   ,9:  [ 64,  64,   0]     # 09 Pedestrian
                   ,10: [  0, 128, 192]     # 10 Bicyclist
                   ,11: [  0,   0,   0]     # 11 Void
                   }
    
elif HP_DATASET_NAME == "MiniCity":
    
    #HP_ORIGIN_IMG_W, HP_ORIGIN_IMG_H = 2048, 1024
    
    HP_LABEL_TOTAL          = 20
    HP_LABEL_VOID           = 19
    HP_LABEL_ONEHOT_ENCODE  = False
    
    HP_DATASET_CLASSES = ("0(Road),1(Sidewalk),2(Building),3(Wall),4(Fence),"
                         +"5(Pole),6(Traffic_light),7(Traffic_sign),8(Vegetation),9(Terrain),"
                         +"10(Sky),11(Person),12(Rider),13(Car),14(Truck),"
                         +"15(Bus),16(Train),17(Motorcycle),18(Bicycle)"
                         )
    
    HP_COLOR_MAP = {0:  [128,  64, 128]     # 00 Road
                   ,1:  [244,  35, 232]     # 01 Sidewalk
                   ,2:  [ 70,  70,  70]     # 02 Building
                   ,3:  [102, 102, 156]     # 03 Wall
                   ,4:  [190, 153, 153]     # 04 Fence
                   ,5:  [153, 153, 153]     # 05 Pole
                   ,6:  [250, 170,  30]     # 06 Traffic light
                   ,7:  [220, 220,   0]     # 07 Traffic sign
                   ,8:  [107, 142,  35]     # 08 Vegetation
                   ,9:  [152, 251, 152]     # 09 Terrain
                   ,10: [ 70, 130, 180]     # 10 Sky
                   ,11: [220,  20,  60]     # 11 Person
                   ,12: [255,   0,   0]     # 12 Rider
                   ,13: [  0,   0, 142]     # 13 Car
                   ,14: [  0,   0,  70]     # 14 Truck
                   ,15: [  0,  60, 100]     # 15 Bus
                   ,16: [  0,  80, 100]     # 16 Train
                   ,17: [  0,   0, 230]     # 17 Motorcycle
                   ,18: [119,  11,  32]     # 18 Bicycle
                   ,19: [  0,   0,   0]     # 19 Void
                   }

elif HP_DATASET_NAME == "CamVid_5Fold":
    #HP_ORIGIN_IMG_W, HP_ORIGIN_IMG_H = 480, 360
    
    HP_LABEL_TOTAL          = 12
    HP_LABEL_VOID           = 11
    HP_LABEL_ONEHOT_ENCODE  = False
    
    HP_DATASET_CLASSES = ("0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"
                         +"6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
                         )
    
    HP_COLOR_MAP = {0:  [128, 128, 128]     # 00 Sky
                   ,1:  [128,   0,   0]     # 01 Building
                   ,2:  [192, 192, 128]     # 02 Column_pole
                   ,3:  [128,  64, 128]     # 03 Road
                   ,4:  [  0,   0, 192]     # 04 Sidewalk
                   ,5:  [128, 128,   0]     # 05 Tree
                   ,6:  [192, 128, 128]     # 06 SignSymbol
                   ,7:  [ 64,  64, 128]     # 07 Fence
                   ,8:  [ 64,   0, 128]     # 08 Car
                   ,9:  [ 64,  64,   0]     # 09 Pedestrian
                   ,10: [  0, 128, 192]     # 10 Bicyclist
                   ,11: [  0,   0,   0]     # 11 Void
                   }

#이미지 별 채널 수
#HP_CHANNEL_RGB = 3                   #RGB 이미지 (입력 이미지)
#HP_CHANNEL_GRAY = 1                  #GRAY 이미지 (라벨 이미지)

#---(HP 7) Data Augm- & ColorJitter 관련
#각 축의 crop 시작좌표 범위 (min, max)
HP_AUGM_RANGE_CROP_INIT = (4, 10)
#회전각도 최대값
HP_AUGM_ROTATION_MAX = 5
#flip 확률
HP_AUGM_PROB_FLIP = 50
#crop 확률
HP_AUGM_PROB_CROP = 70
#rotation 확률
HP_AUGM_PROB_ROTATE = 20

#torchvision transform ColorJitter 옵션
HP_CJ_BRIGHTNESS = (0.6, 1.4)
HP_CJ_CONTRAST   = (0.7, 1.2)
HP_CJ_SATURATION = (0.9, 1.3)
HP_CJ_HUE        = (-0.05, 0.05)

#---(HP 9) Degradation 관련
if HP_DATASET_NAME == "CamVid":
    # Degradation 고정값 (val & test) csv 파일 이름 & 경로 (dataset의 A_set, B_set 폴더에 모두 넣으면 됨)
    HP_DG_CSV_NAME = "degradation_2.csv" # 확장자 .csv 까지 작성할것
    
elif HP_DATASET_NAME == "MiniCity":
    HP_DG_CSV_NAME = "degradation_MiniCity.csv"
    
elif HP_DATASET_NAME == "CamVid_5Fold":
    HP_DG_CSV_NAME = "degradation_2.csv" 

HP_DG_CSV_PATH = PATH_BASE_IN_SUB + HP_DG_CSV_NAME # 고정값

# Scale Factor (배율값, 이 값으로 이미지 길이를 나눠서 Down Sampling 시행) -> x4 배율로 고정됨
HP_DG_SCALE_FACTOR = 4

# resize 방식 ("NEAREST", "BILINEAR", "BICUBIC", "LANCZOS" 중 하나 선택)
HP_DG_RESIZE_OPTION = "BILINEAR"
# 추후에 list 형으로 입력시, 랜덤하게 선택하는 기능을 추가할지 고민중

# Gaussian 노이즈 시그마 범위
HP_DG_RANGE_NOISE_SIGMA = (1, 30)

# 노이즈 종류 (Gray(V) or Color(H or S) 중 Gray 노이즈 확률
HP_DG_NOISE_GRAY_PROB = 40


print("inputs...")
print("dataset (PATH_BASE_IN):", PATH_BASE_IN)
print("degradation_2 (HP_DG_CSV_PATH): ", HP_DG_CSV_PATH)

print("outputs...")
print("logs (PATH_OUT_LOG):", PATH_OUT_LOG)
print("models (PATH_OUT_MODEL):", PATH_OUT_MODEL)
print("images (PATH_OUT_IMAGE):", PATH_OUT_IMAGE)


print("EOF: _opt_v2.py")


"""
#폴더 형태
"Drive Name"
    LAB
        _RUN_on_N
        datasets
            project_use
                "name_dataset"
                    train
                        images
                        labels
                    val
                        images
                        labels
                    test
                        images
                        labels
                        
                "name_dataset_DLC"
                    "name_sub"
                        images
                        labels
                
        result_files
            "name_project"
                logs
                    train
                    val
                    test
                models
                    chech_point
                    state_dict
                images
                    train
                        "epoch_number"
                    val
                        "epoch_number"
                    test
                        "epoch_number"
        for_tests
            model_state_dict
"""