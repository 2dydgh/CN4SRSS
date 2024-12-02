# main_sr_train.py

if __name__ == '__main__':
    from DLCs.super_resolution.model_esrt   import ESRT   # loss: torch.nn.L1Loss()
    from trainers.trainer_total             import *
    from _options                           import *

    #[init]----------------------------
    #Prevent overwriting results
    if os.path.isdir(PATH_BASE_OUT):
        print("실험결과 덮어쓰기 방지기능 작동됨 (Prevent overwriting results function activated)")
        sys.exit("Prevent overwriting results function activated")
    
    # 체크포인트 경로 + 파일명 + 확장자  ("False" 입력시, 자동으로 미입력 처리됨)
    path_check_point = "False"
    
    # (float) 이전 실행시 best score (SR -> PSNR 점수)
    prev_best = None
    
    # log dicts reset
    dict_log_init = {}
    # set computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    update_dict_v2("", "---< init >---"
                  ,"", "실험 날짜: " + HP_INIT_DATE_TIME
                  ,"", "--- Dataset Parameter info ---"
                  ,"", "데이터셋 이름: " + HP_DATASET_NAME
                  ,"", "Dataset from... " + PATH_BASE_IN
                  ,"", ""
                  ,"", "--- Hyper Parameter info ---"
                  ,"", "Device:" + str(device)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [Input HR Image]------------------------
    # 이미지 교체여부 관련
    path_alter_hr_image = None                      # 원본 이미지 사용
    
    update_dict_v2("", ""
                  ,"", "HR Image 관련 정보"
                  ,"", "원본 HR 이미지 사용됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    #[model_sr & Loss]------------------------
    #모델 입력 시 정규화 여부
    # True:  
    # False: MPRNet, ESRT, IMDN, BSRN, RFDN
    
    model_sr_name = "ESRT"

    #ESRT
    model_sr = ESRT(upscale=4)
    update_dict_v2("", ""
                  ,"", "모델 정보: " + model_sr_name
                  ,"", "(github) ESRT"
                  ,"", "pretrained = False"
                  ,"", "upsample = 4"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    criterion_sr = torch.nn.L1Loss()
    update_dict_v2("", "loss 정보"
                  ,"", "loss: L1 Loss"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    is_norm_in_transform_to_tensor  = False
    HP_TS_NORM_MEAN                 = None
    HP_TS_NORM_STD                  = None
    
    
    model_sr.to(device)
    
    #[optimizer]------------------------
    HP_LR = 2e-4                    # ESRT official: 2 x 10^-4
    
    #weight decay를 사용하지 않는 경우
    optimizer = torch.optim.Adam(model_sr.parameters()
                                ,lr=HP_LR
                                )
    update_dict_v2("", ""
                  ,"", "옵티마이저 정보"
                  ,"", "optimizer: " + "torch.optim.Adam"
                  ,"", "learning_rate: " + str(HP_LR)
                  ,"", "weight decay 적용 안됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    #[scheduler]----------------------------------------------
    #https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    
    option_scheduer = "Step"    # StepLR    ESRT official
    
    if option_scheduer == "Step":
        HP_EPOCH                        = 3002          # 학습 종료 epoch
        HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"       # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
        
        HP_SCHEDULER_STEP_SIZE          = 200           # 반감기     ESRT official: 200
        HP_SCHEDULER_GAMMA              = 0.5           # Halfed
        
        scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer
                                             ,step_size = HP_SCHEDULER_STEP_SIZE
                                             ,gamma     = HP_SCHEDULER_GAMMA
                                             )
        
        update_dict_v2("", ""
                          ,"", "스케쥴러 정보"
                          ,"", "업데이트 간격: " + HP_SCHEDULER_UPDATE_INTERVAL
                          ,"", "scheduler: " + "Step (optim.lr_scheduler.StepLR)"
                          ,"", "step size: " + str(HP_SCHEDULER_STEP_SIZE)
                          ,"", "gamma: "     + str(HP_SCHEDULER_GAMMA)
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        
    #=========================================================================================

    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
             # Trainer 모드 설정 ("SR", "SSSR")
             TRAINER_MODE                   = "SR"
             
             # 코드 실행 장소 (colab 여부 확인용, colab == -1)
            ,RUN_WHERE                      = RUN_WHERE
            
             # 사용할 데이터 셋
            ,HP_DATASET_NAME                = HP_DATASET_NAME
            
             # Test 과정에서 저장할 이미지 수를 줄일 것인가?
            ,REDUCE_SAVE_IMAGES             = REDUCE_SAVE_IMAGES
             # Test 과정에서 반드시 저장할 이미지 이름
            ,MUST_SAVE_IMAGE                = MUST_SAVE_IMAGE
             # 저장할 최대 이미지 수 (Val & Test 에서는 REDUCE_SAVE_IMAGES is True 인 경우에만 적용됨)
            ,MAX_SAVE_IMAGES                = MAX_SAVE_IMAGES
            
            
             #(선택) 체크포인트 불러오기 ("False" 입력시, 자동으로 미입력 처리됨)
            ,path_check_point               = path_check_point
            
             #(선택) 이전 실행 시 best score
            ,prev_best                      = prev_best
            
            # 버퍼 크기 (int) -> 버퍼 안쓰는 옵션 사용불가. 양수값 가능
            ,BUFFER_SIZE                    = 60
             # 초기화 기록 dict 이어받기
            ,dict_log_init                  = dict_log_init
             # 랜덤 시드 고정
            ,HP_SEED                        = HP_SEED
            
            
             # 학습 관련 기본 정보(epoch 수, batch 크기(train은 생성할 patch 수), 학습 시 dataset 루프 횟수)
            ,HP_EPOCH                       = HP_EPOCH
            ,HP_BATCH_TRAIN                 = 16                # ESRT official: 16 (HR 기준 192x192 이미지에 VRAM 6GB 미만 사용됨)
            ,HP_BATCH_VAL                   = 1
            ,HP_BATCH_TEST                  = 1
            ,HP_NUM_WORKERS                 = HP_NUM_WORKERS
            ,HP_VALID_WITH_PATCH            = True
            
             # 데이터 입출력 경로, 폴더명
            ,PATH_BASE_IN                   = PATH_BASE_IN
            ,NAME_FOLDER_TRAIN              = NAME_FOLDER_TRAIN
            ,NAME_FOLDER_VAL                = NAME_FOLDER_VAL
            ,NAME_FOLDER_TEST               = NAME_FOLDER_TEST
            ,NAME_FOLDER_IMAGES             = NAME_FOLDER_IMAGES
            ,NAME_FOLDER_LABELS             = NAME_FOLDER_LABELS
            
            # (선택) HR Image 교체용 폴더 경로 (default: None)
            ,PATH_ALTER_HR_IMAGE            = path_alter_hr_image
            
             # (선택) degraded image 불러올 경로
            ,PATH_BASE_IN_SUB               = PATH_BASE_IN_SUB
            
            ,PATH_OUT_IMAGE                 = PATH_OUT_IMAGE
            ,PATH_OUT_MODEL                 = PATH_OUT_MODEL
            ,PATH_OUT_LOG                   = PATH_OUT_LOG
            
             # Patch 생성 관련 강제 Margin 추가여부 (bool, is_use_patch와 동시 사용불가)
            ,is_force_fix                   = True
            ,force_fix_size_hr              = (192, 192)    # ESRT - official 설정 : 192x192 (LR 이미지 기준 48x48)
            
             # 모델 이름 -> 모델에 따라 예측결과 형태가 다르기에 모델입출력물을 조정하는 역할
             # 지원 리스트 = "MPRNet"
            ,model_name                     = model_sr_name
            
             # model, optimizer, scheduler, loss
            ,model                          = model_sr
            ,optimizer                      = optimizer
            ,scheduler                      = scheduler
            ,criterion                      = criterion_sr
             # 스케쥴러 업데이트 간격("epoch" 또는 "batch")
            ,HP_SCHEDULER_UPDATE_INTERVAL   = HP_SCHEDULER_UPDATE_INTERVAL
            
            
             # DataAugm- 관련 (colorJitter 포함)
            ,HP_AUGM_LITE                   = True                      # SR 모드에서 lite 모드 augm 시행여부 (ESRT official: True)
            ,HP_AUGM_LITE_FLIP_HORI         = True                      # FLIP: 수평방향
            ,HP_AUGM_LITE_FLIP_VERT         = False                     # FLIP: 수직방향
            
             # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
            ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
            ,HP_TS_NORM_MEAN                = HP_TS_NORM_MEAN
            ,HP_TS_NORM_STD                 = HP_TS_NORM_STD
            
            
             # Degradation 관련 설정값
            ,HP_DG_CSV_NAME                 = HP_DG_CSV_NAME
            ,HP_DG_SCALE_FACTOR             = 4
            )
    
    
    print("End of main_sr_train.py")
