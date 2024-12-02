# main_sr_train.py

"""

v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의
v 3.0.0 에 맞게 수정 안되었으니, 사용시 주의

"""
if __name__ == '__main__':
    from DLCs.model_fpenet                  import FPENet
    from utils.schedulers                   import PolyLR, Poly_Warm_Cos_LR
    from trainers.trainer_total             import *
    from _options                           import *

    # [init]----------------------------
    #Prevent overwriting results
    if os.path.isdir(PATH_BASE_OUT):
        print("실험결과 덮어쓰기 방지기능 작동됨 (Prevent overwriting results function activated)")
        sys.exit("Prevent overwriting results function activated")
    
    # 체크포인트 경로 + 파일명 + 확장자  ("False" 입력시, 자동으로 미입력 처리됨)
    path_check_point = "False"
    
    # (float) 이전 실행시 best score (SS -> mIoU 점수)
    prev_best = None
    
    #log dicts reset
    dict_log_init = {}
    # set computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    update_dict_v2("", "---< init >---"
                  ,"", "실험 날짜: " + HP_INIT_DATE_TIME
                  ,"", "--- Dataset Parameter info ---"
                  ,"", "Dataset from... " + PATH_BASE_IN
                  ,"", ""
                  ,"", "--- Hyper Parameter info ---"
                  ,"", "Device:" + str(device)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    
    # [Input HR Image]------------------------
    # 이미지 교체여부 관련
    #path_alter_hr_image = None                      # 원본 이미지 사용
    path_alter_hr_image = PATH_ALTER_HR_IMAGE       # 대체 이미지 사용 -> 경로는 options에서 수정할것
    
    
    if path_alter_hr_image is None:
        update_dict_v2("", ""
                      ,"", "HR Image 관련 정보"
                      ,"", "원본 HR 이미지 사용됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        update_dict_v2("", ""
                      ,"", "HR Image 관련 정보"
                      ,"", "HR 이미지 교체됨"
                      ,"", "path: " + path_alter_hr_image
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [model_ss & Loss]------------------------
    #모델 입력 시 정규화 여부
    
    model_ss_name = "FPENet"   # FPENet
    
    model_ss = FPENet(classes=11)
    update_dict_v2("", ""
                  ,"", "모델 정보: " + model_ss_name
                  ,"", "(github) FPENet"
                  ,"", "pretrained = False"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    criterion_ss = torch.nn.CrossEntropyLoss()
    update_dict_v2("", "loss 정보"
                  ,"", "loss: Cross Entropy Loss"
                  ,"", "loss 가중치 적용여부: 적용 안됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    is_norm_in_transform_to_tensor  = False
    HP_TS_NORM_MEAN                 = None
    HP_TS_NORM_STD                  = None
    
    
    model_ss.to(device)
    
    # [optimizer]------------------------
    
    #option_optimizer = "SGD"
    option_optimizer = "ADAM"           # FPENet official: ADAM
    
    if option_optimizer == "SGD":
        #<<< SGD
        HP_LR          = 4.5e-2
        HP_LR_MOMENTUM = 0.9
        HP_WD          = 1e-4
        optimizer = torch.optim.SGD(model_ss.parameters()
                                   ,lr           = HP_LR
                                   ,momentum     = HP_LR_MOMENTUM
                                   ,weight_decay = HP_WD
                                   )
        
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보"
                      ,"", "optimizer: " + "torch.optim.SGD"
                      ,"", "learning_rate: " + str(HP_LR)
                      ,"", "momentum: " + str(HP_LR_MOMENTUM)
                      ,"", "weight decay 적용됨: " + str(HP_WD)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        #>>> SGD
    elif option_optimizer == "ADAM":
        #<<< ADAM
        HP_LR       = 0.0005          # FPENet official: 0.0005
        HP_WD       = 0.0001          # FPENet official: 0.0001
        
        optimizer = torch.optim.Adam(model_ss.parameters()
                                    ,lr=HP_LR
                                    ,weight_decay = HP_WD
                                    )
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR)
                      ,"", "weight decay 적용됨: " + str(HP_WD)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        #>>> ADAM
    
    
    # [scheduler]----------------------------------------------
    
    option_scheduer = "Poly"                           # FPENet official: Poly
    #option_scheduer = "Cosine"
    #option_scheduer = "Poly_Warm_Cosine"
    
    if option_scheduer == "Poly":
        #<<< Poly
        HP_EPOCH                        = 3002         # 학습 종료 epoch
        HP_SCHEDULER_TOTAL_EPOCH        = 3000         # option for Poly
        HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"      # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
        HP_SCHEDULER_POWER              = 0.9          # FPENet official: 0.9
        scheduler = PolyLR(optimizer, max_epoch=HP_SCHEDULER_TOTAL_EPOCH, power=HP_SCHEDULER_POWER)
        
        update_dict_v2("", ""
                          ,"", "스케쥴러 정보"
                          ,"", "업데이트 간격: " + HP_SCHEDULER_UPDATE_INTERVAL
                          ,"", "scheduler: " + "PolyLR"
                          ,"", "total_epoch: " + str(HP_SCHEDULER_TOTAL_EPOCH)
                          ,"", "power: " + str(HP_SCHEDULER_POWER)
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        #>>> Poly
    elif option_scheduer == "Cosine":
        #<<< Cosine
        HP_EPOCH                        = 5002         # 학습 종료 epoch
        HP_SCHEDULER_T_MAX              = 50            # 반주기
        HP_SCHEDULER_ETA_MIN            = 1e-6          # 최소 LR
        HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"      # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer
                                                        ,T_max = HP_SCHEDULER_T_MAX
                                                        ,eta_min = HP_SCHEDULER_ETA_MIN
                                                        )

        update_dict_v2("", ""
                      ,"", "스케쥴러 정보"
                      ,"", "업데이트 간격: " + HP_SCHEDULER_UPDATE_INTERVAL
                      ,"", "scheduler: " + "optim.lr_scheduler.CosineAnnealingLR"
                      ,"", "LR 최대값 도달 epoch 수 (lr 반복주기의 절반)"
                      ,"", "T_max: " + str(HP_SCHEDULER_T_MAX)
                      ,"", "lr 최소값 (default = 0)"
                      ,"", "eta_min: " + str(HP_SCHEDULER_ETA_MIN)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        #>>> Cosine
    elif option_scheduer == "Poly_Warm_Cosine":
        #<<< Poly with Cosine Warm-up
        HP_EPOCH                        = 5002          # 학습 종료 epoch
        
        HP_SCHEDULER_WARM_STEPS         = 500           # (warm-up cos) warm-up 시행 step 수
        
        HP_SCHEDULER_T_MAX              = 50            # (warm-up cos) 반주기
        HP_SCHEDULER_ETA_MIN            = 1e-6          # (warm-up cos) 최소 LR
        
        HP_SCHEDULER_TOTAL_EPOCH        = 5000          # (poly) LR 0.0 도달 step
        HP_SCHEDULER_POWER              = 0.9           # (poly) power 값
        
        HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"       # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
        
        
        scheduler = Poly_Warm_Cos_LR(optimizer
                                    ,warm_up_steps=HP_SCHEDULER_WARM_STEPS
                                    ,T_max=HP_SCHEDULER_T_MAX, eta_min=HP_SCHEDULER_ETA_MIN
                                    ,power=HP_SCHEDULER_POWER, max_epoch=HP_SCHEDULER_TOTAL_EPOCH
                                    )
        
        update_dict_v2("", ""
                      ,"", "스케쥴러 정보"
                      ,"", "업데이트 간격: "      + HP_SCHEDULER_UPDATE_INTERVAL
                      ,"", "scheduler: "      + "Poly warm-up with Cosine"
                      ,"", "warm-up steps: "  + str(HP_SCHEDULER_WARM_STEPS)
                      ,"", "cosine 반주기: "    + str(HP_SCHEDULER_T_MAX)
                      ,"", "cosine 최소 LR: "  + str(HP_SCHEDULER_ETA_MIN) 
                      ,"", "Poly 최종 step: "  + str(HP_SCHEDULER_TOTAL_EPOCH)
                      ,"", "Poly Power: "     + str(HP_SCHEDULER_POWER)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        #>>> Poly with Cosine Warm-up
    
    
    #=========================================================================================

    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
             # Trainer 모드 설정 ("SS", "SR", "SSSR")
             TRAINER_MODE                   = "SS"
             
             # 코드 실행 장소 (colab 여부 확인용, colab == -1)
            ,RUN_WHERE                      = RUN_WHERE
            
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
            ,HP_BATCH_TRAIN                 = 8                 # FPENet official: Batch 크기 = 8
            ,HP_BATCH_VAL                   = 1
            ,HP_BATCH_TEST                  = 1
            
            
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
            
            
             # 데이터(이미지) 크기 (원본 이미지), 이미지 채널 수(이미지, 라벨, 모델출력물)
            ,HP_ORIGIN_IMG_W                = HP_ORIGIN_IMG_W
            ,HP_ORIGIN_IMG_H                = HP_ORIGIN_IMG_H
            
            ,HP_CHANNEL_RGB                 = HP_CHANNEL_RGB
            ,HP_CHANNEL_GRAY                = HP_CHANNEL_GRAY
            ,HP_CHANNEL_HYPO                = HP_CHANNEL_HYPO
            
             # 라벨 정보(원본 데이터 라벨 수(void 포함), void 라벨 번호, 컬러매핑)
            ,HP_LABEL_TOTAL                 = HP_LABEL_TOTAL
            ,HP_LABEL_VOID                  = HP_LABEL_VOID
            ,HP_COLOR_MAP                   = HP_COLOR_MAP
            
             # Patch 생성 관련 (사용여부, 입력 Patch 크기, stride, 시작좌표범위)
            ,is_use_patch                   = False
            ,HP_MODEL_IMG_W                 = None
            ,HP_MODEL_IMG_H                 = None
            ,HP_PATCH_STRIDES               = None
            ,HP_PATCH_CROP_INIT_COOR_RANGE  = None
            
            # 강제 Margin 추가여부 (bool, is_use_patch와 동시 사용불가)
            ,is_force_fix                   = True
            ,force_fix_size_hr              = (480, 360)                # FPENet official: CamVid (W480, H360) 기준 원본크기 그대로 입력
            
             # 모델 이름 -> 모델에 따라 예측결과 형태가 다르기에 모델입출력물을 조정하는 역할
            ,model_name                     = model_ss_name
            
             # model, optimizer, scheduler, loss
            ,model                          = model_ss
            ,optimizer                      = optimizer
            ,scheduler                      = scheduler
            ,criterion                      = criterion_ss
             # 스케쥴러 업데이트 간격("epoch" 또는 "batch")
            ,HP_SCHEDULER_UPDATE_INTERVAL   = HP_SCHEDULER_UPDATE_INTERVAL
            
             # Label Dilation 설정 (Train에만 적용, default = False)
            ,HP_LABEL_DILATED               = False
            
             # DataAugm- 관련 (colorJitter 포함)
            ,HP_AUGM_RANGE_CROP_INIT        = HP_AUGM_RANGE_CROP_INIT
            ,HP_AUGM_ROTATION_MAX           = 10                        # FPENet official: Augm에 - 10' ~ 10' 회전 적용
            ,HP_AUGM_PROB_FLIP              = 50                        # FPENet official: Augm에 horizontal flip 적용 (확률 언급 x)
            ,HP_AUGM_PROB_CROP              = -1                        # FPENet official: Augm에 원본 이미지에 대한 random crop X
            ,HP_AUGM_PROB_ROTATE            = 50                        # FPENet official: Augm에 random rotation 시행 (확률 언급 x)
            ,HP_CJ_BRIGHTNESS               = [1,1]                     # FPENet official: Augm에 colorJitter X
            ,HP_CJ_CONTRAST                 = [1,1]                     # FPENet official: Augm에 colorJitter X
            ,HP_CJ_SATURATION               = [1,1]                     # FPENet official: Augm에 colorJitter X
            ,HP_CJ_HUE                      = [0,0]                     # FPENet official: Augm에 colorJitter X
            # is_force_fix = True의 경우에 적용가능
            ,HP_AUGM_RANDOM_SCALER          = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]     # FPENet official:0.5 ~ 1.75 (세부 값 언급 x)
            
             # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
            ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
            ,HP_TS_NORM_MEAN                = HP_TS_NORM_MEAN
            ,HP_TS_NORM_STD                 = HP_TS_NORM_STD
            
            
             # Degradation 관련 설정값
            ,HP_DG_CSV_NAME                 = HP_DG_CSV_NAME
            ,HP_DG_SCALE_FACTOR             = 4
            )
    
    
    print("End of main_sr_train.py")
