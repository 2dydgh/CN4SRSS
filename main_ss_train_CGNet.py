# main_ss_train.py

if __name__ == '__main__':
    from DLCs.semantic_segmentation.model_cgnet     import Context_Guided_Network as CGNet
    from utils.schedulers                           import PolyLR, Poly_Warm_Cos_LR
    from trainers.trainer_total                     import *
    from _options                                   import *

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
    
    # https://github.com/wutianyiRosun/CGNet/blob/master/camvid_train.py#L266
    # https://github.com/wutianyiRosun/CGNet/blob/master/cityscapes_train.py#L264
    if HP_DATASET_NAME == "CamVid":
        force_fix_size_hr   = (360, 360)    # CGNet - CamVid official 설정: (360, 360)
        HP_BATCH_TRAIN      = 14            # CGNet - CamVid official 설정: 14 (paper), 8 (code)
        HP_CHANNEL_HYPO     = HP_LABEL_TOTAL - 1
    elif HP_DATASET_NAME == "MiniCity":
        force_fix_size_hr   = (680, 680)    # CGNet - Cityscapes official 설정
        HP_BATCH_TRAIN      = 4             # CGNet - Cityscapes official 설정: 14 (paper), 16 (code) -> 재현 어려움 (VRAM 8GB)
        HP_CHANNEL_HYPO     = HP_LABEL_TOTAL - 1
    
    # [model_ss & Loss]------------------------
    #모델 입력 시 정규화 여부
    
    model_ss_name = "CGNet"   #CGNet
    
    model_ss = CGNet(classes=HP_CHANNEL_HYPO)
    update_dict_v2("", ""
                  ,"", "모델 정보: " + model_ss_name
                  ,"", "(github) CGNet"
                  ,"", "pretrained = False"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    if HP_LABEL_ONEHOT_ENCODE:
        # label이 one-hot 형태인 경우
        criterion_ss = torch.nn.CrossEntropyLoss()
    else:
        # label이 one-hot 형태가 아닌 경우
        criterion_ss = torch.nn.CrossEntropyLoss(ignore_index=HP_LABEL_VOID)
    
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
    
    option_optimizer = "ADAM"  #공식에서 ADAM 썼다 언급됨
    
    if option_optimizer == "ADAM":
        #<<< ADAM
        HP_LR       = 1e-3          # CGNet official: 1e-3
        HP_WD       = 5e-4          # CGNet official: 5e-4
        
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
    
    option_scheduer = "Poly"    # CGNet official
    
    if option_scheduer == "Poly":
        #<<< Poly
        HP_EPOCH                        = 3002         # 학습 종료 epoch
        HP_SCHEDULER_TOTAL_EPOCH        = 3000         # option for Poly
        HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"      # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
        HP_SCHEDULER_POWER              = 0.9          # CGNet official: 0.9
        scheduler = PolyLR(optimizer, max_epoch=HP_SCHEDULER_TOTAL_EPOCH, power=HP_SCHEDULER_POWER)
        
        update_dict_v2("", ""
                          ,"", "스케쥴러 정보"
                          ,"", "업데이트 간격: " + HP_SCHEDULER_UPDATE_INTERVAL
                          ,"", "scheduler: " + "PolyLR"
                          ,"", "total_epoch: " + str(HP_SCHEDULER_TOTAL_EPOCH)
                          ,"", "power: " +  str(HP_SCHEDULER_POWER)
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        #>>> Poly
    
    
    #=========================================================================================

    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
             # Trainer 모드 설정 ("SS", "SR", "SSSR")
             TRAINER_MODE                   = "SS"
             
             # 코드 실행 장소 (colab 여부 확인용, colab == -1)
            ,RUN_WHERE                      = RUN_WHERE
            
             # 사용할 데이터 셋
            ,HP_DATASET_NAME                = HP_DATASET_NAME
            ,HP_DATASET_CLASSES             = HP_DATASET_CLASSES
            
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
            ,HP_BATCH_TRAIN                 = HP_BATCH_TRAIN
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
            
             # 라벨 정보(원본 데이터 라벨 수(void 포함), void 라벨 번호, 컬러매핑)
            ,HP_LABEL_TOTAL                 = HP_LABEL_TOTAL
            ,HP_LABEL_VOID                  = HP_LABEL_VOID
            ,HP_COLOR_MAP                   = HP_COLOR_MAP
            
             # Patch 생성 관련 강제 Margin 추가여부 (bool, is_use_patch와 동시 사용불가)
            ,is_force_fix                   = True
            ,force_fix_size_hr              = force_fix_size_hr
            
             # 모델 이름 -> 모델에 따라 예측결과 형태가 다르기에 모델입출력물을 조정하는 역할
            ,model_name                     = model_ss_name
            
             # model, optimizer, scheduler, loss
            ,model                          = model_ss
            ,optimizer                      = optimizer
            ,scheduler                      = scheduler
            ,criterion                      = criterion_ss
             # 스케쥴러 업데이트 간격("epoch" 또는 "batch")
            ,HP_SCHEDULER_UPDATE_INTERVAL   = HP_SCHEDULER_UPDATE_INTERVAL
            
             # Train backward loss 검사여부 (default = False)
            ,HP_DETECT_LOSS_ANOMALY         = False
            
             # Label Dilation 설정 (Train에만 적용, default = False)
            ,HP_LABEL_DILATED               = False
            
             # Label One-Hot encoding 시행여부 (HP_LABEL_DILATED 사용시, 반드시 True)
            ,HP_LABEL_ONEHOT_ENCODE         = HP_LABEL_ONEHOT_ENCODE
            
             # Label Verify 설정 (Train에만 적용, default = False)
            ,HP_LABEL_VERIFY                = HP_LABEL_VERIFY
            ,HP_LABEL_VERIFY_TRY_CEILING    = HP_LABEL_VERIFY_TRY_CEILING
            ,HP_LABEL_VERIFY_CLASS_MIN      = HP_LABEL_VERIFY_CLASS_MIN
            ,HP_LABEL_VERIFY_RATIO_MAX      = HP_LABEL_VERIFY_RATIO_MAX
            
             # DataAugm- 관련 (colorJitter 포함)
            ,HP_AUGM_RANGE_CROP_INIT        = HP_AUGM_RANGE_CROP_INIT
            ,HP_AUGM_ROTATION_MAX           = HP_AUGM_ROTATION_MAX
            ,HP_AUGM_PROB_FLIP              = 50                                # CGNet official: 50% 확률로 mirror
            ,HP_AUGM_PROB_CROP              = -1                                # CGNet official: Augm에 random crop X
            ,HP_AUGM_PROB_ROTATE            = -1                                # CGNet official: Augm에 random rotation X
            ,HP_CJ_BRIGHTNESS               = [1,1]                             # CGNet official: Augm에 colorJitter X
            ,HP_CJ_CONTRAST                 = [1,1]                             # CGNet official: Augm에 colorJitter X
            ,HP_CJ_SATURATION               = [1,1]                             # CGNet official: Augm에 colorJitter X
            ,HP_CJ_HUE                      = [0,0]                             # CGNet official: Augm에 colorJitter X
            ,HP_AUGM_RANDOM_SCALER          = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0]  # is_force_fix = True의 경우에 적용가능
            
             # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
            ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
            ,HP_TS_NORM_MEAN                = HP_TS_NORM_MEAN
            ,HP_TS_NORM_STD                 = HP_TS_NORM_STD
            
            
             # Degradation 관련 설정값
            ,HP_DG_CSV_NAME                 = HP_DG_CSV_NAME
            ,HP_DG_SCALE_FACTOR             = 4
            )
    
    
    print("End of main_ss_train.py")
