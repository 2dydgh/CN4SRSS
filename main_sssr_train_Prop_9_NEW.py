# main_sssr_train.py

if __name__ == '__main__':
    #from _private_models.Prop_9_NEW_Ab_01   import model_proposed, loss_proposed
    #from _private_models.Prop_9_NEW_Ab_02   import model_proposed, loss_proposed
    #from _private_models.Prop_9_NEW_Ab_03   import model_proposed, loss_proposed
    #from _private_models.Prop_9_NEW_Ab_04   import model_proposed, loss_proposed
    
    
    from utils.schedulers                   import Poly_Warm_Cos_LR
    from trainers.trainer_total             import *
    from _options                           import *

    #[init]----------------------------
    #Prevent overwriting results
    if os.path.isdir(PATH_BASE_OUT):
        print("실험결과 덮어쓰기 방지기능 작동됨 (Prevent overwriting results function activated)")
        sys.exit("Prevent overwriting results function activated")
    
    # 체크포인트 경로 + 파일명 + 확장자  ("False" 입력시, 자동으로 미입력 처리됨)
    path_check_point = "False"
    
    # (float) 이전 실행시 best score (SSSR -> mIoU 점수)
    prev_best = 0.20
    
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
    path_alter_hr_image = None                      # 원본 이미지 사용
    
    if path_alter_hr_image is None:
        update_dict_v2("", ""
                      ,"", "HR Image 관련 정보"
                      ,"", "원본 HR 이미지 사용됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        sys.exit("-9")
    
    if HP_DATASET_NAME == "CamVid" or HP_DATASET_NAME == "CamVid_5Fold":
        force_fix_size_hr       = (352, 352)            #
        
        HP_BATCH_TRAIN          = 4                     #
        
        HP_LABEL_ONEHOT_ENCODE  = True
        
        HP_LR       = 2e-3
        HP_WD       = 1e-9
        
        hp_augm_random_scaler          = [1.0, 1.0, 1.0, 1.25, 1.25, 1.5]    # is_force_fix = True의 경우에 적용가능
        
    elif HP_DATASET_NAME == "MiniCity":
        force_fix_size_hr       = (480, 480)
        HP_BATCH_TRAIN          = 5
        HP_LABEL_ONEHOT_ENCODE  = False
        HP_LR       = 2e-3
        HP_WD       = 7e-9
        hp_augm_random_scaler          = [0.5, 0.5, 0.5, 0.75, 0.75, 1.0]
    
    #[model_sssr]------------------------
    
    model_sssr_name = "model_d"
    
    model_sssr = model_proposed()
    update_dict_v2("", ""
                  ,"", "모델 정보: " + model_sssr_name
                  ,"", "(custom)"
                  ,"", "pretrained = False"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    model_sssr.to(device)
    
    
    if HP_LABEL_ONEHOT_ENCODE:
        # one-hot encoded
        criterion = loss_proposed(is_onehot  = True
                                 ,class_void = None
                                 )
        
        update_dict_v2("", ""
                      ,"", "loss 정보"
                      ,"", "proposed loss (one-hot label 사용됨)"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    else:
        # one-hot encode 시행 안함
        #sys.exit("-9")
        criterion = loss_proposed(is_onehot=False
                                 ,class_void=HP_LABEL_VOID
                                 )

        update_dict_v2("", ""
                       , "", "loss 정보"
                       , "", "proposed loss (one-hot label 사용 안됨)"
                       , in_dict=dict_log_init
                       , in_print_head="dict_log_init"
                       )
    
    # 모델 입력 시 정규화 여부
    is_norm_in_transform_to_tensor = False
    HP_TS_NORM_MEAN  = None
    HP_TS_NORM_STD   = None
    
    #[optimizer]------------------------
    
    option_optimizer = "ADAM"
    
    if option_optimizer == "ADAM":
        #<<< ADAM
        
        
        optimizer = torch.optim.Adam(model_sssr.parameters()
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
    
    
    
    #[scheduler]----------------------------------------------
    
    option_scheduer = "Poly_Warm_Cosine"
    
    if option_scheduer == "Poly_Warm_Cosine":
        #<<< Poly with Cosine Warm-up
        HP_EPOCH                        = 602          # 학습 종료 epoch
        
        HP_SCHEDULER_WARM_STEPS         = 100           # (warm-up cos) warm-up 시행 step 수
        
        HP_SCHEDULER_T_MAX              = 50            # (warm-up cos) 반주기
        HP_SCHEDULER_ETA_MIN            = 1e-4          # (warm-up cos) 최소 LR
        HP_SCHEDULER_STYLE              = "floor_4"     # (warm-up cos) 진동 형태
        
        HP_SCHEDULER_TOTAL_EPOCH        = 600          # (poly) LR 0.0 도달 step
        HP_SCHEDULER_POWER              = 0.9           # (poly) power 값
        
        HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"       # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
        
        
        scheduler = Poly_Warm_Cos_LR(optimizer
                                    ,warm_up_steps=HP_SCHEDULER_WARM_STEPS
                                    ,T_max=HP_SCHEDULER_T_MAX, eta_min=HP_SCHEDULER_ETA_MIN, style=HP_SCHEDULER_STYLE
                                    ,power=HP_SCHEDULER_POWER, max_epoch=HP_SCHEDULER_TOTAL_EPOCH
                                    )
        
        update_dict_v2("", ""
                      ,"", "스케쥴러 정보"
                      ,"", "업데이트 간격: "      + HP_SCHEDULER_UPDATE_INTERVAL
                      ,"", "scheduler: "      + "Poly warm-up with Cosine"
                      ,"", "warm-up steps: "  + str(HP_SCHEDULER_WARM_STEPS)
                      ,"", "cosine 반주기: "    + str(HP_SCHEDULER_T_MAX)
                      ,"", "cosine 최소 LR: "  + str(HP_SCHEDULER_ETA_MIN)
                      ,"", "cosine style: "     + str(HP_SCHEDULER_STYLE)
                      ,"", "Poly 최종 step: "  + str(HP_SCHEDULER_TOTAL_EPOCH)
                      ,"", "Poly Power: "     + str(HP_SCHEDULER_POWER)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        #>>> Poly with Cosine Warm-up
    
    
    
    

    #=========================================================================================

    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
             # Trainer 모드 설정 ("SR", "SSSR")
             TRAINER_MODE                   = "SSSR"
            
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
            
            
             # 강제 Margin 추가여부 (bool, is_use_patch와 동시 사용불가, lr 이미지 관련 값은 scale factor에 비례한 값 자동 입력됨)
            ,is_force_fix                   = True
            ,force_fix_size_hr              = force_fix_size_hr
            
             # 모델 이름 -> 모델에 따라 예측결과 형태가 다르기에 모델입출력물을 조정하는 역할
            ,model_name                     = model_sssr_name
            
             # model, optimizer, scheduler, loss
            ,model                          = model_sssr
            ,optimizer                      = optimizer
            ,scheduler                      = scheduler
            ,criterion                      = criterion
             # 스케쥴러 업데이트 간격("epoch" 또는 "batch")
            ,HP_SCHEDULER_UPDATE_INTERVAL   = HP_SCHEDULER_UPDATE_INTERVAL
            
             # Train backward loss 검사여부 (default = False)
            ,HP_DETECT_LOSS_ANOMALY         = True
            
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
            ,HP_AUGM_PROB_FLIP              = HP_AUGM_PROB_FLIP
            ,HP_AUGM_PROB_CROP              = HP_AUGM_PROB_CROP
            ,HP_AUGM_PROB_ROTATE            = HP_AUGM_PROB_ROTATE
            ,HP_CJ_BRIGHTNESS               = HP_CJ_BRIGHTNESS
            ,HP_CJ_CONTRAST                 = HP_CJ_CONTRAST
            ,HP_CJ_SATURATION               = HP_CJ_SATURATION
            ,HP_CJ_HUE                      = HP_CJ_HUE
            ,HP_AUGM_RANDOM_SCALER          = hp_augm_random_scaler
            
             # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
            ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
            ,HP_TS_NORM_MEAN                = HP_TS_NORM_MEAN
            ,HP_TS_NORM_STD                 = HP_TS_NORM_STD
            
            
             # Degradation 관련 설정값
            ,HP_DG_CSV_NAME                 = HP_DG_CSV_NAME
            ,HP_DG_SCALE_FACTOR             = 4
            )
    
    
    print("End of workspace_sssr.py")
