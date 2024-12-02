# main_sssr_train.py

if __name__ == '__main__':
    #from model_dsrl_deeplab import DSRL_D3P, loss_for_dsrl
    from _private_models.Prop_9_NEW_Ab_07_fix_1 import model_proposed, loss_proposed
    
    from trainers.trainer_total import *
    from _options import *

    #[init]----------------------------
    #Prevent overwriting results
    if os.path.isdir(PATH_BASE_OUT):
        print("실험결과 덮어쓰기 방지기능 작동됨 (Prevent overwriting results function activated)")
        sys.exit("Prevent overwriting results function activated")
    
    # 체크포인트 경로 + 파일명 + 확장자  ("False" 입력시, 자동으로 미입력 처리됨)
    path_check_point = "False"
    
    # (float) 이전 실행시 best score (SSSR -> mIoU 점수)
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


    #[model_sssr]------------------------
    #모델 입력 시 정규화 여부
    #DSRL(DeepLab v3 Plus: True)
    #(model_a = False)
    is_norm_in_transform_to_tensor = False
    '''
    model_sssr_name = "DeepLab v3 plus"
    model_sssr = DSRL_D3P(num_classes = 11       #SS labels
                         ,backbone='xception'    #SS backbone
                         ,scale_factor = 4       #SR ScaleFactor
                         )
    update_dict_v2("", ""
                  ,"", "모델 정보: " + model_sssr_name
                  ,"", "(github) DSRL Deeplab v3 plus (xception)"
                  ,"", "pretrained = False"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    '''
    model_sssr_name = "model_a"
    #model_sssr_name = "model_d"     # 기타 해당 모델: model_ab
    #model_sssr_name = "model_aa"    # 기타 해당 모델: model_ac
    model_sssr = model_proposed()
    update_dict_v2("", ""
                  ,"", "모델 정보: " + model_sssr_name
                  ,"", "(custom)"
                  ,"", "pretrained = False"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    model_sssr.to(device)


    #[optimizer]------------------------
    #Adam: torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, maximize=False)
    #https://arxiv.org/abs/1412.6980
    #https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

    if is_weight_decay: #weight decay를 사용한 경우

        optimizer = torch.optim.Adam(model_sssr.parameters()
                                    ,lr=HP_LR_SSSR
                                    ,weight_decay = HP_WD_SSSR
                                    )
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_SSSR)
                      ,"", "weight decay 적용됨: " + str(HP_WD_SSSR)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    else: #weight decay를 사용하지 않는 경우
        optimizer = torch.optim.Adam(model_sssr.parameters()
                                    ,lr=HP_LR_SSSR
                                    )
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_SSSR)
                      ,"", "weight decay 적용 안됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    #[scheduler]----------------------------------------------
    #https://gaussian37.github.io/dl-pytorch-lr_scheduler/

    if HP_SCHEDULER_OPTION_SSSR == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer
                                                        ,T_max = HP_SCHEDULER_T_MAX
                                                        ,eta_min = HP_SCHEDULER_ETA_MIN
                                                        )

        update_dict_v2("", ""
                      ,"", "스케쥴러 정보"
                      ,"", "업데이트 간격: " + HP_SCHEDULER_UPDATE_INTERVAL_SSSR
                      ,"", "scheduler: " + "optim.lr_scheduler.CosineAnnealingLR"
                      ,"", "LR 최대값 도달 epoch 수 (lr 반복주기의 절반)"
                      ,"", "T_max: " + str(HP_SCHEDULER_T_MAX)
                      ,"", "lr 최소값 (default = 0)"
                      ,"", "eta_min: " + str(HP_SCHEDULER_ETA_MIN)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    '''
    elif HP_SCHEDULER_OPTION_SSSR == "CyclicLR":
        #torch.optim.Adam does not support "CyclicLR" (for cycle_momentum=True)
        #Cyclical Learning Rates for Training Neural Networks (https://arxiv.org/abs/1506.01186)
        #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
        
        scheduler = optim.lr_scheduler.CyclicLR(optimizer
                                               ,base_lr = HP_SCHEDULER_BASE_LR
                                               ,max_lr = HP_SCHEDULER_MAX_LR
                                               ,step_size_up = HP_SCHEDULER_STEP_SIZE_UP
                                               ,step_size_down = HP_SCHEDULER_STEP_SIZE_DOWN
                                               ,mode = HP_SCHEDULER_MODE
                                               )
        
        update_dict_v2("", ""
                      ,"", "스케쥴러 정보"
                      ,"", "업데이트 간격: " + HP_SCHEDULER_UPDATE_INTERVAL_SSSR
                      ,"", "scheduler: " + "optim.lr_scheduler.CyclicLR"
                      ,"", "최소 Learning Rate (Lower Bound)"
                      ,"", "HP_SCHEDULER_BASE_LR: " + str(HP_SCHEDULER_BASE_LR)
                      ,"", "최대 Learning Rate (Upper Bound)"
                      ,"", "HP_SCHEDULER_MAX_LR: " + str(HP_SCHEDULER_MAX_LR)
                      ,"", "(base_lr -> max_lr) epoch 수"
                      ,"", "HP_SCHEDULER_STEP_SIZE_UP" + str(HP_SCHEDULER_STEP_SIZE_UP)
                      ,"", "(max_lr -> base_lr) epoch 수"
                      ,"", "HP_SCHEDULER_STEP_SIZE_DOWN" + str(HP_SCHEDULER_STEP_SIZE_DOWN)
                      ,"", "(str) 모드: triangular2 = 주기(step_size_up + step_size_down)마다 max_lr이 반감됨"
                      ,"", "HP_SCHEDULER_MODE" + HP_SCHEDULER_MODE
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    '''

    #[loss]--------------------------------------------------
    '''
    # from model_dsrl
    criterion = loss_for_dsrl()
    
    update_dict_v2("", ""
                  ,"", "loss 정보"
                  ,"", "loss: loss_for_dsrl from model_dsrl.py"
                  ,"", "used default option"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    '''
    
    criterion = loss_proposed()
    update_dict_v2("", ""
                  ,"", "loss 정보"
                  ,"", "proposed loss"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )


    #=========================================================================================

    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
             # Trainer 모드 설정 ("SR", "SSSR")
             TRAINER_MODE = "SR_A"
            
             # 코드 실행 장소 (colab 여부 확인용, colab == -1)
            ,RUN_WHERE = RUN_WHERE
            
             # Test 과정에서 저장할 이미지 수를 줄일 것인가?
            ,REDUCE_SAVE_IMAGES = REDUCE_SAVE_IMAGES
             # Test 과정에서 반드시 저장할 이미지 이름
            ,MUST_SAVE_IMAGE = MUST_SAVE_IMAGE
             # 저장할 최대 이미지 수 (Val & Test 에서는 REDUCE_SAVE_IMAGES is True 인 경우에만 적용됨)
            ,MAX_SAVE_IMAGES = MAX_SAVE_IMAGES
            
             #(선택) COLAB 결과물 저장용 G-Drive 경로 (log 관련, model 관련)
            ,PATH_COLAB_OUT_LOG = PATH_COLAB_OUT_LOG
            ,PATH_COLAB_OUT_MODEL = PATH_COLAB_OUT_MODEL
            
             #(선택) 체크포인트 불러오기 ("False" 입력시, 자동으로 미입력 처리됨)
            ,path_check_point = path_check_point
            
             #(선택) 이전 실행 시 best score
            ,prev_best = prev_best
            
            # 버퍼 크기 (int) -> 버퍼 안쓰는 옵션 사용불가. 양수값 가능
            ,BUFFER_SIZE = 60
             # 초기화 기록 dict 이어받기
            ,dict_log_init = dict_log_init
             # 랜덤 시드 고정
            ,HP_SEED = HP_SEED
            
            
             # 학습 관련 기본 정보(epoch 수, batch 크기(train은 생성할 patch 수), 학습 시 dataset 루프 횟수)
            ,HP_EPOCH = HP_EPOCH
            ,HP_BATCH_TRAIN = HP_BATCH_TRAIN_SSSR
            #,HP_DATASET_LOOP = HP_DATASET_LOOP_SSSR
            ,HP_BATCH_VAL = HP_BATCH_VAL
            ,HP_BATCH_TEST = HP_BATCH_TEST
            
            
             # 데이터 입출력 경로, 폴더명
            ,PATH_BASE_IN = PATH_BASE_IN
            ,NAME_FOLDER_TRAIN = NAME_FOLDER_TRAIN
            ,NAME_FOLDER_VAL = NAME_FOLDER_VAL
            ,NAME_FOLDER_TEST = NAME_FOLDER_TEST
            ,NAME_FOLDER_IMAGES = NAME_FOLDER_IMAGES
            ,NAME_FOLDER_LABELS = NAME_FOLDER_LABELS
            
             # (선택) degraded image 불러올 경로
            ,PATH_BASE_IN_SUB = PATH_BASE_IN_SUB
            
            ,PATH_OUT_IMAGE = PATH_OUT_IMAGE
            ,PATH_OUT_MODEL = PATH_OUT_MODEL
            ,PATH_OUT_LOG = PATH_OUT_LOG
            
            
             # 데이터(이미지) 크기 (원본 이미지), 이미지 채널 수(이미지, 라벨, 모델출력물)
            ,HP_ORIGIN_IMG_W = HP_ORIGIN_IMG_W
            ,HP_ORIGIN_IMG_H = HP_ORIGIN_IMG_H
            
            ,HP_CHANNEL_RGB = HP_CHANNEL_RGB
            ,HP_CHANNEL_GRAY = HP_CHANNEL_GRAY
            ,HP_CHANNEL_HYPO = HP_CHANNEL_HYPO
            
             # 라벨 정보(원본 데이터 라벨 수(void 포함), void 라벨 번호, 컬러매핑)
            ,HP_LABEL_TOTAL = HP_LABEL_TOTAL
            ,HP_LABEL_VOID = HP_LABEL_VOID
            ,HP_COLOR_MAP = HP_COLOR_MAP
            
             # Patch 생성 관련 (사용여부, 입력 Patch 크기, stride, 시작좌표범위)
            ,is_use_patch = True
            ,HP_MODEL_IMG_W = HP_MODEL_SSSR_IMG_W
            ,HP_MODEL_IMG_H = HP_MODEL_SSSR_IMG_H
            ,HP_PATCH_STRIDES = HP_PATCH_STRIDES
            ,HP_PATCH_CROP_INIT_COOR_RANGE = HP_PATCH_CROP_INIT_COOR_RANGE
            
             # 모델 이름 -> 모델에 따라 예측결과 형태가 다르기에 모델입출력물을 조정하는 역할
             # 지원 리스트 = (의미없음)"DeeplabV3Plus"
            ,model_name = model_sssr_name
            
             # model, optimizer, scheduler, loss
            ,model = model_sssr
            ,optimizer = optimizer
            ,scheduler = scheduler
            ,criterion = criterion
             # 스케쥴러 업데이트 간격("epoch" 또는 "batch")
            ,HP_SCHEDULER_UPDATE_INTERVAL = HP_SCHEDULER_UPDATE_INTERVAL_SSSR
            
            
             # DataAugm- 관련 (colorJitter 포함)
            ,HP_AUGM_RANGE_CROP_INIT = HP_AUGM_RANGE_CROP_INIT
            ,HP_AUGM_ROTATION_MAX = HP_AUGM_ROTATION_MAX
            ,HP_AUGM_PROB_FLIP = HP_AUGM_PROB_FLIP
            ,HP_AUGM_PROB_CROP = HP_AUGM_PROB_CROP
            ,HP_AUGM_PROB_ROTATE = HP_AUGM_PROB_ROTATE
            ,HP_CJ_BRIGHTNESS = HP_CJ_BRIGHTNESS
            ,HP_CJ_CONTRAST = HP_CJ_CONTRAST
            ,HP_CJ_SATURATION = HP_CJ_SATURATION
            ,HP_CJ_HUE = HP_CJ_HUE
            
            
             # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
            ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
            ,HP_TS_NORM_MEAN = HP_TS_NORM_MEAN_SSSR
            ,HP_TS_NORM_STD = HP_TS_NORM_STD_SSSR
            
            
             # Degradation 관련 설정값
            ,HP_DG_CSV_NAME = HP_DG_CSV_NAME
            ,HP_DG_SCALE_FACTOR = HP_DG_SCALE_FACTOR
            )
    
    
    print("End of workspace_sssr.py")
