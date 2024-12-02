# main_m_1_52_t_1_2.py (no CJ)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    
    from _private_models.model_1_52         import proposed_model, proposed_loss, proposed_loss_ss
    from utils.schedulers                   import PolyLR, Poly_Warm_Cos_LR
    from _private_models.trainer_1_2        import *
    from _opt_v2_1                          import *
    
    # Prevent overwriting results
    if args_options.overwrite:
        print("\n---[ 실험결과 덮어쓰기 가능 ]---\n")
    elif os.path.isdir(PATH_BASE_OUT):
        print("실험결과 덮어쓰기 방지기능 작동됨 (Prevent overwriting results function activated)")
        sys.exit("Prevent overwriting results function activated")
    
    #--- args_options.ssn
    
    list_opt_ss = ["a", "b", "c"]   # D3P, DABNet, CGNet
    
    if args_options.ssn is not None:
        if args_options.ssn not in ["D3P", "DABNet", "CGNet"]:
            _str = "Wrong parser.ssn, received " + str(args_options.ssn)
            warnings.warn(_str)
            sys.exit(-9)
        else:
            if   args_options.ssn == "D3P":
                _opt_ss = list_opt_ss[0]    # D3P    -> "a"
            elif args_options.ssn == "DABNet":
                _opt_ss = list_opt_ss[1]    # DABNet -> "b"
            elif args_options.ssn == "CGNet":
                _opt_ss = list_opt_ss[2]    # CGNet  -> "c"
        
    else:
        # _opt_ss = list_opt_ss[0]        # D3P
        # _opt_ss = list_opt_ss[1]        # DABNet
        _opt_ss = list_opt_ss[2]        # CGNet
    
    #--- args_options.kd_mode
    
    LIST_KD_MODE = ["kd_origin", "kd_fitnet", "kd_at", "kd_fsp", "kd_fakd", "kd_p_1", "kd_p_2"]
    
    if args_options.kd_mode is not None:
        if args_options.kd_mode not in LIST_KD_MODE:
            _str = "Wrong parser.kd_mode, received " + str(args_options.kd_mode)
            warnings.warn(_str)
            sys.exit(-9)
        else:
            HP_KD_MODE = args_options.kd_mode
    else:
        HP_KD_MODE = LIST_KD_MODE[6]
    
    #--- args_options.wce_alpha
    
    wce_alpha = args_options.wce_alpha
    wce_beta = args_options.wce_beta
    if not isinstance(wce_alpha, float) or not isinstance(wce_beta, float):
        _str  = "parser wce_alpha and wce_beta must be float"
        warnings.warn(_str)
        sys.exit(-9)
    
    #--- args_options.mixup_a
    HP_MIXUP_A = args_options.mixup_a
    if not isinstance(HP_MIXUP_A, float):
        _str  = "parser HP_MIXUP_A must be float"
        warnings.warn(_str)
        sys.exit(-9)
    
    
    #--- args_options.???
    
    
    
    
    
    if _opt_ss == list_opt_ss[0]:   # a
        from DLCs.semantic_segmentation.model_deeplab_v3_plus   import DeepLab_v3_plus
        print("(D3P) _opt_ss :", _opt_ss)
    elif _opt_ss == list_opt_ss[1]: # b
        from DLCs.semantic_segmentation.model_dabnet            import DABNet
        print("(DABNet) _opt_ss :", _opt_ss)
    elif _opt_ss == list_opt_ss[2]: # c
        from DLCs.semantic_segmentation.model_cgnet             import Context_Guided_Network as CGNet
        print("(CGNet) _opt_ss :", _opt_ss)
    
    
    #[init]----------------------------
    
    # 체크포인트 경로 + 파일명 + 확장자  ("False" 입력시, 자동으로 미입력 처리됨)
    path_check_point = "False"
    
    # (float) 이전 실행시 best score (SR -> PSNR 점수)
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
    path_alter_hr_image = None                      # 원본 이미지 사용
    
    update_dict_v2("", ""
                  ,"", "HR Image 관련 정보"
                  ,"", "원본 HR 이미지 사용됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    if HP_DATASET_NAME == "CamVid" or HP_DATASET_NAME == "CamVid_5Fold":
        force_fix_size_hr       = (360, 360)
        HP_BATCH_TRAIN          = 8
        HP_CHANNEL_HYPO         = HP_LABEL_TOTAL - 1    # 12 - 1
        HP_LABEL_ONEHOT_ENCODE  = False
        hp_augm_random_scaler   = [1.0, 1.0, 1.0, 1.25, 1.25]    # is_force_fix = True의 경우에 적용가능
        
    elif HP_DATASET_NAME == "MiniCity":
        force_fix_size_hr       = (512, 512)
        HP_BATCH_TRAIN          = 4
        HP_CHANNEL_HYPO         = HP_LABEL_TOTAL - 1    # 20 - 1
        HP_LABEL_ONEHOT_ENCODE  = False
        hp_augm_random_scaler   = [0.5, 0.75, 1.0, 1.0]
    
    #[model_sr & Loss]------------------------
    #모델 입력 시 정규화 여부
    # True:  
    # False: MPRNet, ESRT, IMDN, BSRN, RFDN, PAN
    
    # model_sr_name = "Proposed"
    model_t = proposed_model(basic_blocks=27) # teacher
    
    model_s = proposed_model(basic_blocks= 9) # student
    # model_s = None
    
    if _opt_ss   == list_opt_ss[0]: # a
        model_m, _str = DeepLab_v3_plus(num_classes = HP_CHANNEL_HYPO, pretrained = False), "D3P (a)"
    elif _opt_ss == list_opt_ss[1]: # b
        model_m, _str = DABNet(classes=HP_CHANNEL_HYPO), "DABNet (b)"
    elif _opt_ss == list_opt_ss[2]: # c
        model_m, _str = CGNet(classes=HP_CHANNEL_HYPO, M=3, N=21), "CGNet (c)"
    
    update_dict_v2("", ""
                  ,"", "모델 정보"
                  ,"", "proposed 2nd Teacher x27"
                  ,"", "proposed 2nd Student x9"
                  ,"", "Segmenation: " + _str
                  ,"", "pretrained = False"
                  ,"", "scale = 4"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    criterion_t = proposed_loss(kd_mode=None)           # teacher
    
    if model_s is not None:
        criterion_s = proposed_loss(kd_mode=HP_KD_MODE)   # student
    else:
        criterion_s = None
    
    criterion_m = proposed_loss_ss(pred_classes    = HP_CHANNEL_HYPO
                                   ,ignore_index    = HP_LABEL_VOID
                                   ,alpha           = wce_alpha
                                   ,beta            = wce_beta
                                   )
    
    criterion_srss  = proposed_loss_ss(pred_classes    = HP_CHANNEL_HYPO
                                      ,ignore_index    = HP_LABEL_VOID
                                      ,alpha           = wce_alpha
                                      ,beta            = wce_beta
                                      )
    
    if criterion_s is not None:
        # teacher - student 적용됨
        update_dict_v2("", "loss 정보"
                      ,"", "loss: proposed Loss"
                      ,"", "KD method: " + str(HP_KD_MODE)
                      ,"", "weighted CE loss alpha n beta: " + str(wce_alpha) + " n " + str(wce_beta)
                      ,"", "(kd_sr_2_ss) HP_MIXUP_A: " + str(HP_MIXUP_A)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    else:
        # teacher 만 적용됨
        update_dict_v2("", "loss 정보"
                      ,"", "loss: proposed Loss"
                      ,"", "KD 적용 안되고 Teacher만 학습됨"
                      ,"", "weighted CE loss alpha n beta: " + str(wce_alpha) + " n " + str(wce_beta)
                      ,"", "(kd_sr_2_ss) HP_MIXUP_A: " + str(HP_MIXUP_A)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    is_norm_in_transform_to_tensor  = False
    HP_TS_NORM_MEAN                 = None
    HP_TS_NORM_STD                  = None
    
    #[optimizer]------------------------
    # teacher (basic block 27개)
    # 터졌음: 2e-3, 1e-3
    HP_LR_T, HP_WD_T = 5e-4, 0
    optimizer_t = torch.optim.Adam(model_t.parameters(), lr = HP_LR_T, weight_decay = HP_WD_T, betas= (0.9, 0.99))  # teacher
    
    # student
    HP_LR_S, HP_WD_S = 1e-3, 0
    optimizer_s = torch.optim.Adam(model_s.parameters(), lr = HP_LR_S, weight_decay = HP_WD_S, betas= (0.9, 0.99))  # student
    
    
    if _opt_ss == list_opt_ss[0]:   # a, "D3P"
        HP_LR_SS, HP_WD_SS          = 2e-3, 1e-9
        HP_LR_L,  HP_WD_L           = 2e-3, 0 # 1e-9
    elif _opt_ss == list_opt_ss[1]: # b, "DABNet"
        HP_LR_SS, HP_WD_SS          = 2e-3, 1e-9
        HP_LR_L,  HP_WD_L           = 1e-3, 0 # 1e-9
    elif _opt_ss == list_opt_ss[2]: # c, "CGNet"
        HP_LR_SS, HP_WD_SS          = 2e-3, 1e-9
        HP_LR_L,  HP_WD_L           = 2e-3, 0 # 1e-9
    
    # ss
    optimizer_m    = torch.optim.Adam(model_m.parameters(), lr = HP_LR_SS,    weight_decay = HP_WD_SS,    betas= (0.9, 0.99))
    
    # srss (model_s + model_l)
    _params_l = list(model_s.parameters()) + list(model_m.parameters())
    optimizer_srss = torch.optim.Adam(_params_l, lr = HP_LR_L, weight_decay = HP_WD_L, betas= (0.9, 0.99))  # srss (model_s + model_l)
    
    update_dict_v2("", ""
                  ,"", "옵티마이저 정보 (Teacher)"
                  ,"", "optimizer: " + "torch.optim.Adam"
                  ,"", "learning_rate: " + str(HP_LR_T)
                  ,"", "weight decay: " + str(HP_WD_T)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    if optimizer_s is not None:
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보 (Student)"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_S)
                      ,"", "weight decay: " + str(HP_WD_S)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    if optimizer_m is not None:
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보 (SS)"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_SS)
                      ,"", "weight decay: " + str(HP_WD_SS)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    if optimizer_srss is not None:
        update_dict_v2("", ""
                      ,"", "옵티마이저 정보 (SRSS)"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_L)
                      ,"", "weight decay: " + str(HP_WD_L)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    #[scheduler]----------------------------------------------
    HP_EPOCH                        = 1502          # 학습 종료 epoch
    HP_SCHEDULER_TOTAL_EPOCH        = 1500         # option for Poly
    HP_SCHEDULER_POWER              = 0.9          #
    # HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"       # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
    
    scheduler_t = PolyLR(optimizer_t, max_epoch=HP_SCHEDULER_TOTAL_EPOCH, power=HP_SCHEDULER_POWER)
    update_dict_v2("", ""
                  ,"", "스케쥴러 정보 (Teacher)"
                  ,"", "업데이트 간격: epoch" 
                  ,"", "scheduler: Poly " + str(HP_SCHEDULER_POWER)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    scheduler_s = PolyLR(optimizer_s, max_epoch=HP_SCHEDULER_TOTAL_EPOCH, power=HP_SCHEDULER_POWER)
    update_dict_v2("", ""
                  ,"", "스케쥴러 정보 (Student)"
                  ,"", "업데이트 간격: epoch" 
                  ,"", "scheduler: Poly " + str(HP_SCHEDULER_POWER)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    HP_SCHEDULER_WARM_STEPS         = 200
    HP_SCHEDULER_T_MAX              = 50            # (warm-up cos) 반주기
    HP_SCHEDULER_ETA_MIN            = 1e-5          # (warm-up cos) 최소 LR
    HP_SCHEDULER_STYLE              = "floor_4"     # (warm-up cos) 진동 형태
    
    scheduler_m = Poly_Warm_Cos_LR(optimizer_m
                                  ,warm_up_steps=HP_SCHEDULER_WARM_STEPS
                                  ,T_max=HP_SCHEDULER_T_MAX, eta_min=HP_SCHEDULER_ETA_MIN, style=HP_SCHEDULER_STYLE
                                  ,power=HP_SCHEDULER_POWER, max_epoch=HP_SCHEDULER_TOTAL_EPOCH
                                  )
    update_dict_v2("", ""
                  ,"", "스케쥴러 정보 (SS)"
                  ,"", "업데이트 간격: epoch" 
                  ,"", "scheduler: Poly_Warm_Cos_LR"
                  ,"", "Power: "   + str(HP_SCHEDULER_POWER)
                  ,"", "Warm up: " + str(HP_SCHEDULER_WARM_STEPS)
                  ,"", "T max: "   + str(HP_SCHEDULER_T_MAX)
                  ,"", "ETA min: " + str(HP_SCHEDULER_ETA_MIN)
                  ,"", "Style: "   + str(HP_SCHEDULER_STYLE)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    scheduler_srss = Poly_Warm_Cos_LR(optimizer_srss
                                     ,warm_up_steps=HP_SCHEDULER_WARM_STEPS
                                     ,T_max=HP_SCHEDULER_T_MAX, eta_min=HP_SCHEDULER_ETA_MIN, style=HP_SCHEDULER_STYLE
                                     ,power=HP_SCHEDULER_POWER, max_epoch=HP_SCHEDULER_TOTAL_EPOCH
                                     )
    update_dict_v2("", ""
                  ,"", "스케쥴러 정보 (SRSS)"
                  ,"", "업데이트 간격: epoch" 
                  ,"", "scheduler: Poly_Warm_Cos_LR"
                  ,"", "Power: "   + str(HP_SCHEDULER_POWER)
                  ,"", "Warm up: " + str(HP_SCHEDULER_WARM_STEPS)
                  ,"", "T max: "   + str(HP_SCHEDULER_T_MAX)
                  ,"", "ETA min: " + str(HP_SCHEDULER_ETA_MIN)
                  ,"", "Style: "   + str(HP_SCHEDULER_STYLE)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    
    
    
    #=========================================================================================
    
    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
             # 사용할 데이터 셋
             HP_DATASET_NAME                = HP_DATASET_NAME
            ,HP_DATASET_CLASSES             = HP_DATASET_CLASSES
            
             # Test 과정에서 저장할 이미지 수를 줄일 것인가?
            ,REDUCE_SAVE_IMAGES             = REDUCE_SAVE_IMAGES
             # Test 과정에서 반드시 저장할 이미지 이름
            ,MUST_SAVE_IMAGE                = MUST_SAVE_IMAGE
             # 저장할 최대 이미지 수 (Val & Test 에서는 REDUCE_SAVE_IMAGES is True 인 경우에만 적용됨)
            ,MAX_SAVE_IMAGES                = MAX_SAVE_IMAGES
            
             #(선택) 이전 실행 시 best score
            # ,prev_best                      = prev_best
            
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
            
            ,is_force_fix                   = True
            ,force_fix_size_hr              = force_fix_size_hr
            
            
             # (T) model, optimizer, scheduler, loss
            ,model_t                        = model_t
            ,optimizer_t                    = optimizer_t
            ,scheduler_t                    = scheduler_t
            ,criterion_t                    = criterion_t
            
             # (S) model, optimizer, scheduler, loss
            ,model_s                        = model_s
            ,optimizer_s                    = optimizer_s
            ,scheduler_s                    = scheduler_s
            ,criterion_s                    = criterion_s
            
             # (kd_sr_2_ss SS) model, optimizer, scheduler, loss
            ,model_m                        = model_m
            ,optimizer_m                    = optimizer_m
            ,scheduler_m                    = scheduler_m
            ,criterion_m                    = criterion_m
            ,HP_MIXUP_A                     = HP_MIXUP_A
            
             # (SRSS) optimizer, scheduler, loss
            ,optimizer_srss                 = optimizer_srss
            ,scheduler_srss                 = scheduler_srss
            ,criterion_srss                 = criterion_srss
            
            
            #<<<
             # Label Dilation 설정 (Train에만 적용, default = False)
            ,HP_LABEL_DILATED               = False
            
             # Label One-Hot encoding 시행여부 (HP_LABEL_DILATED 사용시, 반드시 True)
            ,HP_LABEL_ONEHOT_ENCODE         = False
            
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
            ,HP_CJ_BRIGHTNESS               = [1,1] # HP_CJ_BRIGHTNESS
            ,HP_CJ_CONTRAST                 = [1,1] # HP_CJ_CONTRAST
            ,HP_CJ_SATURATION               = [1,1] # HP_CJ_SATURATION
            ,HP_CJ_HUE                      = [0,0] # HP_CJ_HUE
            ,HP_AUGM_RANDOM_SCALER          = hp_augm_random_scaler
            #>>>
            
            
             # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
            ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
            ,HP_TS_NORM_MEAN                = HP_TS_NORM_MEAN
            ,HP_TS_NORM_STD                 = HP_TS_NORM_STD
            
            
             # Degradation 관련 설정값
            ,HP_DG_CSV_NAME                 = HP_DG_CSV_NAME
            ,HP_DG_SCALE_FACTOR             = 4
            )
    
    
    print("End of main.py")
