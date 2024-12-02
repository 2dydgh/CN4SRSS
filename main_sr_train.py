"""
기반
    main_m_1_53_t_1_3_3_fix_2.py, main_p3.py, main_ss_train.py

디버깅용 커맨드 (v 4.3)

python main_sr_train.py --overwrite True --name "CUDA_RESET" --dataset "MiniCity" --fold "A_set" --scale 4 --srn "ESRT" --patch_length 1000 --batch_train 8 --valid_with_patch False --detect_anomaly False --will_save_image True --skip_test_until 0

python main_sr_train.py --overwrite True --name "_debug_train_sr" --dataset "YCOR" --fold "A_set" --scale 4 --srn "IMDN" --valid_with_patch False --detect_anomaly False --will_save_image True --skip_test_until 0
python main_sr_train.py --overwrite True --name "_debug_train_sr" --dataset "YCOR" --fold "A_set" --scale 4 --srn "ESRT" --valid_with_patch False --detect_anomaly False --will_save_image True --skip_test_until 0
python main_sr_train.py --overwrite True --name "_debug_train_sr" --dataset "YCOR" --fold "A_set" --scale 4 --srn "PAN" --valid_with_patch False --detect_anomaly False --will_save_image True --skip_test_until 0


"""


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    from trainers.trainer_sr                    import *
    from _opt_sr                                import *
    
    # Prevent overwriting results
    if args_options.overwrite:
        print("\n---[ 실험결과 덮어쓰기 가능 ]---\n")
    elif os.path.isdir(PATH_BASE_OUT):
        print("실험결과 덮어쓰기 방지기능 작동됨 (Prevent overwriting results function activated)")
        sys.exit("Prevent overwriting results function activated")
    
    #--- args_options.ssn
    # list_opt_ss = ["a", "b", "c"]   # D3P, DABNet, CGNet
    
    # if args_options.ssn is not None:
        # if args_options.ssn not in ["D3P", "DABNet", "CGNet"]:
            # _str = "Wrong parser.ssn, received " + str(args_options.ssn)
            # warnings.warn(_str)
            # sys.exit(-9)
        # else:
            # if   args_options.ssn == "D3P":
                # _opt_ss = list_opt_ss[0]    # D3P    -> "a"
            # elif args_options.ssn == "DABNet":
                # _opt_ss = list_opt_ss[1]    # DABNet -> "b"
            # elif args_options.ssn == "CGNet":
                # _opt_ss = list_opt_ss[2]    # CGNet  -> "c"
        
    # else:
        # # _opt_ss = list_opt_ss[0]        # D3P
        # # _opt_ss = list_opt_ss[1]        # DABNet
        # _opt_ss = list_opt_ss[2]        # CGNet
    
    
    #--- args_options.srn
    list_opt_sr = ["IMDN", "RFDN", "BSRN", "ESRT", "LAPAR_A", "PAN"]
    
    if args_options.srn is not None:
        if args_options.srn not in list_opt_sr:
            _str = "Wrong parser.srn, received " + str(args_options.srn)
            warnings.warn(_str)
            sys.exit(-9)
        else:
            _opt_sr = args_options.srn
            
    else:
        _opt_sr = list_opt_sr[0]
    
    
    
    #--- args_options.will_save_image
    WILL_SAVE_IMAGE = args_options.will_save_image  # (bool) 결과 이미지를 저장할 것인가?
    
    #--- args_options.skip_test_until
    SKIP_TEST_UNTIL = args_options.skip_test_until  # (int) test 생략할 마지막 epoch (epoch는 1부터 시작할 떄 기준, 100 입력시 101 epoch 부터 test 실행 가능)
    
    #--- args_options.calc_with_logit
    CALC_WITH_LOGIT = args_options.calc_with_logit
    
    #--- args_options.onehot
    if args_options.onehot is not None:
        HP_LABEL_ONEHOT_ENCODE = args_options.onehot
    
    #--- args_options.make_pkl
    make_pkl = args_options.make_pkl                # (bool) 피클 생성기: 학습 대신 피클을 만들 것인가? -> True 시 학습 안하고 피클 만들고 종료됨
    
    #--- args_options.???
    
    
    # if _opt_ss == list_opt_ss[0]:   # a
        # from DLCs.semantic_segmentation.model_deeplab_v3_plus   import DeepLab_v3_plus
        # print("(D3P) _opt_ss :", _opt_ss)
    # elif _opt_ss == list_opt_ss[1]: # b
        # from DLCs.semantic_segmentation.model_dabnet            import DABNet
        # print("(DABNet) _opt_ss :", _opt_ss)
    # elif _opt_ss == list_opt_ss[2]: # c
        # from DLCs.semantic_segmentation.model_cgnet             import Context_Guided_Network as CGNet
        # print("(CGNet) _opt_ss :", _opt_ss)
    
    
    
    
    #[init]----------------------------
    
    # 체크포인트 경로 + 파일명 + 확장자  ("False" 입력시, 자동으로 미입력 처리됨)
    path_check_point = r"D:\LAB\result_files\PAN_1_sr_CamVid_12_2Fold_v4_B_set\models\check_points"
    
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
    # path_alter_hr_image = None                      # 원본 이미지 사용
    path_alter_hr_image = PATH_ALTER_HR_IMAGE
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
                      ,"", "HR 이미지 교체됨: "+ str(path_alter_hr_image)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    # force_fix_size_hr       = (360, 360)
    # HP_BATCH_TRAIN          = 4
    # HP_BATCH_VAL            = 1
    # HP_BATCH_TEST           = 1
    HP_CHANNEL_HYPO         = HP_LABEL_TOTAL - 1    # 12 - 1
    # hp_augm_random_scaler          = [1.0, 1.0, 1.0, 1.25, 1.25, 1.5]
    
    # valid 시행 시 center-cropped patch 쓸지 whole-image 쓸 지 선택
    HP_VALID_WITH_PATCH = args_options.valid_with_patch
    
    #[model_sr & Loss]------------------------
    
    #@<
    #@@@ 작성중
    # _opt_sr 에 따른 옵션 지정
    
    _str = _opt_sr + " (x" + str(args_options.scale) + ")"
    print("SR model:", _str)
    
    if _opt_sr == "IMDN":
        from DLCs.super_resolution.model_imdn import IMDN
        model_srss = IMDN(upscale=args_options.scale)
        criterion_srss = torch.nn.L1Loss()
        _str_loss = "torch.nn.L1Loss()"
        
        # optimizer
        HP_LR_L, HP_WD_L = 2e-4, 0
        HP_BETAS_L       = (0.9, 0.999) # default
        HP_EPS_L         = 1e-8         # default
        
        # official: HR 기준 192x192, batch 16
        force_fix_size_hr       = (192, 192)
        HP_BATCH_TRAIN          = 16
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        
        HP_AUGM_LITE            = True
        HP_AUGM_LITE_FLIP_HORI  = True
        HP_AUGM_LITE_FLIP_VERT  = False
        hp_augm_random_scaler   = [1.0]
        
    elif _opt_sr == "RFDN":
        from DLCs.super_resolution.model_rfdn import RFDN
        model_srss = RFDN(upscale=args_options.scale)
        criterion_srss = torch.nn.L1Loss()
        _str_loss = "torch.nn.L1Loss()"
        
        # optimizer
        HP_LR_L, HP_WD_L = 5e-4, 0
        HP_BETAS_L       = (0.9, 0.999) # default
        HP_EPS_L         = 1e-8         # default
        
        # official: LR 기준 64x64, batch 64
        force_fix_size_hr       = (int(64*args_options.scale), int(64*args_options.scale))
        HP_BATCH_TRAIN          = 64
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        
        HP_AUGM_LITE            = True
        HP_AUGM_LITE_FLIP_HORI  = True
        HP_AUGM_LITE_FLIP_VERT  = False
        hp_augm_random_scaler   = [1.0]
        
    elif _opt_sr == "BSRN":
        from DLCs.super_resolution.model_bsrn import BSRN
        model_srss = BSRN(upscale=args_options.scale)
        criterion_srss = torch.nn.L1Loss()
        _str_loss = "torch.nn.L1Loss()"
        
        # optimizer
        HP_LR_L, HP_WD_L = 1e-3, 0
        HP_BETAS_L       = (0.9, 0.999) # default
        HP_EPS_L         = 1e-8         # default
        
        # official: LR 기준 48x48, batch 64
        force_fix_size_hr       = (int(48*args_options.scale), int(48*args_options.scale))
        HP_BATCH_TRAIN          = 64
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        
        HP_AUGM_LITE            = True
        HP_AUGM_LITE_FLIP_HORI  = True
        HP_AUGM_LITE_FLIP_VERT  = False
        hp_augm_random_scaler   = [1.0]
        
    elif _opt_sr == "ESRT":
        from DLCs.super_resolution.model_esrt import ESRT
        model_srss = ESRT(upscale=args_options.scale)
        criterion_srss = torch.nn.L1Loss()
        _str_loss = "torch.nn.L1Loss()"
        
        # optimizer
        HP_LR_L, HP_WD_L = 2e-4, 0
        HP_BETAS_L       = (0.9, 0.999) # default
        HP_EPS_L         = 1e-8         # default
        
        # official: LR 이미지 기준 48x48, batch 16 (점검 필요함)
        force_fix_size_hr       = (int(48*args_options.scale), int(48*args_options.scale))
        HP_BATCH_TRAIN          = 16
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        
        HP_AUGM_LITE            = True
        HP_AUGM_LITE_FLIP_HORI  = True
        HP_AUGM_LITE_FLIP_VERT  = False
        hp_augm_random_scaler   = [1.0]
        
    elif _opt_sr == "LAPAR_A":
        from DLCs.super_resolution.model_lapar_a import Network as LAPAR_A
        model_srss = LAPAR_A(scale=args_options.scale)
        criterion_srss = torch.nn.L1Loss()
        _str_loss = "torch.nn.L1Loss()"
        
        # optimizer
        HP_LR_L, HP_WD_L = 4e-4, 0
        HP_BETAS_L       = (0.9, 0.999) # default
        HP_EPS_L         = 1e-8         # default
        
        # official: LR 기준 64x64, batch 32
        force_fix_size_hr       = (int(64*args_options.scale), int(64*args_options.scale))
        HP_BATCH_TRAIN          = 32
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        
        HP_AUGM_LITE            = True
        HP_AUGM_LITE_FLIP_HORI  = True
        HP_AUGM_LITE_FLIP_VERT  = True
        hp_augm_random_scaler   = [1.0]
        
    elif _opt_sr == "PAN":
        from DLCs.super_resolution.model_pan import PAN
        model_srss = PAN(scale=args_options.scale)
        criterion_srss = torch.nn.L1Loss()
        _str_loss = "torch.nn.L1Loss()"
        
        # optimizer
        HP_LR_L, HP_WD_L = 1e-3, 0
        HP_BETAS_L       = (0.9, 0.99)
        HP_EPS_L         = 1e-8         # default
        
        # official: HR 기준 256x256, batch 32
        force_fix_size_hr       = (256, 256)
        HP_BATCH_TRAIN          = 32
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        
        HP_AUGM_LITE            = True
        HP_AUGM_LITE_FLIP_HORI  = True
        HP_AUGM_LITE_FLIP_VERT  = False
        hp_augm_random_scaler   = [1.0]
        
    
    # overwrite with argparse
    if args_options.patch_length is not None: # patch length for train
        force_fix_size_hr = (int(args_options.patch_length), int(args_options.patch_length))
    
    # batch_size -> batch_train / batch_val / batch_test
    if args_options.batch_train is not None:
        HP_BATCH_TRAIN = int(args_options.batch_train)
    if args_options.batch_val is not None:
        HP_BATCH_VAL   = int(args_options.batch_val)
    if args_options.batch_test is not None:
        HP_BATCH_TEST  = int(args_options.batch_test)
    
    if args_options.lr_l is not None:
        HP_LR_L = args_options.lr_l
        
    if args_options.wd_l is not None:
        HP_WD_L = args_options.wd_l
    
    optimizer_srss = optim.Adam(model_srss.parameters()
                               ,lr              = HP_LR_L
                               ,betas           = HP_BETAS_L
                               ,eps             = HP_EPS_L
                               ,weight_decay    = HP_WD_L
                               )
    
    update_dict_v2("", ""
                  ,"", "Model: " + str(_opt_sr)
                  ,"", "Loss: " + str(_str_loss)
                  ,"", ""
                  ,"", "옵티마이저 정보"
                  ,"", "optimizer: " + "torch.optim.Adam"
                  ,"", "learning_rate: " + str(HP_LR_L)
                  ,"", "weight decay: " + str(HP_WD_L)
                  ,"", "betas: " + str(HP_BETAS_L)
                  ,"", "eps: " + str(HP_EPS_L)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # scheduler
    HP_EPOCH                        = 3002         # 학습 종료 epoch
    HP_SCHEDULER_TOTAL_EPOCH        = 3000         # option for scheduler
    if _opt_sr == "IMDN":
        HP_SCHEDULER_STEP_SIZE = 500           # 반감기 (IMDN 설정 재현 불가능)
        HP_SCHEDULER_GAMMA     = 0.5           # Halfed
        scheduler_srss = optim.lr_scheduler.StepLR(optimizer = optimizer_srss
                                                  ,step_size = HP_SCHEDULER_STEP_SIZE
                                                  ,gamma     = HP_SCHEDULER_GAMMA
                                                  )
        _str_scheduler = "Step (optim.lr_scheduler.StepLR)"
        
    elif _opt_sr == "RFDN":
        HP_SCHEDULER_STEP_SIZE = 500           # 반감기 (RFDN 설정 재현 불가능)
        HP_SCHEDULER_GAMMA     = 0.5           # Halfed
        scheduler_srss = optim.lr_scheduler.StepLR(optimizer = optimizer_srss
                                                  ,step_size = HP_SCHEDULER_STEP_SIZE
                                                  ,gamma     = HP_SCHEDULER_GAMMA
                                                  )
        _str_scheduler = "Step (optim.lr_scheduler.StepLR)"
        
    elif _opt_sr == "BSRN":
        from utils.schedulers import CosineDecayLR
        HP_ETA_MIN = 1e-7   # BSRN official: code에서 float 1e-7로 설정값 확인
        scheduler_srss = CosineDecayLR(optimizer = optimizer_srss
                                      ,eta_min   = HP_ETA_MIN
                                      ,max_iter  = HP_SCHEDULER_TOTAL_EPOCH
                                      )
        _str_scheduler = "CosineDecayLR (재구현)"
        
    elif _opt_sr == "ESRT":
        HP_SCHEDULER_STEP_SIZE = 200           # 반감기     ESRT official: 200
        HP_SCHEDULER_GAMMA     = 0.5           # Halfed
        scheduler_srss = optim.lr_scheduler.StepLR(optimizer = optimizer_srss
                                                  ,step_size = HP_SCHEDULER_STEP_SIZE
                                                  ,gamma     = HP_SCHEDULER_GAMMA
                                                  )
        _str_scheduler = "Step (optim.lr_scheduler.StepLR)"
        
    elif _opt_sr == "LAPAR_A":
        from DLCs.super_resolution.model_lapar_a import CosineAnnealingLR_warmup
        HP_ETA_MIN = 1e-7   # LAPAR official: 1e-7
        scheduler_srss = CosineAnnealingLR_warmup(optimizer_srss
                                                 ,warm_iter   = 10
                                                 ,warm_factor = 0.1
                                                 ,base_lr     = HP_LR_L
                                                 ,min_lr      = HP_ETA_MIN
                                                 ,t_period    = [1000, 2000, 3000]    # total train iter = 3k
                                                 )
        _str_scheduler = "CosineAnnealingLR_warmup (LAPAR_A 자체 제공)"
        
    elif _opt_sr == "PAN":
        from DLCs.super_resolution.model_pan import CosineAnnealingLR_Restart
        HP_ETA_MIN = 1e-7   # PAN official: 1e-7
        scheduler_srss = CosineAnnealingLR_Restart(optimizer = optimizer_srss
                                                  ,T_period  = [750, 750, 750, 750]  # T_period used before next restart.
                                                                                     # cosine T_period 옵션. restart 전까지만 유효. restarts 보다 1개 많은 element 필요.
                                                  ,restarts  = [750, 1500, 2250]     # make cosine step to zero -> make LR init value
                                                                                     # cos 주기 초기화 iter 값. T_period 보다 1개 적은 element 필요.
                                                  ,weights   = [1, 1, 1]             # restart init LR multiplier. >1 will make larger initial LR.
                                                                                     # cos 주기 초기화 시, 시작 LR 배율값. >1 값 사용시, optimizer 설정값보다 큰 LR로 init
                                                  ,eta_min   = HP_ETA_MIN            # minimum LR. 
                                                                                     # cosine 그래프 최소값. 
                                                  )
        _str_scheduler = "CosineAnnealingLR_Restart (PAN 자체 제공)"
        
    update_dict_v2("", ""
                  ,"", "스케쥴러 정보"
                  ,"", "scheduler: " + str(_str_scheduler)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    #@>
    
    
    # if _opt_ss   == list_opt_ss[0]: # a
        # model_srss, _str        = DeepLab_v3_plus(num_classes = HP_CHANNEL_HYPO, pretrained = False), "D3P (a)"
        # hp_augm_random_scaler   = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        
    # elif _opt_ss == list_opt_ss[1]: # b
        # model_srss, _str        = DABNet(classes=HP_CHANNEL_HYPO), "DABNet (b)"
        # hp_augm_random_scaler   = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        
    # elif _opt_ss == list_opt_ss[2]: # c
        # model_srss, _str        = CGNet(classes=HP_CHANNEL_HYPO, M=3, N=21), "CGNet (c)"
        # hp_augm_random_scaler   = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0]
    
    
    # if HP_LABEL_ONEHOT_ENCODE:
        # criterion_srss = torch.nn.CrossEntropyLoss()
    # else:
        # criterion_srss = torch.nn.CrossEntropyLoss(ignore_index=HP_LABEL_VOID)
    
    
    # update_dict_v2("", ""
                  # ,"", "모델 정보"
                  # ,"", "Segmenation: " + _str
                  # ,"", "pretrained = False"
                  # ,"", ""
                  # ,"", "Loss 정보"
                  # ,"", "CrossEntropyLoss"
                  # ,in_dict = dict_log_init
                  # ,in_print_head = "dict_log_init"
                  # )
    
    is_norm_in_transform_to_tensor  = False
    HP_TS_NORM_MEAN                 = None
    HP_TS_NORM_STD                  = None
    
    #[optimizer]------------------------
    # if _opt_ss == list_opt_ss[0]:   # a, "D3P"
        # HP_LR_L,  HP_WD_L           = 1e-3, 2e-8
    # elif _opt_ss == list_opt_ss[1]: # b, "DABNet"
        # HP_LR_L,  HP_WD_L           = 1e-3, 2e-8
    # elif _opt_ss == list_opt_ss[2]: # c, "CGNet"
        # HP_LR_L,  HP_WD_L           = 1e-3, 2e-8
    
    
        
    # optimizer_srss = torch.optim.Adam(model_srss.parameters()
                                     # ,lr=HP_LR_L
                                     # ,weight_decay = HP_WD_L
                                     # )
    
    # update_dict_v2("", ""
                  # ,"", "옵티마이저 정보"
                  # ,"", "optimizer: " + "torch.optim.Adam"
                  # ,"", "learning_rate: " + str(HP_LR_L)
                  # ,"", "weight decay: "  + str(HP_WD_L)
                  # ,in_dict = dict_log_init
                  # ,in_print_head = "dict_log_init"
                  # )
    
    #[scheduler]----------------------------------------------
    # HP_EPOCH                        = 3002         # 학습 종료 epoch
    # HP_SCHEDULER_TOTAL_EPOCH        = 3000         # option for Poly
    # HP_SCHEDULER_POWER              = 0.9          #
    # # HP_SCHEDULER_UPDATE_INTERVAL    = "epoch"       # 스케줄려 갱신 간격 ("epoch" 또는 "batch")
    
    # HP_SCHEDULER_WARM_STEPS         = 200
    # HP_SCHEDULER_T_MAX              = 50            # (warm-up cos) 반주기
    # HP_SCHEDULER_ETA_MIN            = 1e-4          # (warm-up cos) 최소 LR
    # HP_SCHEDULER_STYLE              = "floor_4"     # (warm-up cos) 진동 형태
    
    # scheduler_srss = PolyLR(optimizer_srss
                           # ,max_epoch = HP_SCHEDULER_TOTAL_EPOCH
                           # ,power = HP_SCHEDULER_POWER
                           # )
    
    # scheduler_srss = Poly_Warm_Cos_LR(optimizer_srss
                                     # ,warm_up_steps=HP_SCHEDULER_WARM_STEPS
                                     # ,T_max=HP_SCHEDULER_T_MAX, eta_min=HP_SCHEDULER_ETA_MIN, style=HP_SCHEDULER_STYLE
                                     # ,power=HP_SCHEDULER_POWER, max_epoch=HP_SCHEDULER_TOTAL_EPOCH
                                     # )
    
    # update_dict_v2("", ""
                  # ,"", "스케쥴러 정보"
                  # ,"", "업데이트 간격: epoch" 
                  # ,"", "scheduler: Poly"
                  # ,"", "Power: "   + str(HP_SCHEDULER_POWER)
                  # # ,"", "Warm up: " + str(HP_SCHEDULER_WARM_STEPS)
                  # # ,"", "T max: "   + str(HP_SCHEDULER_T_MAX)
                  # # ,"", "ETA min: " + str(HP_SCHEDULER_ETA_MIN)
                  # # ,"", "Style: "   + str(HP_SCHEDULER_STYLE)
                  # ,in_dict = dict_log_init
                  # ,in_print_head = "dict_log_init"
                  # )
    
    
    #=========================================================================================
    
    trainer_(# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
            # 피클 생성기 사용여부
            make_pkl                        = make_pkl
            # 이상 감지
            ,HP_DETECT_LOSS_ANOMALY          = HP_DETECT_LOSS_ANOMALY
            # 학습을 빠르게 해주는 옵션
            ,WILL_SAVE_IMAGE                = WILL_SAVE_IMAGE
            ,SKIP_TEST_UNTIL                = SKIP_TEST_UNTIL
            
             # 사용할 데이터 셋
            ,HP_DATASET_NAME                = HP_DATASET_NAME
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
            ,HP_BATCH_VAL                   = HP_BATCH_VAL
            ,HP_BATCH_TEST                  = HP_BATCH_TEST
            ,HP_NUM_WORKERS                 = HP_NUM_WORKERS
            ,HP_VALID_WITH_PATCH            = HP_VALID_WITH_PATCH
            
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
            
            ,CALC_WITH_LOGIT                = CALC_WITH_LOGIT
            
             # (SRSS) optimizer, scheduler, loss
            ,model_srss                     = model_srss
            ,optimizer_srss                 = optimizer_srss
            ,scheduler_srss                 = scheduler_srss
            ,criterion_srss                 = criterion_srss
            
            #<<<
             # Label Dilation 설정 (Train에만 적용, default = False)
            ,HP_LABEL_DILATED               = False
            
             # Label One-Hot encoding 시행여부 (HP_LABEL_DILATED 사용시, 반드시 True)
            ,HP_LABEL_ONEHOT_ENCODE         = HP_LABEL_ONEHOT_ENCODE # True
            
             # Label Verify 설정 (Train에만 적용, default = False)
            ,HP_LABEL_VERIFY                = HP_LABEL_VERIFY
            ,HP_LABEL_VERIFY_TRY_CEILING    = HP_LABEL_VERIFY_TRY_CEILING
            ,HP_LABEL_VERIFY_CLASS_MIN      = HP_LABEL_VERIFY_CLASS_MIN
            ,HP_LABEL_VERIFY_RATIO_MAX      = HP_LABEL_VERIFY_RATIO_MAX
            
             # DataAugm- 관련 (colorJitter 포함)
            ,HP_AUGM_LITE                   = HP_AUGM_LITE              # SR 모드에서 lite 모드 augm 시행여부
            ,HP_AUGM_LITE_FLIP_HORI         = HP_AUGM_LITE_FLIP_HORI    # FLIP: 수평방향
            ,HP_AUGM_LITE_FLIP_VERT         = HP_AUGM_LITE_FLIP_VERT    # FLIP: 수직방향
            
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
            ,HP_DG_SCALE_FACTOR             = args_options.scale
            )
    
    _str = "End of main.py"
    warnings.warn(_str)
    print("End of main.py")
