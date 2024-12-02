# main_sr_test.py

if __name__ == '__main__':
    import torch
    
    from DLCs.super_resolution.model_esrt        import ESRT
    from DLCs.super_resolution.model_imdn        import IMDN
    from DLCs.super_resolution.model_bsrn        import BSRN
    from DLCs.super_resolution.model_rfdn        import RFDN
    from DLCs.super_resolution.model_pan         import PAN
    from DLCs.super_resolution.model_lapar_a import Network as LAPAR_A
    
    #from _private_models.Prop_9_NEW_Ab_51 import model_proposed

    from testers.tester_sr      import tester_sr
    
    
    PATH_BASE = "D:"
    #PATH_BASE = "E:"
    #PATH_BASE = "/content"
    
    #--- CamVid
    
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/train/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/val/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/test/images"

    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_2Fold_v4/B_set/train/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_2Fold_v4/B_set/val/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_2Fold_v4/B_set/test/images"
    
    
    #path_input_lr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_DLC_v1/x4_BILINEAR/images"
    
    
    #--- Minicity
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_2Fold_v1/A_set/train/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_2Fold_v1/A_set/val/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_2Fold_v1/A_set/test/images"
    
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_2Fold_v1/B_set/train/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_2Fold_v1/B_set/val/images"
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_2Fold_v1/B_set/test/images"
    
    #path_input_lr_images = PATH_BASE + "/LAB/datasets/project_use/MiniCity_19_DLC_v1/x4_BILINEAR/images"
    
    
    #--- CamVid 5 Fold
    
    #path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_5Fold_v1/A_set/test/images"
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_DLC_v1/original/images"
    # path_input_lr_images = PATH_BASE + "/LAB/datasets/project_use/CamVid_12_DLC_v1/x4_BILINEAR/images"
    
    
    #--- YCOR
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/A_set/train/images"
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/A_set/val/images"
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/A_set/test/images"
    
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/B_set/train/images"
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/B_set/val/images"
    # path_input_hr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/B_set/test/images"
    
    # list_path_input_hr_images = [PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/A_set/train/images"
                                # ,PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/A_set/val/images"
                                # ,PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/A_set/test/images"
                                # ]
    
    list_path_input_hr_images = [PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/B_set/train/images"
                                ,PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/B_set/val/images"
                                ,PATH_BASE + "/LAB/datasets/project_use/YCOR_9_2Fold_v1/B_set/test/images"
                                ]
    
    
    path_input_lr_images = PATH_BASE + "/LAB/datasets/project_use/YCOR_9_DLC_v1/x4_BILINEAR/images"
    
    
    
    #path_output_images   = None
    # path_output_images   = PATH_BASE + "/LAB/tmp/SR_Results"
    path_output_images   = PATH_BASE + "/LAB/tmp/SR_Results"
    
    method_name = None          # model 사용
    #method_name = "Bilinear"   # interpolation 사용
    
    
    for path_input_hr_images in list_path_input_hr_images:
        
        path_output_images   = PATH_BASE + "/LAB/tmp/SR_Results/" + "/".join(path_input_hr_images.split("/")[-3:])
        if method_name is not None:
            model_sr                        = None
            model_sr_name                   = None
            path_model_state_dict           = None
            path_model_check_point          = None
            is_norm_in_transform_to_tensor  = None
            HP_TS_NORM_STD                  = None
            HP_TS_NORM_MEAN                 = None
            
        else:
            
            # model_sr_name = "IMDN"
            # model_sr_name = "RFDN"
            # model_sr_name = "BSRN"
            # model_sr_name = "ESRT"
            model_sr_name = "PAN"
            # model_sr_name = "LAPAR"
            
            #model_sr_name = "MPRNet"    #prop_9_new도 같은 옵션으로 사용가능
            
            
            #--- IMDN
            if model_sr_name == "IMDN":
                pass
                # _path_msd = "D:/LAB/[보존_자료]/CamVid 12 5 Fold v1/1 SR - 1500/1 IMDN - scheduler v4"
                # path_model_state_dict = _path_msd + "/A1 IMDN_scheduler_v4_CamVid_12_5Fold_v1_A_set - 23.3403" + "/models/state_dicts/" + "1380_model_state_dict.pt"   # A_set
                # path_model_state_dict = _path_msd + "/B1 IMDN_scheduler_v4_CamVid_12_5Fold_v1_B_set - 23.3421" + "/models/state_dicts/" + "1486_model_state_dict.pt"   # B_set
                # path_model_state_dict = _path_msd + "/C1 IMDN_scheduler_v4_CamVid_12_5Fold_v1_C_set - 23.2425" + "/models/state_dicts/" + "1491_model_state_dict.pt"   # C_set
                # path_model_state_dict = _path_msd + "/D1 IMDN_scheduler_v4_CamVid_12_5Fold_v1_D_set - 23.2084" + "/models/state_dicts/" + "1441_model_state_dict.pt"   # D_set
                # path_model_state_dict = _path_msd + "/E1 IMDN_scheduler_v4_CamVid_12_5Fold_v1_E_set - 23.1631" + "/models/state_dicts/" + "1479_model_state_dict.pt"   # E_set
                
                # path_model_state_dict = "D:/LAB/result_files/결과정리하기 - Revision 2nd/IMDN_x4_YCOR_9_2Fold_v1_A_set-25.7041/models/state_dicts/SR_2796_l_msd.pt" # L3
                path_model_state_dict = "D:/LAB/result_files/결과정리하기 - Revision 2nd/IMDN_x4_YCOR_9_2Fold_v1_B_set-25.6739/models/state_dicts/SR_2970_l_msd.pt" # L3
            
            #--- RFDN
            elif model_sr_name == "RFDN":
                pass
                # _path_msd = "D:/LAB/[보존_자료]/CamVid 12 5 Fold v1/1 SR - 1500/2 RFDN"
                # path_model_state_dict = _path_msd + "/A4 RFDN_CamVid_12_5Fold_v1_A_set - 23.0188" + "/models/state_dicts/" + "1500_model_state_dict.pt"   # A_set
                # path_model_state_dict = _path_msd + "/B1 RFDN_CamVid_12_5Fold_v1_B_set - 23.0197" + "/models/state_dicts/" + "1449_model_state_dict.pt"   # B_set
                # path_model_state_dict = _path_msd + "/C5 RFDN_CamVid_12_5Fold_v1_C_set - 22.9950" + "/models/state_dicts/" + "1490_model_state_dict.pt"   # C_set
                # path_model_state_dict = _path_msd + "/D4 RFDN_CamVid_12_5Fold_v1_D_set - 22.9413" + "/models/state_dicts/" + "1500_model_state_dict.pt"   # D_set
                # path_model_state_dict = _path_msd + "/E2 RFDN_CamVid_12_5Fold_v1_E_set - 22.8864" + "/models/state_dicts/" + "1439_model_state_dict.pt"   # E_set
                
                path_model_state_dict = "R:/[보존_자료]/[paper_1st - 2nd revision]/RFDN_x4_Re1_YCOR_9_2Fold_v1_A_set-25.6081/models/state_dicts/SR_2980_l_msd.pt" # L0
                # path_model_state_dict = "R:/[보존_자료]/[paper_1st - 2nd revision]/RFDN_x4_Re1_YCOR_9_2Fold_v1_B_set-25.5821/models/state_dicts/SR_2985_l_msd.pt" # L0
                
            #--- BSRN
            elif model_sr_name == "BSRN":
                pass
                #_ path_msd = "D:/LAB/[보존_자료]/CamVid 12 5 Fold v1/1 SR - 1500/3 BSRN"   # L0
                # _path_msd = "D:/LAB/[보존_자료]/CamVid 12 5_Fold/SR - 1500/3 BSRN"   # L2
                
                # path_model_state_dict = _path_msd + "/A1 BSRN_CamVid_12_5Fold_v1_A_set - 23.1685" + "/models/state_dicts/" + "1270_model_state_dict.pt"   # A_set (L2)
                # path_model_state_dict = _path_msd + "/B4 BSRN_CamVid_12_5Fold_v1_B_set - 23.1744" + "/models/state_dicts/" + "1263_model_state_dict.pt"   # B_set (L0)
                # path_model_state_dict = _path_msd + "/C4 BSRN_CamVid_12_5Fold_v1_C_set - 23.0717" + "/models/state_dicts/" + "1256_model_state_dict.pt"   # C_set (L0)
                # path_model_state_dict = _path_msd + "/D5 BSRN_CamVid_12_5Fold_v1_D_set - 23.0514" + "/models/state_dicts/" + "1268_model_state_dict.pt"   # D_set (L0)
                # path_model_state_dict = _path_msd + "/E1 BSRN_CamVid_12_5Fold_v1_E_set - 23.0357" + "/models/state_dicts/" + "1121_model_state_dict.pt"   # E_set (L0)
                
                # path_model_state_dict = "D:/LAB/result_files/결과정리하기 - Revision 2nd/BSRN_x4_YCOR_9_2Fold_v1_A_set-25.6598/models/state_dicts/SR_1642_l_msd.pt" # L3
                # path_model_state_dict = "R:/[보존_자료]/[paper_1st - 2nd revision]/BSRN_x4_Re1_YCOR_9_2Fold_v1_B_set-25.6363/models/state_dicts/SR_2612_l_msd.pt" # L0
                
            #--- ESRT
            elif model_sr_name == "ESRT":
                pass
                # _path_msd = "D:/LAB/[보존_자료]/CamVid 12 5 Fold v1/SR - 1500/4 ESRT"
                # path_model_state_dict = _path_msd + "/A1 ESRT_CamVid_12_5Fold_v1_A_set - 23.1630" + "/models/state_dicts/" + "1116_model_state_dict.pt"   # A_set
                # path_model_state_dict = _path_msd + "/B1 ESRT_CamVid_12_5Fold_v1_B_set - 23.1645" + "/models/state_dicts/" + "1363_model_state_dict.pt"   # B_set
                # path_model_state_dict = _path_msd + "/C1 ESRT_CamVid_12_5Fold_v1_C_set - 23.0537" + "/models/state_dicts/" + "1428_model_state_dict.pt"   # C_set
                # path_model_state_dict = _path_msd + "/D1 ESRT_CamVid_12_5Fold_v1_D_set - 23.0307" + "/models/state_dicts/" + "1268_model_state_dict.pt"   # D_set
                # path_model_state_dict = _path_msd + "/E1 ESRT_CamVid_12_5Fold_v1_E_set - 22.9789" + "/models/state_dicts/" + "1047_model_state_dict.pt"   # E_set
                
                path_model_state_dict = "D:/LAB/result_files/결과정리하기 - Revision 2nd/ESRT_x4_YCOR_9_2Fold_v1_A_set-25.5547/models/state_dicts/SR_1378_l_msd.pt" # L3
                # path_model_state_dict = "D:/LAB/result_files/결과정리하기 - Revision 2nd/ESRT_x4_YCOR_9_2Fold_v1_B_set-25.5208/models/state_dicts/SR_1297_l_msd.pt" # L3
                
            #--- PAN
            elif model_sr_name == "PAN":
                pass
                # _path_msd = "D:/LAB/[보존_자료]/CamVid 12 5 Fold v1/SR - 1500/5 PAN"
                # path_model_state_dict = _path_msd + "/A1 PAN_CamVid_12_5Fold_v1_A_set - 23.3071" + "/models/state_dicts/" + "1402_model_state_dict.pt"   # A_set
                # path_model_state_dict = _path_msd + "/B1 PAN_CamVid_12_5Fold_v1_B_set - 23.3122" + "/models/state_dicts/" + "1431_model_state_dict.pt"   # B_set
                # path_model_state_dict = _path_msd + "/C1 PAN_CamVid_12_5Fold_v1_C_set - 23.2326" + "/models/state_dicts/" + "1360_model_state_dict.pt"   # C_set
                # path_model_state_dict = _path_msd + "/D1 PAN_CamVid_12_5Fold_v1_D_set - 23.1979" + "/models/state_dicts/" + "1339_model_state_dict.pt"   # D_set
                # path_model_state_dict = _path_msd + "/E1 PAN_CamVid_12_5Fold_v1_E_set - 23.1472" + "/models/state_dicts/" + "1386_model_state_dict.pt"   # E_set
                
                # path_model_state_dict = "R:/[보존_자료]/[paper_1st - 2nd revision]/PAN_x4_Re1_YCOR_9_2Fold_v1_A_set-25.7441/models/state_dicts/SR_2879_l_msd.pt" # L0
                path_model_state_dict = "R:/[보존_자료]/[paper_1st - 2nd revision]/PAN_x4_Re1_YCOR_9_2Fold_v1_B_set-25.7185/models/state_dicts/SR_2789_l_msd.pt" # L0
                
            #--- LAPAR
            elif model_sr_name == "LAPAR":
                pass
                # _path_msd = "D:/LAB/[보존_자료]/CamVid 12 5 Fold v1/SR - 1500/6 LAPAR_A"
                # path_model_state_dict = _path_msd + "/A1 LAPAR_A_CamVid_12_5Fold_v1_A_set - 23.1655" + "/models/state_dicts/" + "1455_model_state_dict.pt"   # A_set
                # path_model_state_dict = _path_msd + "/B1 LAPAR_A_CamVid_12_5Fold_v1_B_set - 23.1674" + "/models/state_dicts/" + "1439_model_state_dict.pt"   # B_set
                # path_model_state_dict = _path_msd + "/C1 LAPAR_A_CamVid_12_5Fold_v1_C_set - 23.0397" + "/models/state_dicts/" + "1375_model_state_dict.pt"   # C_set
                # path_model_state_dict = _path_msd + "/D1 LAPAR_A_CamVid_12_5Fold_v1_D_set - 23.0404" + "/models/state_dicts/" + "1431_model_state_dict.pt"   # D_set
                # path_model_state_dict = _path_msd + "/E1 LAPAR_A_CamVid_12_5Fold_v1_E_set - 22.9335" + "/models/state_dicts/" + "1369_model_state_dict.pt"   # E_set
                
                path_model_state_dict = "R:/[보존_자료]/[paper_1st - 2nd revision]/LAPAR_A_x4_Re1_YCOR_9_2Fold_v1_A_set-25.5206/models/state_dicts/SR_2892_l_msd.pt" # L0
                # path_model_state_dict = "D:/LAB/result_files/결과정리하기 - Revision 2nd/LAPAR_A_x4_YCOR_9_2Fold_v1_B_set-25.5255/models/state_dicts/SR_2892_l_msd.pt" # L3
                
            
            
            
            path_model_check_point  = "some_path"
            is_patch_wise_inference = False
            if model_sr_name == "IMDN":
                # IMDN
                model_sr = IMDN(upscale=4)
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
                
            elif model_sr_name == "RFDN":
                # RFDN
                model_sr = RFDN(upscale=4)
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
                
            elif model_sr_name == "BSRN":
                # BSRN
                model_sr = BSRN(upscale=4)
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
                
            elif model_sr_name == "ESRT":
                # ESRT
                model_sr = ESRT(upscale=4)
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
                # is_patch_wise_inference            = True # Minicity만 True
                
                
            elif model_sr_name == "PAN":
                # PAN
                model_sr = PAN(scale=4)
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
                
            elif model_sr_name == "LAPAR":
                # PAN
                model_sr = LAPAR_A(scale=4)
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
                
            elif model_sr_name == "MPRNet":
                #MPRNet, Prop_9_new Ab_51 ~ 54
                model_sr = model_proposed()
                is_norm_in_transform_to_tensor  = False
                HP_TS_NORM_STD                  = None
                HP_TS_NORM_MEAN                 = None
            
            
            print("path_input_hr_images: ", path_input_hr_images)
            print("path_model_state_dict:", path_model_state_dict)
            print("path_output_images:   ", path_output_images)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_sr.to(device)
        
        
        tester_sr(# SR 모델 혹은 알고리즘으로 SR 시행한 이미지 저장하는 코드
                  #--- (선택 1/2) 알고리즘으로 생성된 이미지 저장하는 옵션 (str) -> None 입력 시 (선택 2/2) 옵션 적용됨
                  method_name                       = method_name
                 ,is_patch_wise_inference           = is_patch_wise_inference
                  #--- (선택 2/2) SR 계열 모델로 생성된 이미지 저장하는 옵션 (model, str)
                 ,model                             = model_sr
                 ,model_name                        = model_sr_name
                  # path_model_state_dict 또는 path_model_check_point 중 1 가지 입력 (path_model_state_dict 우선 적용) (str)
                 ,path_model_state_dict             = path_model_state_dict
                 ,path_model_check_point            = path_model_check_point
                 
                  # 이미지 입출력 폴더 경로 (str)
                 ,path_input_hr_images              = path_input_hr_images
                 ,path_input_lr_images              = path_input_lr_images
                 # ,path_output_images                = None
                 ,path_output_images                = path_output_images + "/" + model_sr_name
                 
                  # 이미지 정규화 여부 (bool) 및 정규화 설정 (list) (정규화 안하면 설정값 생략 가능)
                 ,is_norm_in_transform_to_tensor    = is_norm_in_transform_to_tensor
                 ,HP_TS_NORM_MEAN                   = HP_TS_NORM_MEAN
                 ,HP_TS_NORM_STD                    = HP_TS_NORM_STD
                 )
    
    
    
    print("EoF: main_sr_test.py")