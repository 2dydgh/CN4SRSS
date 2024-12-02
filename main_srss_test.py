# main_sr_test.py

if __name__ == '__main__':
    import torch
    
    #from DLCs.model_esrt        import ESRT
    #from DLCs.model_imdn        import IMDN
    #from DLCs.model_bsrn        import BSRN
    #from DLCs.model_rfdn        import RFDN

    from _private_models.Prop_9_NEW_Ab_51 import model_proposed

    from testers.tester_sr      import tester_sr
    
    #path_input_hr_images    = "D:/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/train/images"
    #path_input_hr_images    = "D:/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/val/images"
    #path_input_hr_images    = "D:/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/test/images"

    #path_input_hr_images = "D:/LAB/datasets/project_use/CamVid_12_2Fold_v4/B_set/train/images"
    #path_input_hr_images = "D:/LAB/datasets/project_use/CamVid_12_2Fold_v4/B_set/val/images"
    path_input_hr_images = "D:/LAB/datasets/project_use/CamVid_12_2Fold_v4/B_set/test/images"

    path_input_lr_images    = "D:/LAB/datasets/project_use/CamVid_12_DLC_v1/x4_BILINEAR/images"
    
    path_output_images      = "D:/LAB/tmp/Ab_07"
    
    


    model_sr_name = "Ab_07"    #prop_9_new도 같은 옵션으로 사용가능


    path_model_state_dict  = "D:/LAB/[보존자료]/CamVid 12 v4/SRSS/Ablation - CGNet/Ab_51 - SR/A2 v3_Prop_9_NEW_Ab_51_CamVid_12_2Fold_v4_A_set - 23.0033/models/state_dicts/1380_model_state_dict.pt"


    path_model_check_point = "some_path"
    

    print("path_input_hr_images", path_input_hr_images)
    print("path_model_state_dict", path_model_state_dict)

    if model_srss_name == "Ab_07":
        # Ab_07
        model_srss = model_proposed()
        is_norm_in_transform_to_tensor  = False
        HP_TS_NORM_STD                  = None
        HP_TS_NORM_MEAN                 = None
        
    elif model_srss_name == "Ab_01":
        # Ab_01
        model_srss = model_proposed()
        is_norm_in_transform_to_tensor  = False
        HP_TS_NORM_STD                  = None
        HP_TS_NORM_MEAN                 = None
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_srss.to(device)
    
    
    tester_srss  (# SR 모델 혹은 알고리즘으로 SR 시행한 이미지 저장하는 코드
                 
                  #--- (선택 2/2) SR 계열 모델로 생성된 이미지 저장하는 옵션 (model, str)
                 ,model                             = model_srss
                 ,model_name                        = model_srss_name
                  # path_model_state_dict 또는 path_model_check_point 중 1 가지 입력 (path_model_state_dict 우선 적용) (str)
                 ,path_model_state_dict             = path_model_state_dict
                 ,path_model_check_point            = path_model_check_point
                 
                  # 이미지 입출력 폴더 경로 (str)
                 ,path_input_hr_images              = path_input_hr_images
                 ,path_input_lr_images              = path_input_lr_images
                 ,path_output_images                = path_output_images
                 
                  # 이미지 정규화 여부 (bool) 및 정규화 설정 (list) (정규화 안하면 설정값 생략 가능)
                 ,is_norm_in_transform_to_tensor    = is_norm_in_transform_to_tensor
                 ,HP_TS_NORM_MEAN                   = HP_TS_NORM_MEAN
                 ,HP_TS_NORM_STD                    = HP_TS_NORM_STD
                 )
    
    
    
    print("EoF: main_srss_test.py")