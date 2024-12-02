# main_sr_test.py

if __name__ == '__main__':
    import torch
    
    from DLCs.model_deeplab_v3_plus     import DeepLab_v3_plus
    from testers.tester_ss              import tester_ss
    from _options                       import HP_LABEL_TOTAL, HP_LABEL_VOID, HP_COLOR_MAP
    
    model_ss_name = "D3P"
    
    path_model_state_dict   = "some_path"
    path_model_check_point  = "some_path"
    
    path_input_hr_images    = "/content/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/test/images"
    path_input_hr_labels    = "/content/LAB/datasets/project_use/CamVid_12_2Fold_v4/A_set/test/labels"
    path_input_lr_images    = "/content/LAB/datasets/project_use/CamVid_12_DLC_v1/x4_BILINEAR/images"
    path_input_sr_images    = "some_path"
    
    path_outputs            = "/content/LAB/result_files"
    
    
    if model_ss_name == "D3P":
        # DeepLab v3 Plus
        model_ss = DeepLab_v3_plus(num_classes = HP_LABEL_TOTAL - 1, pretrained = False)
        
        is_norm_in_transform_to_tensor = False
        HP_TS_NORM_STD = None
        HP_TS_NORM_MEAN = None
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ss.to(device)
    
    tester_ss(# SR 계열 모델로 생성된 이미지 저장하는 코드 (model, str)
              model                             = model_ss
             ,model_name                        = model_ss_name
              # path_model_state_dict 또는 path_model_check_point 중 1 가지 입력 (path_model_state_dict 우선 적용) (str)
             ,path_model_state_dict             = path_model_state_dict
             ,path_model_check_point            = path_model_check_point
             
              # 이미지 입력 폴더 경로 (str)
             ,path_input_hr_images              = path_input_hr_images
             ,path_input_hr_labels              = path_input_hr_labels
             ,path_input_lr_images              = path_input_lr_images
             ,path_input_sr_images              = path_input_sr_images
             ,path_outputs                      = path_outputs
             
              # 이미지 정규화 여부 (bool) 및 정규화 설정 (list) (정규화 안하면 설정값 생략 가능)
             ,is_norm_in_transform_to_tensor    = is_norm_in_transform_to_tensor
             ,HP_TS_NORM_MEAN                   = HP_TS_NORM_MEAN
             ,HP_TS_NORM_STD                    = HP_TS_NORM_STD
             
              # 라벨 정보
             ,HP_LABEL_TOTAL                    = HP_LABEL_TOTAL
             ,HP_LABEL_VOID                     = HP_LABEL_VOID
             ,HP_COLOR_MAP                      = HP_COLOR_MAP
             )
    
    
    
    print("EoF: main_ss_test.py")