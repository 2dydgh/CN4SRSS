"""
Yamaha-CMU-Off-Road (YCOR) dataset

1024x544, 1076 장 (931 train, 145 validation)

Paper: https://www.ri.cmu.edu/app/uploads/2017/11/semantic-mapping-offroad-nav-compressed.pdf
Download: https://theairlab.org/yamaha-offroad-dataset/
Color map: https://gist.github.com/GerardMaggiolino/258a65077d43d4e176e0fb0240a49edb


LABEL_TO_COLOR = OrderedDict({
     "background": [255, 255, 255],
     "high_vegetation": [40, 80, 0],
     "traversable_grass": [128, 255, 0],
     "smooth_trail": [178, 176, 153],
     "obstacle": [255, 0, 0],
     "sky": [1, 88, 255],
     "rough_trial": [156, 76, 30],
     "puddle": [255, 0, 128],
     "non_traversable_low_vegetation": [0, 160, 0]
})



    HP_LABEL_TOTAL          = 9
    HP_LABEL_VOID           = 8
    HP_LABEL_ONEHOT_ENCODE  = False
    
    HP_DATASET_CLASSES = ("0(SmoothTrail),1(Grass),2(RoughTrial),3(Puddle),4(Obstacle),"
                         +"5(LowVegetation),6(HighVegetation),7(Sky),8(Background)"
                         )
    
    HP_COLOR_MAP = {0:  [178, 176, 153]     # 00 SmoothTrail
                   ,1:  [128, 255,   0]     # 01 Grass
                   ,2:  [156,  76,  30]     # 02 RoughTrial
                   ,3:  [255,   0, 128]     # 03 Puddle
                   ,4:  [255,   0,   0]     # 04 Obstacle
                   ,5:  [  0, 160,   0]     # 05 LowVegetation
                   ,6:  [ 40,  80,   0]     # 06 HighVegetation
                   ,7:  [  1,  88, 255]     # 07 Sky
                   ,8:  [255, 255, 255]     # 08 Background
                   }


"""
import os
import sys
import shutil
import random
import datetime

import csv
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def path_merge(path_1, path_2):
    if path_1[-1] != "/":
        path_1 += "/"
    
    if path_2[0] == "/":
        return path_1 + path_2[1:]
    else:
        return path_1 + path_2


def path_check(path_dst):
    if not os.path.exists(path_dst):
        os.makedirs(path_dst)

def label_rgb_2_gray(label_rgb, color_id):
    np_rgb = np.array(label_rgb)
    # print(np_rgb.shape)
    
    _H, _W, _C = np_rgb.shape
    
    # print(_H, _W, _C)
    
    # np_canvas = np.zeros((_H, _W))
    # np_canvas: 1 / RRR / GGG / BBB
    # np_canvas = np.ones((_H, _W), dtype=np.uint64) * 10000000000
    np_canvas = np.zeros((_H, _W), dtype=np.uint64) + 1000000000
    
    np_label = np.zeros((_H, _W), dtype=np.uint8)
    
    # print(np_rgb[:, :, 0])
    
    np_canvas += np_rgb[:, :, 0].astype(np.uint64) * 1000000
    np_canvas += np_rgb[:, :, 1].astype(np.uint64) * 1000
    np_canvas += np_rgb[:, :, 2].astype(np.uint64)
    
    # print(np_rgb)
    # print(np_rgb[:, :, 0])
    # print(np_rgb[:, :, 1])
    # print(np_rgb[:, :, 2])
    # print(np_canvas)
    
    for i_code in color_id:
        i_class = color_id[i_code]
        np_label += np.where(np_canvas==i_code, i_class, 0).astype(np.uint8)
    
    # print(np_label)
    
    return Image.fromarray(np_label)


#PIL 이미지 출력
def imshow_pil(in_pil, **kargs):
    '''
    imshow_pil(#pil image show with plt function
               in_pil
               #(선택) (tuple) 출력 크기
              ,figsize = (,)
               #(선택) (bool) pil 이미지 정보 출력 여부 (default = True)
              ,print_info = 
              )
    '''
    
    try:
        plt.figure(figsize = kargs['figsize'])
    except:
        pass
    plt.imshow(np.array(in_pil))
    plt.show()
    
    try:
        print_info = kargs['print_info']
    except:
        print_info = True
    
    if print_info:
        try:
            print("Format:", in_pil.format, "  Mode:", in_pil.mode, "  Size (w,h):", in_pil.size)
        except:
            print("Format: No Info", "  Mode:", in_pil.mode, "  Size (w,h):", in_pil.size)
    

#=== End of imshow_pil

def imshow_pils(list_pils, list_titles, rows=2, cols=2, title=""):
    fig = plt.figure(figsize=(12,8))
    if title != "":
        fig.suptitle(title)
    
    if len(list_pils) != rows * cols:
        print("length of list_pils are not right.")
        sys.exit(-9)
    
    for i in range(rows * cols):
        i += 1
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(list_pils[i-1])
        ax.set_xlabel(list_titles[i-1])
        
    plt.show()


#IN (2): 
#        (pil)  gray label
#        (dict) label_color_map
#OUT(1): (pil) RGB 이미지
def label_2_RGB(in_pil, in_label_rgb):
    
    in_np = np.array(in_pil)
    in_h, in_w = in_np.shape
    
    out_np_rgb = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    
    for label, rgb in in_label_rgb.items():
        out_np_rgb[in_np == label, :] = rgb
    
    return Image.fromarray(out_np_rgb)

#===

'''
update_dict_v2("", "추가할 내용 2-1"
              ,"", "추가할 내용 2-2"
              ,in_dict_dict = {"a":dict_a, "b":dict_b, "c":dict_c}
              ,in_dict_key = "c"
              ,print_head = "sample"
              ,is_print = True
              )
'''
#IN (* 짝수 개수):
#           홀: (str) key
#           짝: (str) value
#IN (**1 or 2):
#           <case 1>
#           in_dict  : (dict) 내용을 추가할 변수
#           <case 2>
#           in_dict_dict : (dict) in_dict 후보로 구성된 dict
#           in_dict_key  : (str)  이번에 갱신할 dict의 key
#
#IN(** 생략가능)
#           in_print_head : (str) 출력문 말머리 (변경됨: in_dict_name -> in_print_head)
#           is_print: (T/F) 결과 출력여부 (default = True)
#dict 형 요소 추가
def update_dict_v2(*args, **kargs):
    name_func = "[update_dict_v2] ->"
    #dict = {key : value}, 별도 함수 없이 append & 요소 수정 가능 / dict 변수 생성은 불가능 (사전선언 필수)
    
    try:
        in_dict = kargs['in_dict']
    except:
        #후보 dict의 dict
        in_dict_dict = kargs['in_dict_dict']
        #이번에 update 할 dict의 key (str)
        in_dict_key = kargs['in_dict_key']
        #dict 결정
        in_dict = in_dict_dict[in_dict_key]
        
    try:
        print_head = kargs['in_print_head']
    except:
        print_head = "False"
    
    try:
        is_print = kargs['is_print']
    except:
        is_print = True
    
    flag_elem = 0
    
    for elem in args:
        if type(elem) != type(" "):
            print(name_func, "입력값은 str 형태만 가능합니다.")
            sys.exit(1)
            
        if flag_elem == 0:
            in_key = elem
            if in_key == "" or in_key == " ":
                in_key = str(len(in_dict))
            flag_elem = 1
        else:
            in_value = elem
            in_dict[in_key] = in_value
            if print_head == "False":
                if is_print:
                    print(name_func, "{", in_key, ":", in_value, "}")
            else:
                if is_print:
                    print(name_func, print_head, "{", in_key, ":", in_value, "}")
            flag_elem = 0

#=== End of update_dict_v2

#dict를 txt 파일로 저장
#IN (**3 or 4)
#       in_file_path: 저장경로 지정
#       in_file_name: 파일 이름 + (txt or csv)
#       <case 1>
#       in_dict: 딕셔너리 변수
#       <case 2>
#       in_dict_dict : (dict) in_dict 후보로 구성된 dict
#       in_dict_key  : (str)  이번에 저장할 dict의 key
def dict_2_txt_v2(**kargs):
    #파일 경로
    in_file_path = kargs['in_file_path']
    #파일 이름
    in_file_name = kargs['in_file_name']
    
    try:
        #딕셔너리 변수 바로 선택
        in_dict = kargs['in_dict']
    except:
        #후보 dict의 dict
        in_dict_dict = kargs['in_dict_dict']
        #이번에 update 할 dict의 key (str)
        in_dict_key = kargs['in_dict_key']
        #dict 결정
        in_dict = in_dict_dict[in_dict_key]
    
    in_keys = list(in_dict.keys())
    
    if in_file_path[-1] == "/":
        in_file_name = in_file_path + in_file_name
    else:
        in_file_name = in_file_path + "/" + in_file_name
    
    if not os.path.exists(in_file_path):
        os.makedirs(in_file_path)
    
    try:
        #기존 파일 덮어쓰기
        file_txt = open(in_file_name, 'w')
        try:
            list_keys = list(in_dict.keys())
            list_values = list(in_dict.values())
            for i_list in range(len(list_keys)):
                file_txt.write(str(list_keys[i_list]) + "," + str(list_values[i_list]) + "\n")
                
        except:
            print("(exc) dict access FAIL")
            sys.exit(1)
        
        file_txt.close()
        print("dict -> txt finished:", in_file_name)
    except:
        print("(exc) file open FAIL")

#=== End of dict_2_txt_v2

def csv_2_dict(**kargs):
    '''
    dict_from_csv = csv_2_dict(path_csv = "./aaa/bbb.csv")
    '''
    
    #첫 열을 key로, 나머지 열을 value로서 list 형 묶음으로 저장한 dict 변수 생성
    #csv 파일 경로
    path_csv = kargs['path_csv']

    file_csv = open(path_csv, 'r', encoding = 'utf-8')
    lines_csv = csv.reader(file_csv)
    
    dict_csv = {}
    
    for line_csv in lines_csv:
        item_num = 0
        items_tmp = []
        for item_csv in line_csv:
            #print(item_csv)
            if item_num != 0:
                items_tmp.append(item_csv)
            item_num += 1
        dict_csv[line_csv[0]] = items_tmp
        
    file_csv.close()
    
    return dict_csv

#=== End of csv_2_dict


def degradation_total_v7(**kargs):
    func_name = "[degradation_total_v7] -->"
    #--- 변경사항
    # 
    #   1. Degradation 1회만 시행
    #   2. Degradation 결과물을 원본 크기로 복원하는 단계 시행 안함
    #      -> scale_factor 만큼 그대로 작아진 결과물 배출
    #   3. 노이즈 (Color / Gray) 생성 방식은 Real-ESRGAN 방식 그대로 적용
    #      -> Color: RGB 채널에 서로 다른 노이즈 생성 
    #      -> Gray : RGB 채널에 서로 같은 노이즈 생성
    #---
    
    '''
    사용 예시
    pil_img = degradation_total_v7(in_pil =
                                  ,is_return_options = 
                                  #--블러
                                  ,in_option_blur = "Gaussian"
                                  #-- 다운 스케일
                                  ,in_scale_factor = 
                                  ,in_option_resize = 
                                  #--노이즈
                                  ,in_option_noise = "Gaussian"
                                  #노이즈 시그마값 범위 (tuple)
                                  ,in_range_noise_sigma = 
                                  #Gray 노이즈 확룔 (int)
                                  ,in_percent_gray_noise = 
                                  #노이즈 고정값 옵션
                                  ,is_fixed_noise = 
                                  ,in_fixed_noise_channel =
                                  ,in_fixed_noise_sigma   = 
                                  )
    '''

    #IN(**):
    #       (선택, str)        in_path                 : "/..."
    #       (대체 옵션, pil)    in_pil                  : pil_img

    #고정값   (str)             in_option_blur         : "Gaussian"
    #       (선택, 2d npArray) kernel_blur            : np커널
    #
    #중지됨   (int or tuple)    in_resolution          : (사용금지) in_scale_factor로 변경됨
    #       (int or tuple)    in_scale_factor        : 스케일 팩터 (1 ~ ) -> tuple 입력시, 범위 내 균등추출 (소수점 1자리까지 사용)
    #       (str)             in_option_resize       : "AREA", "BILINEAR", "BICUBIC"
    #
    #고정값   (str)             in_option_noise        : "Gaussian"
    #       (tuple)           in_range_noise_sigma   : ((float), (float))
    #       (int)             in_percent_gray_noise  : Gray 노이즈 확률 (그 외엔 Color 노이즈로 생성), 최대 100

    #       (선택, bool)       is_fixed_noise         : 노이즈 옵션지정여부 (val & test용 , default = False)
    #       (선택, str)        in_fixed_noise_channel : 노이즈 발생 채널 지정 (val & test용) ("Color" or "Gray")
    #       (선택, str)        in_fixed_noise_sigma   : 노이즈 시그마값 지정  (val & test용)

    #       (bool)            is_return_options      : degrad- 옵션 return 여부

    #OUT(1):
    #       (PIL)             이미지
    #       (선택, str)             Degradation 옵션
    #--- --- ---

    #degrad- 옵션 return 여부
    try:
        is_return_options = kargs['is_return_options']
    except:
        is_return_options = False
    
    #(str) 사용된 degrad 옵션 저장
    return_option = ""
    
    #(str) 파일 경로 or (pil) 이미지 입력받음
    try:
        in_path = kargs['in_path']
        in_cv = cv2.imread(in_path)
    except:
        in_cv = cv2.cvtColor(np.array(kargs['in_pil']), cv2.COLOR_RGB2BGR)
    
    #입력 이미지 크기
    in_h, in_w, _ = in_cv.shape
    
    #***--- degradation_blur
    #(str) blur 방식
    in_option_blur = kargs['in_option_blur']
    
    return_option += "Blur = " + in_option_blur
    
    #평균 필터
    if in_option_blur == "Mean" or in_option_blur == "mean":
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size) / (kernel_size * kernel_size))
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #가우시안 필터
    elif in_option_blur == "Gaussian" or in_option_blur == "gaussian":
        kernel_size = 3 #홀수만 가능
        kernel_sigma = 0.1
        kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma) * cv2.getGaussianKernel(kernel_size, kernel_sigma).T
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #기타 필터 (sinc 용)
    elif in_option_blur == "Custom" or in_option_blur == "custom":
        kernel = kargs['kernel_blur']
        out_cv_blur = cv2.filter2D(in_cv, -1, kernel)
    
    #***--- degradation_resolution
    
    #scale factor (float, 소수점 2자리 까지 사용)
    if type(kargs['in_scale_factor']) == type((0, 1)):
        in_scale_factor = round(random.uniform(kargs['in_scale_factor'][0], kargs['in_scale_factor'][-1]), 2)
    else:
        in_scale_factor = round(kargs['in_scale_factor'], 2)
    
    #최소값 clipping
    min_scale_factor = 0.25
    if in_scale_factor < min_scale_factor:
        print(func_name, "scale factor clipped to", min_scale_factor)
        in_scale_factor = min_scale_factor
    
    #(str) resize 옵션 ("AREA", "BILINEAR", "BICUBIC" / 소문자 가능)
    try:
        in_option_resize = kargs['in_option_resize']
    except:
        #default: BILINEAR
        in_option_resize = "BILINEAR"
    
    tmp_s_f = 1 / in_scale_factor
    if in_option_resize == "AREA" or in_option_resize == "area":
        tmp_interpolation = cv2.INTER_AREA
    elif in_option_resize == "BILINEAR" or in_option_resize == "bilinear":
        tmp_interpolation = cv2.INTER_LINEAR
    elif in_option_resize == "BICUBIC" or in_option_resize == "bicubic":
        tmp_interpolation = cv2.INTER_CUBIC
    
    out_cv_resize = cv2.resize(out_cv_blur, dsize=(0,0), fx=tmp_s_f, fy=tmp_s_f
                              ,interpolation = tmp_interpolation
                              )
    
    #감소된 크기 계산
    out_h, out_w, _ = out_cv_resize.shape
    
    return_option += ", Downscale(x" + str(in_scale_factor) + ") = " + in_option_resize
    
    #***--- degradation 노이즈 추가 (Color or Gray)
    
    #채널 분할
    in_cv_b, in_cv_g, in_cv_r = cv2.split(out_cv_resize)
    
    #노이즈 옵션 고정값 사용여부
    try:
        is_fixed_noise = kargs['is_fixed_noise']
    except:
        is_fixed_noise = False
    
    #노이즈 종류 선택 (Gaussian or Poisson) -> Poisson 사용 안함
    try:
        in_option_noise = kargs['in_option_noise']
    except:
        in_option_noise = "Gaussian"
    
    #노이즈 생성 (Gaussian)
    if in_option_noise == "Gaussian":
        in_noise_mu = 0 #뮤 =고정값 적용
        
        #노이즈 옵션이 지정된 경우
        if is_fixed_noise:
            #노이즈 발생 채널
            in_noise_channel = kargs['in_fixed_noise_channel']
            #시그마 값
            in_noise_sigma = int(kargs['in_fixed_noise_sigma'])
        #노이즈 옵션이 지정되지 않은 경우
        else:
            #노이즈 발생 채널 추첨
            in_percent_gray_noise = kargs['in_percent_gray_noise']
            in_noise_channel = random.choices(["Color", "Gray"]
                                             ,weights = [(100 - in_percent_gray_noise), in_percent_gray_noise]
                                             ,k = 1
                                             )[0]
            #시그마 값
            in_noise_sigma = int(random.uniform(kargs['in_range_noise_sigma'][0], kargs['in_range_noise_sigma'][-1]))
        
        #Color 노이즈 발생 (채널별 다른 노이즈 발생)
        if in_noise_channel == "Color":
            noise_r = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_g = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_b = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
        #Gray 노이즈 발생 (모든 채널 동일 노이즈 발생)
        elif in_noise_channel == "Gray":
            noise_r = np.random.normal(in_noise_mu, in_noise_sigma, size=(out_h, out_w))
            noise_g = noise_r
            noise_b = noise_r
        
        out_cv_r = np.uint8(np.clip(in_cv_r + noise_r, 0, 255))
        out_cv_g = np.uint8(np.clip(in_cv_g + noise_g, 0, 255))
        out_cv_b = np.uint8(np.clip(in_cv_b + noise_b, 0, 255))
    
    #채널 재 병합
    out_cv_noise = cv2.merge((out_cv_r, out_cv_g, out_cv_b))
    
    #생성옵션 기록갱신
    return_option += ", Noise = (" + in_option_noise + ", " + in_noise_channel
    return_option += ", mu = " + str(in_noise_mu) + ", sigma = " + str(in_noise_sigma) + ")"
    
    if is_return_options:
        #(PIL), (str)
        return Image.fromarray(out_cv_noise) , return_option
    else:
        #(PIL)
        return Image.fromarray(out_cv_noise)


#=== end of degradation_total_v7





if __name__ == "__main__":

    
    
    stage_1 = False     # 원본 데이터셋 -> original 폴더 안에 images, labels_RGB 로 분할
    stage_2 = False     # labels_RGB -> labels
    stage_3 = False     # check data
    stage_4 = False     # 2-fold split
    stage_5 = False     # YCOR -> YCOR_9_DLC_v1
    stage_6 = False     # YCOR_9_DLC_v1 -> x4_BILINEAR
    stage_7 = False     # YCOR_9_DLC_v1/x4_BILINEAR -> YCOR_9_ALTER_HR_Image/x4_BILINEAR
    
    
    
    # stage_1 = True
    # stage_2 = True
    # stage_3 = True
    # stage_4 = True
    # stage_5 = True
    # stage_6 = True
    stage_7 = True
    
    _count = 0
    for i_flag in [stage_1, stage_2, stage_3, stage_4, stage_5, stage_6, stage_7]:
        if i_flag:
            _count += 1
    
    if _count != 1:
        print("should use only one stage")
        sys.exit(-9)
    
    
    
    if stage_1:
        print("\n---[ init stage 1 ]---\n")
        path_src        = "./yamaha_seg(2)/yamaha_v0"
        folder_train    = "train"
        folder_valid    = "valid"
        name_image      = "rgb.jpg"
        name_label_rgb  = "labels.png"
        
        path_src_train = path_merge(path_src, folder_train)
        path_src_valid = path_merge(path_src, folder_valid)
        
        path_dst            = "./YCOR"
        path_dst_image      = path_merge(path_dst,     "images")
        path_dst_label_rgb  = path_merge(path_dst, "labels_RGB")
        
        print("path_src_train",     path_src_train)
        print("path_src_valid",     path_src_valid)
        print("path_dst_image",     path_dst_image)
        print("path_dst_label_rgb", path_dst_label_rgb)
        
        path_check(path_dst_image)
        path_check(path_dst_label_rgb)
        
        print("\n---[ 파일 정리 시작 ]---\n")
        
        lists_name = [sorted(os.listdir(path_src_train))
                     ,sorted(os.listdir(path_src_valid))
                     ]
        
        for path_curr, list_name in [[path_src_train, sorted(os.listdir(path_src_train))], [path_src_valid, sorted(os.listdir(path_src_valid))]]:
            for i_name in list_name:
                if i_name == "index.html":
                    continue
                _path = path_merge(path_curr, i_name)
                print(_path)
                
                if not os.path.exists(path_merge(path_dst_image, i_name + ".png")):
                    image_jpg = Image.open(path_merge(_path, name_image)).convert('RGB')
                    image_jpg.save(path_merge(path_dst_image, i_name + ".png"), 'png')
                
                if not os.path.exists(path_merge(path_dst_label_rgb, i_name + ".png")):
                    shutil.copy(path_merge(_path, name_label_rgb), path_merge(path_dst_label_rgb, i_name + ".png"))
                
            print("\n---\n")
        
        print("\n---[ 파일 정리 종료 ]---\n")
        
        
    
    elif stage_2:
        print("\n---[ init stage 2 ]---\n")
        #                   [  R,   G,   B]
        HP_COLOR_MAP = {0:  [178, 176, 153]     # 00 SmoothTrail        
                       ,1:  [128, 255,   0]     # 01 Grass              
                       ,2:  [156,  76,  30]     # 02 RoughTrial         
                       ,3:  [255,   0, 128]     # 03 Puddle             
                       ,4:  [255,   0,   0]     # 04 Obstacle           
                       ,5:  [  0, 160,   0]     # 05 LowVegetation      
                       ,6:  [ 40,  80,   0]     # 06 HighVegetation     
                       ,7:  [  1,  88, 255]     # 07 Sky                
                       ,8:  [255, 255, 255]     # 08 Background         
                       }
        
        color_id = {}
        
        print("Check HP_COLOR_MAP")
        for i_key in HP_COLOR_MAP:
            i_value = HP_COLOR_MAP[i_key]
            # code: 1 / RRR / GGG / BBB
            color_code = 1000000000 + i_value[0] * 1000000 + i_value[1] * 1000 + i_value[2]
            color_id[color_code] = i_key
            print(i_key, i_value, color_code)
        print("color_id:", color_id)
        
        print("")
        
        path_src            = "./YCOR"
        path_src_label_rgb  = path_merge(path_src, "labels_RGB")
        path_dst_label      = path_merge(path_src, "labels")
        
        path_check(path_dst_label)
        
        flag_show = False
        # flag_show = True
        
        for i_name in sorted(os.listdir(path_src_label_rgb)):
            
            if flag_show:
                # show sample
                flag_show = False
                label_rgb = Image.open(path_merge(path_src_label_rgb, i_name)).convert('RGB')
                label_rgb.show()
                np_rgb = np.array(label_rgb)
                Image.fromarray(np_rgb[:,:,0]).show()   # R
                Image.fromarray(np_rgb[:,:,1]).show()   # G
                Image.fromarray(np_rgb[:,:,2]).show()   # B
                print(np_rgb.shape)
            
            
            if not os.path.exists(path_merge(path_dst_label, i_name)):
                label_rgb  = Image.open(path_merge(path_src_label_rgb, i_name)).convert('RGB')
                label_gray = label_rgb_2_gray(label_rgb, color_id)
                # label_gray.show()
                
                label_gray.save(path_merge(path_dst_label, i_name), 'png')
            
            
            print(path_merge(path_dst_label, i_name))
            # break
        
    
    
    elif stage_3:
        path_src            = "./YCOR"
        path_src_image      = path_merge(path_src, "images")
        path_src_label      = path_merge(path_src, "labels")
        path_src_label_rgb  = path_merge(path_src, "labels_RGB")
        
        #                   [  R,   G,   B]
        HP_COLOR_MAP = {0:  [178, 176, 153]     # 00 SmoothTrail        
                       ,1:  [128, 255,   0]     # 01 Grass              
                       ,2:  [156,  76,  30]     # 02 RoughTrial         
                       ,3:  [255,   0, 128]     # 03 Puddle             
                       ,4:  [255,   0,   0]     # 04 Obstacle           
                       ,5:  [  0, 160,   0]     # 05 LowVegetation      
                       ,6:  [ 40,  80,   0]     # 06 HighVegetation     
                       ,7:  [  1,  88, 255]     # 07 Sky                
                       ,8:  [255, 255, 255]     # 08 Background         
                       }
        
        for i_name in sorted(os.listdir(path_src_image)):
            
            # imshow_pil(Image.open(path_merge(path_src_image,        i_name)))
            # imshow_pil(Image.open(path_merge(path_src_label_rgb,    i_name)).convert('RGB'))
            # imshow_pil(Image.open(path_merge(path_src_label,        i_name)))
            # imshow_pil(label_2_RGB(Image.open(path_merge(path_src_label, i_name)), HP_COLOR_MAP))
            
            imshow_pils([Image.open(path_merge(path_src_image, i_name))
                        ,Image.open(path_merge(path_src_label_rgb, i_name)).convert('RGB')
                        ,Image.open(path_merge(path_src_label,        i_name))
                        ,label_2_RGB(Image.open(path_merge(path_src_label, i_name)), HP_COLOR_MAP)
                        ]
                       ,["Image"
                        ,"label_RGB"
                        ,"label"
                        ,"label -> RGB"
                        ]
                       ,title = str(i_name)
                       )
            
            # break
    
    elif stage_4:
        # 40 등분 후 이미지 분배
        # 각 fold는 train 45%, val 5%, test 50% 로 분배 (total 1076 images)
        XX = None
        fold_1_train = [ 1,  3,  5,  7,  9, 11, 13, XX, 17, 19, 21, 23, 25, 27, 29, 31, 33, XX, 37, 39] # 484 images
        fold_1_val   = [                            15,                                     35        ] #  54 images
        fold_1_test  = [ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40] # 538 images
        
        fold_2_train = [ 2,  4,  6,  8, 10, 12, 14, XX, 18, 20, 22, 24, 26, 28, 30, 32, 34, XX, 38, 40] # 484 images
        fold_2_val   = [                            16,                                     36        ] #  54 images
        fold_2_test  = [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39] # 538 images
        
        fold_1 = True
        fold_2 = True
        
        
        path_src            = "./YCOR"
        path_src_image      = path_merge(path_src, "images")
        path_src_label      = path_merge(path_src, "labels")
        # path_src_label_rgb  = path_merge(path_src, "labels_RGB")
        
        path_dst            = "./YCOR_9_2Fold_v1"
        
        list_name = sorted(os.listdir(path_src_image))
        
        if fold_1:
            print("\n---[ fold 1 ]---\n")
            path_dst_train = path_merge(path_dst, "A_set/train")
            path_dst_val   = path_merge(path_dst, "A_set/val")
            path_dst_test  = path_merge(path_dst, "A_set/test")
            
            i_count = -1
            for i_name in list_name:
                i_count += 1 # start from 0
                
                if (i_count % 40) + 1 in fold_1_train:
                    # print("train\t", (i_count % 40) + 1)
                    path_dst_image = path_merge(path_dst_train, "images")
                    path_dst_label = path_merge(path_dst_train, "labels")
                    
                elif (i_count % 40) + 1 in fold_1_val:
                    # print("val\t\t", (i_count % 40) + 1)
                    path_dst_image = path_merge(path_dst_val, "images")
                    path_dst_label = path_merge(path_dst_val, "labels")
                    
                elif (i_count % 40) + 1 in fold_1_test:
                    # print("test\t", (i_count % 40) + 1)
                    path_dst_image = path_merge(path_dst_test, "images")
                    path_dst_label = path_merge(path_dst_test, "labels")
                    
                
                path_check(path_dst_image)
                path_check(path_dst_label)
                
                # print(path_merge(path_src_image, i_name))
                # print(path_merge(path_dst_image, i_name))
                if not os.path.exists(path_merge(path_dst_image, i_name)):
                    shutil.copy(path_merge(path_src_image, i_name), path_merge(path_dst_image, i_name))
                
                # print(path_merge(path_src_label, i_name))
                # print(path_merge(path_dst_label, i_name))
                if not os.path.exists(path_merge(path_dst_label, i_name)):
                    shutil.copy(path_merge(path_src_label, i_name), path_merge(path_dst_label, i_name))
        
        
        if fold_2:
            print("\n---[ fold 2 ]---\n")
            path_dst_train = path_merge(path_dst, "B_set/train")
            path_dst_val   = path_merge(path_dst, "B_set/val")
            path_dst_test  = path_merge(path_dst, "B_set/test")
            
            i_count = -1
            for i_name in list_name:
                i_count += 1 # start from 0
                
                if (i_count % 40) + 1 in fold_2_train:
                    # print("train\t", (i_count % 40) + 1)
                    path_dst_image = path_merge(path_dst_train, "images")
                    path_dst_label = path_merge(path_dst_train, "labels")
                    
                elif (i_count % 40) + 1 in fold_2_val:
                    # print("val\t\t", (i_count % 40) + 1)
                    path_dst_image = path_merge(path_dst_val, "images")
                    path_dst_label = path_merge(path_dst_val, "labels")
                    
                elif (i_count % 40) + 1 in fold_2_test:
                    # print("test\t", (i_count % 40) + 1)
                    path_dst_image = path_merge(path_dst_test, "images")
                    path_dst_label = path_merge(path_dst_test, "labels")
                    
                path_check(path_dst_image)
                path_check(path_dst_label)
                
                # print(path_merge(path_src_image, i_name))
                # print(path_merge(path_dst_image, i_name))
                if not os.path.exists(path_merge(path_dst_image, i_name)):
                    shutil.copy(path_merge(path_src_image, i_name), path_merge(path_dst_image, i_name))
                
                # print(path_merge(path_src_label, i_name))
                # print(path_merge(path_dst_label, i_name))
                if not os.path.exists(path_merge(path_dst_label, i_name)):
                    shutil.copy(path_merge(path_src_label, i_name), path_merge(path_dst_label, i_name))
        
    elif stage_5:
        path_src_ycor = "./YCOR"
        list_name = sorted(os.listdir(path_merge(path_src_ycor, "images")))
        path_dst_ycor = "./YCOR_9_DLC_v1/original"
        
        print("\n---[ YCOR_9_DLC_v1 original 준비 ]---\n")
        if not os.path.exists(path_dst_ycor):
            shutil.copytree(path_src_ycor, path_dst_ycor)
        
        
        print("\n---[ YCOR_9_DLC_v1 degradation_YCOR.csv 생성 ]---\n")
        path_dst_csv = "./YCOR_9_DLC_v1/degradation_YCOR.csv"
        
        print("/".join(path_dst_csv.split("/")[:-1]))
        print(path_dst_csv.split("/")[-1])
        
        if not os.path.exists(path_dst_csv):
        # if False:
            HP_SEED = 15
            
            HP_DG_NOISE_GRAY_PROB = 40      # gray noise (V 채널) 확률
            HP_DG_NOISE_SIGMA = (1, 30)     # Gaussian 노이즈 시그마 범위 (1st DG)
            
            random.seed(HP_SEED)
            dict_csv = {}
            update_dict_v2("file_name", "1_channel,1_sigma"
                          ,in_dict = dict_csv
                          ,print_head = "dict_csv"
                          ,is_print = True
                          )
            
            
            for i_name in list_name:
                ch_1 = random.choices(["Color", "Gray"]
                                     ,weights = [(100 - HP_DG_NOISE_GRAY_PROB), HP_DG_NOISE_GRAY_PROB]
                                     ,k = 1
                                     )[0]
                sigma_1 = str(round(random.uniform(HP_DG_NOISE_SIGMA[0], HP_DG_NOISE_SIGMA[-1])))
                
                update_dict_v2(i_name, ch_1 + "," + sigma_1
                              ,in_dict = dict_csv
                              ,print_head = "dict_csv"
                              ,is_print = False
                              )
            date_time = datetime.datetime.now()
            str_today = str(date_time.year) + "y" + str(date_time.month) + "m" + str(date_time.day) + "d"
            update_dict_v2("file_info", str_today + ",SEED=" + str(HP_SEED)
                          ,in_dict = dict_csv
                          ,print_head = "dict_csv"
                          ,is_print = True
                          )
            
            dict_2_txt_v2(in_file_path = "/".join(path_dst_csv.split("/")[:-1])
                         ,in_file_name = path_dst_csv.split("/")[-1]
                         ,in_dict = dict_csv
                         )
        
    elif stage_6:
        HP_SCALE_FACTOR = 4
        HP_INTERPOLATION_METHOD = "BILINEAR"
        
        _str = "x"+ str(HP_SCALE_FACTOR) + "_" + HP_INTERPOLATION_METHOD
        print("\n---[ LR", _str, "생성 ]---\n")
        path_src_image = "./YCOR_9_DLC_v1/original/images"
        path_src_csv   = "./YCOR_9_DLC_v1/degradation_YCOR.csv"
        
        path_dst_image = "./YCOR_9_DLC_v1/" + _str + "/images"
        path_check(path_dst_image)
        
        
        list_path_dst_csv = ["/".join(path_src_image.split("/")[:-1])
                            ,"/".join(path_dst_image.split("/")[:-1])
                            ]
        
        for i_path in list_path_dst_csv:
            _path = path_merge(i_path, path_src_csv.split("/")[-1])
            
            if not os.path.exists(_path):
                shutil.copy(path_src_csv, _path)
                print("csv copy:", path_src_csv, "->", _path)
            
        dict_csv = csv_2_dict(path_csv = path_src_csv)
        
        list_name = sorted(os.listdir(path_src_image))
        
        for i_name in list_name:
            if not os.path.exists(path_merge(path_dst_image, i_name)):
                in_pil = Image.open(path_merge(path_src_image, i_name))
                csv_contents = dict_csv[i_name]
                out_pil, out_option = degradation_total_v7(in_pil = in_pil
                                                          ,is_return_options = True
                                                          #--블러
                                                          ,in_option_blur = "Gaussian"
                                                          #-- 다운 스케일
                                                          ,in_scale_factor = HP_SCALE_FACTOR
                                                          ,in_option_resize = HP_INTERPOLATION_METHOD
                                                          #--노이즈
                                                          ,in_option_noise = "Gaussian"
                                                          #노이즈 시그마값 범위 (tuple)
                                                          ,in_range_noise_sigma = [1,30]
                                                          #Gray 노이즈 확룔 (int)
                                                          ,in_percent_gray_noise = 40
                                                          #노이즈 고정값 옵션
                                                          ,is_fixed_noise = True
                                                          ,in_fixed_noise_channel = csv_contents[0]
                                                          ,in_fixed_noise_sigma   = csv_contents[1]
                                                          )
                
                print(i_name, out_option)
                # imshow_pils([in_pil, out_pil], ["In", "Out"], rows=1, cols=2)
                
                out_pil.save(path_merge(path_dst_image, i_name), 'png')
            
            # break
        
    
    elif stage_7:
        HP_SCALE_FACTOR = 4
        HP_INTERPOLATION_METHOD = "BILINEAR"
        _str = "x"+ str(HP_SCALE_FACTOR) + "_" + HP_INTERPOLATION_METHOD
        print("\n---[ HR Alter", _str, "생성 ]---\n")
        path_src_image = "./YCOR_9_DLC_v1/" + _str + "/images"
        path_dst_image = "./YCOR_9_ALTER_HR_Image/" + _str + "/AB_set"
        path_check(path_dst_image)
        # print("cv2.INTER_AREA\t", cv2.INTER_AREA)
        # print("cv2.INTER_LINEAR\t", cv2.INTER_LINEAR)
        # print("cv2.INTER_CUBIC\t", cv2.INTER_CUBIC)
        
        print("Image.BILINEAR\t", Image.BILINEAR)
        print("Image.BICUBIC\t", Image.BICUBIC)
        
        for i_name in sorted(os.listdir(path_src_image)):
            if not os.path.exists(path_merge(path_dst_image, i_name)):
                # in_cv = cv2.imread(path_merge(path_src_image, i_name))
                in_pil = Image.open(path_merge(path_src_image, i_name))
                if HP_INTERPOLATION_METHOD in["AREA", "area"]:
                    # _inter = cv2.INTER_AREA     # 3
                    # _inter = Image.BOX
                    sys.exit(-9)
                elif HP_INTERPOLATION_METHOD in["BILINEAR", "bilinear"]:
                    # _inter = cv2.INTER_LINEAR   # 1
                    _inter = Image.BILINEAR     # 2
                elif HP_INTERPOLATION_METHOD in["BICUBIC", "bicubic"]:
                    # _inter = cv2.INTER_CUBIC    # 2
                    _inter = Image.BICUBIC      # 3
                
                print(i_name, "x", HP_SCALE_FACTOR, "interpolation:", _inter)
                
                # cv_out = cv2.resize(in_cv, dsize=(0,0)
                                   # ,fx=HP_SCALE_FACTOR, fy=HP_SCALE_FACTOR
                                   # ,interpolation = _inter
                                   # )
                
                
                pil_out = in_pil.resize((in_pil.width * HP_SCALE_FACTOR, in_pil.height* HP_SCALE_FACTOR)
                                       ,resample = _inter
                                       )
                
                # cv2.imwrite(path_merge(path_dst_image, i_name), cv_out)
                pil_out.save(path_merge(path_dst_image, i_name), 'png')
                
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("EOF")