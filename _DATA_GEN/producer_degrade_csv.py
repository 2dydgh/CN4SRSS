# -*- coding: utf-8 -*-
"""
#producer_degrade_csv.py
#degradation 옵션 csv파일 생성기
Created on Mon Apr 25 16:38:15 2022

@author: user

Degradation 옵션 지정 csv 생성기
"""
import os
import random
import sys
import datetime

HP_SEED = 15

random.seed(HP_SEED)

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

NAME_FOLDER_TRAIN = "train"
NAME_FOLDER_VAL = "val"
NAME_FOLDER_TEST = "test"
NAME_FOLDER_IMAGES= "images"

def make_list_img(**kargs):
    name_func = "[make_list_img] ->"
    
    #(str) 데이터셋 상위 경로 (PATH_BASE_IN: ./"name_dataset")
    in_path_dataset = kargs['in_path_dataset']
    
    if in_path_dataset[-1] != "/":
        in_path_dataset += "/"
    
    #(str) 데이터 종류 ("train" or "val" or "test")
    #in_category = kargs['in_category']
    
    
    #폴더 분류 유효성 검사
    global NAME_FOLDER_TRAIN, NAME_FOLDER_VAL, NAME_FOLDER_TEST

    
    #---
    #image, label 이미지 파일의 폴더 이름
    global NAME_FOLDER_IMAGES
    
    
    in_category = NAME_FOLDER_TRAIN
    list_file_img_train = os.listdir(in_path_dataset + in_category + "/" + NAME_FOLDER_IMAGES)
    in_category = NAME_FOLDER_VAL
    list_file_img_val = os.listdir(in_path_dataset + in_category + "/" + NAME_FOLDER_IMAGES)
    in_category = NAME_FOLDER_TEST
    list_file_img_test = os.listdir(in_path_dataset + in_category + "/" + NAME_FOLDER_IMAGES)
    
    return list_file_img_train + list_file_img_val + list_file_img_test

#=== End of make_list_img

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

#-------------------------------------------------

list_all = make_list_img(in_path_dataset = "C:/Users/user/.spyder-py3/CamVid_12_2Fold_v4/A_set")
list_all.sort()

print(len(list_all))

dict_all = {}

update_dict_v2("file_name", "1_channel,1_sigma"
              ,in_dict = dict_all
              ,print_head = "dict_all"
              ,is_print = True
              )

#gray noise (V 채널) 확률
HP_DG_NOISE_GRAY_PROB = 40
#Gaussian 노이즈 시그마 범위 (1st DG)
HP_DG_NOISE_SIGMA = (1, 30)



in_percent_gray_noise = HP_DG_NOISE_GRAY_PROB
for i_img in range(len(list_all)):
    
    ch_1 = random.choices(["Color", "Gray"]
                         ,weights = [(100 - in_percent_gray_noise), in_percent_gray_noise]
                         ,k = 1
                         )[0]
    
    sigma_1 = str(round(random.uniform(HP_DG_NOISE_SIGMA[0], HP_DG_NOISE_SIGMA[-1])))
    
    update_dict_v2(list_all[i_img], ch_1 + "," + sigma_1
                  ,in_dict = dict_all
                  ,print_head = "dict_all"
                  ,is_print = False
                  )
date_time = datetime.datetime.now()
str_today = str(date_time.year) + "y" + str(date_time.month) + "m" + str(date_time.day) + "d"

update_dict_v2("file_info", str_today + ",SEED=" + str(HP_SEED)
              ,in_dict = dict_all
              ,print_head = "dict_all"
              ,is_print = False
              )

dict_2_txt_v2(in_file_path = "C:/Users/user/.spyder-py3"
             ,in_file_name  = "check.csv"
             ,in_dict = dict_all
             )

print("EoF producer_degrade_csv.py")