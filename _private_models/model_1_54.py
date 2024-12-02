# 2nd paper EISR network & loss
"""
CamVid 4 배율 기준 ARNet
PSNR, SSIM      / Parameters    / Memory        / Multi-Adds    / FLOPs
23.00, 0.6900   / 190,248       / 131.72        / 3.2387        / 3.2515



*PSRN, SSIM 성능은 편의상 Test best 기록 예정*
v 1
    L0
    init
    
v 1.1
    L2
    perceptual loss 추가
        현재 perceptual loss / L1 loss = 32 ~ 36 -> 학습결과 나쁘면 가중치 조절해서 바꿔보기
    
v 1.2
    L0
    perceptual loss * 0.01, L1_loss * 0.5 -> 가중치 추가
    -> 22.8885, 0.6659
    
v 1.3
    L0
    block_a 수정
    -> 22.9874, 0.6671 / 186,687 / 255.93 / 2.7846 / 2.7942
    
v 1.4
    L2
    loss 비율 수정 (perceptual, L1) -> 0.005, 0.6
    -> 23.1043, 0.6913
    
v 1.5
    nearest interpolation 대신 bilinear interpolation 적용
    -> 23.1551, 0.68705 / 186,687 / 255.93 / 2.7846 / 2.7942
    
v 1.6
    block_a에 layer_1_3에 CCA 도입
    -> 23.1707, 0.6864 / 187,317 / 219.65 / 2.5986 / 2.6082
    
v 1.7
    block_a 10개 에서 8개로 줄여봄
    -> 23.1518, 0.6857 / 151,029 / 201.36 / 2.2453 / 2.2531
    
v 1.8
    block_a 8개 에서 6개로 줄여봄
    -> 23.1112, 0.6833 / 114,741 / 183.07 / 1.8919 / 1.8979
    
v 1.9
    block_a 6개 유지
    라플라시안 커널 타입 바꿔봄 (laplacian -> laplacian_2)
    -> 23.0988, 0.6955 / 114,741 / 183.07 / 1.8919 / 1.8979
    
v 1.10
    block_a 6개 -> 8개
    라플라시안 커널 타입 유지 (laplacian_2)
    -> 23.1368, 0.6974 / 151,029 / 201.36 / 2.2453 / 2.2531
    
*** 라플라시안 커널은 type 1 (-8 있는거)가 type 2 (-4) 보다 좋은듯 ***

v 1.11
    block_a 8개
    라플라시안 커널 타입 변경 (laplacian_2 -> laplacian_3)
    -> 23.1443, 0.6981 / 151,029 / 201.36 / 2.2453 / 2.2531

v 1.12
    block_a 8개 -> 6개
    라플라시안 커널 타입 유지 (laplacian_3)
    -> 23.1183, 0.6931 / 114,741 / 183.07 / 1.8919 / 1.8979

*** 커널 laplacian (type 1) 은 PSNR이 잘 나오고, laplacian_3 (type 3)은 SSIM이 잘 나옴 ***

v 1.13
    laplacian_3
    block 6 개 유지
    loss L1만 넣어봄
    -> 23.1435, 0.6985 / 114,741 / 183.07 / 1.8919 / 1.8979

v 1.14
    laplacian_3
    block 6 -> 8
    loss L1만 넣어봄
    -> 23.1761, 0.6999 / 151,029 / 201.36 / 2.2453 / 2.2531

*** perceptual loss 써봤는데 v 1.13 으론 큰 차이 모르겠음 ***
    perceptual loss 안쓰는게 더 나은듯

v 1.15
    laplacian_3
    mid_c = 42
    block_a * 10
    loss L1
    proposed trainer를 위해 network 분할 (model_I2F, model_F2I)
    
    model_I2F에서 feature의 pre-upsample 구조 제거
    
    -> 23.1980, 0.6870 / 187,317 / 110.78 / 2.0271 / 2.0372
    
    
v 1.16
    mid_c = 42 -> 54
    block_a * 8 -> 6
    -> 23.1611, 0.6786 / 188,649 / 93.16 / 2.0960 / 2.1042
    
    
*** 많은 채널 수 & 얕은 모델 보다 적은 채널 수 & 깊은 모델이 더 성능이 좋음 ***
-> v 1.15로 되돌아감
    
    
v 1.17 (L3)
    v 1.15에서...
        kernel_attention 에서 1x1 conv 제거함
    -> 23.1917, 0.6846 / 182,907 / 92.62 / 1.9795 / 1.9895
    
    
v 1.18 (L3)
    laplacian_3
    mid_c = 42
    block_a * 10 -> 8
    -> 23.1632, 0.6838 / 147,501 / 77.97 / 1.6356 / 1.6439
    
v 1.19 (L0)
    v 1.18 에서...
        block_a 에서 CCA 제거함
    
    laplacian_3
    mid_c = 42
    block_a * 8
    -> 23.1578, 0.6828 / 132,885 / 77.90 / 1.6320 / 1.6402
    
v 1.20 (L0)
    laplacian_3
    mid_c = 42
    block_a * 8 -> 10
    -> 23.1692, 0.6830 / 164,637 / 92.54 / 1.9749 / 1.9850

v 1.21 (L2)
    laplacian_3
    mid_c = 42
    block_a * 10 -> 6
    -> ?, ? / 101,133 / 63.26 / 1.2891 / 1.2955

v 1.22 (L3)
    v 1.17에서...
        fixed kernel convolution을 단일 채널만 고려하도록 group 수와 channel 수를 같게 수정
    -> 23.1995, 0.6857 / 182,907 / 92.62 / 1.9795 / 1.9895
    
v 1.23 (L3)
    laplacian_3
    mid_c = 42
    block_a * 10 -> 8
    -> 23.1817, 0.6851 / 147,501 / 77.97 / 1.6356 / 1.6439

v 1.24 (L0)
    v 1.23에서...
        block_a 에서 CCA 제거함
    laplacian_3
    mid_c = 42
    block_a * 8
    -> 23.1247, 0.6759 / 132,885 / 77.90 / 1.6320 / 1.6402

v 1.25 (L0)
    laplacian_3
    mid_c = 42
    block_a * 8 -> 10
    -> 23.1513, 0.6799 / 164,637 / 92.54 / 1.9749 / 1.9850

*** block a 에서 CCA가 성능향상에 중요한 역할을 하는거같음 ***
-> model 1.22로 되돌아감

v 1.26 (L0)
    laplacian_3
    mid_c = 42
    block_a * 10
    block a에서 CCA -> SASA (5x5 conv)
    -> 23.1704, 0.6835 / 175,137 / 128.87 / 2.0883 / 2.1120

v 1.27 (L0)
    laplacian_3
    mid_c = 42
    block_a * 10
    block a에서 SASA (5x5 conv) -> SASA (3x3 conv)
    -> 23.1435, 0.6771 / 168,417 / 128.85 / 2.0157 / 2.0394

v 1.28 (L0)
    laplacian_3
    mid_c = 42
    block_a * 10
    block a에서 SASA (3x3 conv) -> SASA (1x1 conv)
    -> 23.1886, 0.6855 / 165,057 / 128.83 / 1.9794 / 2.0031


*** PKA 에서 conv와 sigmoid 순서 바꿔보기 **

v 1.29 (L0)
    laplacian_3
    mid_c = 42
    block_a * 10
    block a에서 SASA (1x1 conv)
    v 1.28 에서...
        PKA에서...
            conv와 sigmoid 순서 바꿈
    -> 23.1304, 0.6826 / 165,057 / 128.83 / 1.9794 / 2.0031

v 1.30 (L0)
    v 1.28 에서...
        laplacian_3
        mid_c = 42
        block_a * 10
        block a에서 SASA (1x1 conv)
        * v 1.29의 변화 롤백
    model_I2F 구조에서 1x1 conv - Leaky ReLU - 1x1 conv 삭제
    ->  23.1856, 0.6861 / 161,529 / 121.56 / 1.9413 / 1.9645
    
*** sigmoid는 module 맨 끝에 위치해야 좋은 결과를 얻는듯 ***
    (v 1.28 <-> v 1.29)

v 1.31 (L0)
    v 1.30에서...
        model_I2F 구조에서 residual 연결 추가 (1개)
    -> 23.1900, 0.6852 / 161,529 / 121.56 / 1.9413 / 1.9645

v 1.32 (L0)
    v 1.31에서...
        model_I2F 구조에서 residual 연결 추가 (2개)
        basic block 묶음 (x10) 구조를 basic block 묶음 2개 (x5 + x5) 구조로 변경
    -> 23.1665, 0.6809 / 161,529 / 121.56 / 1.9413 / 1.9645

v 1.33 (L0)
    v 1.32에서...
        model_I2F 구조에서 residual 연결 구조 변경 (2개)
    -> 23.1493, 0.6789 / 161,529 / 121.56 / 1.9413 / 1.9645

*** PKA 에 학습 가능한 요소 추가해보기 ***

v 1.34 (L0)
    v 1.30에서...
        PKA 구조 변경 1 (G conv 추가)
    -> 23.1764, 0.6851 / 163,419 / 139.71 / 1.9618 / 1.9850

v 1.35 (L0)
    v 1.30에서...
        PKA 구조 변경 2 (G conv 추가, sigmoid 제거, 곱연산 합연산으로 변경)
    -> 모델 터짐 / 163,419 / 139.71 / 1.9618 / 1.9781

v 1.36 (L0)
    v 1.30에서...
        PKA 구조 변경 3 (G conv 추가, sigmoid 위치 변경, 곱-합연산 교체)
    -> 모델 터짐 / 163,419 / 139.71 / 1.9618 / 1.9850


*** v 1.31 ~ v 1.36 ***
basic block은 연속적으로 배치해야 효과가 좋다
-> 중간에 갈라서 뭔가를 하면 성능이 떨어진다.

Grouped conv w/ 3x3 라플라시안 (Group = Channel)는 반드시 sigmoid와 함께 사용해야한다
-> sigmoid 없으면 residual (Add, Multiply 모두) 연결에서 feature를 망가트려서 모델 터트린다.


v 1.37 (L0)
    v 1.30에서...
        PKA 구조 변경 4
    -> 23.1731, 0.6847 / 163,419 / 139.71 / 1.9618 / 1.9918


v 1.38 (L0)
    v 1.30에서...
        PKA 구조 변경 5
    -> 23.1817, 0.6857 / 163,419 / 139.71 / 1.9618 / 1.9918

*** v 1.34 ~ v 1.38 ***
PKA에 residual 경로를 3갈래 이상으로 늘리면 성능이 떨어진다
-> Residual은 2갈래만 쓰자

v 1.39 (L0)
    v 1.31에서...
        PKA 구조 변경 6
    -> 23.2142 ,0.6851 / 163,419 / 139.71 / 1.9618 / 1.9850


v 1.40 (L0)
    v 1.39에서...
        model I2Mask 구조에서 I2I로 변경
    -> 23.1780, 0.6863 / 163,419 / 139.71 / 1.9618 / 1.9850
    
*** 일단 v 1.39로 teacher 설계함 ***
    
v 1.41 (L2)
    Teacher network 설계
    v 1.39에서...
        block_a * 10 -> 30
    -> ?, ? / 485,559 / 395.02 / 5.4409 / 5.5094


v 1.42
    model v 1.39 기반
    Teacher (basic block 9*3 개), Student (basic block 3*3 개)
    -> (T 27) ?, ? / 437,238 / 356.72 / 4.9190 / 4.9808
    -> (S  9) ?, ? / 147,312 / 126.95 / 1.7878 / 1.8087
    
v 1.43
    v 1.42에서 기존 내용 수정사항 없음
    proposed_loss_ss 추가

v 1.44
    v 1.43에서...
    proposed_loss_ss 에 kd_p_1, kd_p_2 추가
    * 정정 *
    proposed_loss 에 kd_p_1, kd_p_2 추가

v 1.45
    proposed_loss_ss에서 CE loss 계산에 Target을 one-hot 형태로 안쓰게 변경
    *trainer v 0.8 요구됨

v 1.46
    proposed_loss_ss에서...
        CE loss weight 계산에 Target을 one-hot 형태로 안쓰게 변경
        weight 계산 수식 변경 (sample에 대한 std 활용, _alpha=1.0, _beta=1.0)
    *trainer v 1.0 요구됨
    

v 1.47
    proposed_loss_ss에서...
        CE loss에서 weight 사용 안하게 변경

v 1.48
    proposed_loss_ss에서...
        CE loss에서 weight 사용을 통해 mIoU 향상이 확인됨
        v 1.44 수식을 one-hot 형태 데이터 없이 사용하게 작성

v 1.49
    sr model의 basic block stack 앞에 CCA 추가 (총 3개)
    -> (T 27) ?, ? / 448,074 / 356.77 / 4.9204 / 4.9821
    -> (S  9) ?, ? / 158,148 / 126.99 / 1.7892 / 1.8101

v 1.50
    sr model의 basic block stack 뒤에 ESA 추가 (총 3개)
    -> (T 27) ?, ? / 455,529 / 371.87 / 4.9454 / 5.0323
    -> (S  9) ?, ? / 165,603 / 142.09 / 1.8142 / 1.8603

* ESA는 mIoU 성능 향상에 도움 안되는듯

v 1.51
    proposed_loss_srss 추가
    
    sr model에서 ESA 제거, basic block 전 후로 CCA 추가 (CCA 총 6개 사용됨)
    -> (T 27) ?, ? / 458,910 / 356.81 / 4.9217 / 4.9835
    -> (S  9) ?, ? / 168,984 / 127.04 / 1.7905 / 1.8115
    
v 1.52
    sr model에서 basic block 뒤에만 CCA 추가 (CCA 총 3개 사용됨)
    -> (T 27) ?, ? / 448,074 / 356.77 / 4.9204 / 4.9821
    -> (S  9) ?, ? / 158,148 / 126.99 / 1.7892 / 1.8101

* v 1.48 에 여러 모듈 추가한 (v 1.49 ~ v 1.52) 경우, 성능이 하락됨
    -> 채널간 정보를 혼합하지 않는 모델 특성상 기존 attention은 별로 안좋은듯
    -> v 1.48로 되돌아감
    * trainer 1.2 -> 1.3 (SRSS에서 SR 결과 loss에 반영되게 변경)은 mIoU에 긍정적 효과르 보임

v 1.53
    model v 1.48 구조로 회귀
    proposed_loss_ss, proposed_loss_srss가 log(in_pred)를 바탕으로 연산되도록 변경
    * in_pred는 확률값 표현
    
    -> (T 27) ?, ? / 437,238 / 356.72 / 4.9190 / 4.9808
    -> (S  9) ?, ? / 147,312 / 126.95 / 1.7878 / 1.8087
    
v 1.54
    proposed_loss_ss 에서
        의미없는 _weight 계산 주석처리
        edge 를 기반으로 수행하는 loss_b 추가
"""
# model version
_mv = "1.54"

#=======================================================================================
import time
import warnings
_str = "\n\n---[ Proposed 2nd paper model version: " + str(_mv) + " ]---\n"
warnings.warn(_str)
# time.sleep(3)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import vgg19
from torchvision import transforms
import sys
import numpy as np
import copy



from PIL import Image

from torchinfo import summary
from thop import profile as profile_thop    # https://github.com/Lyken17/pytorch-OpCounter
from DLCs.FLOPs import profile


# Conv 2D with stride=1 & dilation=1
def Conv_(in_c, out_c, k_size=3, groups=1, bias=False, padding_mode='replicate'):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    p_size = int((k_size - 1)//2) # padding size
    
    if k_size-1 != 2*p_size:
        print("Kernel size should be odd value. Currnet k_size:", k_size)
        sys.exit(9)
    
    return nn.Conv2d(in_channels=in_c, out_channels=out_c
                    ,kernel_size=k_size, stride=1
                    ,padding=p_size, dilation=1
                    ,groups=groups
                    ,bias=bias, padding_mode=padding_mode
                    )

# Conv 2D with kernel=3 & stride=1
def Conv_3x3(in_c, out_c, d_size=1, groups=1, bias=False, padding_mode='replicate'):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    p_size = int(d_size) # set padding size with dilation value
    
    return nn.Conv2d(in_channels=in_c, out_channels=out_c
                    ,kernel_size=3, stride=1
                    ,padding=p_size, dilation=d_size
                    ,groups=groups
                    ,bias=bias, padding_mode=padding_mode
                    )



# single-channel aware spatial attention
class SASA(nn.Module):
    def __init__(self, in_c, k_size):
        super(SASA, self).__init__()
        print("SASA with", k_size, "x", k_size, "conv")
        self.layer_gconv_sig = nn.Sequential(Conv_(in_c, in_c, k_size=k_size, groups=in_c)
                                            ,nn.Sigmoid()
                                            )
    
    def forward(self, in_x):
        return in_x * self.layer_gconv_sig(in_x)


class fixed_kernel(nn.Module):
    #def __init__(self, k_type=None, k_dim=4, k_ch=3, device=None):
    def __init__(self, k_type=None, in_c=3, out_c=3, device=None):
        super(fixed_kernel, self).__init__()
        # [info]
        # fixed kernel을 통한 2d Conv 시행
        # k_type = 커널 타입
        # in_c = 입력 feature 채널 수 
        self.in_c = in_c
        # out_c = 출력 feature 채널 수 
        self.out_c = out_c
        # device = torch device setting
        #
        # [memo]
        # kernel 바꿔과면서 ablation 돌리기
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if k_type is None:
            k_type = "laplacian"
        
        if k_type == "laplacian":
            print("fixed_kernel is laplacian")
            k_weight = torch.tensor(np.array([[1, 1, 1]
                                             ,[1,-8, 1]
                                             ,[1, 1, 1]
                                             ])).float().to(device)
            
        elif k_type == "laplacian_2":
            print("fixed_kernel is laplacian_2")
            k_weight = torch.tensor(np.array([[0, 1, 0]
                                             ,[1,-4, 1]
                                             ,[0, 1, 0]
                                             ])).float().to(device)
            
        elif k_type == "laplacian_3":
            print("fixed_kernel is laplacian_3")
            k_weight = torch.tensor(np.array([[-1,-1,-1]
                                             ,[-1, 8,-1]
                                             ,[-1,-1,-1]
                                             ])).float().to(device)
            
        
        
        k_weight = k_weight.unsqueeze(dim=0)
        _k_weight = k_weight.clone().detach()
        while(False):
            _C, _, _ = k_weight.shape
            if _C >= in_c:
                break
            else:
                k_weight = torch.cat((k_weight, _k_weight), dim=0)
        
        k_weight = k_weight.unsqueeze(dim=0)
        _k_weight = k_weight.clone().detach()
        while(True):
            _C, _, _, _ = k_weight.shape
            if _C >= out_c:
                break
            else:
                k_weight = torch.cat((k_weight, _k_weight), dim=0)
        
        self.register_buffer('weight', k_weight)
        
        # print("self.weight", self.weight.shape)
        
    def forward(self, in_x):
        return F.conv2d(in_x, self.weight
                       ,bias=None, stride=1, padding="same", dilation=1, groups=self.in_c
                       )


# pre-defined kernel attention
class PKA(nn.Module):
    def __init__(self, in_c):
        super(PKA, self).__init__()
        
        self.fixed_conv = fixed_kernel(k_type="laplacian_3", in_c=in_c, out_c=in_c)
        
        self.train_conv = Conv_(in_c=in_c, out_c=in_c, k_size=3, groups=in_c)
        
        self.layer_sig  = nn.Sigmoid()
    
    # def contrast(self, in_feat):
        # # CCA에서 가져옴
        # # 사용법
        # # 1. 선언
        # # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # #
        # # 2. 사용
        # # feat_contrast = self.contrast(x) + self.avg_pool(x)
        # #
        
        # _, _, _h, _w = in_feat.shape
        # _hw = _h * _w
        
        # feat_sum        = in_feat.sum(3, keepdim=True).sum(2, keepdim=True)
        # feat_mean       = feat_sum / _hw
        # feat_diff       = torch.abs(in_feat - feat_mean) + 1e-3
        # feat_pow        = feat_diff * feat_diff
        # feat_pow_sum    = feat_pow.sum(3, keepdim=True).sum(2, keepdim=True) / _hw
        # feat_var        = torch.sqrt(feat_pow_sum)
        # return torch.nan_to_num(feat_var
                               # ,nan=1e-3, posinf=1e-3, neginf=1e-3
                               # )
    
    def forward(self, in_x):
        
        feat_sig = self.layer_sig(self.fixed_conv(in_x) + self.train_conv(in_x))
        
        return in_x * feat_sig
        

def stack_block(block, count, recursive=True):
    stacks = torch.nn.Sequential()
    
    if recursive:
        for i_count in range(count):
            stacks.add_module(str(i_count), block)
    else:
        for i_count in range(count):
            stacks.add_module(str(i_count), copy.deepcopy(block))
    
    return stacks



class block_a(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(block_a, self).__init__()
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.layer_1_1 = nn.Sequential(Conv_3x3(in_c//2, mid_c, d_size=1, groups=1)
                                      ,PKA(mid_c)
                                      ,Conv_3x3(mid_c, in_c//2, d_size=1, groups=1)
                                      )
        
        self.layer_1_2 = nn.Sequential(Conv_3x3(in_c - in_c//2, mid_c, d_size=1, groups=1)
                                      ,self.act
                                      ,Conv_3x3(mid_c, in_c - in_c//2, d_size=1, groups=1)
                                      )
        
        self.layer_1_3 = SASA(in_c, 1)
        
    def forward(self, in_x):
        _, _C, _, _ = in_x.shape
        
        f_1_1 = self.layer_1_1(in_x[:, :_C//2,:,:])
        f_1_2 = self.layer_1_2(in_x[:, _C//2:,:,:])
        f_1_3 = self.layer_1_3(in_x)
        
        return torch.cat((f_1_1,f_1_2),dim=1) + f_1_3



class model_I2F(nn.Module):
    # def __init__(self, in_c=3, mid_c=42):
    def __init__(self, in_c, mid_c, basic_blocks):
        super(model_I2F, self).__init__()
        
        self.layer_init = nn.Sequential(Conv_(in_c=in_c, out_c=mid_c, k_size=3, groups=1)
                                       ,
                                       )
        
        print("\nmodel with", (basic_blocks//3)*3, "basic_blocks")
        
        self.layer_mid_1 = stack_block(block_a(mid_c, mid_c//2, mid_c), count=basic_blocks//3, recursive=False)
        self.layer_mid_2 = stack_block(block_a(mid_c, mid_c//2, mid_c), count=basic_blocks//3, recursive=False)
        self.layer_mid_3 = stack_block(block_a(mid_c, mid_c//2, mid_c), count=basic_blocks//3, recursive=False)
        
        
    def forward(self, in_x):
        
        f_init_1 = self.layer_init(in_x)
        
        f_mid_1 = self.layer_mid_1(f_init_1)
        f_mid_2 = self.layer_mid_2(f_mid_1)
        f_mid_3 = self.layer_mid_3(f_mid_2)
        
        return f_mid_3 + f_init_1, [f_mid_1, f_mid_2, f_mid_3]


class model_F2I(nn.Module):
    # def __init__(self, mid_c=42, out_c=3, scale=4):
    def __init__(self, mid_c, out_c, scale):
        super(model_F2I, self).__init__()
        
        self.scale = scale
        
        self.layer_last_1 = nn.Sequential(Conv_(mid_c, out_c, k_size=3, groups=1)
                                         ,nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                         ,Conv_(out_c, out_c, k_size=3, groups=1)
                                         )
        
    def forward(self, in_x, in_feat):
        f_last_1 = self.layer_last_1(F.interpolate(in_feat
                                                  ,scale_factor=self.scale
                                                  ,mode='bilinear'
                                                  ,align_corners=None
                                                  )
                                    )
        
        return f_last_1 + F.interpolate(in_x
                                       ,scale_factor=self.scale
                                       ,mode='bilinear'
                                       ,align_corners=None
                                       )



class proposed_model(nn.Module):
    def __init__(self, in_c=3, mid_c=42, out_c=3, scale=4, basic_blocks=None):
        super(proposed_model, self).__init__()
        
        self.layer_I2F = model_I2F(in_c=in_c, mid_c=mid_c, basic_blocks=basic_blocks)
        
        self.layer_F2I = model_F2I(mid_c=mid_c, out_c=out_c, scale=scale)
        
    
    def forward(self, in_x):
        
        mid_feat, list_feats = self.layer_I2F(in_x)
        
        return [self.layer_F2I(in_x, mid_feat), list_feats]
        # return self.layer_F2I(in_x, mid_feat)
    


#===================================================================================



class proposed_loss(nn.Module):
    def __init__(self, is_amp=True, kd_mode=None):
        super(proposed_loss, self).__init__()
        # self.loss_perceptual = PerceptualLoss()
        self.loss_l1         = torch.nn.L1Loss()
        # self.is_amp          = is_amp
        self.kd_mode         = kd_mode
        
        print("kd_mode:", self.kd_mode)
        
    # def kd_origin(in_feat):
        # # original KD
        # return in_feat
    
    # def kd_fitnet(in_feat):
        # # FitNet KD
        # return in_feat
    
    def kd_at(self, in_feat):
        # attention transfer (AT)
        eps = 1e-6
        am = torch.pow(torch.abs(in_feat), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        return torch.div(am, norm+eps)
    
    def kd_fsp(self, in_feat_1, in_feat_2):
        # flow of solution procedure (FSP)
        _B, _C, _H, _W = in_feat_2.shape
        in_feat_1 = F.adaptive_avg_pool2d(in_feat_1, (_H, _W))
        in_feat_1 = in_feat_1.view(_B, _C, -1)
        in_feat_2 = in_feat_2.view(_B, _C, -1).transpose(1,2)
        return torch.bmm(in_feat_1, in_feat_2) / (_H*_W)
    
    def kd_fakd(self, in_feat):
        # Feature-affinity based knowledge distillation (FAKD)
        _B, _C, _H, _W = in_feat.shape
        in_feat = in_feat.view(_B, _C, -1)
        norm_fm = in_feat / (torch.sqrt(torch.sum(torch.pow(in_feat,2), 1)).unsqueeze(1).expand(in_feat.shape) + 1e-8)
        sa = norm_fm.transpose(1,2).bmm(norm_fm)
        return sa.unsqueeze(1)
    
    def kd_p_1(self, in_feat):
        # proposed KD 1
        try:
            return self.kd_p_1_conv(in_feat)
        except:
            _B, _C, _H, _W = in_feat.shape
            self.kd_p_1_conv = fixed_kernel(k_type="laplacian_3", in_c=_C, out_c=1)
            return self.kd_p_1_conv(in_feat)
    
    def kd_p_2(self, in_feat):
        # proposed KD 1
        try:
            return self.kd_p_2_conv(in_feat)
        except:
            _B, _C, _H, _W = in_feat.shape
            self.kd_p_2_conv = fixed_kernel(k_type="laplacian_3", in_c=_C, out_c=_C)
            return self.kd_p_2_conv(in_feat)
    
    
    def forward(self, in_pred, in_ans, *args):
        
        if isinstance(in_pred, list):
            in_pred_sr = in_pred[0]
            in_pred_feat = in_pred[1]
        else:
            in_pred_sr = in_pred
            in_pred_feat = None
        
        try:
            in_teacher = args[0]
            if isinstance(in_teacher, list):
                in_teacher_sr = in_teacher[0]
                in_teacher_feat = in_teacher[1]
            else:
                in_teacher_sr = in_teacher
                in_teacher_feat = None
            
        except:
            # print("in_teacher is None")
            in_teacher = None
        
        if in_teacher is None:
            
            return self.loss_l1(in_pred_sr, in_ans)
        else:
            # _alpha = 1 - self.loss_l1(in_teacher_sr, in_ans)    # teacher의 정답 수준에 비례한 가중치 활용 -> loss 범위가 0 ~ 0.06 이라 의미 없는듯
            
            _alpha = 0.3
            
            loss_kd   = _alpha * self.loss_l1(in_pred_sr,in_teacher_sr)
            loss_feat = 0.0
            if in_teacher_feat is not None:
                _feat_count = len(in_teacher_feat)
            else:
                _feat_count = None
            
            if   self.kd_mode == "kd_origin":
                pass
                
            elif self.kd_mode == "kd_fitnet":
                for i_feat in range(_feat_count):
                    loss_feat += _alpha * self.loss_l1(in_pred_feat[i_feat], in_teacher_feat[i_feat])
                
            elif self.kd_mode == "kd_at":
                for i_feat in range(_feat_count):
                    loss_feat += _alpha * self.loss_l1(self.kd_at(in_pred_feat[i_feat]), self.kd_at(in_teacher_feat[i_feat]))
                
            elif self.kd_mode == "kd_fsp":
                _feat_count = _feat_count - 1
                for i_feat in range(_feat_count):
                    loss_feat += _alpha * self.loss_l1(self.kd_fsp(in_pred_feat[i_feat], in_pred_feat[i_feat + 1]), self.kd_fsp(in_teacher_feat[i_feat], in_teacher_feat[i_feat + 1]))
                
            elif self.kd_mode == "kd_fakd":
                for i_feat in range(_feat_count):
                    loss_feat += _alpha * self.loss_l1(self.kd_fakd(in_pred_feat[i_feat]), self.kd_fakd(in_teacher_feat[i_feat]))
            
            elif self.kd_mode == "kd_p_1":
                for i_feat in range(_feat_count):
                    loss_feat += _alpha * self.loss_l1(self.kd_p_1(in_pred_feat[i_feat]), self.kd_p_1(in_teacher_feat[i_feat]))
            elif self.kd_mode == "kd_p_2":
                for i_feat in range(_feat_count):
                    loss_feat += _alpha * self.loss_l1(self.kd_p_2(in_pred_feat[i_feat]), self.kd_p_2(in_teacher_feat[i_feat]))
            
            
            # elif self.kd_mode is None:
                # _feat_count = None
                # loss_feat = None
                
            
            if _feat_count is None:
                return (1 - _alpha) * self.loss_l1(in_pred_sr, in_ans) + _alpha * loss_kd
            else:
                return (1 - _alpha) * self.loss_l1(in_pred_sr, in_ans) + _alpha * (loss_kd + loss_feat / _feat_count)



class proposed_loss_ss(nn.Module):
    def __init__(self, is_amp=True, pred_classes=None, ignore_index=-100, alpha=1.0, beta=1.0):
        super(proposed_loss_ss, self).__init__()
        self.eps = 1e-9 # epsilon
        
        if pred_classes is None:
            # 예측을 시행할 class 수
            _str = "pred_classes must be specified"
            warnings.warn(_str)
            sys.exit(-9)
        
        self.pred_classes   = pred_classes
        print("\nproposed_loss_ss pred_classes is", self.pred_classes)
        print("index should start from 0")
        
        self.ignore_index   = ignore_index
        print("proposed_loss_ss ignore_index is", self.ignore_index)
        print("ignore_index should be last number of index")
        
        # self.alpha          = alpha     # 가중치 기준값
        # self.beta           = beta      # 표준편차 반영 가중치
        # print("proposed_loss_ss alpha and beta is", self.alpha, self.beta)
        if self.ignore_index < 0:
            self.edge_pred = fixed_kernel(k_type="laplacian_3", in_c=self.pred_classes + 1, out_c=self.pred_classes + 1)
        else:
            self.edge_pred = fixed_kernel(k_type="laplacian_3", in_c=self.pred_classes, out_c=self.pred_classes)
        self.edge_ans  = fixed_kernel(k_type="laplacian_3", in_c=1,                 out_c=1)
        print("edge_pred, edge_ans layer initialized")
        
    # def calc_weight(self, in_ans):
        # _ans  = in_ans.clone().detach()
        
        # # v 1.47 방식 폐기
        # # v 1.44 수식을 one-hot 없이 사용하게 재구현
        # _B, _, _ = _ans.shape
        # _ans = _ans.view([-1])  # (B, H, W) -> (BHW)
        # _bin = torch.bincount(_ans, minlength=self.pred_classes)[:self.pred_classes] # (BHW) -> (Index except void)
        # _norm = 1 - torch.nn.functional.normalize(_bin.float(), p=1, dim=0)
        
        # return _norm
        
    def forward(self, in_pred, in_ans):
        # _weight = self.calc_weight(in_ans)
        # print("\n_weight:", _weight)
        
        # return torch.nn.functional.cross_entropy(in_pred, in_ans, ignore_index=self.ignore_index)
        # return torch.nn.functional.cross_entropy(torch.log(in_pred + self.eps), in_ans, ignore_index=self.ignore_index)
        
        loss_a = torch.nn.functional.cross_entropy(torch.log(in_pred + self.eps)
                                                  ,in_ans
                                                  ,ignore_index=self.ignore_index
                                                  )
        
        loss_b = torch.nn.functional.l1_loss(torch.sum(self.edge_pred(in_pred), dim=1, keepdim=True)
                                            ,self.edge_ans(in_ans.unsqueeze(dim=1).type(in_pred.dtype))
                                            )
        
        return loss_a + loss_b


class proposed_loss_srss(nn.Module):
    def __init__(self, is_amp=True, pred_classes=None, ignore_index=-100, alpha=1.0, beta=1.0):
        super(proposed_loss_srss, self).__init__()
        self.eps = 1e-9 # epsilon
        
        if pred_classes is None:
            # 예측을 시행할 class 수
            _str = "pred_classes must be specified"
            warnings.warn(_str)
            sys.exit(-9)
        
        # if ignore_index == -100:
            # _str = "ignore_index must be specified"
            # warnings.warn(_str)
            # sys.exit(-9)
        
        self.pred_classes   = pred_classes
        self.ignore_index   = ignore_index
        self.alpha          = alpha     # 가중치 기준값
        self.beta           = beta      # 표준편차 반영 가중치
        print("\nproposed_loss_srss pred_classes is", self.pred_classes)
        print("proposed_loss_srss ignore_index is", self.ignore_index)
        # print("proposed_loss_srss alpha and beta is", self.alpha, self.beta)
        # print("ignore_index should be last number of index")
        # print("index should start from 0")
        
        self.loss_l1         = torch.nn.L1Loss()
        
    
    def calc_weight(self, in_ans):
        _ans  = in_ans.clone().detach()
        
        # v 1.47 방식 폐기
        # v 1.44 수식을 one-hot 없이 사용하게 재구현
        _B, _, _ = _ans.shape
        _ans = _ans.view([-1])  # (B, H, W) -> (BHW)
        _bin = torch.bincount(_ans, minlength=self.pred_classes)[:self.pred_classes] # (BHW) -> (Index except void)
        _norm = 1 - torch.nn.functional.normalize(_bin.float(), p=1, dim=0)
        
        return _norm
        
    def forward(self, in_sr, in_hr, in_pred, in_ans):
        # in_pred는 probability (확률 표현) 형태를 가져야 함
        
        loss_sr = self.loss_l1(in_sr, in_hr)
        _weight = self.calc_weight(in_ans)
        # print("\n_weight:", _weight)
        # loss_ss = torch.nn.functional.cross_entropy(in_pred, in_ans, ignore_index=self.ignore_index)
        loss_ss = torch.nn.functional.cross_entropy(torch.log(in_pred + self.eps), in_ans, ignore_index=self.ignore_index)
        return loss_sr + loss_ss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #try:
    if True:
        _B, _C, _H, _W = 1, 3, 90, 120
        
        for _basic_blocks in [9, 27]:
            print("\n_basic_blocks:", _basic_blocks)
            model = proposed_model(basic_blocks=_basic_blocks)
            model.to(device)
            
            print("\n --- Info ---\n")
            summary(model, input_size=(_B, _C, _H, _W)) 
            
            print("\n --- THOP ---\n")
            input = torch.randn(_B, _C, _H, _W).to(device)
            macs, params = profile_thop(model, inputs=(input, ))
            _giga = 1000000000
            print("THOP", macs, macs/_giga, "G", params)
            print("Multi-Adds (G):", round(macs/_giga,4))
            
            print("\n --- FLOPs ---\n")
            flops, params = profile(model, input_size=(_B, _C, _H, _W))
            print('Input: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format( _H, _W,flops/(1e9),params))
            print("FLOPS (G):", round(flops/(1e9),4))
        
    # except:
        # pass
    
    
    if False:
        in_pil = Image.open("C:/Lenna.png").convert("RGB")
        in_pil = in_pil.resize((64,64))
        #in_pil.show()
        PIL_To_Tensor = transforms.ToTensor()
        Tensor_To_PIL = transforms.ToPILImage()
        
        in_ts = PIL_To_Tensor(in_pil).requires_grad_(False)
        in_ts = in_ts.unsqueeze(dim=0)
        in_ts = torch.cat((in_ts,in_ts))
        
        model = proposed_model(basic_blocks=9)
        criterion = proposed_loss()
        
        out_ts = model(in_ts)
        if isinstance(out_ts, list):
            print("feat shape", type(out_ts[1]), len(out_ts[1]))
            print(type(out_ts[1][0]), out_ts[1][0].shape)
            out_ts = out_ts[0]
        else:
            out_ts = out_ts
        print("out_ts.shape", out_ts.shape)
        
        loss = criterion(out_ts, out_ts)
        print(loss)
        out_pil = Tensor_To_PIL(out_ts[0])
        #out_pil.show()
        print("Emulate Finished")
    # except:
        # pass
        
    print("\n---\n")
    
    
print("EOF: model.py")