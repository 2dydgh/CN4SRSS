###########################################################
#
#       이 파일은 Prop_9_NEW v 101 입니다.
#
#       Ablation:   07
#
#       Info
#           Number of COB   : 2
#           RFM apply?      : O
#           Loss stage 3 in : 1, 2 (Yes)
#                               *block 개수와 동일하면 Yes human vision standard guide for sr image
#                               *block 개수와 다르다면 No  human vision standard guide for sr image
#       
#       Run At...   L
#
###########################################################

print("Model: Prop_9_NEW v101 - Ablation")
# loss stage 조정
# epoch             stage
# 1                 3
# 2 ~               1
# front_freeze ~    2
# front_melt ~      3
#

front_freeze = 10
front_melt   = 15
loss_ce_weights = None


import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

out_c_lab = 11  #CamVid
#out_c_lab = 19  # MiniCity

#from DLCs.model_cgnet import Context_Guided_Network as CGNet
#back_model_ss = CGNet(classes=out_c_lab, M=3, N=21)
#print("back_model_ss: CGNet")

#from DLCs.model_dabnet import DABNet
#back_model_ss = DABNet(classes=out_c_lab)
#print("back_model_ss: DABNet")

from DLCs.semantic_segmentation.model_deeplab_v3_plus import DeepLab_v3_plus
back_model_ss = DeepLab_v3_plus(num_classes = out_c_lab, pretrained = False)
print("back_model_ss: DeepLab_v3_plus")


def _debug(in_str, in_ts):
    print(in_str, in_ts.shape)

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

# Channel Attention Layer (MPRNet)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False, groups=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias, groups=groups)
                                    ,nn.ReLU(inplace=True)
                                    ,nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias, groups=groups)
                                    ,nn.Sigmoid()
                                    )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Channel Attention Block (MPRNet)
class CAB(nn.Module):
    def __init__(self, n_feat, k_size, reduction, bias, act, groups=1):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(Conv_(n_feat, n_feat, k_size, bias=bias, groups=groups))
        modules_body.append(act)
        modules_body.append(Conv_(n_feat, n_feat, k_size, bias=bias, groups=groups))

        self.CA = CALayer(n_feat, reduction, bias=bias, groups=groups)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


#<<< (IMDN) contrast-aware channel attention module -> Group option added

class CCA(nn.Module):
    
    def contrast(self, in_feat):
        # fix 2: gradient 오류 방지목적 수정
        # stdv_channels -> contrast
        # 외부 함수 -> class 내장함수
        _, _, _h, _w = in_feat.shape
        _hw = _h * _w
        
        feat_sum        = in_feat.sum(3, keepdim=True).sum(2, keepdim=True)
        feat_mean       = feat_sum / _hw
        feat_diff       = torch.abs(in_feat - feat_mean) + 1e-3
        feat_pow        = feat_diff * feat_diff
        feat_pow_sum    = feat_pow.sum(3, keepdim=True).sum(2, keepdim=True) / _hw
        feat_var        = torch.sqrt(feat_pow_sum)
        return torch.nan_to_num(feat_var
                               ,nan=1e-3, posinf=1e-3, neginf=1e-3
                               )
        
    
    def __init__(self, n_feat, k_size=1, reduction=16, bias=True, act=nn.ReLU(inplace=True), groups=1):
        super(CCA, self).__init__()
        
        #self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # self.conv_du에서 오류발생 지목됨
        # stdv_channels의 eps를 1e-6에서 1e-3으로 키워봄
        self.conv_du = nn.Sequential(nn.Conv2d(n_feat, n_feat // reduction, k_size, padding=0, bias=True, groups=groups)
                                    ,act
                                    ,nn.Conv2d(n_feat // reduction, n_feat, k_size, padding=0, bias=True, groups=groups)
                                    ,nn.Sigmoid()
                                    )
        
    
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        z = self.conv_du(y)
        return x * z
#>>> (IMDN) contrast-aware channel attention module -> Group option added


#<<< (RFDN)
class ESA(nn.Module):
    def __init__(self, n_feat, reduction, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feat // reduction
        self.conv1 = conv(n_feat, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feat, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

#>>> (RFDN)


#>>> (MPRNet)
# Supervised Attention Module
"""
class SAM(nn.Module): 
    def __init__(self, n_feat, k_size, bias):
        super(SAM, self).__init__()
        self.conv1 = Conv_(n_feat, n_feat, k_size, bias=bias)
        self.conv2 = Conv_(n_feat, 3,      k_size, bias=bias)
        self.conv3 = Conv_(3,      n_feat, k_size, bias=bias)

    def forward(self, x_feat, x_img):
        x1 = self.conv1(x_feat)
        img = self.conv2(x_feat) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x_feat
        return x1, img
"""
#>>> (MPRNet)




# Feature DownSampler (x1/2)
class DownSampler(nn.Module):
    def __init__(self, in_feat, out_feat, groups=1):
        super(DownSampler, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
                                 ,nn.Conv2d(in_feat, out_feat, 1, stride=1, padding=0, bias=False, groups=groups)
                                 )

    def forward(self, x):
        x = self.down(x)
        return x


# Feature UpSampler (지정크기로 변환)
class UpSampler(nn.Module):
    def __init__(self, in_feat, out_feat, groups=1):
        super(UpSampler, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_feat, out_feat, 1, stride=1, padding=0, bias=False, groups=groups)
                               )

    def forward(self, x, size_h_w):
        x = self.up(F.interpolate(x, size=size_h_w, mode='bilinear', align_corners=True))
        return x



####################################################################################### <<< 새로운 시도

class Encoder_Decoder(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.PReLU(), groups=1, _receive_prev=True):
        super(Encoder_Decoder, self).__init__()
        
        self.layer_init     = nn.Sequential(CCA(n_feat=n_feat, k_size=1, reduction=reduction, bias=bias,          groups=groups)
                                           ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                           )
        
        self.layer_enc      = nn.Sequential(CCA(n_feat=n_feat, k_size=1, reduction=reduction, bias=bias,          groups=groups)
                                           ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                           )
        
        self.layer_down     = DownSampler(n_feat, n_feat)
        
        self.layer_deep     = nn.Sequential(CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                           ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                           )
        
        self.layer_up       = UpSampler(n_feat, n_feat, groups=groups)
        
        self.layer_dec      = nn.Sequential(CCA(n_feat=n_feat, k_size=1, reduction=reduction, bias=bias,          groups=groups)
                                           ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                           )
        
        self.layer_last     = nn.Sequential(CCA(n_feat=n_feat, k_size=1, reduction=reduction, bias=bias,          groups=groups)
                                           ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                           )
        
        if _receive_prev:
            # 이전 feature 입력받은 경우
            self.insert_down = Conv_(n_feat, n_feat, k_size=3, bias=bias, groups=groups)
            self.insert_up   = Conv_(n_feat, n_feat, k_size=3, bias=bias, groups=groups)
        
    def forward(self, x_init, prev=None):
        _, _, H, W = x_init.shape
        if prev is None:
            
            y_feat  = self.layer_init(x_init)           # n_feat
            
            y_enc   = self.layer_enc(y_feat)            # n_feat
            
            y_down  = self.layer_down(y_enc)            # n_feat
            
            y_deep  = self.layer_deep(y_down)           # n_feat
            
            y_up    = self.layer_up(y_deep, (H, W))     # n_feat
            
            y_dec   = self.layer_dec(y_up)              # n_feat
            
            y_out   = self.layer_last(y_dec)            # n_feat
            
            return y_out, [y_down, y_up]
        
        else:
            prev_down, prev_up = prev
            
            y_feat  = self.layer_init(x_init)           # n_feat
            
            y_enc   = self.layer_enc(y_feat)            # n_feat
            
            y_down  = self.layer_down(y_enc)            # n_feat
            
            _, _, _h, _w = y_down.shape
            y_deep  = self.layer_deep(y_down            # n_feat
                                     +self.insert_down(F.interpolate(prev_down, size=(_h, _w), mode='bilinear', align_corners=True))
                                     )
            
            y_up    = self.layer_up(y_deep, (H, W))     # n_feat
            
            _, _, _h, _w = y_up.shape
            y_dec   = self.layer_dec(y_up               # n_feat
                                    +self.insert_up(F.interpolate(prev_up, size=(_h, _w), mode='bilinear', align_corners=True))
                                    )
            
            y_out   = self.layer_last(y_dec)            # n_feat
            
            return y_out, [y_down, y_up]


# DFM -> RFM
class RFM(nn.Module): # Re-Focus Module
    def __init__(self, n_feat, in_feat=3, bias=False,act=nn.PReLU()):
        super(RFM, self).__init__()
        self.layer_init = nn.Sequential(Conv_(in_feat, n_feat, k_size=3, bias=False)
                                       ,act
                                       ,Conv_(n_feat, n_feat, k_size=3, bias=False)
                                       ,nn.Sigmoid()
                                       )
        
    def forward(self, in_feat, in_img):
        
        y_feat = self.layer_init(in_img)
        
        return in_feat + in_feat*y_feat



class Make_Feat_2_Img(nn.Module):
    def __init__(self, in_feat, upscale, k_size=3, act=nn.PReLU()):
        super(Make_Feat_2_Img, self).__init__()
        
        self.upscale = upscale
        
        n_feat = in_feat // 2
        
        self.layer_init = nn.Sequential(Conv_(in_feat, n_feat, k_size=3, bias=False)
                                       ,act
                                       ,Conv_(n_feat, n_feat, k_size=3, bias=False)
                                       ,act
                                       ,Conv_(n_feat, 3, k_size=3, bias=False)
                                       )
        
    def forward(self, in_feat):
        return self.layer_init(F.interpolate(in_feat, scale_factor=self.upscale, mode='bilinear'))


####################################################################################### >>> 새로운 시도

############################################################################################
## MODEL ## MODEL ## MODEL ## MODEL ## MODEL ## MODEL ## MODEL ## MODEL ## MODEL ## MODEL ##
############################################################################################

class BasicBlock(nn.Module):
    def __init__(self,n_feat,reduction,act=nn.PReLU(),bias=False,groups=1,is_head=False,is_tail=False,scale=1):
        super(BasicBlock, self).__init__()
        
        # n_feat    중간 과정 feature channels
        # reduction CCA 또는 CAB 의 reduction
        # is_head   현재 BB의 첫 블록 여부 -> input 형태와 연관됨
        # is_tail   현재 BB의 마지막 블록 여부 -> output 형태와 연관됨
        
        self.n_feat = n_feat
        
        self.is_head = is_head
        self.is_tail = is_tail
        if is_tail:
            self.scale = scale
        else:
            self.scale = 1
        
        if is_head:
            self.BB_init = nn.Sequential(Conv_(3, n_feat, k_size=3, bias=bias, groups=groups)
                                        ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                        ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                        ,CAB(n_feat=n_feat, k_size=3, reduction=reduction, bias=bias, act=act, groups=groups)
                                        )
            self.BB_ED   = Encoder_Decoder(n_feat=n_feat//2, reduction=reduction//2, bias=bias, act=act, groups=groups, _receive_prev=False)
        else:
            self.BB_ED   = Encoder_Decoder(n_feat=n_feat//2, reduction=reduction//2, bias=bias, act=act, groups=groups, _receive_prev=True)
        
        self.BB_bypass   = nn.Sequential(Conv_(n_feat//2, n_feat//2, k_size=1, bias=bias, groups=groups)
                                        )
        
        self.BB_merge    = nn.Sequential(ESA(n_feat=n_feat, reduction=reduction)
                                        ,CCA(n_feat=n_feat, k_size=1, reduction=reduction, bias=bias,          groups=groups)
                                        )
        
        self.BB_to_img = Make_Feat_2_Img(in_feat=n_feat, upscale=self.scale, k_size=3, act=act)
        
        if not is_tail:
            self.BB_RFM = RFM(n_feat=n_feat, in_feat=3, bias=bias,act=act)
        
    def forward(self, _input, in_img=None):
        if self.is_head:
            # _input = 이미지 텐서
            bb_init         = self.BB_init(_input)                                          # 최초 이미지로 생성된 feature
            bb_ed, bb_mid   = self.BB_ED(bb_init[:,0:self.n_feat//2,:,:])                   # E-D Block 통과한 feature
            bb_bypass       = self.BB_bypass(bb_init[:,self.n_feat//2:self.n_feat,:,:])     # 최소한의 가공만 시행한 feature
            
        else:
            # _input = [이전 feature 텐서, ED 중간 텐서]
            bb_prev         = _input[0]                                                     # 이전 BB의 bb_merge feature
            bb_ed, bb_mid   = self.BB_ED(bb_prev[:,0:self.n_feat//2,:,:], _input[1])
            bb_bypass       = self.BB_bypass(bb_prev[:,self.n_feat//2:self.n_feat,:,:])
        
        bb_merge            = self.BB_merge(torch.cat([bb_ed, bb_bypass], dim=1))           # 갈랐던 featue 병합
        
        bb_correct      = self.BB_to_img(bb_merge)
        
        if not self.is_tail:
            bb_merge_focus  = self.BB_RFM(bb_merge, bb_correct)
            
            return in_img + bb_correct, [bb_merge_focus, bb_mid]
            
        else:
            return F.interpolate(in_img, scale_factor=self.scale, mode='bilinear') + bb_correct
            




class model_proposed(nn.Module):
    def __init__(self, in_c = 3, out_c_img = 3, out_c_lab = out_c_lab
                ,n_feat=36
                ,reduction=6, act=nn.GELU() #nn.PReLU()
                ,bias=False
                ,groups = 1
                ,upscale=4
                ,front_freeze = front_freeze
                ,front_melt   = front_melt
                ,back_model_ss = back_model_ss
                ):
        super(model_proposed, self).__init__()
        
        # in_c          초기 Input Image 채널 수     -> 반드시 3 채널
        # out_c_img     최종 Output Image 채널 수
        self.out_c_img = out_c_img
        # out_c_lab     최종 Output Label 채널 수
        self.out_c_lab = out_c_lab
        
        # n_feat        Encoder & Decoder 입력 & 출력 채널 수
        
        # k_size        kernel 크기 (3 고정)
        # d_size        (block_c) dilation 크기
        #
        # act           Activation 함수
        # reduction     channel attention 시 reduction 배율 
        #
        # bias          Bias
        
        
        ##-------------------------------------------##
        ##-------------- Front   --------------------##
        ##-------------------------------------------##
        self.STAGE_FRONT_BasicBlock_LAYER_1 = BasicBlock(n_feat,reduction,act=act,bias=bias,groups=groups,is_head=True)
        #self.STAGE_FRONT_BasicBlock_LAYER_2 = BasicBlock(n_feat,reduction,act=act,bias=bias,groups=groups)
        self.STAGE_FRONT_BasicBlock_LAYER_3 = BasicBlock(n_feat,reduction,act=act,bias=bias,groups=groups,is_tail=True,scale=upscale)
        
        
        ##-------------------------------------------##
        ##-------------- Back    --------------------##
        ##-------------- Stage 1 --------------------##
        ##-------------------------------------------##
        
        self.STAGE_BACK_MODEL_SS = back_model_ss 
        
        ##-------------------------------------------##
        ##-------------- SWITCH  --------------------##
        ##-------------------------------------------##
        self._device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._prev_mode  = False        # self.training: Train=True, Eval=False
        self._curr_epoch = 0            # 현재 epoch 값
        self._curr_stage = 1            # 현재 stage 값
        self._front_freeze = front_freeze
        self._front_melt   = front_melt
        
    
    
    def _freezing(self, key):
        print("[model] freezing", key)
        for name, param in self.named_parameters():
            if key in name:
                print("Freezed:", key)
                param.requires_grad = False
    
    def _melting(self, key):
        print("[model] melting", key)
        for name, param in self.named_parameters():
            if key in name:
                print("Melted:", key)
                param.requires_grad = True
    
    
    
    def forward(self, x_img):
        ##-------------------------------------------##
        ##-------------- SWITCH  --------------------##
        ##-------------------------------------------##
        
        if self._prev_mode != self.training:
            if self.training:
                # Eval -> Train
                self._curr_epoch += 1
                print("\n[model] epoch", self._curr_epoch, "train init")
                self._prev_mode = self.training
                
                
                if self._curr_epoch == self._front_freeze:
                    self._freezing("STAGE_FRONT_BasicBlock_LAYER")
                
                if self._curr_epoch == self._front_melt:
                    self._melting("STAGE_FRONT_BasicBlock_LAYER")
                
                
                
            else:
                # Train -> Eval
                print("\n[model] epoch", self._curr_epoch, "val init")
                self._prev_mode = self.training
                
        
        ##-------------------------------------------##
        ##--------------  Front  --------------------##
        ##--------------  Stage  --------------------##
        ##-------------------------------------------##
        init_B, _, init_H , init_W = x_img.shape
        
        y_img_1, y_BB_1 = self.STAGE_FRONT_BasicBlock_LAYER_1(x_img,  in_img=x_img)
        #y_img_2, y_BB_2 = self.STAGE_FRONT_BasicBlock_LAYER_2(y_BB_1, in_img=x_img)
        #y_img_3         = self.STAGE_FRONT_BasicBlock_LAYER_3(y_BB_2, in_img=x_img)
        y_img_3         = self.STAGE_FRONT_BasicBlock_LAYER_3(y_BB_1, in_img=x_img)
        
        #return [y_img_3, y_img_2, y_img_1]
        
        ##-------------------------------------------##
        ##--------------  Back   --------------------##
        ##--------------  Stage  --------------------##
        ##-------------------------------------------##
        
        if self._curr_epoch <= 1 or self._curr_epoch >= self._front_freeze:
            y_lab_1 = self.STAGE_BACK_MODEL_SS(y_img_3)
        else:
            sr_B, _, sr_H, sr_W = y_img_3.shape
            y_lab_1 = torch.zeros([sr_B, self.out_c_lab, sr_H , sr_W], device=self._device)
        
        
        #y_lab_1 = self.STAGE_BACK_MODEL_SS(y_img_3)
        
        # label, image, image_stage_1, image_stage_2(IF)
        #return y_lab_1, y_img_3, y_img_1, y_img_2
        
        # block 1 out, block 2 out, ... , block n out(= SR Image), SS Net out(=SR Label)
        #return y_img_1, y_img_2, y_img_3, y_lab_1 # COB: 3
        return y_img_1, y_img_1, y_img_3, y_lab_1 # COB: 2 -> 편의상 y_img_2 자리에 y_img_1 복사


















##########################################################################################
## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ## LOSS ##
##########################################################################################

class loss_proposed(nn.Module):
    # charbonnier_loss -> 보완된 L1 loss에 해당
    def charb_loss(self, pred_img, ans_img, is_AMP=False, eps=1e-3):
        if is_AMP:
            diff = pred_img.to(torch.float32) - ans_img.to(torch.float32)
        else:
            diff = pred_img - ans_img
        
        return torch.mean(torch.sqrt((diff*diff) + (eps*eps)))
    
    def __init__(self
                ,front_freeze       = front_freeze          # (int) 동결 시점
                ,front_melt         = front_melt            # (int) 해동 시점
                ,loss_ce_weights    = loss_ce_weights       # (list) cross entropy loss의 weight
                ,is_onehot          = None                  # (bool) GT label one-hot encode 시행 여부
                ,class_void         = None                  # (int) void class 번호 -> is_onehot is False인 경우에만 반영됨
                ):
        super(loss_proposed, self).__init__()
        
        ##-------------------------------------------##
        ##-------------- SWITCH  --------------------##
        ##-------------------------------------------##
        
        self._prev_mode  = False        # self.training: Train=True, Eval=False
        self._curr_epoch = 0            # 현재 epoch 값
        self._curr_stage = 1            # 현재 stage 값
        self._front_freeze = front_freeze
        self._front_melt   = front_melt
        
        self._device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        if is_onehot:
            # GT label one-hot encode 시행됨
            if loss_ce_weights is not None:
                try:
                    self.loss_ce_weights = torch.FloatTensor(loss_ce_weights)
                    self.loss_ce_weights = self.loss_ce_weights.to(self._device)
                    self.loss_ce  = nn.CrossEntropyLoss(weight=self.loss_ce_weights)   # loss = self.loss_ce (predict, answer)
                    print("\n[loss] loss_ce's weight:", self.loss_ce_weights)
                    print("[loss] one-hot encode applied")
                except:
                    self.loss_ce  = nn.CrossEntropyLoss()   # loss = self.loss_ce (predict, answer)
                    print("\n[loss] loss_ce with no weight")
                    print("[loss] one-hot encode applied")
            else:
                self.loss_ce  = nn.CrossEntropyLoss()
                print("\n[loss] loss_ce with no weight")
                print("[loss] one-hot encode applied")
        else:
            # GT label one-hot encode 시행 안됨
            if loss_ce_weights is not None:
                try:
                    self.loss_ce_weights = torch.FloatTensor(loss_ce_weights)
                    self.loss_ce_weights = self.loss_ce_weights.to(self._device)
                    self.loss_ce  = nn.CrossEntropyLoss(weight=self.loss_ce_weights,ignore_index=class_void)
                    print("\n[loss] loss_ce's weight:", self.loss_ce_weights)
                    print("[loss] one-hot encode NOT applied")
                except:
                    self.loss_ce  = nn.CrossEntropyLoss(ignore_index=class_void)
                    print("\n[loss] loss_ce with no weight")
                    print("[loss] one-hot encode NOT applied")
            else:
                self.loss_ce  = nn.CrossEntropyLoss(ignore_index=class_void)
                print("\n[loss] loss_ce with no weight")
                print("[loss] one-hot encode NOT applied")
        
        
        
        
    
    def calc_v6(self, pred_label, pred_sr, ans_label, ans_sr
               ,pred_img_1=None, pred_img_2=None, pred_img_3=None
               ,is_AMP = None, is_onehot = None
               ):
        
        ##-------------------------------------------##
        ##-------------- SWITCH  --------------------##
        ##-------------------------------------------##
        
        is_init_epoch = False
        
        if pred_img_1.requires_grad:
            # Model Train mode
            if self._prev_mode is False:
                # Eval -> Train switched
                self._prev_mode = True
                self._curr_epoch += 1
                print("[loss] epoch", self._curr_epoch, "train init")
                is_init_epoch = True
            
        else:
            # Model Eval mode
            if self._prev_mode is True:
                self._prev_mode = False
                print("[loss] epoch", self._curr_epoch, "val init")
                is_init_epoch = True
        
        
        if self._curr_epoch <= 1 or self._curr_epoch >= self._front_melt:
            LOSS_STAGE = 3
        elif self._curr_epoch >= self._front_freeze and self._curr_epoch < self._front_melt:
            LOSS_STAGE = 2
        else:
            LOSS_STAGE = 1
            
        
        if is_init_epoch:
            print("Loss Stage:", LOSS_STAGE)
        
        ##-------------------------------------------##
        ##--------------  CALC   --------------------##
        ##-------------------------------------------##
        _, _, _h, _w = ans_sr.shape
        
        #--- COB = 3
        if LOSS_STAGE == 1:
            # Front stage
            return torch.nan_to_num(self.charb_loss(F.interpolate(pred_img_1, size=(_h, _w), mode='bilinear'), ans_sr, is_AMP=is_AMP)   * 0.1
                                   #+self.charb_loss(F.interpolate(pred_img_2, size=(_h, _w), mode='bilinear'), ans_sr, is_AMP=is_AMP)   * 0.1
                                   +self.charb_loss(pred_sr,                                                   ans_sr, is_AMP=is_AMP)   * 0.1
                                   ,nan=1e-6, posinf=1e-6, neginf=1e-6
                                   )
        elif LOSS_STAGE == 2:
            # Back stage
            return torch.nan_to_num(self.loss_ce(pred_label, ans_label)                                                                 * 1.0
                                   ,nan=1e-6, posinf=1e-6, neginf=1e-6
                                   )
        
        elif LOSS_STAGE == 3:
            # All stage
            return torch.nan_to_num(self.charb_loss(F.interpolate(pred_img_1, size=(_h, _w), mode='bilinear'), ans_sr, is_AMP=is_AMP)   * 0.1
                                   #+self.charb_loss(F.interpolate(pred_img_2, size=(_h, _w), mode='bilinear'), ans_sr, is_AMP=is_AMP)   * 0.1
                                   +self.charb_loss(pred_sr,                                                   ans_sr, is_AMP=is_AMP)   * 0.1
                                   +self.loss_ce(pred_label, ans_label)                                                                 * 1.0
                                   ,nan=1e-6, posinf=1e-6, neginf=1e-6
                                   )
    
    
    