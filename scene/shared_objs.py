import torch
import numpy as np
import cv2
import torch.nn as nn
import copy
import math

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):  
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class SharedCam(nn.Module):
    def __init__(self, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super().__init__()
        self.cam_idx = torch.zeros((1)).int()
        self.R = torch.eye(3,3).float()
        self.t = torch.zeros((3)).float()
        self.FoVx = torch.tensor([FoVx])
        self.FoVy = torch.tensor([FoVy])
        self.image_width = torch.tensor([image.shape[1]])
        self.image_height = torch.tensor([image.shape[0]])
        self.cx = torch.tensor([cx])
        self.cy = torch.tensor([cy])
        self.fx = torch.tensor([fx])
        self.fy = torch.tensor([fy])
        
        self.intrisic_matrix = torch.zeros((4, 4)).float().cuda()
        self.intrisic_matrix[0, 0] = fx
        self.intrisic_matrix[1, 1] = fy
        self.intrisic_matrix[0, 2] = cx
        self.intrisic_matrix[1, 2] = cy
        self.intrisic_matrix[3, 3] = 1.0
        
        self.original_image = torch.from_numpy(image).float().permute(2,0,1)/255
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(self.R, self.t, trans, scale).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def update_matrix(self):
        self.world_view_transform[:,:] = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        self.full_proj_transform[:,:] = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center[:] = self.world_view_transform.inverse()[3, :3]
    
    def setup_cam(self, R, t, rgb_img, depth_img):
        self.R[:,:] = torch.from_numpy(R)
        self.t[:] = torch.from_numpy(t)
        self.update_matrix()
        self.original_image[:,:,:] = torch.from_numpy(rgb_img).float().permute(2,0,1)/255
        self.original_depth_image[:,:,:] = torch.from_numpy(depth_img).float().unsqueeze(0)
    
    def on_cuda(self):
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()
        
        self.original_image = self.original_image.cuda()
        self.original_depth_image = self.original_depth_image.cuda()