#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch_cluster
from utils.ops import (
    distance_to_gaussian_surface,
    K_nearest_neighbors,
    qvec2rotmat_batched,
)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_scaling_activation = torch.log

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args, device=torch.device('cuda:0')):
        self.args = args
        self.active_sh_degree = 0
        self.device = device
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.bg_color = torch.empty(0)
        self.confidence = torch.empty(0)
        self.Fs = torch.tensor([3.0], device=self.device)
        self.Ks = torch.tensor([0.0], device=self.device)
        self.Fs = self.Fs.unsqueeze(0)
        self.Ks = self.Ks.unsqueeze(0)
        self.optimize_defocus = False        

   
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.Fs,
            self.Ks
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.Fs,
        self.Ks) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def rotmat(self):
        return qvec2rotmat_batched(self._rotation)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            # print(camera.image_name, xyz[...,0].max(), xyz[...,0].min(), xyz[...,0].mean())
            # print(camera.image_name, xyz[...,1].max(), xyz[...,1].min(), xyz[...,1].mean())
            # print(camera.image_name, xyz[...,2].max(), xyz[...,2].min(), xyz[...,2].mean())   
            # print(camera.image_name, R, T)          
            xyz_cam = xyz @ R + T[None, :]
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            


            # print(camera.image_name, x.max(), x.min(), x.mean())
            # print(camera.image_name, y.max(), y.min(), y.mean())
            # print(camera.image_name, z.max(), z.min(), z.mean())

            # print('$$$$$$$$$$$$$$$$$')
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            # print(camera.image_name, x.max(), x.min(), x.mean())
            # print(camera.image_name, y.max(), y.min(), y.mean())
            # print(camera.image_name, z.max(), z.min(), z.mean())
            # print(camera.focal_x, camera.focal_y, camera.image_width, camera.image_height)
            # print('---------------')
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")

        
    def setup_defocus_params(self, N_view, camera_translate, dataset_path, mask_act='binary', COC_MAX=15):
        z_median = 1.0 / torch.quantile(1.0 / self._xyz[:, 2], 0.5, interpolation='nearest').item()  + camera_translate[2]
        print('z.median is ', z_median)

        # determine the initial value of K according to the depth range of the scene and the specified max coc value
        z_max = 1.0 / torch.quantile(1.0 / self._xyz[:,2], 0.05, interpolation='nearest').item() 
        z_min = 1.0 / torch.quantile(1.0 / self._xyz[:,2], 0.95, interpolation='nearest').item() 
        K_value = COC_MAX / (np.abs(1.0 / z_min - 1.0/ z_max) + 1e-6)
        print('K_value is ', K_value)
        K_shrink_factor = (K_value - 100.0) / 1000.0
        
        self.Fs = torch.ones(N_view, 1)
        self.Fs = self.Fs * z_median                         
        self.Fs = self.Fs / 10.0
        #  * z_median / 10.0
        self.Fs = self.Fs.to(self.device)
        self.Ks = torch.ones(N_view, 1) * 10.0 * K_shrink_factor
        self.Ks = self.Ks.to(self.device)
        
        self.Fs = nn.Parameter(self.Fs.requires_grad_(True))
        self.Ks = nn.Parameter(self.Ks.requires_grad_(True))
        self.optimize_defocus = False
        self.mask_net = MaskNet(N_view, act_type=mask_act)
        self.mask_net = self.mask_net.to(self.device)

    def join_optim_defocus_params(self, training_args):
        self.optimizer.add_param_group({'params': [self.Fs], 'lr': training_args.fdist_lr, "name": "fdist"})
        self.optimizer.add_param_group({'params': [self.Ks], 'lr': training_args.kapert_lr, "name": "kapert"})
       
    def join_optim_mlp_params(self, training_args):
        for name, p in self.mask_net.named_parameters():
            self.optimizer.add_param_group({'params': [p], 'lr': training_args.mask_nn_lr, 'name': f'maskNet.{name}'})
            self.const_group_list.append(f'maskNet.{name}')
    
    def set_F_optimizer(self, training_args):
        l = [
            {'params': [self.Fs], 'lr': training_args.F_stage_lr, "name": "fdist"}
        ]
        if training_args.F_optimizer_type == 'adam':
            self.optimizer_F = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif training_args.F_optimizer_type == 'sgd':
            self.optimizer_F = torch.optim.SGD(l, lr=0.0)
        elif training_args.F_optimizer_type == 'rprop':
            self.optimizer_F = torch.optim.Rprop(l, lr=0.0)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}        
        ]
        
        if self.optimize_defocus: 
            self.join_optim_defocus_params(training_args)

        # the group list within which the group params won't be pruned or densitified.
        self.const_group_list = ["fdist", "kapert"] 
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        return l

    def save_defocus_params(self, path):
        state_dict ={}
        state_dict['focal_distance'] = self.Fs.detach().cpu()
        state_dict['K_aperture'] = self.Ks.detach().cpu()
        state_dict['mask_net_dict'] = self.mask_net.cpu().state_dict()
        torch.save(state_dict, path)
        self.mask_net.to(self.device)
        
    def load_defocus_params(self, path):
        state_dict = torch.load(path)
        self.Fs = state_dict['focal_distance']
        self.Ks = state_dict['K_aperture']
        mask_net_dict = state_dict['mask_net_dict']
        
        self.Fs = self.Fs.to(self.device)
        self.Ks = self.Ks.to(self.device)
        self.mask_net.load_state_dict(mask_net_dict)
        self.mask_net = self.mask_net.to(self.device)
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
    def reset_opacity_with_filter3D(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = inverse_sigmoid(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
        

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "name" not in group:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self.const_group_list:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, iter):
        if iter > self.args.prune_from_iter:
            valid_points_mask = ~mask
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self.confidence = self.confidence[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in self.const_group_list:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)


    def densify_by_shrink_then_compatness(self, shrink_factor: float = 1.5, K: int = 3):
        self._scaling = self._scaling / shrink_factor
        return self.densify_by_compatness(K=K)
     
    def add_extra_points(self, M=1e5, k=4, dist_thres=15, range_factor=0.1):
        def scale_mins(value, factor=0.25):
            value = value - factor * torch.abs(value)
            return value
        
        def scale_maxs(value, factor=0.25):
            value = value + factor * torch.abs(value)
            return value 

        # Find the min and max values for x, y, z in the original point cloud
        x_min, y_min, z_min = torch.min(self._xyz, dim=0).values
        x_max, y_max, z_max = torch.max(self._xyz, dim=0).values

        x_min, y_min, z_min = scale_mins(x_min, range_factor), scale_mins(y_min, range_factor), scale_mins(z_min, range_factor)
        x_max, y_max, z_max = scale_maxs(x_max, range_factor), scale_maxs(y_max, range_factor), scale_maxs(z_max, range_factor)

                
        # Sample new points from a uniform distribution within the min-max range
        new_points = torch.stack([
            torch.cuda.FloatTensor(int(M * 2)).uniform_(x_min, x_max),
            torch.cuda.FloatTensor(int(M * 2)).uniform_(y_min, y_max),
            torch.cuda.FloatTensor(int(M * 2)).uniform_(z_min, z_max)
        ], dim=1)
                         
    
        # Compute the knn graph for the new points
        edge_index = torch_cluster.knn(self._xyz, new_points, k, cosine=False, num_workers=1)
        indexes = edge_index[1].reshape(-1, k)
        neighbors = self._xyz[indexes]
        #  Calculate the distances between each new point and its neighbors
        diff = new_points.unsqueeze(1) - neighbors
        distances = torch.norm(diff, dim=2)
          
        # Find the valid new points according to its distances to neighbors
        valid_indexes = torch.min(distances, dim=1).values < dist_thres
        print('valid_indexes.shape', valid_indexes.shape)
        valid_neighbors = indexes[valid_indexes]
        valid_distances = distances[valid_indexes]
        # the weights for interpolation, inversely proportional to the distances
        valid_weights = 1 / (valid_distances + 1e-5)
        valid_weights /= torch.sum(valid_weights, dim=1, keepdim=True)
        
        # fetch attributes for interpolation
        n_feature_dc = self._features_dc[valid_neighbors]
        n_feature_rest = self._features_rest[valid_neighbors]
        n_opacity = self._opacity[valid_neighbors]
        
        #print(n_feature_dc.shape, n_feature_rest.shape, n_opacity.shape)
        #interpolation
        new_xyz = new_points[valid_indexes]
        new_scaling = torch.log(torch.clamp(torch.min(valid_distances, dim=1).values, 1e-6, 10.0))[...,None].repeat(1, 3)
        new_rotation = torch.zeros(new_xyz.shape[0], 4, device='cuda') 
        new_rotation[:, 0] = 1.0
        new_features_dc = torch.sum(valid_weights.unsqueeze(-1).unsqueeze(-1) * n_feature_dc, dim=1)
        new_features_rest = torch.sum(valid_weights.unsqueeze(-1).unsqueeze(-1) * n_feature_rest, dim=1)
        new_opacity = torch.sum(valid_weights.unsqueeze(-1) * n_opacity, dim=1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_by_compatnes_with_idx(self, idx, _rotmat):                
        nn_svec = self._scaling[idx]
        nn_rotmat = _rotmat[idx]
        nn_pos = self._xyz[idx]

        nn_gaussian_surface_dist = distance_to_gaussian_surface(
            nn_pos, nn_svec, nn_rotmat, self._xyz
        )
        gaussian_surface_dist = distance_to_gaussian_surface(
            self._xyz, self._scaling, _rotmat, nn_pos
        )

        dist_to_nn = torch.norm(nn_pos - self._xyz, dim=-1)
        mask = (gaussian_surface_dist + nn_gaussian_surface_dist) < dist_to_nn
        new_direction = (nn_pos - self._xyz.data) / dist_to_nn[..., None]
        new_mean = (
            self._xyz.data
            + new_direction
            * (dist_to_nn + gaussian_surface_dist - nn_gaussian_surface_dist)[..., None]
            / 2.0
        )[mask]
        new_feature_dc = self._features_dc.data[mask]
        new_feature_rest = self._features_rest.data[mask]
        new_opacities = self._opacity.data[mask]
        new_rotation = self._rotation.data[mask]
        # print(torch.ones_like(self.svec.data[mask]).shape)
        # print(
        #     (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask].shape
        # )
        new_raw_svec = self.inverse_scaling_activation(
            torch.clamp(torch.ones_like(self._scaling.data[mask])
            * (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask][
                ..., None
            ]
            / 6.0, 1e-6, 10.0)
        )
        new_params = {
            "xyz": new_mean,
            "f_dc": new_feature_dc,
            "f_rest": new_feature_rest,
            "opacity": new_opacities,
            "scaling": new_raw_svec,
            "rotation": new_rotation
            }
        return new_params


    def densify_by_compatness(self, K=1):
         # Compute the knn graph for the new points
        rotmat = self.rotmat
        edge_index = torch_cluster.knn(self._xyz, self._xyz, int(K+1), cosine=False, num_workers=4)
        idx = edge_index[1].reshape(-1, int(K+1))

        num_densified = 0
        new_params_list = []
        for i in range(K):
            new_params = self.densify_by_compatnes_with_idx(idx[:, i], rotmat)
            new_params_list.append(new_params)
        new_params = {}
        for key in new_params_list[0].keys():
            new_params[key] = torch.cat([p[key] for p in new_params_list], dim=0)
        num_densified = new_params["xyz"].shape[0]
        self.densification_postfix(new_params['xyz'], new_params['f_dc'], new_params['f_rest'], 
                                   new_params['opacity'], new_params['scaling'], new_params['rotation'] )
        return num_densified


    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def cal_adaptive_scale_mask(self, extent, scale_start=0.1, scale_end=0.6):
        with torch.no_grad():
            _, _, z_min = torch.min(self._xyz, dim=0).values
            _, _, z_max = torch.max(self._xyz, dim=0).values
            weight = torch.abs(self._xyz[:, 2].detach() - z_min) / (z_max - z_min + 1e-6) * (scale_end - scale_start) + scale_start
            scaled_extent = weight * extent 
            big_points_ws = ((self.get_scaling.max(dim=1).values - scaled_extent) > 0.0).squeeze()
        return big_points_ws
    
    def cal_adaptive_prune_mask(self, min_opacity, wp=0.3):
        with torch.no_grad():
            _, _, z_min = torch.min(self._xyz, dim=0).values
            _, _, z_max = torch.max(self._xyz, dim=0).values
            weight = torch.abs(self._xyz[:, 2:].detach() - z_max) / (z_max - z_min + 1e-6) * (1.0 - wp) + wp
            #print('weight.shape is ', (weight * min_opacity).shape)
            prune_mask = ((self.get_opacity - weight * min_opacity) < 0.0).squeeze()
        return prune_mask


    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, iter)
        if iter < 2000:
            self.proximity(extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask, iter)
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1