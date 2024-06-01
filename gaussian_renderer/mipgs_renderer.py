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
from diff_gaussian_rasterization_freqConstrain import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn as nn 


def mpgs_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
          kernel_size=0.1, band_limit=True, subpixel_offset=None, scaling_modifier = 1.0, override_color = None):
    select_view = 0
    tone_map=None
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    #print('in renderer, the kernel size and band_limit are ', kernel_size, band_limit)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        vIdx = select_view,
        coc2z_coeff = 0.0,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    if band_limit:
        opacity = pc.get_opacity_with_3D_filter
    else:
        opacity = pc.get_opacity
    
    focal_distance = torch.ones(1,1).cuda()
    K_apert = torch.zeros(1,1).cuda()

    #print('In version79 renderer, the band_limit parameters is set to ', band_limit)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if band_limit:
            scales = pc.get_scaling_with_3D_filter
        else:
            scales = pc.get_scaling
            
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features #　shape is [#n_points, 16, 3]
    else:
        colors_precomp = override_color
        

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, xy, coc_alpha, radii, radii_diff = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp, 
        F=focal_distance,
        K=K_apert)
    

    if tone_map is not None:
        rendered_image = tone_map(rendered_image)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "radii_diff": radii_diff,
            "coc_alpha": coc_alpha}




class ToneMapping(nn.Module):
    def __init__(self, map_type: str):
        super(ToneMapping, self).__init__()
        assert map_type in ['none', 'gamma', 'learn', 'ycbcr']
        self.map_type = map_type
        if map_type == 'learn':
            self.linear = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, x):
        if self.map_type == 'none':
            return x
        elif self.map_type == 'learn':
            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            res_x = self.linear(x_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            return x_out.reshape(ori_shape)
        elif self.map_type == 'gamma':
            return (x + 1e-10) ** (1. / 2.2)
        else:
            assert RuntimeError("map_type not recognized")