"""
Visualization tool for 3D Gaussian Splatting with semantic labels.
This module provides interactive visualization capabilities for 3D scenes
with semantic information using Open3D and custom rendering.
"""

# Standard library imports
import os
import time
import argparse
from copy import deepcopy
import math

# Third-party imports
import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.multiprocessing as mp
import open3d as o3d

# Local imports
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.shared_objs import SharedCam
from scene.utils import *

def on_click(x, y, button, pressed, mouse_pos):
    """Mouse click event handler that updates shared mouse position."""
    if button == button.left and pressed:
        mouse_pos[0] = x
        mouse_pos[1] = y
        print(f"Mouse clicked at position: {x}, {y}")

def start_listener(mouse_pos):
    """Initialize mouse listener in a separate process."""
    from pynput.mouse import Listener
    with Listener(on_click=lambda x, y, button, pressed: on_click(x, y, button, pressed, mouse_pos)) as listener:
        listener.join()

class MyVisualizer():
    def __init__(self, args, mouse_pos):
        """
        Initialize the visualizer with configuration and scene parameters.
        
        Args:
            args: Parsed command line arguments
            mouse_pos: Shared list for mouse position
        """
        # Configuration parameters
        self.config = args.config
        self.scene_npz = args.scene_npz
        self.dataset_type = args.dataset_type
        self.view_scale = args.view_scale
        self.mouse_pos = mouse_pos
        self.label_nums = args.label_nums
        self.render_label_nums = args.render_label_nums
        self.output_path = args.output_path

        # Display parameters
        self.test_rgb_img = np.zeros((680, 1200, 3), dtype=np.uint8)    # For replica dataset
        self.test_depth_img = np.zeros((680, 1200), dtype=np.float32)   # For replica dataset

        # Load camera parameters
        with open(self.config) as camera_parameters_file:
            camera_parameters_ = camera_parameters_file.readlines()
            camera_parameters = camera_parameters_[2].split()
            self.W = int(camera_parameters[0])   
            self.H = int(camera_parameters[1])
            self.fx = float(camera_parameters[2])
            self.fy = float(camera_parameters[3])
            self.cx = float(camera_parameters[4])
            self.cy = float(camera_parameters[5])
            self.depth_scale = float(camera_parameters[6])
            self.depth_trunc = float(camera_parameters[7])

        # UI offset parameters
        self.x_off = 141  # X-axis UI offset， ！！may need manual adjustment
        self.y_off = 127  # Y-axis UI offset， ！！may need manual adjustment

        # State variables
        self.capture = False
        self.start_w2c = None
        self.view_mode = "color"
        self.object_view_mode = False
        self.change_views = False
        self.current_label = None
        self.mapping_cam_index = 0
        self.mouse_x = 0
        self.mouse_y = 0
        
        # Transform parameters
        self.xyz_trans = torch.zeros(3).cuda()
        self.xyz_per_trans = 0.005
        self.changed_w2c = np.eye(4)
        
        # Semantic related variables
        torch.manual_seed(0)
        self.label_2_feature = torch.rand(self.label_nums, 3, dtype=torch.float32).cuda()    # Color storage for semantic info
        self.label_2_feature[0] = torch.zeros((3)).cuda()    # Color for unlabeled points
        self.label_map = torch.zeros((self.render_label_nums, self.H, self.W), dtype=torch.float32, device='cuda').contiguous()
        self.label_2_render_label = torch.zeros((self.label_nums), dtype=torch.int32, device='cuda').contiguous()
        self.mapping_gs_record = torch.zeros((50, self.H, self.W), dtype=torch.int, device='cuda').contiguous()
        self.mapping_gs_label_record = torch.zeros((50, self.H, self.W), dtype=torch.int, device='cuda').contiguous()

        # Camera intrinsic matrix
        k = torch.tensor([self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3).cuda()

        # Initialize camera parameters
        rendered_cam = SharedCam(FoVx=self.focal2fov(self.fx, self.W), FoVy=self.focal2fov(self.fy, self.H), 
                                      image=self.test_rgb_img, depth_image=self.test_depth_img,
                                      cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)

        # Load scene parameters
        scene_npz = np.load(self.scene_npz)
        self.xyz = torch.tensor(scene_npz["xyz"]).cuda().float().contiguous()
        self.opacity = torch.tensor(scene_npz["opacity"]).cuda().float().contiguous()
        self.scales = torch.tensor(scene_npz["scales"]).cuda().float().contiguous()
        self.rotation = torch.tensor(scene_npz["rotation"]).cuda().float().contiguous()
        self.shs = torch.tensor(scene_npz["shs"]).cuda().float().contiguous()
        self.with_sem_labels = False
        if "sem_labels" in scene_npz:
            print("With semantic labels")
            self.with_sem_labels = True
            self.sem_labels = torch.tensor(scene_npz["sem_labels"]).cuda().float()
            self.colors_precomp = self.sem_labels.unsqueeze(1).unsqueeze(2).repeat(1,1,3).cuda().contiguous()

        # Check for R_list and T_list
        if "R_list" not in scene_npz or "T_list" not in scene_npz:
            print("No R_list or T_list in scene_npz")
            self.R_list = [np.eye(3)]
            self.t_list = [np.zeros(3)]
        else:
            # Load per-frame RT
            R_list = scene_npz["R_list"]
            t_list = scene_npz["T_list"]
            self.R_list = R_list
            self.t_list = t_list

        # Initialize first frame camera
        rendered_cam.R = torch.tensor(self.R_list[0]).float().cuda()
        rendered_cam.t = torch.tensor(self.t_list[0]).float().cuda()
        rendered_cam.update_matrix()
        rendered_cam.on_cuda()
        w2c = np.eye(4)
        if self.start_w2c is not None:
            w2c = self.start_w2c
        else:
            w2c[:3, :3] = self.R_list[-1]
            w2c[:3, 3] = self.t_list[-1]

        # Initialize visualization
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=int(self.W * self.view_scale), height=int(self.H * self.view_scale), visible=True)
        vis.register_key_callback(ord("T"), self.on_T_key_press)  # Toggle between color and label display modes.
        vis.register_key_callback(ord("J"), self.on_J_key_press)  # Toggle between showing all objects or a single object.
        vis.register_key_callback(ord("K"), self.on_K_key_press)  # Capture the current view.
        vis.register_key_callback(ord("A"), self.on_A_key_press)  # Translate the object along the x-axis by +0.01.
        vis.register_key_callback(ord("S"), self.on_S_key_press)  # Translate the object along the y-axis by +0.01.
        vis.register_key_callback(ord("D"), self.on_D_key_press)  # Translate the object along the z-axis by +0.01.
        vis.register_key_callback(ord("Z"), self.on_Z_key_press)  # Translate the object along the x-axis by -0.01.
        vis.register_key_callback(ord("X"), self.on_X_key_press)  # Translate the object along the y-axis by -0.01.
        vis.register_key_callback(ord("C"), self.on_C_key_press)  # Translate the object along the z-axis by -0.01.
        vis.register_key_callback(ord("F"), self.on_F_key_press)  # Rotate the object around the x-axis by +1 degree.
        vis.register_key_callback(ord("G"), self.on_G_key_press)  # Rotate the object around the y-axis by +1 degree.
        vis.register_key_callback(ord("H"), self.on_H_key_press)  # Rotate the object around the z-axis by +1 degree.
        vis.register_key_callback(ord("V"), self.on_V_key_press)  # Rotate the object around the x-axis by -1 degree.
        vis.register_key_callback(ord("B"), self.on_B_key_press)  # Rotate the object around the y-axis by -1 degree.
        vis.register_key_callback(ord("N"), self.on_N_key_press)  # Rotate the object around the z-axis by -1 degree.
        vis.register_key_callback(ord("O"), self.on_O_key_press)  # Output the current camera view matrix.
        vis.register_key_callback(ord("M"), self.on_M_key_press)  # Switch to the next mapping camera view.
        vis.register_key_callback(ord("L"), self.on_L_key_press)  # Increase the scale of all Gaussians.
        vis.register_key_callback(ord("P"), self.on_P_key_press)  # Downsample Gaussians using a voxel grid.

        # Initialize first point cloud
        self.bg = torch.tensor([0.0, 0.0, 0.0]).float().cuda()
        with torch.no_grad():
            time1  = time.time()
            im, depth, visibility_filter = self.render(rendered_cam, self.xyz, self.opacity, self.scales, self.rotation, self.shs, self.bg)
            print(f"Render Time cost: {time.time() - time1}")
        init_pts, init_cols = self.rgbd2pcd(im, depth, w2c, k)
        pcd = o3d.geometry.PointCloud()
        pcd.points = init_pts
        pcd.colors = init_cols
        vis.add_geometry(pcd)

        # Initialize view control
        view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        view_w2c = w2c.astype(np.float64)
        view_k = deepcopy(k) * self.view_scale
        view_k = view_k.cpu().numpy().astype(np.float64)
        view_k[2, 2] = 1
        cparams.extrinsic = view_w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height =  int(self.H * self.view_scale)
        cparams.intrinsic.width = int(self.W * self.view_scale)
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
        render_options = vis.get_render_option()
        render_options.point_size = float(self.view_scale)
        render_options.light_on = False

        # State variables
        last_mouse_x = 0
        last_mouse_y = 0

        # Edit mode variables
        edit_label_mask = None

        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / self.view_scale
            k[2,2] = 1
            w2c = cam_params.extrinsic

            if self.change_views:
                # Change view
                w2c = self.changed_w2c
                self.change_views = False
                # Update camera extrinsics
                cparams.extrinsic = w2c.astype(np.float64)
                view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

            self.w2c = w2c

            R = torch.tensor(w2c[:3,:3]).cuda()
            t = torch.tensor(w2c[:3,3]).cuda()
            rendered_cam.R = R
            rendered_cam.t = t
            rendered_cam.update_matrix()
            rendered_cam.on_cuda()

            im, depth, visibility_filter = self.render(rendered_cam, self.xyz, self.opacity, self.scales, self.rotation, self.shs, self.bg)

            if self.with_sem_labels:
                with torch.no_grad():
                    unique_label, counts = torch.unique(self.sem_labels[visibility_filter], return_counts=True)
                    if unique_label.shape[0] > 0:
                        if unique_label[0] != 0:
                            unique_label = torch.cat([torch.tensor([0]).cuda(), unique_label])
                            counts = torch.cat([torch.tensor([0]).cuda(), counts])
                        if unique_label.shape[0] > self.render_label_nums:
                            unique_label = unique_label[:self.render_label_nums]
                            counts = counts[:self.render_label_nums]
                            print("Too many to show")
                    else:
                        unique_label = torch.cat([torch.tensor([0]).cuda(), unique_label])
                        counts = torch.cat([torch.tensor([0]).cuda(), counts])
                    # Mapping tensor
                    self.label_2_render_label.zero_()
                    self.label_2_render_label[unique_label.long()] = torch.arange(unique_label.shape[0]).cuda().int()

                self.label_map.zero_()
                self.mapping_gs_record.zero_()
                self.mapping_gs_label_record.zero_()
                with torch.no_grad():
                    self.render_3_sem(rendered_cam, self.xyz, self.opacity, self.scales, self.rotation, self.colors_precomp, self.bg, self.label_map, self.mapping_gs_record, self.mapping_gs_label_record, self.label_2_render_label)
                    label_map1 = torch.argmax(self.label_map, dim=0)
                    label_map = unique_label[label_map1]

                    label_map = label_map.cpu().detach().numpy()

                if self.view_mode == "label":
                    im = self.label_2_feature[label_map.astype(np.int32)]
                    im = im.permute(2, 0, 1)    # (W,H,C) -> (C,W,H)

                if self.current_label is not None and self.view_mode == "color":
                    if self.object_view_mode==False:
                        img_mask = label_map == self.current_label
                        im[:, img_mask] += 40.0/255.0
                        im[im>1] = 1

                if self.object_view_mode:
                    # Show single object only
                    if self.with_sem_labels:
                        if self.current_label is not None:
                            img_mask = label_map == self.current_label
                            im[:, img_mask==False] = 1

            pts, cols = self.rgbd2pcd(im, depth, w2c, k)

            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if self.capture:
                # Save image
                timestamps = time.time()
                output_name = f"{self.output_path}/{timestamps}.png"
                cv2.imwrite(output_name, cv2.cvtColor(im.cpu().detach().numpy().transpose(1,2,0)*255, cv2.COLOR_BGR2RGB))
                print(f"Saving image to {output_name}")
                self.capture = False

            if not vis.poll_events():
                break
            vis.update_renderer()

            # Print mouse position
            mouse_x = self.mouse_pos[0] - self.x_off
            mouse_y = self.mouse_pos[1] - self.y_off
            if (mouse_x != last_mouse_x or mouse_y != last_mouse_y) and self.with_sem_labels:
                print(f"Mouse moved to position: x={mouse_x}, y={mouse_y}")
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y

                if self.object_view_mode and self.current_label is not None:
                    continue

                # Check if within window bounds
                if mouse_x >= 0 and mouse_x < (self.W * self.view_scale) and mouse_y >= 0 and mouse_y < (self.H * self.view_scale):
                    # Calculate corresponding 3D point
                    edit_mode = True
                    scales_2_img = self.view_scale
                    x = int(mouse_x / scales_2_img)
                    y = int(mouse_y / scales_2_img)

                    # Get label with most occurrences in a 20x20 patch
                    patch_size = 20
                    if x-patch_size//2 < 0:
                        x = patch_size//2
                    if x+patch_size//2 >= self.W:
                        x = self.W - patch_size//2 - 1
                    if y-patch_size//2 < 0:
                        y = patch_size//2
                    if y+patch_size//2 >= self.H:
                        y = self.H - patch_size//2 - 1
                    patch_label = label_map[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
                    unique_label, counts = np.unique(patch_label, return_counts=True)
                    if unique_label[0] == 0:
                        unique_label = unique_label[1:]
                        counts = counts[1:]
                    if torch.sum(torch.tensor(counts)) == 0:
                        edit_mode = False
                        continue
                    print(f"Unique label: {unique_label}, counts: {counts}")
                    label = unique_label[np.argmax(counts)]
                    print(f"Choose label: {label}")

                    # Output label's xyz range
                    edit_label_mask = self.sem_labels == label
                    edit_label_xyz = self.xyz[edit_label_mask]
                    print(f"Label {label} has {edit_label_xyz.shape[0]} points")

                    self.current_label = label
                    self.xyz_trans = torch.zeros(3).cuda()

        vis.destroy_window()
        
                

    #----------------- Interactive Functions -----------------

    def rotate_around_axis(self, vis, axis, angle_degrees):
        """
        Rotate selected object around specified axis.
        
        Args:
            vis: Open3D visualizer instance
            axis: Rotation axis (0=X, 1=Y, 2=Z)
            angle_degrees: Rotation angle in degrees
        """
        gs_label_mask = self.sem_labels == self.current_label
        center = torch.mean(self.xyz[gs_label_mask], dim=0)
        print(f"Rotating around {axis}-axis by {angle_degrees} degrees")
        
        # Calculate quaternion
        angle_rad = torch.deg2rad(torch.tensor(angle_degrees / 2)).cuda().float()
        rot_r = torch.zeros(4, device="cuda").float()
        rot_r[3] = torch.cos(angle_rad)
        sin_val = torch.sin(angle_rad)
        rot_r[axis] = sin_val
        
        # Build rotation matrix
        rot_m = build_rotation(rot_r.view(1, 4)).squeeze(0)
        
        with torch.no_grad():
            self.xyz[gs_label_mask] = torch.matmul(rot_m, (self.xyz[gs_label_mask] - center).T).T + center
            rotation = self.rotation[gs_label_mask]
            cur_rot = unitquat_to_rotmat(rotation)
            new_rot = torch.matmul(rot_m.unsqueeze(0), cur_rot)
            new_quat = rotmat_to_unitquat(new_rot)
            self.rotation[gs_label_mask] = new_quat
        return True

    def translate_along_axis(self, vis, axis, delta):
        """
        Translate selected object along specified axis.
        
        Args:
            vis: Open3D visualizer instance
            axis: Translation axis (0=X, 1=Y, 2=Z)
            delta: Translation distance
        """
        self.xyz_trans.zero_()
        self.xyz_trans[axis] += delta
        print(f"Translating {['x', 'y', 'z'][axis]}-axis by {delta}")
        with torch.no_grad():
            if self.current_label is not None:
                gs_label_mask = self.sem_labels == self.current_label
                self.xyz[gs_label_mask] += self.xyz_trans
                center = torch.mean(self.xyz[gs_label_mask], dim=0)
                print(f"Current Center: {center}")
        return True

    # Keyboard interaction handlers
    def on_T_key_press(self, vis):
        """Toggle between color and label display modes."""
        if self.view_mode == "color":
            print("Switching to label display mode")
            self.view_mode = "label"
        else:
            print("Switching to color display mode")
            self.view_mode = "color"
        return True

    def on_O_key_press(self, vis):
        """Output current camera view matrix."""
        print(self.w2c)
        return True

    def on_J_key_press(self, vis):
        """Toggle between showing all objects or single object."""
        if self.object_view_mode:
            print("Showing all objects")
            self.object_view_mode = False
        else:
            print("Showing single object")
            self.object_view_mode = True
        return True

    def on_K_key_press(self, vis):
        """Capture current view."""
        print("Capturing current view")
        self.capture = True
        return True

    def on_M_key_press(self, vis):
        """Switch to next mapping camera view."""
        self.mapping_cam_index += 1
        if self.mapping_cam_index >= len(self.R_list):
            self.mapping_cam_index = 0
        R = self.R_list[self.mapping_cam_index]
        t = self.t_list[self.mapping_cam_index]
        self.changed_w2c[:3, :3] = R
        self.changed_w2c[:3, 3] = t
        self.change_views = True
        print(f"Switched to camera {self.mapping_cam_index}")
        return True

    def on_L_key_press(self, vis):
        """Increase scale of all Gaussians."""
        print("Increasing Gaussian scales")
        with torch.no_grad():
            self.scales += 0.001
            self.scales[self.scales > 1] = 1
        return True

    def on_P_key_press(self, vis):
        """Downsample Gaussians using voxel grid."""
        print("Downsampling Gaussians")
        with torch.no_grad():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.xyz.cpu().detach().numpy())
            voxel_size = 0.03
            downpcd = pcd.voxel_down_sample(voxel_size)
            down_pts = np.asarray(downpcd.points)
            tree = cKDTree(self.xyz.cpu().detach().numpy())
            _, idx = tree.query(down_pts)
            idx = torch.tensor(idx).cuda()
            self.xyz = self.xyz[idx]
            self.opacity = self.opacity[idx]
            self.scales = self.scales[idx]
            self.rotation = self.rotation[idx]
            self.shs = self.shs[idx]
            if self.with_sem_labels:
                self.sem_labels = self.sem_labels[idx]
                self.colors_precomp = self.colors_precomp[idx]
        return True

    # Translation key handlers
    def on_A_key_press(self, vis): return self.translate_along_axis(vis, axis=0, delta=0.01)
    def on_S_key_press(self, vis): return self.translate_along_axis(vis, axis=1, delta=0.01)
    def on_D_key_press(self, vis): return self.translate_along_axis(vis, axis=2, delta=0.01)
    def on_Z_key_press(self, vis): return self.translate_along_axis(vis, axis=0, delta=-0.01)
    def on_X_key_press(self, vis): return self.translate_along_axis(vis, axis=1, delta=-0.01)
    def on_C_key_press(self, vis): return self.translate_along_axis(vis, axis=2, delta=-0.01)

    # Rotation key handlers
    def on_F_key_press(self, vis): return self.rotate_around_axis(vis, axis=0, angle_degrees=1)
    def on_G_key_press(self, vis): return self.rotate_around_axis(vis, axis=1, angle_degrees=1)
    def on_H_key_press(self, vis): return self.rotate_around_axis(vis, axis=2, angle_degrees=1)
    def on_V_key_press(self, vis): return self.rotate_around_axis(vis, axis=0, angle_degrees=-1)
    def on_B_key_press(self, vis): return self.rotate_around_axis(vis, axis=1, angle_degrees=-1)
    def on_N_key_press(self, vis): return self.rotate_around_axis(vis, axis=2, angle_degrees=-1)

    
    #----------------- Mouse Interaction Functions -----------------
    def on_mouse_click(self, vis, button, action, mods):
        print(f"Mouse clicked: button={button}, action={action}, mods={mods}")
        return True  # Return False as there's no need to update geometry here
    
    def on_mouse_move(self, vis, x, y):
        self.mouse_x = x
        self.mouse_y = y
        print(f"Mouse moved to position: x={self.mouse_x}, y={self.mouse_y}")
        return True

    #----------------- Render Functions -----------------
    def rgbd2pcd(self, color, depth, w2c, intrinsics):
        """
        Convert RGBD image to point cloud.
        
        Args:
            color: RGB image tensor
            depth: Depth image tensor
            w2c: World to camera transform matrix
            intrinsics: Camera intrinsic parameters
            
        Returns:
            tuple: (points, colors) as Open3D Vector3dVector objects
        """
        width, height = color.shape[2], color.shape[1]
        CX = intrinsics[0][2]
        CY = intrinsics[1][2]
        FX = intrinsics[0][0]
        FY = intrinsics[1][1]

        # Compute pixel coordinates
        xx = torch.tile(torch.arange(width).cuda(), (height,))
        yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
        xx = (xx - CX) / FX
        yy = (yy - CY) / FY
        z_depth = depth[0].reshape(-1)

        # Transform to world coordinates
        pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(torch.tensor(w2c).cuda().float())
        pts = (c2w @ pts4.T).T[:, :3]

        # Convert to Open3D format
        pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
        return pts, cols
    
    def fov2focal(self, fov, pixels):
        """Convert field of view to focal length."""
        return pixels / (2 * math.tan(fov / 2))

    def focal2fov(self, focal, pixels):
        """Convert focal length to field of view."""
        return 2*math.atan(pixels/(2*focal))
    
    def render(self, viewpoint_camera, xyz: torch.Tensor, opicity: torch.Tensor, scales:torch.Tensor, 
            rotations:torch.Tensor, shs:torch.Tensor, bg_color: torch.Tensor, scaling_modifier = 1.0):
        """
        Render the scene using Gaussian Splatting.
        
        Args:
            viewpoint_camera: Camera parameters for rendering
            xyz: 3D positions of Gaussians
            opicity: Opacity values for each Gaussian
            scales: Scale values for each Gaussian
            rotations: Rotation values for each Gaussian
            shs: Spherical harmonics coefficients
            bg_color: Background color tensor (must be on GPU)
            scaling_modifier: Scale modifier for rendering
            
        Returns:
            tuple: (rendered_image, depth_image, visibility_filter)
        """
        # Initialize screen space points
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=False, device="cuda")

        # Setup rasterization parameters
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        active_sh_degree = 0
        
        # Configure rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Perform rasterization
        depth_image, rendered_image, silhouette, radii, is_used = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = shs,
            colors_precomp = None,
            opacities = opicity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
        
        visibility_filter = radii > 0
        return rendered_image, depth_image, visibility_filter

    def render_3_sem(self, viewpoint_camera, xyz: torch.Tensor, opicity: torch.Tensor, scales:torch.Tensor, 
            rotations:torch.Tensor, colors_precomp:torch.Tensor, bg_color: torch.Tensor, label_map: torch.Tensor,
            mapping_gs_record: torch.Tensor, mapping_gs_label_record: torch.Tensor, label_2_render_label: torch.Tensor, 
            scaling_modifier = 1.0):
        """
        Render the scene with semantic labels.
        
        Args:
            viewpoint_camera: Camera parameters for rendering
            xyz: 3D positions of Gaussians
            opicity: Opacity values for each Gaussian
            scales: Scale values for each Gaussian
            rotations: Rotation values for each Gaussian
            colors_precomp: Pre-computed colors for semantic rendering
            bg_color: Background color tensor (must be on GPU)
            label_map: Tensor for storing label mapping
            mapping_gs_record: Tensor for recording Gaussian mapping
            mapping_gs_label_record: Tensor for recording label mapping
            label_2_render_label: Mapping from labels to render labels
            scaling_modifier: Scale modifier for rendering
        """
        # Initialize screen space points
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=False, device="cuda")
        
        # Setup rasterization parameters
        tanfovx = math.tan(float(viewpoint_camera.FoVx[0]) * 0.5)
        tanfovy = math.tan(float(viewpoint_camera.FoVy[0]) * 0.5)
        active_sh_degree = 0
        
        # Configure rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Perform semantic rasterization
        rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opicity,
            scales = scales,
            rotations = rotations,
            label_map = label_map,
            mapping_gs_record = mapping_gs_record,
            mapping_gs_label_record = mapping_gs_label_record,
            label_2_render_label = label_2_render_label,
            render_labels = True,
            cov3D_precomp = None)
        
        return None
    

# ----------------- Main Entry Point -----------------
if __name__ == "__main__":
    """
    Main entry point for the visualization tool.
    Sets up command line arguments and initializes the visualizer.
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Visualization Tool")
    # parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--config", type=str, help="Path to the camera parameters")
    parser.add_argument("--scene_npz", type=str, help="Path to the scene npz file")
    parser.add_argument("--dataset_type", type=str, help="Type of the dataset")
    parser.add_argument("--view_scale", type=float, default=1.0, help="Scale of the view")
    args = parser.parse_args()
    
    # Set default parameters for testing
    args.dataset_type = "replica"
    args.config = "./configs/Replica/caminfo.txt"

    # Configure visualization parameters
    args.view_scale = 2.0
    args.label_nums = 10000
    args.render_label_nums = 300
    args.output_path = os.path.join(os.path.dirname(args.scene_npz), "vis")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # Initialize mouse position sharing between processes
    manager = mp.Manager()
    mouse_pos = manager.list([0, 0])
    
    # Start mouse listener in separate process
    listener_process = mp.Process(target=start_listener, args=(mouse_pos,))
    listener_process.start()
    
    # Initialize and run visualizer
    vis = MyVisualizer(args, mouse_pos)
    
    # Wait for mouse listener to finish
    listener_process.join()
    
