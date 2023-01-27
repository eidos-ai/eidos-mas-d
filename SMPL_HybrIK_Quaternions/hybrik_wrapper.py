import os
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from easydict import EasyDict as edict
import math

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.bbox import _box_to_center_scale, _center_scale_to_box
from hybrik.utils.transforms import get_affine_transform, im_to_torch
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box, get_max_iou_box
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
import pytorch3d
import pytorch3d.renderer
from pytorch3d.transforms.rotation_conversions import quaternion_multiply, quaternion_apply, standardize_quaternion
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from hybrik.models.layers.smpl.SMPL import SMPL_layer
from hybrik.models.layers.smpl.lbs import quat_to_rotmat

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def quats2rotmat(quats):
    ''' convert quaternions rotation matrixes

        Parameters:
        ----------
        quats: torch.Tensor Bx24x4

        Returns
        -------
        rot_mats: torch.tensor Bx24x3x3
            24 rotation matrixes for every frame 
    '''

    batch_size = quats.shape[0]

    rot_mats = quat_to_rotmat(quats.reshape(batch_size * 24, 4)).reshape(batch_size, 24, 3, 3)
    return rot_mats

# creates a camera intrinsecal matrix tensor
def create_Mint(fx, fy, Ox, Oy):
    assert fx.shape[0] == fy.shape[0] == Ox.shape[0] == Oy.shape[0], "Input intrinsecal matrix params dim 0 doesn't match!"
    
    Mint = torch.zeros((fx.shape[0], 3, 4)).to(Ox.device)
    Mint[:, 0, 0] = fx
    Mint[:, 1, 1] = fy
    Mint[:, 2, 2] = 1
    Mint[:, 0, 2] = Ox
    Mint[:, 1, 2] = Oy
    
    return Mint

# creates a camera extrinsecal matrix tensor from camera translations tensor and 
# camera rotations (in axis angle format) tensor
def create_Mext(cam_t, cam_rt):
    assert cam_t.shape[0] == cam_rt.shape[0], "camera translations and camera rotations dim 0 doesn't match!"
    Mext = torch.zeros((cam_t.shape[0], 4, 4)).to(cam_t.device)
    cam_rt = axis_angle_to_matrix(cam_rt)
    
    Mext[:, :3, :3] = cam_rt
    Mext[:, :3, 3] = cam_t[:, 0, :]
    Mext[:, 3, 3] = 1
    
    return Mext

# creates projection matrix from extrinsecal and intrinsecal matrix
def create_P(Mext, Mint):
    assert Mext.device == Mint.device, "Matrixes not in same device"
    P = torch.matmul(Mint, Mext)
    return P

def project_kps(P, kps):
    ones_to_append = torch.ones((kps.shape[0], kps.shape[1], 1)).to(kps.device)
    kps_homo = torch.cat((kps, ones_to_append), dim=2)
    points = torch.einsum('bij,bkj->bki', P, kps_homo)
    points = torch.div(points, points[:, :, 2:3])
    points = points[:,:,0:2]
    return points

def world2cam(Mext, kps):
    ones_to_append = torch.ones((kps.shape[0], kps.shape[1], 1)).to(kps.device)
    kps_homo = torch.cat((kps, ones_to_append), dim=2)
    points = torch.einsum('bij,bkj->bki', Mext, kps_homo)
    points = points[:,:,0:3]
    return points

class YoloV5():
    def __init__(self, model_name='yolov5l6'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        # detect only persons
        self.model.classes = [0]

    def __call__(self, rgb_frames):
        """
        INPUT
        rgb_frames: list of RGB OpenCV images

        OUTPUT
        output: ---> list of dictionaries with keys:
                                            'boxes': Tensor (n_detections, 4)
                                            'labels': Tensor (n_detections)
                                            'scores': Tensor (n_detections)
        """
        output = []
        results = self.model(rgb_frames).tolist()
        for frame_results in results:
            boxes = frame_results.pred[0][:, 0:4]
            labels = frame_results.pred[0][:, 5]
            scores = frame_results.pred[0][:, 4]

            frame_output = {"boxes": boxes,
                            "labels": labels,
                            "scores": scores}
            output.append(frame_output)

        return output

    def cuda(self, device):
        self.model.cuda(device)

    def eval(self):
        self.model.eval()

class HybrIK():
    def __init__(self, inference=True, yolo_model="yolov5l6"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.yolo_model = yolo_model
        # load models and transformations needed for inference and forward kinematics
        if inference:
            self._load_models()
            self._load_transformations()
        else:
            # if inference is False load only smpl layer needed for forward kinematics
            self._load_smpl()


        self.prev_box = None

    def _load_smpl(self):
        class Placeholder():
            def __init__(self):
                self.smpl = None

        src_dir = os.path.dirname(os.path.abspath(__file__))
        hybrik_dir = os.path.join(src_dir, "HybrIK")
        
        h36m_jregressor_path = os.path.join(hybrik_dir, "model_files", "J_regressor_h36m.npy")
        h36m_jregressor = np.load(h36m_jregressor_path)

        lbs_pkl_path = os.path.join(hybrik_dir, "model_files", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")

        self.model = Placeholder()
        self.model.smpl = SMPL_layer(
                                    lbs_pkl_path,
                                    h36m_jregressor=h36m_jregressor,
                                    dtype=torch.float32
                                    )
        self.model.smpl.cuda(self.device)

    def _load_models(self):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        hybrik_dir = os.path.join(src_dir, "HybrIK")

        cfg_file = os.path.join(hybrik_dir, "configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml")
        trained_backbone_path = os.path.join(src_dir, "pretrained_hrnet.pth")

        # load cfg
        self.cfg = update_config(cfg_file)

        #self.det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.det_model = YoloV5(self.yolo_model)

        # build HybrIK model
        os.chdir(hybrik_dir)
        self.model = builder.build_sppe(self.cfg.MODEL)

        # load backbone weights
        print(f'Loading model from {trained_backbone_path}...')
        os.chdir(src_dir)
        self.model.load_state_dict(torch.load(trained_backbone_path, map_location='cpu'))

        if torch.cuda.is_available() and self.device != "cpu":
            self.det_model.cuda(self.device)
            self.model.cuda(self.device)
        self.det_model.eval()
        self.model.eval()

    def _load_transformations(self):
        bbox_3d_shape = getattr(self.cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        dummpy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
                    })
        self.det_transform = T.Compose([T.ToTensor()])

        self.transformation = SimpleTransform3DSMPLCam(
                                            dummpy_set, scale_factor=self.cfg.DATASET.SCALE_FACTOR,
                                            color_factor=self.cfg.DATASET.COLOR_FACTOR,
                                            occlusion=self.cfg.DATASET.OCCLUSION,
                                            input_size=self.cfg.MODEL.IMAGE_SIZE,
                                            output_size=self.cfg.MODEL.HEATMAP_SIZE,
                                            depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
                                            bbox_3d_shape=bbox_3d_shape,
                                            rot=self.cfg.DATASET.ROT_FACTOR, sigma=self.cfg.MODEL.EXTRA.SIGMA,
                                            train=False, add_dpg=False,
                                            loss_type=self.cfg.LOSS['TYPE'])


    def detect_person(self, bgr_frame):
        # Convert frame from bgr to rgb
        rgb_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # transform input from (H, W, 3) numpy array to normalized (3, H, W) tensor
        #det_input = self.det_transform(rgb_image).to(self.device)
        det_input = rgb_image

        # Run object Detection model to get bboxes
        det_output = self.det_model([det_input])[0]

        """
        det_output ---> dictionary with keys:
                                            'boxes': Tensor (n_detections, 4)
                                            'labels': Tensor (n_detections)
                                            'scores': Tensor (n_detections)
        """

        # process detections and return a single bbox
        if self.prev_box is None:
            tight_bbox = get_one_box(det_output)  # xyxy
            if tight_bbox is None:
                return None
        else:
            tight_bbox = get_max_iou_box(det_output, self.prev_box)  # xyxy

        self.prev_box = tight_bbox

        return tight_bbox, rgb_image

    def transform_bbox(self, bbox):
        # scale bbox and make it square
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio=self.transformation._aspect_ratio, 
            scale_mult=self.transformation._scale_mult)
        scale = scale * 1.0

        bbox_xyxy = _center_scale_to_box(center, scale)

        return bbox_xyxy, center, scale

    def transform_frame(self, frame, center, scale):
        # transform det_input into (3, 256, 256) cropped tensor (see assets/HRNet_input.jpg)

        input_size = self.transformation._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(frame, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        img_center = np.array([float(frame.shape[1]) * 0.5, float(frame.shape[0]) * 0.5])
        img = img.to(self.device)[None, :, :, :]

        return img, img_center

    def process_pose_output(self, pose_output, bbox_xywh):
        transl = pose_output.transl
        transl[:, 2] = transl[:, 2] * 256 / bbox_xywh[2]
        
        joints_orients = pose_output.pred_theta_mats[:, 4:]
        root_orient = pose_output.pred_theta_mats[:, 0:4]
        
        # quaternions format is w x y z
        output = {"joints_orients": joints_orients,
                  "root_orient": root_orient,
                  "betas": pose_output.pred_delta_shape,
                  "verts": pose_output.pred_vertices,
                  "transl": transl.unsqueeze(1),
                  "bbox": bbox_xywh,
                  "pose_output": pose_output}
        
        return output
    
    def process_frame(self, bgr_frame):
        # Run object detection and bbox preprocesing
        bbox, rgb_frame = self.detect_person(bgr_frame)

        # further transform bbox by scaling it x1.25 (ensures the person is entirely within)
        # and matching aspect ratio for HRNet input
        bbox_xyxy, bbox_center, bbox_wh = self.transform_bbox(bbox)
        bbox_xywh = [bbox_center[0], bbox_center[1], bbox_wh[0], bbox_wh[1]]

        # Transform input frame and scale bbox
        pose_input, img_center = self.transform_frame(rgb_frame, bbox_center, bbox_wh)

        # Run HRNet + HybrIK
        pose_output = self.model(
            pose_input, flip_test=False,
            bboxes=torch.from_numpy(np.array(bbox_xyxy)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float())

        return self.process_pose_output(pose_output, bbox_xywh)
        #return pose_output, bbox, tight_bbox, img_center
    
    
    def smpl_forward(self, quats, beta, root_orient=None, transl=None):
        """
        quats: Tensor (num_frames, num_joints, 4)
        beta: Tensor (10)
        root_orient: Tensor (num_frames, 1, 4)
        transl: Tensor (num_frames, 1, 4)
        """

        batch_size = quats.shape[0]

        beta_input = beta.repeat(batch_size, 1)
        
        if root_orient is None:
            root_orient_input = None
            quats_input = quats.reshape(batch_size, 96)
        else:
            root_orient_input = root_orient.reshape(batch_size, 4)
            quats_input = quats.reshape(batch_size, 92)

        kps_output = torch.zeros(batch_size, 29, 3, device=quats.device, dtype=quats.dtype)

        smpl_output = self.model.smpl(quats_input, beta_input, global_orient=root_orient_input, transl=transl)

        kps_output[:, :24] = smpl_output.joints
        leaf_number = [411, 2445, 5905, 3216, 6617]
        leaf_vertices = smpl_output.vertices[:, leaf_number].clone()
        kps_output[:, 24:] = leaf_vertices
        
        return smpl_output.vertices, kps_output
    
    def render_vertices(self, verts, transl, focal, bkg_img):
        device=verts.device
        height, width, _ = bkg_img.shape
        bs = verts.shape[0]
        faces = torch.from_numpy(self.model.smpl.faces.astype(np.int32))

        # add the translation
        vertices = verts

        # upside down the mesh
        # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
        rot_z = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        rot_z = torch.from_numpy(rot_z).to(device).expand(bs, 3, 3)
        rot_x = Rotation.from_euler('x', 180, degrees=True).as_matrix().astype(np.float32)
        rot_x = torch.from_numpy(rot_x).to(device).expand(bs, 3, 3)
        faces = faces.expand(bs, *faces.shape).to(device)

        vertices = torch.matmul(rot_x, vertices.transpose(1, 2)).transpose(1, 2)
        vertices += transl
        vertices = torch.matmul(rot_z, vertices.transpose(1, 2)).transpose(1, 2)

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
        textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
        mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

        # Initialize a camera.
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=((2 * focal / min(height, width), 2 * focal / min(height, width)),),
            device=device,
        )

        # Define the settings for rasterization and shading.
        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=(height, width),   # (H, W)
            # image_size=height,   # (H, W)
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Define the material
        materials = pytorch3d.renderer.Materials(
            ambient_color=((1, 1, 1),),
            diffuse_color=((1, 1, 1),),
            specular_color=((1, 1, 1),),
            shininess=64,
            device=device
        )

        # Place a directional light in front of the object.
        lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

        # Create a phong renderer by composing a rasterizer and a shader.
        renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=pytorch3d.renderer.SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                materials=materials
            )
        )

        # Do rendering
        color_batch = renderer(mesh)

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        alpha = 0.9
        image_vis = alpha * color[:, :, :3] * valid_mask + (
            1 - alpha) * bkg_img * valid_mask + (1 - valid_mask) * bkg_img

        image_vis = image_vis.astype(np.uint8)

        return image_vis
    
    def project_kps(self, kps, princpts, t=None, cam_rt=None, cam_focal=None):
        bs = kps.shape[0]
        
        # create extrinsecal matrix of camera
        if cam_rt is None: 
            cam_rt = torch.zeros((bs, 3)).to(self.device)
            cam_rt[:, 0] = math.pi
        if t is None: t = torch.zeros((bs, 1, 3)).to(self.device)
        Mext = create_Mext(t, cam_rt)
        # create intrinsecal matrix of camera for every video frame
        if cam_focal is None: cam_focal = (torch.ones((bs, 2))*1000).to(self.device)
        Mint = create_Mint(cam_focal[:, 0], cam_focal[:, 1],
                        princpts[:,0], 
                        princpts[:, 1])

        # create projection matrix from extrinsecal and intrinsecal matrix
        P = create_P(Mext, Mint)

        # use projection matrix to project the 3D keypoints to the frame
        projected_kps = project_kps(P, kps)
        
        return projected_kps
    
    def world2cam(self, kps, t=None, cam_rt=None):
        bs = kps.shape[0]
        
        # create extrinsecal matrix of camera
        if cam_rt is None: 
            cam_rt = torch.zeros((bs, 3)).to(self.device)
            cam_rt[:, 0] = math.pi
        if t is None: t = torch.zeros((bs, 1, 3)).to(self.device)
        Mext = create_Mext(t, cam_rt)
        
        return world2cam(Mext, kps)
    
    def render_kps(self, projected_kps, bkg_img, color=(255, 255, 0), render_idx=True):
        output_img = bkg_img.copy()
        kps_i = 0
        for kp in projected_kps[0].cpu():
            cv2.circle(output_img, tuple(np.array([int(kp[0]), int(kp[1])])), 4, color, -1)
            if render_idx:
                cv2.putText(img=output_img, text=str(kps_i), org=(int(kp[0]), int(kp[1]+4)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=color, lineType=cv2.LINE_AA, thickness=1)
            kps_i += 1
        return output_img


def hybrik_render_vertices(vertices, transl, bbox_xywh, smpl_faces, image_bgr):
    """render vertices in image using author's process
    """
    focal = 1000.0

    focal = focal / 256 * bbox_xywh[2]

    verts_batch = vertices
    transl_batch = transl.squeeze(1)
    smpl_faces = torch.from_numpy(smpl_faces.astype(np.int32))

    color_batch = render_mesh(
        vertices=verts_batch, faces=smpl_faces,
        translation=transl_batch,
        focal_length=focal, height=image_bgr.shape[0], width=image_bgr.shape[1])

    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()

    color = image_vis_batch[0]
    valid_mask = valid_mask_batch[0].cpu().numpy()
    alpha = 0.9
    image_vis = alpha * color[:, :, :3] * valid_mask + (
        1 - alpha) * image_bgr * valid_mask + (1 - valid_mask) * image_bgr

    image_vis = image_vis.astype(np.uint8)

    return image_vis

def hybrik_render_kps(uvd_jts, bbox_xywh, image_bgr):
    """render keypoints in image using author's process
    """
    image = image_bgr.copy()
    uv_29 = uvd_jts.reshape(29, 3)[:, :2]

    pts = uv_29 * bbox_xywh[2]
    pts[:, 0] = pts[:, 0] + bbox_xywh[0]
    pts[:, 1] = pts[:, 1] + bbox_xywh[1]

    for pt in pts:
        x, y = pt
        cv2.circle(image, (int(x), int(y)), 3, (255, 136, 132), 3)
    
    return image

def rotate_quat_x_pi(quats: torch.Tensor) -> torch.Tensor:
    rot_quat = torch.zeros_like(quats)
    rot_quat[:, :, 1] = 1

    return quaternion_multiply(rot_quat, quats)

def rotate_points_x_pi(points: torch.Tensor) -> torch.Tensor:
    num_frames, num_points, _ = points.shape
    rot_quat = torch.zeros((num_frames, num_points, 4)).to(points.device)
    rot_quat[:, :, 1] = 1

    return quaternion_apply(rot_quat, points)



if __name__ == "__main__":
    # read image used for testing
    test_img_path = "sample_pose.png"
    test_img = cv2.imread(test_img_path)
    input_h, input_w, _ = test_img.shape

    hybrik_wrapper = HybrIK()
    hybrik_wrapper_smpl = HybrIK(inference=False)

    # process frame with HybrIK
    pose_output = hybrik_wrapper.process_frame(test_img)
    joints_orients = pose_output["joints_orients"].detach()
    root_orient = pose_output["root_orient"].detach()
    transl = pose_output["transl"].detach()
    beta = pose_output["betas"].detach()
    vertices = pose_output["verts"].detach()
    uvd_jts = pose_output["pose_output"].pred_uvd_jts
    bbox_xywh = pose_output["bbox"]

    # render keypoints and vertices from hybrik output using author's render process and model's output
    output_original = np.zeros((input_h, input_w*2, 3), dtype=np.uint8)
    original_rendered_vertices = hybrik_render_vertices(vertices, transl, bbox_xywh, 
                                                        hybrik_wrapper.model.smpl.faces, test_img)
    original_rendered_kps = hybrik_render_kps(uvd_jts, bbox_xywh, test_img)
    output_original[:, 0:input_w, :] = original_rendered_vertices
    output_original[:, input_w:, :] = original_rendered_kps
    cv2.imwrite("test_original.jpg", output_original)

    # render keypoints and vertices from quaternions, transl and beta
    output_wrapper = np.zeros((input_h, input_w*2, 3), dtype=np.uint8)
    joints_orients = joints_orients.reshape(1, 23, 4)
    root_orient = root_orient.reshape(1, 1, 4)
    root_orient = rotate_quat_x_pi(standardize_quaternion(root_orient))
    wrapper_vertices, wrapper_kps = hybrik_wrapper_smpl.smpl_forward(joints_orients, beta, root_orient=root_orient)

    Oy, Ox, _ = (np.array(test_img.shape)/2).tolist()
    princpt = torch.Tensor([[Ox, Oy]]).to(hybrik_wrapper_smpl.device)
    wrapper_rendered_vertices = hybrik_wrapper_smpl.render_vertices(wrapper_vertices, transl, 1000, test_img)
    wrapper_2d_kps = hybrik_wrapper_smpl.project_kps(wrapper_kps, princpt, t=transl)
    wrapper_rendered_kps = hybrik_wrapper_smpl.render_kps(wrapper_2d_kps, test_img)
    output_wrapper[:, 0:input_w, :] = wrapper_rendered_vertices
    output_wrapper[:, input_w:, :] = wrapper_rendered_kps
    cv2.imwrite("test_wrapper.jpg", output_wrapper)