import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cv2
import onnxruntime
import numpy as np
import os

from torchvision.transforms import functional as F


class PoseDetectorOnnx():
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.ort_sess = onnxruntime.InferenceSession(self.onnx_path)
        self.input_name = self.ort_sess.get_inputs()[0].name
        self.dst_w = 192
        self.dst_h = 256
        self.input_size = [self.dst_w, self.dst_h]
        self.img_rgb = None
        self.img_p = None
        self.img_t = None
        self.img_s = None
        self.p_boxes = []
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.preds = []
        self.maxvals = []

    @staticmethod
    def box2cs(input_size, box):
        x, y, w, h = box[:4]
        aspect_ratio = input_size[0] / input_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        # keep the ratio box_w/box_h = input_size[0] / input_size[1], and the short size will be scale large
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)  # divide 200 for what? change to float?
        scale = scale * 1.25  # scale w,h
        return center, scale

    @staticmethod
    def get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):
        assert len(center) == 2
        assert len(scale) == 2
        assert len(output_size) == 2
        assert len(shift) == 2

        def _rotate_point(pt, angle_rad):
            assert len(pt) == 2
            sn, cs = np.sin(angle_rad), np.cos(angle_rad)
            new_x = pt[0] * cs - pt[1] * sn
            new_y = pt[0] * sn + pt[1] * cs
            rotated_pt = [new_x, new_y]
            return rotated_pt

        def _get_3rd_point(a, b):
            assert len(a) == 2
            assert len(b) == 2
            direction = a - b
            third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
            return third_pt

        # pixel_std is 200.
        scale_tmp = scale * 200.0

        shift = np.array(shift)
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = _rotate_point([0., src_w * -0.5], rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    @staticmethod
    def affine_transform(pt, trans_mat):
        assert len(pt) == 2
        new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])
        return new_pt

    @staticmethod
    def keypoints_from_heatmaps(heatmaps, center, scale, post_process=True, unbiased=False, kernel=11):
        def _get_max_preds(heatmaps):
            assert isinstance(heatmaps, np.ndarray), ('heatmaps should be numpy.ndarray')
            assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

            N, K, _, W = heatmaps.shape
            heatmaps_reshaped = heatmaps.reshape((N, K, -1))
            idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
            maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

            preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
            preds[:, :, 0] = preds[:, :, 0] % W
            preds[:, :, 1] = preds[:, :, 1] // W

            preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
            return preds, maxvals

        def _transform_preds(coords, center, scale, output_size):
            assert coords.shape[1] in (2, 4, 5)
            assert len(center) == 2
            assert len(scale) == 2
            assert len(output_size) == 2

            target_coords = coords.copy()
            trans = PoseDetectorOnnx.get_affine_transform(center, scale, 0, output_size, inv=True)
            for p in range(coords.shape[0]):
                target_coords[p, 0:2] = PoseDetectorOnnx.affine_transform(coords[p, 0:2], trans)
            return target_coords

        preds, maxvals = _get_max_preds(heatmaps)
        N, K, H, W = heatmaps.shape

        if post_process:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25

        # Transform back to the image
        for i in range(N):
            preds[i] = _transform_preds(preds[i], center[i], scale[i], [W, H])

        return preds, maxvals

    def get_pose_result(self, cv_img, p_boxes, bool_vis):
        self.preds = []
        self.maxvals = []
        self.img_rgb = cv_img[:, :, ::-1]
        self.p_boxes = p_boxes

        for p_box in p_boxes:
            # TopDownAffine
            center, scale = PoseDetectorOnnx.box2cs(self.input_size, p_box)  # keep ratio w/h and scale
            rotate = 0
            trans = PoseDetectorOnnx.get_affine_transform(center, scale, rotate, self.input_size)  # get three pair points
            self.img_p = cv2.warpAffine(self.img_rgb, trans, (int(self.input_size[0]), \
                                                          int(self.input_size[1])), flags=cv2.INTER_LINEAR)

            self.img_t = F.to_tensor(self.img_p)  # ToTensor
            self.img_t = F.normalize(self.img_t, mean=self.mean, std=self.std)  # NormalizeTensor
            self.img_t = self.img_t.unsqueeze(0)

            self.img_t = self.img_t.numpy()

            output = self.ort_sess.run(None, {self.input_name: self.img_t})
            print(output[0].shape)

            preds, maxvals = PoseDetectorOnnx.keypoints_from_heatmaps(output[0], [center],\
                                                                  [scale], post_process=True, unbiased=False)
            self.preds.append(preds)
            self.maxvals.append(maxvals)

        if bool_vis:
            self.vis_pose()

        return self.preds, self.maxvals, self.img_s

    def vis_pose(self):
        # show kp
        self.img_s = self.img_rgb[:, :, ::-1].astype(np.uint8)
        for pred_batches in self.preds:
            for pred_batch in pred_batches:
                for k_p in pred_batch:
                    cv2.circle(self.img_s, (int(k_p[0]), int(k_p[1])), 5, (0, 255, 0), -1)

        # show p-box
        for p_box in self.p_boxes:
            x, y, w, h = [int(x) for x in p_box[:4]]
            cv2.rectangle(self.img_s, (x, y), (x + w, y + h), (255, 0, 255))


if __name__ == "__main__":
    logger.info("Start Proc...")
    onnx_path = "/home/liyongjing/Egolee/scripts/models/mmpose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.onnx.sim"
    pose_detector_onnx = PoseDetectorOnnx(onnx_path)

    # Start pose detector
    img_path = "/home/liyongjing/Egolee/programs/mmpose-master/tests/data/coco/000000000785.jpg"
    person_boxes = [[280.79, 44.73, 218.7, 346.68]]
    img = cv2.imread(img_path)
    preds, maxvals, img = pose_detector_onnx.get_pose_result(img, person_boxes, True)

    cv2.namedWindow("img", 0)
    cv2.imshow("img", img)
    wait_key = cv2.waitKey(0)
    if wait_key == 0:
        exit(1) 
