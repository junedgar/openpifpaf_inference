import torch
import cv2
import math
import decoder
import os.path as osp
import numpy as np
import torchvision
from models import create_pifpaf

normalize = torchvision.transforms.Normalize(  # pylint: disable=invalid-name
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

image_transform = torchvision.transforms.Compose([  # pylint: disable=invalid-name
        torchvision.transforms.ToTensor(),
        normalize,
    ])

line_color = [(180,119,31),(232, 199, 174),(14,127,255),(  120,187,  255),
              (120,187,255),(44,160, 44),(138, 223, 152),(40,39,214),
              (150,152,255),( 189,103, 148),(213, 176,197),(75,86,140),
              (148, 156,196),(194,119,227), (210,  182,247),(  127,127,  127),
              (199,199,  199),(34,189,188),(141,219, 219),(207,190,  23),
              (255,239,213)]

COCO_PERSON_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]]


class PosePifPaf(object):
    def __init__(self, basenet='resnet50', weights=None, score_th=0.2,
                 scale=0.5, use_gpu=True, stickwidth=3):
        """
        Parameters for init
            basenet: str, must in resnet50, shufflenetv2
            weights: the weigths path
            score_th: score threshold for keypoints
            scale:  resize scale
            force_gpu: Bool, if True use gpu
        """

        if basenet == 'resnet50' and weights is None:
            self.checkpoint = osp.join(osp.dirname(__file__), './weights/openpifpaf_resnet50block5.pth')
        elif basenet == 'shufflenetv2' and weights is None:
            self.checkpoint = osp.join(osp.dirname(__file__), './weights/shufflenetv2x2_pifpaf.pth')
        else:
            self.checkpoint = weights

        if use_gpu:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.scale      = scale
        self.score_th   = score_th
        self.stickwidth = stickwidth
        self.model = create_pifpaf(basenet)
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model  = self.model.to(self.device)
        self.decode = decoder.factory_decode(self.model,
                                        seed_threshold=0.5,
                                        debug_visualizer=False)
        self.processor = decoder.Processor( self.model, self.decode,
                                            instance_threshold=0.1,
                                            keypoint_threshold=0.001,
                                            debug_visualizer=None,
                                            profile=None,
                                            worker_pool=2,
                                            device=None)
    @staticmethod
    def bbox_from_keypoints(kps, score, h, w):

        kps = kps[kps[:, 2] > 0]
        bbox = np.array([np.min(kps[:, 0]), np.min(kps[:, 1]),
                         np.max(kps[:, 0]), np.max(kps[:, 1]),
                         score])
        bbox[0:4] = bbox[0:4]
        b_w = bbox[2] - bbox[0]
        b_h = bbox[3] - bbox[1]
        bbox[0] = np.maximum(bbox[0] - 0.15 * b_w, 0)
        bbox[2] = np.minimum(bbox[2] + 0.15 * b_w, w)
        bbox[1] = np.maximum(bbox[1] - 0.15 * b_h, 0)
        bbox[3] = np.minimum(bbox[3] + 0.15 * b_h, h)

        return bbox

    @staticmethod
    def facebox_from_keypoints(kps, h, w):
        face_kps = kps[0:5]
        if np.all(face_kps[:, 2] > 0):
            min_x = np.min(face_kps[:, 0])
            min_y = np.min(face_kps[:, 1])
            max_x = np.max(face_kps[:, 0])
            max_y = np.max(face_kps[:, 1])
            b_w = max_x - min_x
            # b_h = max_y - min_y

            min_x = np.maximum(min_x - 0.15 * b_w, 0)
            max_x = np.minimum(max_x + 0.15 * b_w, w)
            min_y = np.maximum(min_y - 0.3 * b_w, 0)
            max_y = np.minimum(max_y + 0.7 * b_w, h)

            return [min_x, min_y, max_x, max_y]

        return None

    def __call__(self, image):
        im_c = image.copy()
        resize_img = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        h, w, _ = image.shape
        bg_img = im_c.copy()
        img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)

        im_t = image_transform(img).unsqueeze(0)

        im_t = im_t.to(self.device)

        # fields_batch = self.processor.fields(im_t)
        # pred_batch = self.processor.annotations_batch(fields_batch, debug_images=None)
        # keypoint_sets, scores = self.processor.keypoint_sets_from_annotations(pred_batch[0])

        fields = self.processor.fields(im_t)[0]
        keypoint_sets, scores = self.processor.keypoint_sets(fields)

        bboxes = []
        face_boxes = []
        for kps, score in zip(keypoint_sets, scores):
            if score < self.score_th:
                continue
            kps[:, :2] = kps[:, :2] / self.scale
            for i, keys in enumerate(COCO_PERSON_SKELETON):
                if (kps[keys[0]][2]) > 0.1 and (kps[keys[1]][2]) > 0.1:
                    X = (int(kps[keys[0]][0]), int(kps[keys[1]][0]))
                    Y = (int(kps[keys[0]][1]), int(kps[keys[1]][1]))
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), self.stickwidth), int(angle),
                                               0, 360, 1)
                    cv2.fillConvexPoly(bg_img, polygon, line_color[i])
            bboxes.append(self.bbox_from_keypoints(kps, score, h, w))
            face_box = self.facebox_from_keypoints(kps, h, w)
            if face_box:
                face_boxes.append(np.array(face_box))
            else:
                face_boxes.append(None)
        im_c = cv2.addWeighted(bg_img, 0.9, im_c, 0.3, 0)
        # print("draw time:", time.time() - start_time)
        return im_c, np.array(bboxes), face_boxes


if __name__ == '__main__':
    # a = torch.load('/Users/edgar/Desktop/resnet50block5.pkl')
    pose = PosePifPaf(  basenet='shufflenetv2',
                        score_th=0.25)

    img = cv2.imread("./image/2.jpg")

    img_show, bboxes, face_boxes = pose(img)

    for box in bboxes:
        box = list(map(int, box[0:4]))
        cv2.rectangle(img_show, tuple(box[0:2]), tuple(box[2:4]), (0, 255, 0), 1)

    for box in face_boxes:
        if box is None:
            continue
        box = list(map(int, box))
        cv2.rectangle(img_show, tuple(box[0:2]), tuple(box[2:4]), (0, 0, 255), 1)
    cv2.imshow("aaa", img_show)
    cv2.waitKey()
