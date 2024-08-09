# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import sys
sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# the above path is needed by pynq_dpu
from pynq_dpu import DpuOverlay
import matplotlib.pyplot as plt
from pynq_dpu import DpuOverlay
import rclpy
from rclpy.node import Node

# from std_msgs.msg import String
from geometry_msgs.msg import Twist

from std_srvs.srv import Empty

from sensor_msgs.msg import Image
# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

import numpy as np
from ctypes import *
from typing import List
import pathlib
import time
import argparse
import glob
import subprocess
import re
import os

frame_size = (640, 480)


class RosaiCameraYolov3(Node):

    def __init__(self):
        super().__init__('rosai_camera_demo')
        self.scb = StatusControlBlock()
        self.subscriber_ = self.create_subscription(Image, 'image_raw',
                                                    self.listener_callback, 10)
        self.get_logger().info('[INFO] __init__, Create Subscription ...')
        self.subscriber_  # prevent unused variable warning
        self.publisher1 = self.create_publisher(Image, 'vision/asl', 10)
        # self.publisher2 = self.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher2 = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        anchor_list = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        anchor_float = [float(x) for x in anchor_list]
        self.anchors = np.array(anchor_float).reshape(-1, 2)
        # Overlay the DPU and Vitis-AI .xmodel file
        self.overlay = DpuOverlay("dpu.bit")
        self.model_path = '/home/root/jupyter_notebooks/pynq-dpu/tf_yolov3_voc.xmodel'
        self.get_logger().info("MODEL=" + self.model_path)
        self.overlay.load_model(self.model_path)

        # Create DPU runner
        self.dpu = self.overlay.runner

        self.get_logger().info('[INFO] __init__ exiting...')

    def get_class(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    # This function resizes the image with unchanged aspect ratio using padding.

    def letterbox_image(self, image, size):
        ih, iw, _ = image.shape
        w, h = size
        scale = min(w / iw, h / ih)

        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.ones((h, w, 3), np.uint8) * 128
        h_start = (h - nh) // 2
        w_start = (w - nw) // 2
        new_image[h_start:h_start + nh, w_start:w_start + nw, :] = image
        return new_image

    # This function performs pre-processing by helping us in converting
    # the image into an array which can be fed for processing.
    def pre_process(self, image, model_image_size):
        image = image[..., ::-1]
        image_h, image_w, _ = image.shape

        if model_image_size != (None, None):
            assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = self.letterbox_image(
                image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32),
                              image_h - (image_h % 32))
            boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        return image_data

    # This function gets information on box position, its size
    # along with confidence and box class probabilities
    def get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
        # self.get_logger().info("feats="+str(np.shape(feats)))
        # self.get_logger().info("num_classes="+str(num_classes))
        grid_size = np.shape(feats)[1:3]
        nu = num_classes + 5
        predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
        grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis=-1)
        grid = np.array(grid, dtype=np.float32)
        box_xy = (1 / (1 + np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
        box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
        box_confidence = 1 / (1 + np.exp(-predictions[..., 4:5]))
        box_class_probs = 1 / (1 + np.exp(-predictions[..., 5:]))
        return box_xy, box_wh, box_confidence, box_class_probs

    # This function is used to correct the bounding box position by scaling it
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape, dtype=np.float32)
        image_shape = np.array(image_shape, dtype=np.float32)
        new_shape = np.around(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    # This function is used to get information on the
    # valid objects detected and their scores
    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self.get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # This function suppresses non-maximal boxes by eliminating boxes which are lower than the threshold
    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.55)[0]  # threshold
            order = order[inds + 1]

        return keep

    # This function gives essential information about the objects detected like bounding box information, score of the object
    # detected and the class associated with it
    def evaluate(self, yolo_outputs, image_shape, class_names, anchors):
        score_thresh = 0.2
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = np.shape(yolo_outputs[0])[1: 3]
        input_shape = np.array(input_shape) * 32

        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(
                yolo_outputs[i], anchors[anchor_mask[i]], len(class_names),
                input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        mask = box_scores >= score_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(class_names)):
            class_boxes_np = boxes[mask[:, c]]
            class_box_scores_np = box_scores[:, c]
            class_box_scores_np = class_box_scores_np[mask[:, c]]
            nms_index_np = self.nms_boxes(class_boxes_np, class_box_scores_np)
            class_boxes_np = class_boxes_np[nms_index_np]
            class_box_scores_np = class_box_scores_np[nms_index_np]
            classes_np = np.ones_like(class_box_scores_np, dtype=np.int32) * c
            boxes_.append(class_boxes_np)
            scores_.append(class_box_scores_np)
            classes_.append(classes_np)
        boxes_ = np.concatenate(boxes_, axis=0)
        scores_ = np.concatenate(scores_, axis=0)
        classes_ = np.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

    def listener_callback(self, msg):
        self.get_logger().info("Starting of listener callback...")
        bridge = CvBridge()
        cv2_image_org = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        y1 = (128)
        y2 = (128 + 280)
        x1 = (208)
        x2 = (208 + 280)
        roi_img = cv2_image_org[y1:y2, x1:x2, :]
        resized_image = cv2.resize(roi_img, (28, 28), interpolation=cv2.INTER_LINEAR)
        roi_img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        cv2_image_normal = np.asarray(roi_img_gray / 255, dtype=np.float32)
        cv2_image = np.expand_dims(cv2_image_normal, axis=2)

        # classes_path = "/root/ros2_ws/src/rosai_camera/rosai_camera/configs/coco_classes.txt"
        classes_path = "/root/ros2_ws/src/rosai_camera/rosai_camera/configs/voc_classes.txt"
        class_names = self.get_class(classes_path)

        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()
        shapeIn = tuple(inputTensors[0].dims)

        # shapeOut = tuple(outputTensors[0].dims)
        shapeOut0 = (tuple(outputTensors[0].dims))  # (1, 13, 13, 75)
        shapeOut1 = (tuple(outputTensors[1].dims))  # (1, 26, 26, 75)
        shapeOut2 = (tuple(outputTensors[2].dims))  # (1, 52, 52, 75)

        # outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])
        # outputSize0 = int(
        #     outputTensors[0].get_data_size() / shapeIn[0])  # 12675
        # outputSize1 = int(
        #     outputTensors[1].get_data_size() / shapeIn[0])  # 50700
        # outputSize2 = int(
        #     outputTensors[2].get_data_size() / shapeIn[0])  # 202800

        # Setup Buffers
        input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
        output_data = [np.empty(shapeOut0, dtype=np.float32, order="C"),
                       np.empty(shapeOut1, dtype=np.float32, order="C"),
                       np.empty(shapeOut2, dtype=np.float32, order="C")]
        image = input_data[0]
        frame = cv2_image_org
        # Function to perform pre-processing, model predictions and decoding output

        # Pre-processing
        image_size = frame.shape[:2]
        image_data = np.array(self.pre_process(frame, (416, 416)), dtype=np.float32)

        # Fetch data to DPU and trigger it
        image[0, ...] = image_data.reshape(shapeIn[1:])
        job_id = self.dpu.execute_async(input_data, output_data)
        self.dpu.wait(job_id)
        
        # Retrieve output data
        conv_out0 = np.reshape(output_data[0], shapeOut0)
        conv_out1 = np.reshape(output_data[1], shapeOut1)
        conv_out2 = np.reshape(output_data[2], shapeOut2)
        yolo_outputs = [conv_out0, conv_out1, conv_out2]
        # self.get_logger().info("conv_out0=" + str(np.shape(conv_out0)))
        # self.get_logger().info("conv_out1=" + str(np.shape(conv_out1)))
        # self.get_logger().info("conv_out2=" + str(np.shape(conv_out2)))
        # self.get_logger().info("image_size=" + str(np.shape(image_size)))
        # self.get_logger().info("class_names=" + str(class_names))
        # self.get_logger().info("image_data=" + str(np.shape(image_data)))
        # Decode output from YOLOv3
        boxes, scores, classes = self.evaluate(yolo_outputs, image_size, class_names, self.anchors)

        # Store information of the detected object which has the highest score. And store the class information of this object
        if len(scores) == 0:
            pass
        else:
            self.get_logger().info("scores=" + str(scores))
    
            best_score = np.argmax(scores)
            
            prediction = class_names[classes[best_score]]
            if prediction == 'person':
                self.get_logger().info("best_score=" + str(best_score))
                # Draw bounding box
                y_min, x_min, y_max, x_max = map(int, boxes[best_score])
                rects = []
                p1, p2 = ((int(x_min), int(y_min)), (int(x_max), int(y_max)))
                rects.append([p1, p2])
            ###################################################
                # Create message to backup (+ve value on x axis)
                self.scb.follow_target(rects)
                msg = Twist()
                msg.linear.x = float(self.scb.control_speed)
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = float(self.scb.control_turn)
                self.publisher2.publish(msg)

                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=255)

                # Label
                text = f"{class_names[classes[best_score]]}: {scores[best_score]:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                frame = cv2.putText(frame, text, (x_min, y_min - text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA)
                # DISPLAY
                cv2_bgr_image = cv2.cvtColor(cv2_image_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('rosai_demo', cv2_bgr_image)
                cv2.waitKey(1)
                # CONVERT BACK TO ROS & PUBLISH
                image_ros = bridge.cv2_to_imgmsg(cv2_image)
                self.publisher1.publish(image_ros)


class StatusControlBlock:
    
    def __init__(self):
        self.mid_x = frame_size[0] / 2  # 图像中心x
        self.last_x = self.mid_x  # 上一次目标的中心x
        self.last_w = 0  # 上一次目标的宽度

        self.speed_max = 0.45  # 最大速度太快容易撞  可调整
        self.back_speed = -0.15  # 后退速度
        self.t = 0.4  # 两张图片的时间间隔 (s)  可调整
        self.half_rad = 0.7  # 半角弧度
        self.base_w = self.mid_x * 0.6  # 基准框宽度 (像素)
        self.bash_y = 1  # 基准距离 (m )

        self.x_l_min = self.mid_x * 0.8  # 向左转向阈值
        self.x_r_min = self.mid_x * 1.2  # 向右转向阈值
        self.x_l_max = self.mid_x * 0.2  # 向左转向阈值
        self.x_r_max = self.mid_x * 1.8  # 向右转向阈值
        self.w_b_min = self.mid_x * 0.8  # 小车后退阈值
        self.w_f_min = self.mid_x * 0.5  # 小车前进阈值

        self.control_speed = 0  # 控制前后速度
        self.control_turn = 0  # 控制转向速度

    # 根据框框的位置判定该往哪儿走
    def follow_target(self, rects):
        target = []
        num = len(rects)

        if num == 0:  # 无框
            self.last_x = self.mid_x  # 上一次目标的中心x
            self.last_w = 0
            self.control_turn = 0
            self.control_speed = 0
        elif num == 1:
            target = rects[0]
        else:
            target = self.find_target_rect(rects).all()

        if target:
            self.last_x = (target[1][0] + target[0][0]) / 2  # 中心x
            self.last_w = target[1][0] - target[0][0]  # 宽度
            self.tran_turn()
            self.tran_speed()

    # 计算小车转向
    def tran_turn(self):
        if self.x_l_min <= self.last_x <= self.x_r_min:
            # 不转向
            self.control_turn = 0

        elif self.last_x < self.x_l_min:
            # 左转
            turn_rad = self.half_rad * (self.x_l_min - self.last_x) / self.mid_x  # 该转动的弧度
            self.control_turn = turn_rad / self.t
        else:
            # 右转
            turn_rad = self.half_rad * (self.x_r_min - self.last_x) / self.mid_x
            self.control_turn = turn_rad / self.t
        self.control_turn /= 2

    # 计算小车速度
    def tran_speed(self):

        if self.w_f_min <= self.last_w <= self.w_b_min:
            # 不运动
            self.control_speed = 0

        elif self.last_w > self.w_b_min:
            # 小车后退
            self.control_speed = self.back_speed
        else:
            if self.control_speed <= 0.2:
                self.control_speed = 0.2
            # 小车前进
            if self.last_x <= self.x_l_min or self.last_x >= self.x_r_max:
                # 人在边上，可能只识别半个人，速度不能太快
                # self.control_speed = 0.2
                pass
            else:
                distance = self.bash_y * (self.base_w / self.last_w) - self.bash_y  # 该移动的距离
                speed = distance / self.t  # 理应速度
                if speed > self.control_speed:
                    if self.control_speed < self.speed_max:
                        self.control_speed += 0.05  # 匀变速

                else:
                    self.control_speed = speed
                print("--- caculate distance = {} , speed = {}".format(distance, speed))

    # 目标
    def find_target_rect(self, rects):
        assert len(rects) > 1
        gap = self.mid_x * 2  # 寻找和上一个点差值最小的点
        cur_rect = []
        for rect in rects:
            cur_x = (rect[1][0] + rect[0][0]) / 2
            cur_gap = abs(cur_x - self.last_x)
            if cur_gap < gap:
                cur_rect = rect
        self.last_x = (cur_rect[1][0] + cur_rect[0][0]) / 2
        return cur_rect


def main(args=None):
    rclpy.init(args=args)
    rosai_camera_demo = RosaiCameraYolov3()
    rclpy.spin(rosai_camera_demo)


if __name__ == '__main__':
    main()
