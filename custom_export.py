import os
import sys
from ultralytics.engine.exporter import Exporter
from ultralytics import YOLO
import numpy as np
import cv2
import torch

import inspect

class CustomYOLOModel(torch.nn.Module):
    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model
        self.model = base_model.model
        for k in model.__dict__.keys():
            if inspect.ismethod(getattr(model, k)):
                continue

            setattr(self, k, getattr(model, k))
    def forward(self):
        output = self.model
        return self.postprocess(output)

    def fuse(self):
        self.base_model.fuse()

    def postprocess(self, output):
            """
            Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

            Args:
                input_image (numpy.ndarray): The input image.
                output (numpy.ndarray): The output of the model.

            Returns:
                numpy.ndarray: The input image with detections drawn on it.
            """

            # Transpose and squeeze the output to match the expected shape
            outputs = np.transpose(np.squeeze(output[0]))

            # Get the number of rows in the outputs array
            rows = outputs.shape[0]

            # Lists to store the bounding boxes, scores, and class IDs of the detections
            boxes = []
            scores = []
            class_ids = []

            # Calculate the scaling factors for the bounding box coordinates
            x_factor = self.img_width / self.input_width
            y_factor = self.img_height / self.input_height

            # Iterate over each row in the outputs array
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = outputs[i][4:]

                # Find the maximum score among the class scores
                max_score = np.amax(classes_scores)

                # If the maximum score is above the confidence threshold
                if max_score >= self.confidence_thres:
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)

                    # Extract the bounding box coordinates from the current row
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    # Calculate the scaled coordinates of the bounding box
                    left = int((x - w / 2) * x_factor)
                    top = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # Add the class ID, score, and box coordinates to the respective lists
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([left, top, width, height])

            # Apply non-maximum suppression to filter out overlapping bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

            # Iterate over the selected indices after non-maximum suppression
            box_results = []
            score_results = []
            class_id_results = []
            for i in indices:
                # Get the box, score, and class ID corresponding to the index
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]

                
                # TODO: check dimensions
                box_results.append(box)
                score_results.append(score)
                class_id_results.append(class_id)
                # Draw the detection on the input image
                # self.draw_detections(input_image, box, score, class_id)

            return (box_results, score_results, class_id_results)
        # Return the modified input image
        # return input_image

if __name__ == '__main__':

    
    assert len( sys.argv ) > 0
    model_path =  sys.argv[1] #'path/to/model.pt'
    print(f'Loading model path at: {model_path}')
    yolo_model = YOLO(model_path)


    # image_path = 'path/to/image.png'

## Get 'real' model
    model = yolo_model.model


## Custom model ouput
# preds[0] shape: 1 x 5 x box_nums
    # preds = model([image_path])
# TODO: custom model
    # custom_model = CustomYOLOModel( model )
    custom_model = CustomYOLOModel( yolo_model )

## Attach custom model to yolo
# Custom model change number of 
# yolo_model.model = custom_model

## Export with custom format
    custom = {'imgsz': model.args['imgsz'], 'batch': 1, 'data': None, 'verbose': False}  # method defaults
    kwargs = { } # as in YOLO epxort's args
    args = {**yolo_model.overrides, **custom, **kwargs, 'mode': 'export'}  # highest priority args on the right
    saved_export_path =  Exporter(overrides=args )(model=custom_model)

# saved_export_path = yolo_model.export(format='onnx')

###############
# CFG_DET = 'yolov8n.yaml'
# exporter = Exporter()
# # exporter.add_callback('on_export_start', test_func)
# # assert test_func in exporter.callbacks['on_export_start'], 'callback test failed'
# f = exporter(model=YOLO(CFG_DET).model)
