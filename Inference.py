import os
SAMPLES = 10
NUM_CLASSES = 1
TEST_DIR = '/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Demo/Images'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pathlib
import numpy as np

import six.moves.urllib as urllib
import sys
import tarfile

import zipfile
import pandas as pd
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import glob
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# What model to download.
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Demo/fine_tuned_model/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Demo/tfrecords/label_map.pbtxt"


from IPython import get_ipython
# %matplotlib inline

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(category_index)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 5 images:
# from image1.jpg
# to image5.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.



ext = ['png', 'jpg',"JPG","JPEG","jpeg"]    # Add image formats here

test_filenames = []
[test_filenames.extend(glob.glob(TEST_DIR + '*.' + e)) for e in ext]


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


count = 0
for image_path in test_filenames:
  if(count==SAMPLES):
    break
  if image_path.find(".jpg")!=-1 or  image_path.find(".png")!=-1:
    count= count+1
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    print(image_path)
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    # print(image_np)
    output_path = f"/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Output/brainy_image_{count}.jpg"
    cv2.imwrite(output_path, image_np)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)










# import tensorflow as tf
# import time
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# import numpy as np
# from PIL import Image
# import glob
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# TEST_DIR = '/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Demo/Images'
# SAMPLES = 10

# PATH_TO_SAVED_MODEL="/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Demo/fine_tuned_model/saved_model"

# print('Loading model...', end='')

# # Load saved model and build the detection function
# detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

# print('Done!')


# category_index=label_map_util.create_category_index_from_labelmap("/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Demo/tfrecords/label_map.pbtxt",use_display_name=True)

# # img=['/home/models/research/test/img (1).jpg']

# ext = ['png', 'jpg',"JPG","JPEG","jpeg"]    # Add image formats here

# img = []
# [img.extend(glob.glob(TEST_DIR + '*.' + e)) for e in ext]


# def load_image_into_numpy_array(path):
#     return np.array(Image.open(path))

# count=0
# for image_path in img:
#     if(count==SAMPLES):
#         break
#     if image_path.find(".jpg")!=-1 or  image_path.find(".png")!=-1:
#         count= count+1

#     print('Running inference for {}... '.format(image_path), end='')
#     image_np=load_image_into_numpy_array(image_path)
#     input_tensor=tf.convert_to_tensor(image_np)
#     input_tensor=input_tensor[tf.newaxis, ...]
#     detections=detect_fn(input_tensor)
#     num_detections=int(detections.pop('num_detections'))
#     detections={key:value[0,:num_detections].numpy()
#                    for key,value in detections.items()}
#     detections['num_detections']=num_detections
#     detections['detection_classes']=detections['detection_classes'].astype(np.int64)
#     image_np_with_detections=image_np.copy()
#     viz_utils.visualize_boxes_and_labels_on_image_array(
#           image_np_with_detections,
#           detections['detection_boxes'],
#           detections['detection_classes'],
#           detections['detection_scores'],
#           category_index,
#           use_normalized_coordinates=True,
#           max_boxes_to_draw=100,     
#           min_score_thresh=.7,#0.0001      
#           agnostic_mode=False)
# #     %matplotlib inline
#     plt.figure()
#     plt.imshow(image_np_with_detections)
#     print('Done')
#     plt.axis('off')
#     plt.show()

#     # to save the output image 
#     nimg = Image.fromarray(image_np_with_detections)
#     nimg.save('/home/pi/Sai_Rohith/pedestrian-detection-and-tracking/Output/'+'brainypi_image_'+count+'.jpg')
