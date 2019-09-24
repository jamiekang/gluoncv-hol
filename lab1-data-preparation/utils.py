import numpy as np
import itertools
from matplotlib import pyplot as plt
import cv2
import json

def create_lst(gt_manifest, gt_job_name):
    number_no_label, number_no_label_val = 0, 0
    number_label = 0

    im_fnames = list()

    with open(gt_manifest, 'r') as f:
        lines = f.readlines()
        lst_list = list()
        for idx,line in enumerate(lines):
            label_json = json.loads(line)
            src_ref = label_json['source-ref']
            im_fname = src_ref.split('/')[-1]
            
            if gt_job_name not in label_json:
                continue
                
            annotations = label_json[gt_job_name]['annotations']

            im_fnames.append(im_fname)
            im_size = label_json[gt_job_name]['image_size']
            im_height = im_size[0]['height']
            im_width = im_size[0]['width']
            im_depth = im_size[0]['depth']
            im_shape = (im_height, im_width, im_depth)

            number_label = number_label + 1    
            bbox_list = list()
            class_list = list()

            for annotation in annotations:
                class_id = annotation['class_id']
                width = annotation['width']
                top = annotation['top']
                height = annotation['height']
                left = annotation['left']
                bbox = [left, top, left+width, top+height]
                bbox_list.append(bbox)
                class_list.append(class_id)

            all_boxes = np.array(bbox_list)
            all_ids = np.array(class_list)

            if len(annotations) > 0:
                lst_str = write_line(im_fname, im_shape, all_boxes, all_ids, idx)

            else:
                number_no_label = number_no_label + 1
                lst_str = None

            if lst_str != None:
                lst_list.append(lst_str) 

    return lst_list, number_label, number_no_label

def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    
    return line

def write_line_no_object(img_path, im_shape, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    ids = np.array([99])
    boxes = np.array([[0,0,0,0]])
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    
    return line

def write_lst_to_file(lst_list, lst_fname):
    with open(lst_fname,'w') as fw:
        for lst in lst_list:
            fw.write(lst)
            
# def get_topK(class_ids, class_scores, bboxes, filter_class_id=0, K=5, threshold=0.5):
#     cnt = 0
#     scores = list()
#     bboxes_list = list()
#     for class_id, class_score, bbox in zip(class_ids[0,:,0], class_scores[0,:,0], bboxes[0]):
#         class_id = class_id.asscalar()
#         class_score = class_score.asscalar()
#         if class_id == filter_class_id:
#             if class_score >= threshold:
#                 cnt = cnt + 1
#                 scores.append(class_score)
#                 bboxes_list.append(bbox)
#             if cnt == K:
#                 break
                
#     return scores,bboxes_list

# def render_as_image(a, channel_swap=False):
#     img = a.asnumpy() # convert to numpy array
#     if channel_swap:
#         img = img.transpose((1, 2, 0))  # Move channel to the last dimension
#     img = img.astype(np.uint8)  # use uint8 (0-255)

#     plt.imshow(img)
#     plt.show()
    
# def draw_bbox(img, bbox_list, score_list, label=0, number_of_flags=0,number_of_flags_watermark=0):
#     new_img = img.copy()
#     rec_color = (255,0,0) if label == 0 else (0,0,255)
    
#     for flag_bbox in bbox_list[:]:
#         flag_bbox = flag_bbox.asnumpy()
#         p1 = (int(flag_bbox[0]), int(flag_bbox[1]))
#         p2 = (int(flag_bbox[2]), int(flag_bbox[3]))
#         new_img = cv2.rectangle(new_img, p1, p2, rec_color, 4)
        
#     num_boxes = len(bbox_list)
    
#     score_list_to_display = [round(s+0.005,2) for s in score_list]
    
#     if label == 0:
#         txt_color = (255,0,0) if num_boxes > 0 else (255,255,255)
#         txt_str = '{}-{}-{} flag(s) detected {}'.format(number_of_flags_watermark, number_of_flags, num_boxes, score_list_to_display)
#         txt_position = (50,100)
#     elif label == 1:
#         txt_color = (0,0,255) if num_boxes > 0 else (255,255,255)
#         txt_str = '{} graphic detected {}'.format(num_boxes, score_list_to_display)
#         txt_position = (50,200)
#     else:    
#         txt_color = (0,0,255) if num_boxes > 0 else (255,255,255)
#         txt_str = '{} other detected {}'.format(num_boxes, score_list_to_display)
#         txt_position = (50,300)
        
#     cv2.putText(new_img, txt_str, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color, 4, cv2.LINE_AA)
    
#     return new_img 
    
# def get_trained_model_names(base_model, dataset):

#     if base_model == 'resnet50':
#         # For ResNet
#         model_fname = 'ssd_512_resnet50_v1_{}_flag_detection.params'.format(dataset)
#         custom_model_name = 'ssd_512_resnet50_v1_custom'
#     elif base_model == 'resnet18':
#         # For ResNet
#         model_fname = 'ssd_512_resnet18_v1_{}_flag_detection.params'.format(dataset)
#         custom_model_name = 'ssd_512_resnet18_v1_custom'
#     elif base_model == 'mobilenet':
#         # For mobilenet
#         model_fname = 'ssd_512_mobilenet1.0_{}_flag_detection.params'.format(dataset)
#         custom_model_name = 'ssd_512_mobilenet1.0_custom'
#     else:
#         # For VGG
#         model_fname = 'ssd_512_vgg16_atrous_{}_flag_detection.params'.format(dataset)
#         custom_model_name = 'ssd_512_vgg16_atrous_custom'
        
#     return model_fname, custom_model_name

# def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
#     """pretty print for confusion matrixes"""
#     columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
#     empty_cell = " " * columnwidth
#     # Print header
#     print("    " + empty_cell, end=" ")
#     for label in labels:
#         print("%{0}s".format(columnwidth) % label, end=" ")
#     print()
#     # Print rows
#     for i, label1 in enumerate(labels):
#         print("    %{0}s".format(columnwidth) % label1, end=" ")
#         for j in range(len(labels)):
#             cell = "%{0}.1f".format(columnwidth) % cm[i, j]
#             if hide_zeroes:
#                 cell = cell if float(cm[i, j]) != 0 else empty_cell
#             if hide_diagonal:
#                 cell = cell if i != j else empty_cell
#             if hide_threshold:
#                 cell = cell if cm[i, j] > hide_threshold else empty_cell
#             print(cell, end=" ")
#         print()
        