import streamlit as st
import tensorflow as tf 
import cv2
import numpy as np
import random
import colorsys
from PIL import Image
import io
import imageio
class_file_name = "./custom.names"
#original_image = cv2.imread(image_path) 
 
 
#fuctions
 
 
def convtnp(image_path,input_size=416):
    original_image = image_path
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))        
    image_data = image_data / 255.
    img = []
    img.append(image_data)        
    image_np = np.asarray(img).astype(np.float32)
    return image_np
 
 
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names 
 
# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes
#draw box 
 
def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label=True, allowed_classes= list(read_class_names(class_file_name).values()) ):
    classes = read_class_names(class_file_name)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
 
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
 
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.3
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        if class_name not in allowed_classes:
            continue
        else:
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
 
            if info:
                print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))
 
            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
 
                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
 
            if counted_classes != None:
                height_ratio = int(image_h / 25)
                offset = 15
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    offset += height_ratio
    return image
 
# function to count objects, can return total classes or count per class
def count_objects(data, by_class = True, allowed_classes =list(read_class_names(class_file_name).values())):
    boxes, scores, classes, num_objects = data
 
    #create dictionary to hold count of objects
    counts = dict()
 
    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names("/content/drive/My Drive/custom.names")
 
        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
                counts['total object'] = num_objects
            else:
                continue
    
    return counts
@st.cache(allow_output_mutation= True)
def load_model():
  loaded = tf.saved_model.load("./new")
  return loaded 
loaded = load_model()
#infer = loaded.signatures["serving_default"]  
infer = load_model() 
def predict(image_path):
   img=convtnp(image_path)
   batch_data = tf.constant(img)            
   pred_bbox = infer(batch_data)   
   #for key, value in pred_bbox.items():         
         #boxes = value[:, :, 0:4]                
         #pred_conf = value[:, :, 4:]
   boxes = pred_bbox[:, :, 0:4]                   
   pred_conf = pred_bbox[:, :, 4:]   
   boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=80,
            max_total_size=100,
            iou_threshold=0.1,
            score_threshold=0.1
        )
   
    
 
    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
   original_image = image_path
   original_h, original_w, _ = original_image.shape
   bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)
        
   # hold all detection data in one variable
   pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
 
   # read in all class names from config
   class_names = read_class_names(class_file_name)
 
   # by default allow all classes in .names file
   allowed_classes = list(class_names.values())
   image = draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)   
   image = Image.fromarray(image.astype(np.uint8))
  
   image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB) 
   count = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
   return image,count
def main():
  st.title("BLOOD CELL COUNTER ") 
  st.set_option('deprecation.showfileUploaderEncoding', False)
  image_path = st.file_uploader("Upload a file", type=["jpeg","png", "jpg"])
  
  if image_path is None:
    st.write("Please upload your file")
  else:
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)    
    image_path = cv2.imdecode(file_bytes, 1)
    imag,count = predict(image_path)
    st.image(imag)
    for key, value in count.items():    
      st.write(key , value)
   
if __name__=='__main__':
  main()