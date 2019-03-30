#!/usr/bin/env python3
"""
#!/home/sankalp/anaconda3/envs
#!/usr/bin/env python3
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:07:00 2019

@author: sankalp
"""

import numpy as np
import os
import mysql.connector
# os.system(". ~/anaconda3/bin/activate tf-gpu")
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = 'diztro_graph_11/frozen_inference_graph.pb'

PATH_TO_LABELS = 'data/label_map.pbtxt'

NUM_CLASSES = 4
inputimg=sys.argv[1]
storeid=sys.argv[2]
fridgeid=sys.argv[3]

try:

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

  def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


  PATH_TO_TEST_IMAGES_DIR = 'test_images/'
  #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1,3) ]

  
  image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,inputimg)

  IMAGE_SIZE = (12, 8)

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      sess.run(tf.global_variables_initializer())
      #img = 1
      #for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imsave('RESULTS/' + str('Result_'+inputimg), image_np)
      resultimg='RESULTS/' + 'Result_'+inputimg 
      #img += 1

      taco = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.1 ]
      df=pd.DataFrame(taco)
      #a=df.groupby(['id','name']).size().reset_index(name='counts')
      #df['count']=df.groupby(['id','name']).transform('count')
      a=df.groupby(['id','name']).id.agg('count').to_frame('count').reset_index()
      #print(a)
      #print(len(taco))
      dict=[]
      #Runid=1
      
      #insert into mysql db
      
      mydb = mysql.connector.connect( host="localhost",  user="siddas", passwd="siddas@123",  database="diztro")
      mycursor = mydb.cursor()
      sql = "INSERT INTO dldump (storeid, fname,ProcessedPath,status,fridgeid) VALUES (%s, %s,%s, %s,%s)"
      val = (int(storeid), PATH_TO_TEST_IMAGES_DIR+inputimg,resultimg,1,int(fridgeid))
      # print (val)
      mycursor.execute(sql, val)
      mydb.commit()
      Runid=mycursor.lastrowid
      # print('inserted to dldump')
      # print (Runid)
      
      for index, row in a.iterrows(): 
        dict.append((Runid,row['id'],row['count']))
	           
      
      sql1="INSERT INTO image_processing_results (Runid, itemid,count) VALUES (%s, %s,%s)"
      mycursor.executemany(sql1, dict)
      mydb.commit()
      
      print('inserted to image_processing_results')
      
      mycursor.close()
      mydb.close()

except Exception as e:
  print(e)
  mydb = mysql.connector.connect( host="localhost",  user="siddas", passwd="siddas@123",  database="diztro")
  mycursor = mydb.cursor()
  sql = "INSERT INTO dldump (storeid, fname,status,fridgeid) VALUES (%s, %s, %s,%s)"
  val = (int(storeid), PATH_TO_TEST_IMAGES_DIR+inputimg,-1,int(fridgeid))
  # print (val)
  mycursor.execute(sql, val)
  print ('failure inserted')
  mydb.commit()
  mycursor.close()
  mydb.close()
#finally:

