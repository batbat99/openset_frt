import xml.etree.ElementTree as ET
import os
import numpy as np
import pandas as pd
import facial_recognition as fr
import cv2

"""
The methods in this module are used for preparing the dataset for the training and testing
"""

def get_stream_path(stream):
    """
    A function to get the path of a recording in the dataset using the name as the parameter
    """
    dataset_path = "./dataset"
    for subdir, dirs, files in os.walk(dataset_path):
        if os.path.basename(subdir) == stream:
            return subdir
    return None

def get_stream_as_df(stream):
    """
    A function to get a recording from the dataset as a data frame,
    takes recording name as an argument 
    and returns a data frame with the data extracted from the xml file
    with the features extracted from the frame to be used for identification
    """
    xml_data = open("./dataset/groundtruth/"+stream+".xml").read()  # Read file
    root = ET.XML(xml_data)  # Parse XML
    img_path = get_stream_path(stream)
    if not img_path:
        return None

    data = []
    frame = []
    embs = np.empty((0,128), dtype=np.float32)
    for i, child in enumerate(root):
        if list(child) == []: 
            continue
        else: 
            for person in child:
                id  = person.attrib["id"]
                number = child.attrib["number"]
                eyes = [(int(subchild.attrib["x"]),int(subchild.attrib["y"])) for subchild in person]
                img = cv2.imread(img_path+"/"+number+".jpg")
                cropped = fr.crop_and_align_image(img, margin=10, eyes=eyes, min_neighbors=3)
                if isinstance(cropped,np.ndarray):
                    cropped = np.expand_dims(cropped, axis = 0)
                    emb = fr.calc_embs(cropped)
                    embs = np.append(embs, [emb], axis=0)
                    frame.append(number)
                    data.append(int(id))
    df = pd.DataFrame({"frame":frame, "person_ID":data})
    df = pd.concat([df, pd.DataFrame(embs)], axis=1)
    return df