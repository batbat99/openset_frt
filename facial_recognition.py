import numpy as np
import os
import cv2
import pickle
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skimage.transform import resize
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import EVM, scipy
from uuid import uuid4


cascade_path = './model/cv2/haarcascade_frontalface_alt2.xml'
image_dir_basepath = './data/images/'

image_size = 160

class person:
    """a basic class for storing information for a person"""
    def __init__(self,name,embs):
        self.id = name
        self.embs = embs
        
def generate_label(class_ = 'person'):
    """
    generates a unique id for a new person, 
    if it is decided that more than one class of objects
    is going to be detected the first character will represent
    the class
    """
    if class_ == 'person':
        unique_id = "P" + str(uuid4())
    else:
        unique_id = "V" + str(uuid4())
    return unique_id


model_path = './model/keras/facenet_keras.h5'
model = tf.keras.models.load_model(model_path)




def prewhiten(x):
    """perform transformation on input images before detection"""
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    #normalizing the output of the neural network
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output




def load_and_align_images(filepaths, margin= 10):
    cascade = cv2.CascadeClassifier(cascade_path)
    
    aligned_images = []
    for filepath in filepaths:
        try:
            img = imread(filepath)

            faces = cascade.detectMultiScale(img,
                                             scaleFactor=1.1,
                                             minNeighbors=3)
            if len(faces)==1:
                (x, y, w, h) = faces[0]
                cropped = img[y-margin//2:y+h+margin//2,
                              x-margin//2:x+w+margin//2, :]
                aligned = resize(cropped, (image_size, image_size), mode='reflect')
                aligned_images.append(aligned)
        except:
            pass
    return np.array(aligned_images)

def crop_and_align_image(person, margin=0, eyes=None, min_neighbors=3):
    cascade = cv2.CascadeClassifier(cascade_path)
    
    aligned_img = []
    img = person
    try:
        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=min_neighbors)
        
        if eyes:
            for face in faces:
                if all([face[0] < eye[0] < face[0] + face[2] and face[1] < eye[1] < face[1] + face[3] for eye in eyes]):
                    (x, y, w, h) = face
                    break 
        else:
            (x, y, w, h) = faces[0]
        cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
        aligned = resize(cropped, (image_size, image_size))


        aligned_img = np.array(aligned)
        
    except:
        return 0
    
    return aligned_img    
    




def calc_embs(croped, batch_size=1):
    #calculates features for the given data
    aligned_images = prewhiten(croped)
    pd = model.predict_on_batch(aligned_images)
    embs = l2_normalize(np.concatenate(pd))

    return embs



def train(dir_basepath= image_dir_basepath, max_num_img=10):
    """trains a MEVM classifiers
    
    Parameters
    ----------
    dir_basepath : str
        a string that represents the path to the data directory
    max_num_img : int
        the maximum number of samples to use per class
    """
    person_list = []
    embs = []
    
    for subdir, dirs, files in os.walk(dir_basepath):
        print(subdir,files)
        if files:
            name= os.path.basename(subdir)
            filepaths = [os.path.join(subdir, f) for f in files][:max_num_img]
            cropped = load_and_align_images(filepaths)
        
            embs_ = calc_embs(cropped)
            p = person(name,embs_)
            embs.append(embs_)
            person_list.append(p)
            
    print("creating mevm")        
    mevm = EVM.MultipleEVM(tailsize=20, cover_threshold = 0.6, distance_multiplier =0.53, distance_function=scipy.spatial.distance.euclidean)  
    mevm.train(embs)
    with open('person_list.pkl', 'wb') as output:
        pickle.dump(person_list, output, pickle.HIGHEST_PROTOCOL)
    with open('mevm.pkl', 'wb') as output:
        pickle.dump(mevm, output, pickle.HIGHEST_PROTOCOL)
    return mevm, person_list

def add_evm(mevm,pperson,nperson_l):
    """trains an EVM and adds it to a MEVM object
    
    Parameters
    ----------
    mevm : MEVM object
        an instance of a MEVM
    pperson : person object
        a person object that represents the new class to be trained
    nperson_l : list
        a list that represents negative examples for the classifiers from other classes
        
    """
    tailsize = mevm.tailsize
    cover_threshold = mevm.cover_threshold
    distance_multiplier = mevm.distance_multiplier
    distance_function = mevm.distance_function
    include_cover_probability = mevm.include_cover_probability
    
    pembs = pperson.embs
    nembs = []
    for nperson in nperson_l:
        nembs.append(nperson.embs)
    nembs = nembs[0]
    print(nembs)
    new_evm = EVM.EVM(tailsize=tailsize, cover_threshold=cover_threshold, distance_multiplier=distance_multiplier, distance_function=distance_function, include_cover_probability=include_cover_probability)
    new_evm.train(positives = pembs, negatives = nembs)

    mevm.evms.append(new_evm)

    return mevm





def infer(mevm, embs, person_list):
    mevm = mevm
    rslt= mevm.max_probabilities(embs)
    print(rslt)
    prob= rslt[0]
    indices = [x[0] for x in rslt[1]]
    print(max(prob))
    if max(prob)<0.5:
        p = person(generate_label(),embs)
        person_list.append(p)
        n_indices = list(set(indices))
        n_person = []
        for i in n_indices:
            n_person.append(person_list[i])
        mevm = add_evm(mevm,p,n_person)
        with open('person_list.pkl', 'wb') as output:
            pickle.dump(person_list, output, pickle.HIGHEST_PROTOCOL)
        with open('mevm.pkl', 'wb') as output:
            pickle.dump(mevm, output, pickle.HIGHEST_PROTOCOL)
    return prob, indices, mevm, person_list


def load_clf():
    try:
        with open('mevm.pkl', 'rb') as input:
            mevm = pickle.load(input)
        with open('person_list.pkl', 'rb') as input:
            person_list = pickle.load(input)
    except:
        mevm, person_list= train()
    print("classifier loaded")
    return mevm, person_list

# things to do:
# reimplement so that it uses the classifiers.py module