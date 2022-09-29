
from yolo_deepsort import yolo_deepsort as yd
print('yd imported')
import facial_recognition as fr
print('fr imported')
import matplotlib.pyplot as plt
print('plt imported')
import numpy as np
print('np imported')
import cv2
print('cv2 imported')

frame_num = 0
mevm,person_list = fr.load_clf()
while True:
    frame, start_time, image_data, frame_num = yd.read_frame(frame_num)
    
    tracker = yd.detection(frame, image_data)
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 2:
            continue
        
        bbox = track.to_tlbr()
        class_name = track.get_class()
        if track.index is not None:
            label = person_list[track.index].id[:7] +"..."
        else:
            label = str(track.track_id)
        if class_name== "person":
            num_classes = len(person_list)
            person = frame[int(bbox[0]):int(bbox[2]),int(bbox[1]):int(bbox[3]), :]
            cropped = fr.crop_and_align_image(person)
            
            
            if isinstance(cropped,np.ndarray):
                cropped = np.expand_dims(cropped, axis = 0)
                emb = fr.calc_embs(cropped)
                track.embs.append(emb)
                if len(track.embs)== 10:
                    prob, indices, mevm, person_list = fr.infer(mevm,track.embs,person_list)
                    track.class_name = "ID"
                    if num_classes != len(person_list):
                        track.index = len(person_list) - 1
                    else:
                        track.index = int(max(set(indices), key = indices.count))
                    print( prob)
                    print( indices)
                    print( mevm.evms)
                    print( person_list, track.index)
                    
        
        
        
        # draw bbox on screen
        color = yd.colors[int(track.track_id) % len(yd.colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(label))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(label),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
    yd.out_video(frame, start_time)