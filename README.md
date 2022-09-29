# openset_frt
A repository that contains all the related code for my thesis work titled "Open set classification for facial recognition applications"

## How to use

### installing dependencies
Use conda to install all the required dependencies

```bash
conda env create -f openset_frt.yml
conda activate openset_frt
```

### Downloading YOLOv4 Weights
The official YOLOv4 pre-trained weigts can be found on the following link: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

the downloaded file must be copied to yolo_deepsort/data

the .weights file must be converted into TensorFlow model which will be saved to a checkpoints folder

```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 
```

### Adding models
create a folder with the name model and download your keras facenet model that can be found [here](https://github.com/davidsandberg/facenet) and save it in model/keras/

finaly add yor HAAR Cascade classifier to model/cv2/


## license
```
All the work found in the yolo_deepsort folder is licensed under the GPL-3.0 license found in that folder
Any other work found in this repo is licensed under the MIT license
```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This work contains peaces of code borrowed from the following repositories

* []() [yolov4-deepsort](https://github.com/lab176344/yolov4-deepsort)
* []() [PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook)
* []() [keras-facenet](https://github.com/nyoki-mtl/keras-facenet)
