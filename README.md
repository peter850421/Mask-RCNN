## Step 1 Data Preprocessing ##

Data cleaner can remove the trash img out of dataset. It trains on the resnet50 by using 3994 imgs(positive imgs and negative imgs).
Validation imgs includes 238 imgs(positive imgs and negative imgs). The accuracy in validation set is 99.8%.

### Data Preprocessing - Training and Valid ###
    python train.py
### Data Preprocessing - Testing(Cleaning) ###
	python predict.py --dir=高架

### Data Preprocessing Data_path ###
Put the data just like below.

```bash
|-- train.py
|-- model.py
|-- predict.py
|-- data_utils.py
|-- best_model
	|--Classify
    		|-- model.pth
|-- data(image)
    |-- A
    |-- B
    |-- C
    |-- D
    |-- ....

```
---

## Step2 Data annotation ##
### [Label me](https://github.com/wkentaro/labelme) ###
<p align="center"><img src="https://github.com/peter850421/Mask-RCNN/blob/master/img/labelme.PNG"/></p>

When we finish the annotation from labelme, we next transform the format to COCO format by the [command](https://github.com/wkentaro/labelme/tree/master/examples/instance_segmentation ) . But I suggest the other way to transform the format.
Put this [file](https://github.com/lindylin1817/labelme2coco/blob/master/labelme2COCO.py )  in to the image annotation folder, then try below: 

	 python labelme2coco.py
	 
It will generate a new.json that is the annotation file of coco format. Next, we change the file name to train.json. Validation dataset repeat the above step to get the coco format.
Now we have the dataset below:

```bash
|-- dataset
	|-- train
		|--0.jpg
		|--1.jpg
		|-- ...
	|-- val
		|--1000.jpg
		|--1001.jpg
		|-- ...
	|--annotation
		|--train.json
		|--val.json
```

---
## Step3 Installation ##
Follow the installation in [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN#installation)

## Step4 Training and Evaluation ##
Training
```bash
python ./train201908.py --training_dataset train --validation_dataset valid --year 201908 train
```
Evaluate
```bash
python ./train201908.py --training_dataset train --validation_dataset valid --year 201908 evaluate
```
also see [train_valid_visual](https://github.com/peter850421/Mask-RCNN/blob/master/train/train_evaluate_visualize.ipynb)

coco_eval replaced by [cocoeval.py](https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/tools/cocoeval.py) (remember to change the class number)
## TO DO LIST

- [X] Data Clearer
- [x] Annotation
- [X] Traingin MASK RCNN
- [X] Visualize Result

