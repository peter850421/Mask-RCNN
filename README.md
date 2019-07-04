# Data Preprocessing #

Data cleaner can remove the trash img out of dataset. It trains on the resnet50 by using 3994 imgs(positive imgs and negative imgs).
Validation imgs includes 238 imgs(positive imgs and negative imgs). The accuracy in validation set is 99.8%.

## Training and Valid ##
    python train.py
## Testing(Cleaning) ##
	python predict.py --dir=高架

### Data_path
Put the data just like below.

```bash
|-- train.py
|-- model.py
|-- predict.py
|-- data_utils.py
|-- best_model
	|--Classify
    	|-- model.pth
|-- data(高風險照片)
    |-- A
    |-- B
    |-- C
    |-- D
    |-- ....

```