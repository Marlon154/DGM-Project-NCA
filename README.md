# DGM-Project-NCA
For the course DGM at TU Darmstadt



## Download images

To download the images, you can use the script `download_images.py` in the `src` folder.
The script requires a file with the urls of the images to download.
The ca{{ should have the following format:

```
python src/download_images.py bloodmnist --size 28 --split test --index 0
```

Possible datasets:
'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist', 'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d'