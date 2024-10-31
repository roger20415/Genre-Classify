# Genre-Classify
>The term project of NYCU MLP class 2023-12  
  
![Static Badge](https://img.shields.io/badge/Deep%20Learning-gray)
![Static Badge](https://img.shields.io/badge/Tensorflow-blue)



## Discription
>This work is to classify music data in to different genres by MFCC and Deep learning

## Folders and files direction
* "genre original" folder and "genres_original_debug" folder
  * storing music data in .wav
> _Note that there should be "training" and "testing" folders under "genre original" folder, and genre folders under "training" and "testing" folders_
* "main1.py" files
  * load wav and extract training and testing feautures and lables to "features" in 4 .npy files
* "main2.py" files
  * from "features" import features and do deep learning classifying
* "model" files
  * saving the trained model file
<br>

## Operating guide
### 1. Put the music data in corressponding folders correctly
>Follow _Folders and files direction_ section  
  
### 2. Run main1.py  
_`Note that there are some parameters have to revise according to your project，such as frame_size, frame_shift, data path`_  
**main1.py will**  
>* Load files path from folders  
>* Read time domain digital data from file paths
>* Cut the digital seqence by frame_size and frame shift, extract MFCC features from digital data in frames  
, and combine the freqency domain data into _(n * width, 256)_ ndarray shape
>*  Normalize the data and reshape the data into shape _(n, width, 256, 1)_
>*  Save the ndarray features and labels into .npy files in "feature" folders

### 2. Run main2.py  
_`Note that there are some parameters have to revise according to your project`_  
**main2.py will**  
>* Load features and labels from "feature" folder
>* Train the model
>* Save the trained model in "model" folder
>* Test the model
>* Generate classification report and confusion matrix

## Reference 
https://www.kaggle.com/code/jvedarutvija/music-genre-classification  
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
