# Genre-Classify
The term project of NYCU MLP class 2023-12

Classifying music genre by MFCC and deep learning. 
"genre original"/ "genres_original_debug": storing music data in .wav(ex: ./genre original/training/blues/.wav)
"main1.py": load wav and extract training and testing feautures and lables to "features" in 4 .npy files
"main2.py": from "features" import features and do deep learning classifying
"model": saving the trained model file

Package suggestion:
# Name                    Version                   Build  Channel
_tflow_select             2.2.0                     eigen
absl-py                   0.15.0             pyhd3eb1b0_0
appdirs                   1.4.4                    pypi_0    pypi
astor                     0.8.1                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
audioread                 3.0.0                    pypi_0    pypi
blas                      1.0                         mkl
ca-certificates           2023.05.30           haa95532_0
cached-property           1.5.2                    pypi_0    pypi
cachetools                4.2.4                    pypi_0    pypi
certifi                   2021.5.30        py36haa95532_0
cffi                      1.15.1                   pypi_0    pypi
charset-normalizer        2.0.12                   pypi_0    pypi
colorama                  0.4.5                    pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
cython                    3.0.0                    pypi_0    pypi
dataclasses               0.8                      pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
flatbuffers               1.12                     pypi_0    pypi
gast                      0.2.2                    pypi_0    pypi
google-auth               2.22.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0              pyhd3eb1b0_0
grpcio                    1.32.0                   pypi_0    pypi
h5py                      3.1.0                    pypi_0    pypi
hdf5                      1.10.4               h7ebc959_0
icc_rt                    2022.1.0             h6049295_2
idna                      3.4                      pypi_0    pypi
importlib-metadata        4.8.3                    pypi_0    pypi
importlib-resources       5.4.0                    pypi_0    pypi
intel-openmp              2023.1.0         h59b6b97_46319
joblib                    1.1.1                    pypi_0    pypi
keras                     2.2.4                         0
keras-applications        1.0.8                      py_1
keras-base                2.2.4                    py36_0
keras-preprocessing       1.1.2              pyhd3eb1b0_0
kiwisolver                1.3.1                    pypi_0    pypi
libprotobuf               3.17.2               h23ce68f_1
librosa                   0.9.2                    pypi_0    pypi
llvmlite                  0.36.0                   pypi_0    pypi
markdown                  3.3.7                    pypi_0    pypi
matplotlib                3.3.4                    pypi_0    pypi
mkl                       2020.2                      256
mkl-service               2.3.0            py36h196d8e1_0
mkl_fft                   1.3.0            py36h46781fe_0
mkl_random                1.1.1            py36h47e9c7a_0
natsort                   8.2.0                    pypi_0    pypi
numba                     0.53.1                   pypi_0    pypi
numpy                     1.19.5                   pypi_0    pypi
numpy-base                1.19.2           py36ha3acd2a_0
oauthlib                  3.2.2                    pypi_0    pypi
openssl                   1.1.1v               h2bbff1b_0
opt_einsum                3.3.0              pyhd3eb1b0_1
packaging                 21.3                     pypi_0    pypi
pandas                    1.1.5                    pypi_0    pypi
pillow                    8.4.0                    pypi_0    pypi
pip                       21.2.2           py36haa95532_0
pooch                     1.6.0                    pypi_0    pypi
protobuf                  3.19.6                   pypi_0    pypi
pyasn1                    0.5.0                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pycparser                 2.21                     pypi_0    pypi
pyparsing                 3.1.1                    pypi_0    pypi
pyreadline                2.1                      py36_1
python                    3.6.13               h3758d61_0
python-dateutil           2.8.2                    pypi_0    pypi
pytz                      2023.3.post1             pypi_0    pypi
pyworld                   0.3.4                    pypi_0    pypi
pyyaml                    5.4.1            py36h2bbff1b_1
requests                  2.27.1                   pypi_0    pypi
requests-oauthlib         1.3.1                    pypi_0    pypi
resampy                   0.4.2                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
scikit-learn              0.24.2                   pypi_0    pypi
scipy                     1.5.2            py36h9439919_0
seaborn                   0.11.2                   pypi_0    pypi
setuptools                58.0.4           py36haa95532_0
six                       1.15.0                   pypi_0    pypi
soundfile                 0.12.1                   pypi_0    pypi
sqlite                    3.41.2               h2bbff1b_0
tensorboard               1.12.2                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow                1.15.0          eigen_py36h932cce6_0
tensorflow-estimator      1.15.1                   pypi_0    pypi
tensorflow-gpu            1.12.0                   pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
threadpoolctl             3.1.0                    pypi_0    pypi
tqdm                      4.64.1                   pypi_0    pypi
typing-extensions         3.7.4.3                  pypi_0    pypi
urllib3                   1.26.16                  pypi_0    pypi
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
werkzeug                  2.0.3                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0
wincertstore              0.2              py36h7fe50ca_0
wrapt                     1.12.1           py36he774522_1
yaml                      0.2.5                he774522_0
zipp                      3.6.0                    pypi_0    pypi
zlib                      1.2.13               h8cc25b3_0


Reference: 
https://www.kaggle.com/code/jvedarutvija/music-genre-classification
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
