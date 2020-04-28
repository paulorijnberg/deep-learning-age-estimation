# Age estimation based on facial images using Deep Learning techniques
Hi, 

Welcome to my first repo that I created using deep learning techniques (especially CNNs). This GitHub repo explores the possibilities of using deep learning techniques on facial images to estimate someones age. 

The repo consists of the following folders:
* `final-paper`: the final paper that summarizes the results of the experience and was submitted for the Deep Learning course.
* `literature-research`: the collection of papers used in the process of doing the experiments and writing the final paper.
* `notebooks`: collection of notebooks, modules and datasets.

Because the folder `notebook` is such a deeply nested folders, I want to also give some more in dept information about it.

<br><br>

**Folder: notebooks**
- Modules: 
  - `general_functions.py`: This module contains selfmade functions such as e.g. plotting results, confusion matrix, displaying image activations, etc.
  - `dependencies.py`: Module that containts all the libaries.

<br>

- Notebooks: the most important notebooks of the experiment are directly visible when you go into deep-learning-age-estimation/notebooks. 
  - `summarized-results.ipynb`: this notebook shows all the results for the best X performing models.
  - `new-structure-dataset-UTKface_croppped-balanced.ipynb`: module transforms the original `UTKFace_cropped` dataset into a new folder structure. It creates 3 folders (training, validation, test) and applies undersampling. When you place the original images of in the folder `deep-learning-age-estimation/notebooks/UTKFace_cropped`, create folder `deep-learning-age-estimation/notebooks/UTKface_cropped-new-structure-balanced` and run all the cells from notebook `new-structure-dataset-UTKface_croppped-balanced.ipynb`, you apply all those mentioned steps.
  - `neural-networks-UTKface_cropped-balanced.ipynb`: this notebook uses the folder `UTKface_cropped-new-structure-balanced` to train multiple neural networks. This notebook is the collection of multiple approaches to our problem. Each model has its own chapter (e.g. 1.0 is LeNet-5, 2.0 is proposed LeNet-5 from Sithungu and Van der Haar, etc.) When we for example try to tweak model 1.0, we do that under chapter 1.1, etc.
  - `neural-networks-resnet-UTKface_cropped-balanced.ipynb`: this notebook also uses the folder `UTKface_cropped-new-structure-balanced` to train multiple neural networks, but in this case we only focus in this notebook on neural networks with the ResNet architecture.

<br>

- folder `models-and-results-UTKface_cropped-balanced`: we used this folder to save the model, results of each epoch and intermediate results when training the model (checkpoints).

<br>

- folder `datasets`: here the user can insert the datasets that will be transformed when running the `new-structure-dataset-UTKface...` scripts.

<br>

- folder `older-experiments`: this folder contains the results/notebooks of the previously done experiments. The final paper discusses the results of using a the balanced cropped images from the UTKFace dataset, but I also experimented using inbalanced and/or 'in the wild' images. If people are curious, you can take a look at those results.
