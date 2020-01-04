# ESLR Recruitment Project 2019

## Resources

Subject: https://drive.google.com/open?id=1VV6E7x1FfE8PHIY0-B6TA3uWLK6knjhv  
Dataset: https://drive.google.com/open?id=1e6X56yLlmXzaeavJ-Nn8oG2kEtCHnC4T

## k-means

### Usage

To launch the kmeans please type the folowing commands:
```
cd kmeans
make
./kmeans 3 20 1.0 2351 200000 /afs/cri/resources/teach/LSE-IA/ember/Xtest.dat out.dat
```

To check the accuracy (if you are in kmeans directory):
```
cd ../python
pip install -r requirements.txt --user
python eval.py ../out.dat  /afs/cri/resources/teach/LSE-IA/ember/Ytrain.dat
```

## Classifiers and neural networks

### Usage

Jupyter notebooks and .py source files are located in specific subfolders. However, we suggest using `Google Colab` to run our experiments.

### Step 2: Classification with machine learning

- Path: `python/step2-classifier_machine_learning`
- Google Colab: []()

### Step 3: Classification using Deep Neural Network

- Path: `python/step3-deep_neural_network`
- Google Colab: []()

## Bonus

### Usage

Jupyter notebooks and .py source files are located in specific subfolders. However, we suggest using `Google Colab` to run our experiments.

### Metric Learning: Siamese network

- Path: `python/bonus_siamese_network`
- Google Colab: []()
