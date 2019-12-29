# ESLR Recruitment Project 2019

## Resources

Subject: https://drive.google.com/open?id=1VV6E7x1FfE8PHIY0-B6TA3uWLK6knjhv
Dataset: https://drive.google.com/open?id=1e6X56yLlmXzaeavJ-Nn8oG2kEtCHnC4T

## Usage

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
