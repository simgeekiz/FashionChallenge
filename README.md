# Machine Learning In Practice

## How to Run 2: VM

- Start the VM: https://console.cloud.google.com/compute/instances?project=mlip-team-mmndpm
- SSH into VM: ```gcloud compute --project "mlip-team-mmndpm" ssh --zone "europe-west1-b" "fashion"```
- ```sudo bash``` to get root
- ```export LD_LIBRARY_PATH=/usr/local/cuda/lib64``` (else cudnn is not found)
- ```export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64``` (else cudnn is not found)
- ^ There is supposed to be a way in linux to save exports and not do this every time
- ```cd /home/dljva/``` this is where the current files are stored (you need to run the notebook from here)
- ```jupyter-notebook --no-browser --port=5000 --allow-root``` to run jupyter notebook
- Navigate to http://35.195.202.233:5000 (password: fashion)
- Stop VM after use (https://console.cloud.google.com/compute/instances?project=mlip-team-mmndpm)
  

## How to Run

-   Download ```model``` directory from github (e.g. to ```C:/python/mip/model/```). Edit ```train.py``` to change the model.

-   Install Google Cloud SDK (https://cloud.google.com/sdk/docs/)

-   Change ```job_name``` and run the following command in the command prompt:

   ```
   gcloud ml-engine jobs submit training job_name --stream-logs --runtime-version 1.4 --job-dir gs://mmndpm-europe-west --package-path C:/python/mip/model/trainer --module-name trainer.train --region europe-west1 --config C:/python/mip/model/trainer/config.yaml -- --train-file gs://mmndpm-europe-west/
  ```
  
-   View the list of jobs here: https://console.cloud.google.com/mlengine/jobs?project=mlip-team-mmndpm
-   <b> Reading and writing files on the google bucket only works with ```file_io``` from ```tensorflow.python.lib.io```</b> 
-   We are using python 2

## Google Bucket structure
```gs://mmndpm-europe-west/```

```- data```

```-- train``` directory with separate training image (jpg) files

```-- test``` directory with separate test image (jpg) files

```-- validation``` directory with separate validation image (jpg) files

```-- train.json``` contains image URLS and label annotiations for training set

```-- test.json``` contains image URLS and label annotiations for test set

```-- validation.json``` contains image URLS and label annotiations for validation set

```-- y_train.pickle``` gzip compressed pickle file with one-hot encoded labels (column 0 is imageId, columns 1-229 indicate label presence)

```-- y_validation.pickle``` gzip compressed pickle file with one-hot encoded labels (column 0 is imageId, columns 1-229 indicate label presence)
