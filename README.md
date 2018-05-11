# Machine Learning In Practice

## How to Run

-   Download ```model``` directory from github (e.g. to ```C:/python/mip/model/```). Edit ```train.py``` to change the model.

-   Install Google Cloud SDK (https://cloud.google.com/sdk/docs/)

-   Change ```job_name``` and run the following command in the command prompt:

   ```
   gcloud ml-engine jobs submit training job_name --stream-logs --runtime-version 1.4 --job-dir gs://mmndpm-europe-west --package-path C:/python/mip/model/trainer --module-name trainer.train --region europe-west1 --config C:/python/mip/model/trainer/config.yaml -- --train-file gs://mmndpm-europe-west/
  ```
  
-   View the list of jobs here: https://console.cloud.google.com/mlengine/jobs?project=mlip-team-mmndpm
-   <b> Reading and writing files on the google bucket only works with ```file_io``` from ```tensorflow.python.lib.io```</b> 

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
