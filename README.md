# Machine Learning In Practice

# How to Run

-   Download ```model``` directory (e.g. to ```C:/python/mip/model/```). Edit ```train.py``` to change the model.

-   Install Google Cloud SDK (https://cloud.google.com/sdk/docs/)

-   Change ```job_name``` and run the following command in the command prompt:

   ```
   gcloud ml-engine jobs submit training job_name --stream-logs --runtime-version 1.4 --job-dir gs://mmndpm-europe-west --package-path C:/python/mip/model/trainer --module-name trainer.train --region europe-west1 --config C:/python/mip/model/trainer/config.yaml -- --train-file gs://mmndpm-europe-west/
  ```
  
-   View the list of jobs here: https://console.cloud.google.com/mlengine/jobs?project=mlip-team-mmndpm
