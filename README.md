## numerai-gcp

Building models for Numerai tournament using GCP

Some notes:
  - Script for building a model is put in a Docker container with a click-based CLI that has, at minimum, `train` and `inference` commands
  - Each model will have different sets of parameters, each representing a run.  Each run will have its own saved model (.pkl) and metrics.
  - Will use Google Cloud Batch to run batch jobs that will build models for each set of parameters.
  - Data will be saved to Cloud Storage (see file structure below)


```
data/
  datasets/
    <version>/
      *.parquet
  artifacts/
    <model_id>/
      <run_id>/
        params.json
        metrics.json
        model.pkl
```
