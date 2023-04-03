## numerai-gcp

Building models for Numerai tournament using GCP

Some notes:
  - Script for building a model is put in a Docker container with a click-based CLI that has, at minimum, `train` and `inference` commands
  - Each model will have different sets of parameters, each representing a run.  Each run will have its own saved model (.pkl) and metrics.
  - Any parameters calculated DURING training loop (e.g. features to neutralize, target to use) should be put in `config.json`
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
        config.json
        model.pkl or models/
  submissions/
    <model_id>/
      <run_id>/
        live_predictions_{round}.csv
```

Note: for local testing, create a symbolic link to `data` folder using the following command:
```
ln -s ../../data data
```
