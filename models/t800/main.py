import os, gc, re, json
from typing import List

import click
import pandas as pd
from numerapi import NumerAPI
from lightgbm import LGBMRegressor

from utils import (
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL
)

MODEL_ID = 't800'


@click.group()
@click.option('--run-id', default=os.environ.get('BATCH_TASK_INDEX', 0))
@click.option('--data-dir', default="data")
@click.option('--test/--no-test', default=False)
@click.option('--overwrite/--no-overwrite', default=False)
@click.pass_context
def cli(ctx, run_id, data_dir, test, overwrite):
    ctx.ensure_object(dict)
    ctx.obj['RUN_ID'] = int(run_id)
    ctx.obj['TEST'] = test
    ctx.obj['OVERWRITE'] = overwrite

    if test:
        run_id = "test"

    # this is where we'll store model, params, and metrics
    ctx.obj['MODEL_RUN_PATH'] = os.path.join(data_dir, 'artifacts', MODEL_ID, str(run_id))
    os.makedirs(ctx.obj['MODEL_RUN_PATH'], exist_ok=True)

    if test:
        with open('params_test.json', 'r') as f:
            ctx.obj['PARAMS'] = json.load(f)
    else:
        with open('params.json', 'r') as f:
            ctx.obj['PARAMS'] = json.load(f)[int(run_id)]

    # create paths to input datasets
    dataset_version = ctx.obj['PARAMS']['dataset_params']['version']
    os.makedirs(os.path.join(data_dir, 'datasets', dataset_version), exist_ok=True)

    napi = NumerAPI()
    try:
        current_round = napi.get_current_round()
    except ValueError:
        # in case current round not open for submissions
        current_round = 'na'
    make_path = lambda x: os.path.join(data_dir, 'datasets', dataset_version, x)
    ctx.obj['DATASETS'] = {
        'train': make_path('train.parquet'),
        'validation': make_path('validation.parquet'),
        'live': make_path(f'live_{current_round}.parquet'),
        'validation_example_preds': make_path('validation_example_preds.parquet'),
        'features': make_path('features.json')
    }

    # this is where we'll save our submissions
    submission_dir = os.path.join(data_dir, 'submissions', MODEL_ID, str(run_id))
    os.makedirs(submission_dir, exist_ok=True)
    ctx.obj['SUBMISSION_PATH'] = os.path.join(submission_dir, f"live_predictions_{current_round}.csv")


@cli.command()
@click.option('--dataset')
@click.pass_context
def download(ctx, dataset):
    # return <version>/<dataset>.parquet from path
    def get_api_dataset_from_path(path: str) -> str:
        out = os.path.join(*path.split('/')[-2:])
        return re.sub('_[0-9]+\.parquet$', '.parquet', out)

    out = ctx.obj['DATASETS'][dataset]
    api_dataset = get_api_dataset_from_path(out)

    if os.path.exists(out) and not ctx.obj['OVERWRITE']:
        print(f"{api_dataset} already exists!")
    else:
        print(f'Downloading {api_dataset}...')
        try:
            os.remove(out)
        except FileNotFoundError:
            pass
        napi = NumerAPI()
        napi.download_dataset(api_dataset, out)


@cli.command()
@click.pass_context
def download_all(ctx):
    for dataset in ctx.obj['DATASETS'].keys():
        ctx.invoke(download, dataset=dataset)


@cli.command()
@click.pass_context
def download_for_training(ctx):
    if ctx.obj['RUN_ID'] != 0:
        print("Skipping download because not first task")
        return
    for dataset in ctx.obj['DATASETS'].keys():
        if dataset == 'live':
            continue
        ctx.invoke(download, dataset=dataset)


def get_read_columns(ctx: click.Context) -> List[str]:
    with open(ctx.obj['DATASETS']['features'], "r") as f:
        feature_metadata = json.load(f)
    try:
        feature_set = ctx.obj['PARAMS']['dataset_params']['feature_set']
        features = feature_metadata["feature_sets"][feature_set]
        print(f"Using feature set: {feature_set}")
    except KeyError:
        features = list(feature_metadata["feature_stats"].keys())

    print(f"Number of features: {len(features)}")
    read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
    return read_columns


def load_training_data(ctx: click.Context) -> (pd.DataFrame, pd.DataFrame):
    read_columns = get_read_columns(ctx)

    # note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
    # if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
    training_data = pd.read_parquet(ctx.obj['DATASETS']['train'], columns=read_columns)
    validation_data = pd.read_parquet(ctx.obj['DATASETS']['validation'], columns=read_columns)

    validation_preds = pd.read_parquet(ctx.obj['DATASETS']['validation_example_preds'])
    validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]

    return (training_data, validation_data)


def load_live_data(ctx: click.Context) -> pd.DataFrame:
    read_columns = get_read_columns(ctx)
    return pd.read_parquet(ctx.obj['DATASETS']['live'], columns=read_columns)


def load_model(ctx: click.Context):
    path = os.path.join(ctx.obj['MODEL_RUN_PATH'], 'model.pkl')
    if os.path.exists(path) and not ctx.obj['OVERWRITE']:
        model = pd.read_pickle(path)
    else:
        model = False
    return model


@cli.command()
@click.pass_context
def train(ctx):
    # determine if model has already been trained
    model = load_model(ctx)
    if model:
        print("Model has already been trained")
        return

    # load data
    print("Loading data")
    training_data, validation_data = load_training_data(ctx)
    features = list(training_data.filter(like='feature_').columns)

    # reduce the number of eras to every 4th era to speed things up
    if ctx.obj['TEST']:
        every_4th_era = training_data[ERA_COL].unique()[::4]
        training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]
        every_4th_era = validation_data[ERA_COL].unique()[::4]
        validation_data = validation_data[validation_data[ERA_COL].isin(every_4th_era)]

    # train model
    print("Training model")
    params = ctx.obj['PARAMS']['model_params']
    model = LGBMRegressor(**params)
    model.fit(training_data.filter(like='feature_', axis='columns'),
              training_data[TARGET_COL])

    gc.collect()

    # generate predictions on validation
    print("Generating predictions on validation data")
    model_expected_features = model.booster_.feature_name()
    validation_data.loc[:, "pred"] = model.predict(
        validation_data.loc[:, model_expected_features])

    # neutralize our predictions to the riskiest features (biggested change in
    #   corr vs. target between halves of training data)
    print("Determing features to neutralize against")
    all_feature_corrs = training_data.groupby(ERA_COL).apply(
        lambda era: era[features].corrwith(era[TARGET_COL])
    )
    n_neutralize_features = ctx.obj['PARAMS']['neutralize_params']['n_features']
    riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize_features)

    del training_data
    gc.collect()

    print("Neutralizing predictions on validation data")
    validation_data["pred_neutralized"] = neutralize(
        df=validation_data,
        columns=["pred"],
        neutralizers=riskiest_features,
        proportion=1.0,
        normalize=True,
        era_col=ERA_COL
    )

    gc.collect()

    # calculate metrics
    print("Calculating metrics on validation data")
    validation_stats = validation_metrics(
        validation_data, ["pred", "pred_neutralized"], example_col=EXAMPLE_PREDS_COL,
        target_col=TARGET_COL, fast_mode=ctx.obj['TEST']
    )
    print(validation_stats[["mean", "sharpe"]].to_markdown())

    # final model config
    model_config = {
        "neutralize_features": riskiest_features
    }

    # save model, params, and metrics
    print("Saving model, final config, parameters, and metrics")
    pd.to_pickle(model, os.path.join(ctx.obj['MODEL_RUN_PATH'], 'model.pkl'))
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'params.json'), 'w') as f:
        json.dump(ctx.obj['PARAMS'], f, indent=4, separators=(',', ': '))
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'metrics.json'), 'w') as f:
        f.write(validation_stats.to_json(orient='index', indent=4))
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=4, separators=(',', ': '))


@cli.command()
@click.option('--numerai-model-name', default=None)
@click.pass_context
def inference(ctx, numerai_model_name):
    # load model
    print("Loading model")
    model = load_model(ctx)
    if not model:
        print("Trained model not found!")
        raise

    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json'), 'r') as f:
        model_config = json.load(f)

    # load data
    print("Loading live data")
    ctx.invoke(download, dataset='live')
    live_data = load_live_data(ctx)
    features = list(live_data.filter(like='feature_').columns)

    # generate predictions
    print("Generating predictions")
    model_expected_features = model.booster_.feature_name()
    assert set(model_expected_features) == set(features)
    live_data.loc[:, "pred"] = model.predict(
        live_data.loc[:, model_expected_features])

    # neutralize
    print("Neutralizing predictions")
    live_data["pred_neutralized"] = neutralize(
        df=live_data,
        columns=["pred"],
        neutralizers=model_config['neutralize_features'],
        proportion=1.0,
        normalize=True,
        era_col=ERA_COL
    )

    # export
    print("Exporting predictions")
    live_data["prediction"] = live_data['pred_neutralized'].rank(pct=True)
    live_data["prediction"].to_csv(ctx.obj['SUBMISSION_PATH'])

    # upload predictions
    if numerai_model_name is not None:
        print("Uploading predictions to Numerai")
        napi = NumerAPI()
        model_id = napi.get_models()[numerai_model_name]
        napi.upload_predictions(ctx.obj['SUBMISSION_PATH'], model_id=model_id)


if __name__ == '__main__':
    cli(obj={})
