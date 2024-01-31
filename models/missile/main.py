import os, gc, re, json
from typing import List

import click
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI
from lightgbm import LGBMRegressor

from utils import (
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL,
    META_MODEL_COL
)

from numerai_tools.scoring import neutralize


MODEL_ID = 'missile'


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
    ctx.obj['DATA_DIR'] = data_dir

    if test:
        run_id = "test"

    # this is where we'll store model, params, and metrics
    ctx.obj['MODEL_RUN_PATH'] = os.path.join(data_dir, 'artifacts', MODEL_ID, str(run_id))
    os.makedirs(ctx.obj['MODEL_RUN_PATH'], exist_ok=True)

    with open(os.path.join('params', f'{run_id}.json'), 'r') as f:
        ctx.obj['PARAMS'] = json.load(f)

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
        'train': make_path('train_int8.parquet'),
        'validation': make_path('validation_int8.parquet'),
        'validation_example_preds': make_path('validation_example_preds.parquet'),
        'features': make_path('features.json'),
        'meta_model': make_path('meta_model.parquet')
    }

    # this is where we'll save our submissions
    submission_dir = os.path.join(data_dir, 'submissions', MODEL_ID, str(run_id))
    os.makedirs(os.path.join(submission_dir, dataset_version), exist_ok=True)
    ctx.obj['SUBMISSION_PATH'] = os.path.join(submission_dir, f"live_predictions_{current_round}.csv")
    ctx.obj['DATASETS']['live'] = os.path.join(submission_dir, dataset_version, f'live_int8_{current_round}.parquet')


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
def download_datasets(ctx):
    for dataset in ctx.obj['DATASETS'].keys():
        ctx.invoke(download, dataset=dataset)


@cli.command()
@click.pass_context
def download_datasets_all(ctx):
    if int(os.environ.get('BATCH_TASK_INDEX', -1)) != 0:
        print("Skipping download because not first task")
        return

    with open('params.json', 'r') as f:
        params = json.load(f)
    
    dataset_versions = set()
    for p in params:
        dataset_versions.add(p['dataset_params']['version'])

    ctx.obj['DATASETS'] = {}
    for dv in dataset_versions:
        make_path = lambda x: os.path.join(ctx.obj['DATA_DIR'], 'datasets', dv, x)
        ctx.obj['DATASETS'][f'train_{dv}'] = make_path('train_int8.parquet')
        ctx.obj['DATASETS'][f'validation_{dv}'] = make_path('validation_int8.parquet')
        ctx.obj['DATASETS'][f'validation_example_preds_{dv}'] = make_path('validation_example_preds.parquet')
        ctx.obj['DATASETS'][f'meta_model_{dv}'] = make_path('meta_model.parquet')
        ctx.obj['DATASETS'][f'features_{dv}'] = make_path('features.json')

    for dataset in ctx.obj['DATASETS'].keys():
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

    # get targets
    try:
        targets = feature_metadata["targets"]
    except KeyError:
        schema = pq.read_schema(ctx.obj['DATASETS']['train'], memory_map=True)
        targets = [t for t in schema.names if t.startswith('target_')]

    read_columns = features + targets + [ERA_COL, DATA_TYPE_COL]
    return read_columns


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return list(df.filter(like='feature_').columns)


def get_fnc_features(ctx: click.Context, features: List[str]) -> List[str]:
    # medium feature set for fncv3
    with open(ctx.obj['DATASETS']['features'], "r") as f:
        feature_metadata = json.load(f)

    fncv3_features = feature_metadata["feature_sets"]['fncv3_features']
    fncv3_features = list(set(fncv3_features).intersection(set(features)))
    return fncv3_features


def load_training_data(ctx: click.Context) -> (pd.DataFrame, pd.Index, pd.Index):
    read_columns = get_read_columns(ctx)

    # note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
    # if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
    training_data = pd.read_parquet(ctx.obj['DATASETS']['train'], columns=read_columns)
    validation_data = pd.read_parquet(ctx.obj['DATASETS']['validation'], columns=read_columns)

    validation_preds = pd.read_parquet(ctx.obj['DATASETS']['validation_example_preds'])
    validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]

    meta_model = pd.read_parquet(ctx.obj['DATASETS']['meta_model'])
    validation_data[META_MODEL_COL] = meta_model["numerai_meta_model"]

    # list of features
    features = get_feature_columns(training_data)

    # reduce the number of eras to every 4th era to speed things up
    if ctx.obj['TEST']:
        every_4th_era = training_data[ERA_COL].unique()[::4]
        training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]
        every_4th_era = validation_data[ERA_COL].unique()[::4]
        validation_data = validation_data[validation_data[ERA_COL].isin(every_4th_era)]

    # get all the data to possibly use for training
    all_data = pd.concat([training_data, validation_data])

    training_index = training_data.index
    validation_index = validation_data.index

    del training_data
    del validation_data
    gc.collect()

    # fill in NAs
    na_counts = all_data[features].isna().sum()
    na_impute = all_data[features].median(skipna=True).to_dict()
    if na_counts.sum() > 0:
        print("Cleaning up NAs")
        # na_impute are floats so the .fillna will cast to float which blows up memory
        # recast to int8 afterwards but this will cause a memory spike
        all_data[features] = all_data[features].fillna(na_impute)
        all_data[features] = all_data[features].astype("int8")

    return (all_data, training_index, validation_index, na_impute)


def load_live_data(ctx: click.Context) -> pd.DataFrame:
    read_columns = get_read_columns(ctx)
    return pd.read_parquet(ctx.obj['DATASETS']['live'], columns=read_columns)


def load_model(ctx: click.Context, model_key: str):
    path = os.path.join(ctx.obj['MODEL_RUN_PATH'], 'models', f'{model_key}.pkl')
    if os.path.exists(path) and not ctx.obj['OVERWRITE']:
        model = pd.read_pickle(path)
    else:
        model = False
    return model


def load_model_config(ctx: click.Context):
    path = os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json')
    if os.path.exists(path) and not ctx.obj['OVERWRITE']:
        with open(path, 'r') as f:
            model_config = json.load(f)
    else:
        model_config = False
    return model_config


def neutralize_target(df: pd.DataFrame, neutralizers: List[str], proportion: float, target_col: str) -> pd.Series:
    return (
        df
        .groupby(ERA_COL, group_keys=True)
        .apply(
            lambda d: neutralize(
                d[[target_col]],
                d[neutralizers],
                proportion=proportion
            )
        )
        .reset_index()
        .set_index("id")
        [target_col]
    )


def train_model(ctx: click.Context, model_key: str, target_col: str, params: dict,
                all_data: pd.DataFrame, training_index: pd.Index, validation_index: pd.Index = None) :
    model = load_model(ctx, model_key)
    if model:
        print(f"{model_key} model has already been trained")
    else:
        print(f"Training {model_key} model")

        target_train_index = all_data.loc[training_index, target_col].dropna().index
        features = list(all_data.filter(like='feature_').columns)

        model = LGBMRegressor(**params)
        model.fit(
            all_data.loc[target_train_index, features],
            all_data.loc[target_train_index, target_col]
        )

        os.makedirs(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'models'), exist_ok=True)
        out = os.path.join(ctx.obj['MODEL_RUN_PATH'], 'models', f'{model_key}.pkl')
        pd.to_pickle(model, out)

    if validation_index is not None:
        print(f"Adding predictions for {target_col} to validation data")
        model_expected_features = model.booster_.feature_name()
        all_data.loc[validation_index, "pred"] = model.predict(
            all_data.loc[validation_index, model_expected_features])

    del model
    gc.collect()


@cli.command()
@click.pass_context
def train(ctx):
    # try loading model config to see if we've already trained
    model_config = load_model_config(ctx)
    if model_config:
        print("Model has already been trained")
        return

    # load data
    print("Loading data")
    all_data, training_index, validation_index, na_impute = load_training_data(ctx)
    features = get_feature_columns(all_data)
    fnc_features = get_fnc_features(ctx, features)

    # neutralize target
    all_data["target_neutral"] = neutralize_target(
        df=all_data,
        neutralizers=fnc_features,
        proportion=1.0,
        target_col=ctx.obj['PARAMS']['target_params']['target_col']
    )
    gc.collect()

    # train model
    train_model(
        ctx,
        model_key='train',
        target_col='target_neutral',
        params=ctx.obj['PARAMS']['model_params'],
        all_data=all_data,
        training_index=training_index,
        validation_index=validation_index
    )

    # calculate metrics
    print("Calculating metrics on validation data")
    pred_cols = [EXAMPLE_PREDS_COL, "pred"]
    validation_stats = validation_metrics(
        all_data.loc[validation_index, :], 
        pred_cols=pred_cols, example_col=EXAMPLE_PREDS_COL, target_col=TARGET_COL,
        features_for_neutralization=fnc_features
    )
    validation_stats['last_era'] = all_data[ERA_COL].max()
    print(validation_stats[["mean", "sharpe", "max_drawdown", "feature_neutral_mean", "mmc_mean"]].to_markdown())

    gc.collect()

    # train final model
    if not ctx.obj['TEST']:
        train_model(
            ctx,
            model_key='all_data',
            target_col='target_neutral',
            params=ctx.obj['PARAMS']['model_params'],
            all_data=all_data,
            training_index=all_data.index
        )

    # final model config
    model_config = {
        "model_key": "train" if ctx.obj['TEST'] else "all_data",
        "na_impute": na_impute,
    }

    # save params, metrics, and configuration
    print("Saving final config, parameters, and metrics")
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'params.json'), 'w') as f:
        json.dump(ctx.obj['PARAMS'], f, indent=4, separators=(',', ': '))
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'metrics.json'), 'w') as f:
        f.write(validation_stats.to_json(orient='index', indent=4))
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=4, separators=(',', ': '))


@cli.command()
@click.pass_context
def refresh_metrics(ctx):
    # load data
    print("Loading data")
    all_data, training_index, validation_index, na_impute = load_training_data(ctx)
    features = get_feature_columns(all_data)
    fnc_features = get_fnc_features(ctx, features)

    # add OOS predictions for validation
    # NOTE: ignore overwrite
    print("Adding predictions to validation data")
    ctx.obj["OVERWRITE"] = False
    model = load_model(ctx, "train")
    assert model

    model_expected_features = model.booster_.feature_name()
    all_data.loc[validation_index, "pred"] = model.predict(
        all_data.loc[validation_index, model_expected_features])

    del model
    gc.collect()

    # calculate metrics
    print("Calculating metrics on validation data")
    pred_cols = [EXAMPLE_PREDS_COL, "pred"]
    validation_stats = validation_metrics(
        all_data.loc[validation_index, :],
        pred_cols=pred_cols, example_col=EXAMPLE_PREDS_COL, target_col=TARGET_COL,
        features_for_neutralization=fnc_features
    )
    validation_stats['last_era'] = all_data[ERA_COL].max()
    print(validation_stats[["mean", "sharpe", "max_drawdown", "feature_neutral_mean", "mmc_mean"]].to_markdown())

    gc.collect()

    # save params, metrics, and configuration
    print("Saving metrics")
    with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'metrics.json'), 'w') as f:
        f.write(validation_stats.to_json(orient='index', indent=4))


@cli.command()
@click.option('--numerai-model-name', default=None)
@click.pass_context
def inference(ctx, numerai_model_name):
    # load model config
    print("Loading model config")
    try:
        with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json'), 'r') as f:
            model_config = json.load(f)
    except FileNotFoundError:
        print("Trained model(s) not found!")
        raise

    # load data
    print("Loading live data")
    ctx.invoke(download, dataset='live')
    live_data = load_live_data(ctx)
    features = list(live_data.filter(like='feature_').columns)

    # fill in NAs
    print("Cleaning up NAs")
    live_data[features] = live_data[features].fillna(model_config['na_impute'])
    live_data[features] = live_data[features].astype("int8")

    # generate predictions
    print("Generating predictions")
    model = load_model(ctx, model_config['model_key'])
    model_expected_features = model.booster_.feature_name()
    assert set(model_expected_features) == set(features)
    live_data.loc[:, 'pred'] = model.predict(
        live_data.loc[:, model_expected_features])
    gc.collect()

    # export
    print("Exporting predictions")
    live_data["prediction"] = live_data['pred'].rank(pct=True)
    live_data["prediction"].to_csv(ctx.obj['SUBMISSION_PATH'])

    # upload predictions
    if numerai_model_name is not None:
        print("Uploading predictions to Numerai")
        napi = NumerAPI()
        model_id = napi.get_models()[numerai_model_name]
        napi.upload_predictions(ctx.obj['SUBMISSION_PATH'], model_id=model_id)


if __name__ == '__main__':
    cli(obj={})
