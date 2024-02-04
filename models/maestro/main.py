import os, gc, re, json, glob
from typing import List

import click
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from numerapi import NumerAPI
from lightgbm import LGBMRegressor

from cvxopt import matrix, solvers

from utils import (
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL,
    META_MODEL_COL
)

from numerai_tools.scoring import neutralize, numerai_corr


MODEL_ID = 'maestro'


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
        'validation': make_path('validation_int8.parquet'),
        'validation_example_preds': make_path('validation_example_preds.parquet'),
        'validation_benchmark_models': make_path('validation_benchmark_models.parquet'),
        'features': make_path('features.json')
    }

    # this is where we'll save our submissions
    submission_dir = os.path.join(data_dir, 'submissions', MODEL_ID, str(run_id))
    os.makedirs(os.path.join(submission_dir, dataset_version), exist_ok=True)
    ctx.obj['SUBMISSION_PATH'] = os.path.join(submission_dir, f"live_predictions_{current_round}.csv")
    ctx.obj['DATASETS']['live'] = os.path.join(submission_dir, dataset_version, f'live_benchmark_models_{current_round}.parquet')


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

    params = []
    for p in glob.glob('params/*.json'):
        if os.path.basename(p) == 'test.json':
            continue
        with open(p, 'r') as f:
            params.append(json.load(f))

    dataset_versions = set()
    for p in params:
        dataset_versions.add(p['dataset_params']['version'])

    make_path = lambda x: os.path.join(ctx.obj['DATA_DIR'], 'datasets', dv, x)

    ctx.obj['DATASETS'] = {}
    for dv in dataset_versions:
        ctx.obj['DATASETS'][f'validation_{dv}'] = make_path('validation_int8.parquet')
        ctx.obj['DATASETS'][f'validation_example_preds_{dv}'] = make_path('validation_example_preds.parquet')
        ctx.obj['DATASETS'][f'validation_benchmark_models_{dv}'] = make_path('validation_benchmark_models.parquet')
        ctx.obj['DATASETS'][f'features_{dv}'] = make_path('features.json')

    for dataset in ctx.obj['DATASETS'].keys():
        ctx.invoke(download, dataset=dataset)


def neutralize_by_era(df: pd.DataFrame, fields_to_neuralize: List[str], neutralizers: List[str], proportion: float = 1.0):
    return (
        df
        .groupby(ERA_COL, group_keys=True)
        .apply(
            lambda d: neutralize(
                d[fields_to_neuralize],
                d[neutralizers],
                proportion=proportion
            )
        )
        .reset_index()
        .set_index("id")
    )


def load_training_data(ctx: click.Context) -> (pd.DataFrame, pd.DataFrame):
    # load benchmark models
    benchmark_models = pd.read_parquet(ctx.obj['DATASETS']['validation_benchmark_models']) 

    # reduce the number of eras to every 4th era to speed things up
    if ctx.obj['TEST']:
        every_4th_era = benchmark_models[ERA_COL].unique()[::4]
        benchmark_models = benchmark_models[benchmark_models[ERA_COL].isin(every_4th_era)]

    # identify the most dataset for each target
    def parse_model_name(model_name: str) -> (str, str, str):
        try:
            return re.compile(r'(v[0-9]{1,2})_lgbm_([a-z]+)(20|60)').match(model_name).groups()
        except AttributeError:
            return (None, None, None)

    pred_cols = [c for c in benchmark_models.columns if c.startswith("v4")]
    pred_cols = (
        pd.DataFrame.from_records([parse_model_name(m) for m in pred_cols], columns=['dataset', 'target', 'time'])
        .assign(model_name=pred_cols)
        .dropna()
        .assign(dataset=lambda x: pd.Categorical(x.dataset, categories=['v4', 'v41', 'v42', 'v43']))
        .sort_values(by=['target', 'time', 'dataset'], ascending=[True, True, False])
        .groupby(['target', 'time'])
        .head(1)
    )

    pred_cols_final = pred_cols['model_name'].tolist()

    # pull in targets
    target_col = ctx.obj['PARAMS']['target_params']['target_col']
    targets = pd.read_parquet(ctx.obj['DATASETS']['validation'], columns=[target_col])
    benchmark_models[target_col] = targets[target_col]

    # pull in example predictions
    example_preds = pd.read_parquet(ctx.obj['DATASETS']['validation_example_preds'])
    benchmark_models[EXAMPLE_PREDS_COL] = example_preds["prediction"]

    # drop nas
    benchmark_models = (
        benchmark_models
        .filter([
            ERA_COL,
            EXAMPLE_PREDS_COL,
            target_col,
            *pred_cols_final
        ])
        .dropna()
    )

    # neutralize target by example preds
    corr_col = target_col
    if ctx.obj['PARAMS']['model_params'].get('neutralize_by_example_preds', 'no') == 'yes':
        print("Neutralizing target by example preds")
        benchmark_models['target_neutral'] = neutralize(
            df=benchmark_models[[target_col]],
            neutralizers=benchmark_models[[EXAMPLE_PREDS_COL]]
        )[target_col]
        corr_col = 'target_neutral'
    elif ctx.obj['PARAMS']['model_params'].get('neutralize_by_example_preds', 'no') == 'by_era':
        print("Neutralizing target by example preds within each era")
        benchmark_models['target_neutral'] = neutralize_by_era(
            df=benchmark_models,
            fields_to_neuralize=[target_col],
            neutralizers=[EXAMPLE_PREDS_COL]
        )[target_col]
        corr_col = 'target_neutral'

    # era correlations
    era_corr = (
        benchmark_models
        .groupby(ERA_COL)
        .apply(lambda d: numerai_corr(d[pred_cols_final], d[corr_col]))
    )

    return benchmark_models, era_corr


def load_model_config(ctx: click.Context):
    path = os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json')
    if os.path.exists(path) and not ctx.obj['OVERWRITE']:
        with open(path, 'r') as f:
            model_config = json.load(f)
    else:
        model_config = False
    return model_config


# minimizes variance while achieving a return target
def optimize(target_return, returns, covar, min_weight, max_weight):
    # min (1/2) x^TPx + q^T x
    # Gx <= h
    # Ax = b
    n = returns.size
    P = matrix(2*covar)
    q = matrix(np.zeros(n))
    G = matrix(np.vstack((np.diag(-np.ones(n)), np.diag(np.ones(n)), [-returns])))
    h = matrix(np.concatenate((np.repeat(-min_weight, n), np.repeat(max_weight, n), [-target_return])))
    A = matrix(np.ones(returns.size).reshape(1,-1))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    return (sol['primal objective'], np.array(list(sol['x'])))


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
    all_data, era_corr = load_training_data(ctx)

    # build portfolio of benchmark models
    target_return = ctx.obj['PARAMS']['model_params'].get('optimization_target_return', 0.02)
    min_portfolio_weight = ctx.obj['PARAMS']['model_params'].get('min_portfolio_weight', 0.01)
    max_portfolio_weight = ctx.obj['PARAMS']['model_params'].get('max_portfolio_weight', 1)

    portfolio_obj, portfolio_weights = optimize(
        target_return=target_return, 
        returns=era_corr.mean(axis=0).values, 
        covar=era_corr.cov().values,
        min_weight=0, 
        max_weight=max_portfolio_weight
    )

    portfolio_models = era_corr.mean(axis=0).index.values
    portfolio_models = portfolio_models[portfolio_weights > min_portfolio_weight]
    portfolio_weights = portfolio_weights[portfolio_weights > min_portfolio_weight]
    portfolio_weights /= portfolio_weights.sum()

    portfolio = {m: w for m, w in zip(portfolio_models, portfolio_weights)}
    print(portfolio)

    # generate predictions
    portfolio_models = list(portfolio.keys())
    portfolio_weights = list(portfolio.values())
    all_data['pred'] = (all_data[portfolio_models] * portfolio_weights).sum(axis=1)

    # calculate metrics
    print("Calculating metrics on validation data")
    pred_cols = [EXAMPLE_PREDS_COL, "pred"]
    validation_stats = validation_metrics(
        validation_data=all_data, 
        pred_cols=pred_cols, example_col=EXAMPLE_PREDS_COL, target_col=TARGET_COL,
    )
    validation_stats['last_era'] = all_data[ERA_COL].max()
    print(validation_stats[["mean", "sharpe", "max_drawdown", "mmc_mean"]].to_markdown())

    gc.collect()

    # final model config
    model_config = {
        "portfolio": portfolio
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
    # load model config
    print("Loading model config")
    try:
        with open(os.path.join(ctx.obj['MODEL_RUN_PATH'], 'config.json'), 'r') as f:
            model_config = json.load(f)
    except FileNotFoundError:
        print("Trained model(s) not found!")
        raise

    # load data
    print("Loading data")
    all_data, _ = load_training_data(ctx)

    # generate predictions
    portfolio_models = list(model_config['portfolio'].keys())
    portfolio_weights = list(model_config['portfolio'].values())
    all_data['pred'] = (all_data[portfolio_models] * portfolio_weights).sum(axis=1)

    # calculate metrics
    print("Calculating metrics on validation data")
    pred_cols = [EXAMPLE_PREDS_COL, "pred"]
    validation_stats = validation_metrics(
        validation_data=all_data, 
        pred_cols=pred_cols, example_col=EXAMPLE_PREDS_COL, target_col=TARGET_COL
    )
    validation_stats['last_era'] = all_data[ERA_COL].max()
    print(validation_stats[["mean", "sharpe", "max_drawdown", "mmc_mean"]].to_markdown())

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
    live_data = pd.read_parquet(ctx.obj['DATASETS']['live'])

    # generate predictions
    portfolio_models = list(model_config['portfolio'].keys())
    portfolio_weights = list(model_config['portfolio'].values())
    live_data['pred'] = (live_data[portfolio_models] * portfolio_weights).sum(axis=1)

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
