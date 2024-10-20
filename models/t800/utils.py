import numpy as np
import pandas as pd
from pathlib import Path
import json
from numerai_tools.scoring import correlation_contribution, numerai_corr, feature_neutral_corr


ERA_COL = "era"
TARGET_COL = "target_cyrusd_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"
META_MODEL_COL = "meta_model"

MODEL_FOLDER = "models"
MODEL_CONFIGS_FOLDER = "model_configs"
PREDICTION_FILES_FOLDER = "prediction_files"


def save_prediction(df, name):
    try:
        Path(PREDICTION_FILES_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    df.to_csv(f"{PREDICTION_FILES_FOLDER}/{name}.csv", index=True)


def save_model(model, name):
    try:
        Path(MODEL_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    pd.to_pickle(model, f"{MODEL_FOLDER}/{name}.pkl")


def load_model(name):
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model


def save_model_config(model_config, model_name):
    try:
        Path(MODEL_CONFIGS_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    with open(f"{MODEL_CONFIGS_FOLDER}/{model_name}.json", "w") as fp:
        json.dump(model_config, fp)


def load_model_config(model_name):
    path_str = f"{MODEL_CONFIGS_FOLDER}/{model_name}.json"
    path = Path(path_str)
    if path.is_file():
        with open(path_str, "r") as fp:
            model_config = json.load(fp)
    else:
        model_config = False
    return model_config


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def get_biggest_corr_change_negative_eras(data, features, prediction_col, n):
    target_corr = (
        data
        .groupby(ERA_COL)
        .apply(lambda d: numerai_corr(d[prediction_col], d[TARGET_COL]))
    )

    negative_eras = target_corr[target_corr < 0].index.tolist()
    positive_eras = target_corr[target_corr >= 0].index.tolist()

    feature_corrs = (
        data
        .groupby(ERA_COL)
        .apply(lambda era: era[features].corrwith(era[TARGET_COL]))
    )

    negative_era_feature_corrs = (
        feature_corrs
        .loc[feature_corrs.index.isin(negative_eras), :]
        .mean()
    )

    positive_era_feature_corrs = (
        feature_corrs
        .loc[feature_corrs.index.isin(positive_eras), :]
        .mean()
    )

    corr_diffs = negative_era_feature_corrs - positive_era_feature_corrs
    return corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()


def validation_metrics(
    validation_data,
    pred_cols,
    example_col=EXAMPLE_PREDS_COL,
    target_col=TARGET_COL,
    features_for_neutralization=None,
):
    validation_stats = pd.DataFrame()

    # calculate correlations
    validation_correlations = (
        validation_data
        .filter(pred_cols + [TARGET_COL, ERA_COL])
        .dropna()
        .groupby(ERA_COL).apply(lambda d: numerai_corr(d[pred_cols], d[TARGET_COL]))
    )

    # average corr and sharpe
    mean = validation_correlations.mean()
    std = validation_correlations.std(ddof=0)
    sharpe = mean / std

    validation_stats.loc["mean", pred_cols] = mean
    validation_stats.loc["std", pred_cols] = std
    validation_stats.loc["sharpe", pred_cols] = sharpe

    # max drawdown
    rolling_max = (
        (validation_correlations + 1)
        .cumprod()
        .rolling(window=9000, min_periods=1)  # arbitrarily large
        .max()
    )
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    validation_stats.loc["max_drawdown", pred_cols] = max_drawdown

    # apy
    payout_scores = validation_correlations.clip(-0.25, 0.25)
    payout_daily_value = (payout_scores + 1).cumprod()

    apy = (
        ((payout_daily_value.dropna().iloc[-1]) ** (1 / len(payout_scores)))
        ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
        - 1
    ) * 100

    validation_stats.loc["apy", pred_cols] = apy

    # feature neutral corr
    validation_stats.loc["feature_neutral_mean", pred_cols] = (
        validation_data
        .dropna(subset=pred_cols + [TARGET_COL])
        .groupby(ERA_COL).apply(
            lambda d: feature_neutral_corr(d[pred_cols], d[features_for_neutralization], d[TARGET_COL])
        )
        .mean()
    )

    # mmc / bmc
    validation_stats.loc["mmc_mean", pred_cols] = (
        validation_data
        .dropna(subset=pred_cols + [TARGET_COL, example_col])
        .groupby(ERA_COL).apply(
            lambda d: correlation_contribution(d[pred_cols], d[example_col], d[TARGET_COL])
        )
        .mean()
    )

    # Check correlation with example predictions
    validation_stats.loc["corr_with_example_preds", pred_cols] = (
        validation_data
        .dropna(subset=pred_cols + [example_col])
        .groupby(ERA_COL)
        .apply(lambda d: d[pred_cols].corrwith(d[example_col], method='spearman'))
        .mean()
    )

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()
