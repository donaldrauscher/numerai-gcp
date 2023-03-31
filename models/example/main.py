import click, os
from numerapi import NumerAPI


@click.group()
@click.option('--overwrite/--no-overwrite', default=False)
@click.pass_context
def cli(ctx, overwrite):
    ctx.ensure_object(dict)
    ctx.obj['OVERWRITE'] = overwrite


@cli.command()
@click.option('--dataset')
@click.pass_context
def download(ctx, dataset):
    napi = NumerAPI()
    current_round = napi.get_current_round()
    version = os.path.split(dataset)[0]

    out = dataset
    if dataset.endswith('live.parquet'):
        out = f"{version}/live_{current_round}.parquet"
    out = os.path.join('data', 'datasets', out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if os.path.exists(out) and not ctx.obj['OVERWRITE']:
        print(f"{out} already exists!")
    else:
        print(f'Downloading {dataset}...')
        try:
            os.remove(out)
        except FileNotFoundError:
            pass
        napi.download_dataset(dataset, out)


@cli.command()
@click.option('--version')
@click.pass_context
def download_all(ctx):
    download(f"{version}/train.parquet")
    download(f"{version}/validation.parquet")
    download(f"{version}/live.parquet")
    download(f"{version}/validation_example_preds.parquet")
    download(f"{version}/features.json")


if __name__ == '__main__':
    cli(obj={})
