import json, os, datetime, time
from typing import List

import click
import pandas as pd

from google.cloud import batch_v1


PROJECT_ID = "blog-180218"
REGION = "us-central1"
CLOUD_STORAGE_BUCKET = "djr-data"
CLOUD_STORAGE_PATH = "numerai"


batch_client = batch_v1.BatchServiceClient()


def create_container_runnable(model_id: str, commands: List[str]) -> batch_v1.Runnable:
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = f"gcr.io/{PROJECT_ID}/numerai:{model_id}"
    runnable.container.commands = ['--data-dir', '/mnt/disks/share'] + commands
    runnable.environment = batch_v1.Environment(variables={
        'NUMERAI_PUBLIC_ID': os.environ['NUMERAI_PUBLIC_ID'],
        'NUMERAI_SECRET_KEY': os.environ['NUMERAI_SECRET_KEY']
    })
    return runnable


def create_training_task(ctx: click.Context) -> batch_v1.TaskSpec:
    def add_overwrite(commands: List[str]):
        if ctx.obj['OVERWRITE']:
            commands = ['--overwrite'] + commands
        return commands

    # download-for-training will only run for BATCH_TASK_INDEX=0
    runnable1 = create_container_runnable(ctx.obj['MODEL_ID'], add_overwrite(["download-for-training"]))

    # this ensures that BATCH_TASK_INDEX>1 waits for BATCH_TASK_INDEX=0 to complete download
    runnable2 = batch_v1.Runnable()
    runnable2.barrier = batch_v1.Runnable.Barrier()

    runnable3 = create_container_runnable(ctx.obj['MODEL_ID'], add_overwrite(["train"]))

    task = batch_v1.TaskSpec()
    task.runnables = [runnable1, runnable2, runnable3]
    return task


def create_inference_task(ctx: click.Context, run_id: str, numerai_model_name: str) -> batch_v1.TaskSpec:
    commands = ["--run-id", run_id, "inference"]
    if ctx.obj['OVERWRITE']:
        commands = ['--overwrite'] + commands
    if numerai_model_name:
        commands += ['--numerai-model-name', numerai_model_name]

    runnable = create_container_runnable(ctx.obj['MODEL_ID'], commands)
    task = batch_v1.TaskSpec()
    task.runnables = [runnable]
    return task


def create_batch_job(job_name: str, task: batch_v1.TaskSpec, task_count: int) -> batch_v1.Job:
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 30000
    resources.memory_mib = 120000
    task.compute_resource = resources

    task.max_retry_count = 0
    task.max_run_duration = "43200s"

    gcs_bucket = batch_v1.GCS()
    gcs_bucket.remote_path = os.path.join(CLOUD_STORAGE_BUCKET, CLOUD_STORAGE_PATH) + '/'
    gcs_volume = batch_v1.Volume()
    gcs_volume.gcs = gcs_bucket
    gcs_volume.mount_path = '/mnt/disks/share'

    task.volumes = [gcs_volume]

    group = batch_v1.TaskGroup()
    group.task_spec = task
    group.task_count_per_node = 1
    group.task_count = task_count
    group.parallelism = task_count

    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = "c2-standard-30"

    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = job_name
    create_request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    return batch_client.create_job(create_request)


@click.group()
@click.option('--model-id', type=str)
@click.option('--overwrite/--no-overwrite', default=False)
@click.pass_context
def cli(ctx, model_id, overwrite):
    ctx.ensure_object(dict)
    ctx.obj['MODEL_ID'] = model_id
    ctx.obj['OVERWRITE'] = overwrite


@cli.command()
@click.pass_context
def train(ctx):
    with open(os.path.join('models', ctx.obj['MODEL_ID'], 'params.json'), 'r') as f:
        task_count = len(json.load(f))
    job_name = f"numerai-{ctx.obj['MODEL_ID']}-train-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_training_task(ctx)
    job = create_batch_job(job_name, task, task_count)
    print(job)


@cli.command()
@click.option('--run-id', type=str)
@click.option('--numerai-model-name', default=None)
@click.pass_context
def inference(ctx, run_id, numerai_model_name):
    job_name = f"numerai-{ctx.obj['MODEL_ID']}-{run_id}-inference-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_inference_task(ctx, run_id, numerai_model_name)
    job = create_batch_job(job_name, task, task_count=1)
    print(job)


if __name__ == '__main__':
    cli(obj={})
