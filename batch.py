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


def create_batch_job(ctx: click.Context, job_name: str, task_count: int) -> batch_v1.Job:
    def add_overwrite(commands: List[str]):
        if ctx.obj['OVERWRITE']:
            commands = ['--overwrite'] + commands
        return commands

    runnable1 = create_container_runnable(ctx.obj['MODEL_ID'], add_overwrite(["download-for-training"]))
    runnable2 = batch_v1.Runnable()
    runnable2.barrier = batch_v1.Runnable.Barrier()
    runnable3 = create_container_runnable(ctx.obj['MODEL_ID'], add_overwrite(["train"]))

    task = batch_v1.TaskSpec()
    task.runnables = [runnable1, runnable2, runnable3]

    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 15000
    resources.memory_mib = 64000
    task.compute_resource = resources

    task.max_retry_count = 0
    task.max_run_duration = "21600s"

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
    policy.machine_type = "c2-standard-16"
    policy.provisioning_model = batch_v1.AllocationPolicy.ProvisioningModel(3) # pre-emptible

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
    job_name = f"numerai-{ctx.obj['MODEL_ID'].replace('_', '-')}-train-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    job = create_batch_job(ctx, job_name, task_count=task_count)
    print(job)


if __name__ == '__main__':
    cli(obj={})
