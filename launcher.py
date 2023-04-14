import json, os, datetime, time, re
from typing import List
from textwrap import dedent

import click, docker
import pandas as pd
from jinja2 import Template

from google.cloud import batch_v1, storage, workflows


PROJECT_ID = "blog-180218"
REGION = "us-central1"
CLOUD_STORAGE_BUCKET = "djr-data"
CLOUD_STORAGE_PATH = "numerai"


batch_client = batch_v1.BatchServiceClient()
storage_client = storage.Client()
workflows_client = workflows.WorkflowsClient()
docker_client = docker.from_env()


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


def build_and_push_image(model_id: str):
    print("Building image")
    _, build_logs = docker_client.images.build(
        path=os.path.join('models', model_id),
        tag=f"gcr.io/blog-180218/numerai:{model_id}"
    )
    for l in build_logs:
        if 'stream' in l.keys():
            print(l['stream'].strip('\n'))

    print("Pushing image")
    push_logs = docker_client.images.push(
        repository="gcr.io/blog-180218/numerai",
        tag=model_id,
        stream=True,
        decode=True
    )
    for l in push_logs:
        if 'status' in l.keys():
            print(l['status'].strip('\n'))

    print("Pruning dangling images")
    docker_client.images.prune(filters={'dangling': True})


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
    build_and_push_image(ctx.obj['MODEL_ID'])

    with open(os.path.join('models', ctx.obj['MODEL_ID'], 'params.json'), 'r') as f:
        task_count = len(json.load(f))

    job_name = f"numerai-{ctx.obj['MODEL_ID']}-train-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_training_task(ctx)
    job = create_batch_job(job_name, task, task_count)
    print(job)


@cli.command()
@click.option('--run-id', type=str, required=True)
@click.option('--numerai-model-name', default=None)
@click.pass_context
def inference(ctx, run_id, numerai_model_name):
    job_name = f"numerai-{ctx.obj['MODEL_ID']}-{run_id}-inference-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_inference_task(ctx, run_id, numerai_model_name)
    job = create_batch_job(job_name, task, task_count=1)
    print(job)


@cli.command()
@click.pass_context
def get_metrics(ctx):
    prefix = os.path.join(CLOUD_STORAGE_PATH, 'artifacts', ctx.obj['MODEL_ID'])
    blobs = [b for b in storage_client.list_blobs(CLOUD_STORAGE_BUCKET, prefix=prefix) \
             if os.path.basename(b.name) == 'metrics.json']

    metrics = []
    for b in blobs:
        df = pd.read_json(b.download_as_text(), orient='index')
        df.reset_index(inplace=True)
        param_index = re.compile(r'([0-9]+)/metrics\.json$').search(b.name).group(1)
        df.insert(0, 'param_index', int(param_index))
        metrics.append(df)
    metrics = pd.concat(metrics, axis=0)

    print(metrics)


@cli.command()
@click.option('--run-id', type=str, required=True)
@click.option('--numerai-model-name', required=True)
@click.pass_context
def create_workflow(ctx, run_id, numerai_model_name):
    workflow_contents = Template(dedent("""\
    main:
      params: [args]
      steps:
        - init:
            assign:
              - projectId: ${sys.get_env("GOOGLE_CLOUD_PROJECT_ID")}
              - region: "us-central1"
              - batchApi: "batch.googleapis.com/v1"
              - batchApiUrl: ${"https://" + batchApi + "/projects/" + projectId + "/locations/" + region + "/jobs"}
              - modelId: "{{ model_id }}"
              - runId: "{{ run_id }}"
              - imageUri: ${"gcr.io/" + projectId + "/numerai:" + modelId}
              - jobId: ${"numerai-" + modelId + "-" + runId + "-inference-" + string(int(sys.now()))}
              - numeraiSecretKey: "{{ secret_key }}"
              - numeraiPublicId: "{{ public_id }}"
              - numeraiModelName: "{{ numerai_model_name }}"
        - createAndRunBatchJob:
            call: http.post
            args:
              url: ${batchApiUrl}
              query:
                job_id: ${jobId}
              headers:
                Content-Type: application/json
              auth:
                type: OAuth2
              body:
                taskGroups:
                  taskSpec:
                    runnables:
                      - container:
                          imageUri: ${imageUri}
                          commands:
                            - "--data-dir"
                            - "/mnt/disks/share"
                            - "--run-id"
                            - ${runId}
                            - "inference"
                            - "--numerai-model-name"
                            - ${numeraiModelName}
                          volumes: "/mnt/disks/share:/mnt/disks/share:rw"
                        environment:
                          variables:
                            NUMERAI_SECRET_KEY: ${numeraiSecretKey}
                            NUMERAI_PUBLIC_ID: ${numeraiPublicId}
                    computeResource:
                      cpuMilli: 30000
                      memoryMib: 120000
                    maxRunDuration:
                      seconds: 43200
                    volumes:
                      gcs:
                        remotePath: "djr-data/numerai/"
                      mountPath: "/mnt/disks/share"
                  taskCount: 1
                  taskCountPerNode: 1
                  parallelism: 1
                allocationPolicy:
                  instances:
                    policy:
                      machineType: "c2-standard-30"
                logsPolicy:
                  destination: CLOUD_LOGGING
            result: createAndRunBatchJobResponse
        - getJob:
            call: http.get
            args:
              url: ${batchApiUrl + "/" + jobId}
              auth:
                type: OAuth2
            result: getJobResult
        - logState:
            call: sys.log
            args:
              data: ${"Current job state " + getJobResult.body.status.state}
        - checkState:
            switch:
              - condition: ${getJobResult.body.status.state == "SUCCEEDED"}
                next: returnResult
              - condition: ${getJobResult.body.status.state == "FAILED"}
                next: failExecution
            next: sleep
        - sleep:
            call: sys.sleep
            args:
              seconds: 10
            next: getJob
        - returnResult:
            return:
              jobId: ${jobId}
        - failExecution:
            raise:
              message: ${"The underlying batch job " + jobId + " failed"}
    """))
    workflow_contents = workflow_contents.render(
        model_id=ctx.obj['MODEL_ID'],
        run_id=run_id,
        public_id=os.environ['NUMERAI_PUBLIC_ID'],
        secret_key=os.environ['NUMERAI_SECRET_KEY'],
        numerai_model_name=numerai_model_name
    )

    workflow = workflows.Workflow()
    workflow.name = f"numerai-submit-{numerai_model_name}"
    workflow.description = "Workflow which downloads live round, generates predictions, and submits to Numerai"
    workflow.source_contents = workflow_contents

    create_request = workflows.CreateWorkflowRequest()
    create_request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    create_request.workflow = workflow
    create_request.workflow_id = f"numerai-submit-{numerai_model_name}"

    operation = workflows_client.create_workflow(request=create_request)
    response = operation.result()
    print(response)


if __name__ == '__main__':
    cli(obj={})
