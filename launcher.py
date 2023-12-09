import json, os, datetime, time, re
from typing import List
from textwrap import dedent

import click, docker
import pandas as pd
from jinja2 import Template
from numerapi import NumerAPI

from google.cloud import batch_v1, storage, workflows, scheduler_v1, functions_v1


PROJECT_ID = "blog-180218"
REGION = "us-central1"
CLOUD_STORAGE_BUCKET = "djr-data"
CLOUD_STORAGE_PATH = "numerai"
COMPUTE_SERVICE_ACCOUNT = "89590359009-compute@developer.gserviceaccount.com"
NUMERAI_MODEL_PREFIX = "djr"
WEBHOOK_FUNCTION_NAME = "numerai-webhook"

MACHINES = {
    "C60": {
        "machine_type": "c2-standard-60",
        "cpu_milli": 60000,
        "memory_mib": 240000
    },
    "C30": {
        "machine_type": "c2-standard-30",
        "cpu_milli": 30000,
        "memory_mib": 120000
    },
    "C16": {
        "machine_type": "c2-standard-16",
        "cpu_milli": 16000,
        "memory_mib": 64000
    }
}

batch_client = batch_v1.BatchServiceClient()
storage_client = storage.Client()
workflows_client = workflows.WorkflowsClient()
scheduler_client = scheduler_v1.CloudSchedulerClient()
functions_client = functions_v1.CloudFunctionsServiceClient()
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


def create_training_task(ctx: click.Context, command: str = "train") -> batch_v1.TaskSpec:
    def add_args(commands: List[str]):
        add_to_commands = []
        if ctx.obj['OVERWRITE']:
            add_to_commands += ['--overwrite']
        if ctx.obj['RUN_ID'] is not None:
            add_to_commands += ["--run-id", ctx.obj['RUN_ID']]
        return add_to_commands + commands

    # unsure why this isn't working...
    # # download-for-training will only run for BATCH_TASK_INDEX=0
    # runnable1 = create_container_runnable(ctx.obj['MODEL_ID'], add_args(["download-for-training"]))

    # # this ensures that BATCH_TASK_INDEX>1 waits for BATCH_TASK_INDEX=0 to complete download
    # runnable2 = batch_v1.Runnable()
    # runnable2.barrier = batch_v1.Runnable.Barrier()

    runnable3 = create_container_runnable(ctx.obj['MODEL_ID'], add_args([command]))

    task = batch_v1.TaskSpec()
    task.runnables = [runnable3]
    #task.runnables = [runnable1, runnable2, runnable3]
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


def create_batch_job(job_name: str, task: batch_v1.TaskSpec, task_count: int, machine: str) -> batch_v1.Job:
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = MACHINES[machine]['cpu_milli']
    resources.memory_mib = MACHINES[machine]['memory_mib']
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
    policy.machine_type = MACHINES[machine]["machine_type"]

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


def get_numerai_model_name(model_id: str, run_id: str) -> str:
    return "{}_{}_{}".format(
        NUMERAI_MODEL_PREFIX,
        model_id.replace('-', '_'),
        run_id
    )


def get_workflow_id(numerai_model_name: str) -> str:
    return "numerai-submit-{}".format(
        numerai_model_name.replace('_', '-')
    )


@click.group()
@click.option('--model-id', type=str)
@click.option('--run-id', type=str)
@click.option('--overwrite/--no-overwrite', default=False)
@click.pass_context
def cli(ctx, model_id, run_id, overwrite):
    ctx.ensure_object(dict)
    ctx.obj['MODEL_ID'] = model_id
    ctx.obj['RUN_ID'] = run_id
    ctx.obj['OVERWRITE'] = overwrite

    if run_id:
        ctx.obj['NUMERAI_MODEL_NAME'] = get_numerai_model_name(model_id, run_id)


@cli.command()
@click.pass_context
def build_and_push_image(ctx):
    model_id = ctx.obj['MODEL_ID']

    print("Building image")
    _, build_logs = docker_client.images.build(
        path=os.path.join('models', model_id),
        tag=f"gcr.io/{PROJECT_ID}/numerai:{model_id}"
    )
    for l in build_logs:
        if 'stream' in l.keys():
            print(l['stream'].strip('\n'))

    print("Pushing image")
    push_logs = docker_client.images.push(
        repository=f"gcr.io/{PROJECT_ID}/numerai",
        tag=model_id,
        stream=True,
        decode=True
    )
    for l in push_logs:
        if 'status' in l.keys():
            print(l['status'].strip('\n'))

    print("Pruning dangling images")
    docker_client.images.prune(filters={'dangling': True})


@cli.command()
@click.pass_context
def train(ctx):
    ctx.invoke(build_and_push_image)

    if ctx.obj['RUN_ID'] is not None:
        model_name = f"{ctx.obj['MODEL_ID']}-{ctx.obj['RUN_ID']}"
        task_count = 1
    else:
        model_name = ctx.obj['MODEL_ID']
        with open(os.path.join('models', ctx.obj['MODEL_ID'], 'params.json'), 'r') as f:
            task_count = len(json.load(f))

    job_name = f"numerai-{model_name}-train-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_training_task(ctx, command="train")
    job = create_batch_job(job_name, task, task_count, machine="C60")
    print(job)


@cli.command()
@click.pass_context
def refresh_metrics(ctx):
    ctx.invoke(build_and_push_image)

    if ctx.obj['RUN_ID'] is not None:
        model_name = f"{ctx.obj['MODEL_ID']}-{ctx.obj['RUN_ID']}"
        task_count = 1
    else:
        model_name = ctx.obj['MODEL_ID']
        with open(os.path.join('models', ctx.obj['MODEL_ID'], 'params.json'), 'r') as f:
            task_count = len(json.load(f))

    job_name = f"numerai-{model_name}-refresh-metrics-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_training_task(ctx, command="refresh-metrics")
    job = create_batch_job(job_name, task, task_count, machine="C30")
    print(job)


@cli.command()
@click.pass_context
def download_datasets(ctx):
    ctx.invoke(build_and_push_image)

    model_name = ctx.obj['MODEL_ID']
    task_count = 1

    job_name = f"numerai-{model_name}-download-datasets-{datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')}"
    task = create_training_task(ctx, command="download-datasets-all")
    job = create_batch_job(job_name, task, task_count, machine="C16")
    print(job)


@cli.command()
@click.option('--upload/--no-upload', default=False)
@click.pass_context
def inference(ctx, upload):
    job_name = "numerai-{}-{}-inference-{}".format(
        ctx.obj['MODEL_ID'],
        ctx.obj['RUN_ID'],
        datetime.datetime.now().strftime('%Y-%m-%dt%H-%M-%S')
    )
    if upload:
        task = create_inference_task(ctx, run_id, ctx.obj['NUMERAI_MODEL_NAME'])
    else:
        task = create_inference_task(ctx, run_id, None)
    job = create_batch_job(job_name, task, task_count=1, machine="inference")
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

    metrics = (
        pd.concat(metrics, axis=0)
        .filter(['param_index', 'index', 'mean', 'std', 'sharpe', 'max_drawdown', 'corr_with_example_preds', 'feature_neutral_mean', 'last_era'])
    )

    print(metrics)


@cli.command()
@click.pass_context
def create_webhook(ctx):

    print("Creating cloud function for Numerai webhook")
    function_path = functions_client.cloud_function_path(PROJECT_ID, REGION, WEBHOOK_FUNCTION_NAME)

    function = functions_v1.CloudFunction()
    function.name = function_path
    function.description = "Function to receive Numerai webhook calls"
    function.runtime = 'python310'
    function.entry_point = 'trigger_workflow'
    function.source_archive_url = os.path.join(f'gs://{CLOUD_STORAGE_BUCKET}', CLOUD_STORAGE_PATH, 'functions/webhook.zip')
    function.https_trigger = {
        'url': f'https://{REGION}-{PROJECT_ID}.cloudfunctions.net/{WEBHOOK_FUNCTION_NAME}'
    }

    # deploy function
    operation = functions_client.create_function(
        request={
            'location': functions_client.common_location_path(PROJECT_ID, REGION),
            'function': function
        }
    )
    response = operation.result()
    print('Function deployed successfully')

    # make function accessible without authentication
    set_iam_policy_request_body = {
        "bindings": [
            {
              "role": "roles/cloudfunctions.invoker",
              "members": ["allUsers"],
            },
        ]
    }
    functions_client.set_iam_policy({
        'resource': function_path,
        'policy': set_iam_policy_request_body
    })

    print(f"Webhook URL: {response.https_trigger.url}")


@cli.command()
@click.pass_context
def delete_webhook(ctx):
    print("Deleting webhook")
    delete_request = functions_v1.DeleteFunctionRequest()
    delete_request.name = functions_client.cloud_function_path(PROJECT_ID, REGION, WEBHOOK_FUNCTION_NAME)
    operation = functions_client.delete_function(request=delete_request)
    operation.result()


@cli.command()
@click.pass_context
def create_workflow(ctx):
    assert ctx.obj['NUMERAI_MODEL_NAME'] is not None
    workflow_id = get_workflow_id(ctx.obj['NUMERAI_MODEL_NAME'])

    print("Creating workflow")

    workflow_contents = Template(dedent("""\
    main:
      params: [args]
      steps:
        - init:
            assign:
              - projectId: ${sys.get_env("GOOGLE_CLOUD_PROJECT_ID")}
              - region: "{{ region }}"
              - batchApi: "batch.googleapis.com/v1"
              - batchApiUrl: ${"https://" + batchApi + "/projects/" + projectId + "/locations/" + region + "/jobs"}
              - modelId: "{{ model_id }}"
              - runId: "{{ run_id }}"
              - imageUri: ${"gcr.io/" + projectId + "/numerai:" + modelId}
              - jobId: ${"numerai-" + modelId + "-" + runId + "-inference-" + string(int(sys.now()))}
              - numeraiSecretKey: "{{ secret_key }}"
              - numeraiPublicId: "{{ public_id }}"
              - numeraiModelName: "{{ numerai_model_name }}"
        - runBatchJobAndWait:
            try:
              steps:
                - makeJobId:
                    assign:
                      - jobId: ${"numerai-" + modelId + "-" + runId + "-inference-" + string(int(sys.now()))}
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
                              cpuMilli: {{ machine['cpu_milli'] }}
                              memoryMib: {{ machine['memory_mib'] }}
                            maxRunDuration:
                              seconds: 43200
                            volumes:
                              gcs:
                                remotePath: "{{ gcs_path }}"
                              mountPath: "/mnt/disks/share"
                          taskCount: 1
                          taskCountPerNode: 1
                          parallelism: 1
                        allocationPolicy:
                          instances:
                            policy:
                              machineType: "{{ machine['machine_type'] }}"
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
                      data: ${"Batch job " + jobId + " has state " + getJobResult.body.status.state}
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
            retry:
              max_retries: 3
              backoff:
                initial_delay: 900
    """))
    workflow_contents = workflow_contents.render(
        model_id=ctx.obj['MODEL_ID'],
        run_id=ctx.obj['RUN_ID'],
        public_id=os.environ['NUMERAI_PUBLIC_ID'],
        secret_key=os.environ['NUMERAI_SECRET_KEY'],
        numerai_model_name=ctx.obj['NUMERAI_MODEL_NAME'],
        gcs_path=os.path.join(CLOUD_STORAGE_BUCKET, CLOUD_STORAGE_PATH) + '/',
        region=REGION,
        machine=MACHINES["C16"]
    )

    workflow = workflows.Workflow()
    workflow.name = workflow_id
    workflow.description = "Workflow which downloads live round, generates predictions, and submits to Numerai"
    workflow.source_contents = workflow_contents

    create_request = workflows.CreateWorkflowRequest()
    create_request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    create_request.workflow = workflow
    create_request.workflow_id = workflow_id

    operation = workflows_client.create_workflow(request=create_request)
    response = operation.result()
    print(response)

    # update webhook
    napi = NumerAPI()
    model_id = napi.get_models()[ctx.obj['NUMERAI_MODEL_NAME']]
    webhook_url = f'https://{REGION}-{PROJECT_ID}.cloudfunctions.net/{WEBHOOK_FUNCTION_NAME}?model_name={ctx.obj["NUMERAI_MODEL_NAME"]}'
    napi.set_submission_webhook(
        model_id=model_id,
        webhook=webhook_url
    )
    print('Updated webhook')


@cli.command()
@click.pass_context
def delete_workflow(ctx):
    assert ctx.obj['NUMERAI_MODEL_NAME'] is not None
    workflow_id = get_workflow_id(ctx.obj['NUMERAI_MODEL_NAME'])

    print("Deleting workflow")
    delete_request = workflows.DeleteWorkflowRequest()
    delete_request.name = f"projects/{PROJECT_ID}/locations/{REGION}/workflows/{workflow_id}"
    operation = workflows_client.delete_workflow(request=delete_request)
    operation.result()


@cli.command()
@click.pass_context
def round_performance(ctx):
    napi = NumerAPI()
    performances = []
    for model_name, model_id in napi.get_models().items():
        if model_name.endswith('_test'):
            continue
        
        def get_scores(row):
            ss = row['submissionScores']
            del row['submissionScores']
            perc = (
                pd.DataFrame.from_records(ss)
                .assign(displayName=lambda x: x.displayName + '_perc')
                .set_index('displayName')
                .percentile.to_dict()
            )
            val = (
                pd.DataFrame.from_records(ss)
                .set_index('displayName')
                ['value'].to_dict()
            )
            return {
                **row,
                **val,
                **perc
            }


        performance_raw = napi.round_model_performances_v2(model_id)

        performance = (
            pd.DataFrame.from_records([get_scores(p) for p in performance_raw if p['submissionScores']])
            .query('roundNumber >= 470')
            .filter(['v2_corr20', 'v2_corr20_perc', 'bmc', 'bmc_perc', 'mmc', 'mmc_perc'])
            .assign(combined_percentile=lambda x: x.v2_corr20_perc + 2*x.mmc_perc)
            .mean()
            .to_frame()
            .transpose()
        )
        performance.insert(0, 'model', model_name)
        performances.append(performance)

    performances = (
        pd.concat(performances, axis=0)
        .sort_values(by=['combined_percentile'], ascending=False)
    )
    print(performances)


if __name__ == '__main__':
    cli(obj={})
