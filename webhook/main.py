import os
from google.cloud import workflows_v1
from google.cloud.workflows import executions_v1
from google.api_core.exceptions import NotFound


PROJECT_ID = "blog-180218"
REGION = "us-central1"


def get_workflow_id(numerai_model_name: str) -> str:
    return "numerai-submit-{}".format(
        numerai_model_name.replace('_', '-')
    )


def trigger_workflow(request):
    # set up clients
    execution_client = executions_v1.ExecutionsClient()
    workflows_client = workflows_v1.WorkflowsClient()

    # construct the fully qualified location path
    model_name = request.args.get("model_name")
    if model_name is None:
        error_message = "'model_name' parameter is required"
        return error_message, 400

    workflow_id = get_workflow_id(model_name)
    parent = workflows_client.workflow_path(PROJECT_ID, REGION, workflow_id)

    # execute the workflow
    try:
        response = execution_client.create_execution(request={"parent": parent})
        return f"Created execution: {response.name}"
    except NotFound:
        error_message = f"Could not find '{workflow_id}' workflow"
        return error_message, 400
