import os
from google.cloud import workflows_v1
from google.cloud.workflows import executions_v1

PROJECT_ID = "blog-180218"
REGION = "us-central1"

def trigger_workflow(request):
    # set up clients
    execution_client = executions_v1.ExecutionsClient()
    workflows_client = workflows_v1.WorkflowsClient()

    # construct the fully qualified location path
    parent = workflows_client.workflow_path(PROJECT_ID, REGION, os.environ['WORKFLOW_NAME'])

    # execute the workflow
    response = execution_client.create_execution(request={"parent": parent})
    return f"Created execution: {response.name}"
