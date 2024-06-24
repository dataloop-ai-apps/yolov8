import json
import dtlpy as dl
import time


def publish_dpk_and_install_app(project: dl.Project, dpk_json: dict) -> dl.App:
    pass


def create_pipeline(project: dl.Project, pipeline_json: dict) -> dl.Pipeline:
    # Update pipeline template
    pipeline_json["name"] = f'{pipeline_json["name"]}-{project.id}'[:35]  # TODO: append git sha
    pipeline_json["projectId"] = project.id
    pipeline = project.pipelines.create(pipeline_json=pipeline_json)

    # TODO: identifier in order to delete test pipelines
    return pipeline


# TODO: Remove if not needed
def _validate_pipeline_execution(pipeline_execution: dl.PipelineExecution, pipeline_type):
    # TODO: Validate the SDK to wait for pipeline cycle to finish
    pipeline = pipeline_execution.pipeline
    in_progress_statuses = ["pending", "in-progress"]
    while pipeline_execution.status in in_progress_statuses:
        time.sleep(5)
        pipeline_execution = pipeline.pipeline_executions.get(pipeline_execution_id=pipeline_execution.id)
    assert pipeline_execution.status == "success"
