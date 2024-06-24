import json
import dtlpy as dl
import time


def publish_dpk_and_install_app(project: dl.Project, dpk_json: dict):
    dpk = dl.Dpk.from_json(_json=dpk_json, client_api=dl.client_api, project=project)
    dpk.name = f"{dpk.name}-predict-{project.name}"  # TODO: use sha
    dpk.display_name = dpk.name
    dpk.codebase = None

    dpk = project.dpks.publish(dpk=dpk)
    app = project.apps.install(dpk=dpk)
    return dpk, app

def get_installed_app_model(project: dl.Project, app_name: str) -> dl.App:
    filters = dl.Filters(resource=dl.FiltersResource.MODEL)
    filters.add(field="app.id", values=cls.app.id)
    models = cls.project.models.list(filters=filters)
    if isinstance(models, dl.entities.PagedEntities):
        models = models.all()
    cls.model = models[0]


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
