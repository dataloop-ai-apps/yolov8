import dtlpy as dl
import configparser
import json
import time
import uuid
import os
import pathlib


def create_dataset_with_tags(project: dl.Project, dpk_name: str, dataset_folder: str) -> dl.Dataset:
    identifier = str(uuid.uuid4())[:8]
    dataset_name = f"{dpk_name}-{identifier}"  # TODO: append git sha

    # Create dataset
    dataset: dl.Dataset = project.datasets.create(dataset_name=dataset_name)
    items_path = os.path.join(dataset_folder, 'items')
    annotations_path = os.path.join(dataset_folder, 'json')
    # dataset.items.upload(local_path=items_path, local_annotations_path=annotations_path)

    # Upload items with their annotations and tags
    item_binaries = sorted(list(filter(lambda x: x.is_file(), pathlib.Path(items_path).rglob('*'))))
    annotation_jsons = sorted(list(pathlib.Path(annotations_path).rglob('*.json')))
    for item_binary, annotation_json in zip(item_binaries, annotation_jsons):
        # Load annotation json
        with open(annotation_json, 'r') as f:
            annotation_data = json.load(f)

        # Extract tags
        item_metadata = dict()
        tags_metadata = annotation_data.get("metadata", dict()).get("system", dict()).get('tags', None)
        if tags_metadata is not None:
            item_metadata.update({"system": {"tags": tags_metadata}})

        # Extract metadata (outside of system)
        for key, value in annotation_data.get("metadata", dict()).items():
            if key not in ["system"]:
                item_metadata.update({key: value})

        # Upload item
        dataset.items.upload(
            local_path=str(item_binary),
            local_annotations_path=str(annotation_json),
            item_metadata=item_metadata
        )
    return dataset


def publish_dpk_and_install_app(project: dl.Project, dpk_name: str) -> (dl.Dpk, dl.App):
    # Find dpk json
    dataloop_cfg_filepath = '.dataloop.cfg'
    with open(dataloop_cfg_filepath, 'r') as f:
        content = f.read()
    dataloop_cfg = json.loads(content)
    dpk_json = None
    for manifest in dataloop_cfg.get("manifests", list()):
        dpk_json_filepath = manifest
        with open(dpk_json_filepath, 'r') as f:
            dpk_json = json.load(f)
        if dpk_json["name"] == dpk_name:
            break
        dpk_json = None

    # Throw error if dpk not found
    if dpk_json is None:
        raise ValueError(f"Could not find dpk with name {dpk_name} in {dataloop_cfg_filepath}")

    # Update the dpk
    identifier = str(uuid.uuid4())[:8]
    dpk = dl.Dpk.from_json(_json=dpk_json, client_api=dl.client_api, project=project)
    dpk.name = f"{dpk.name}-{identifier}"  # TODO: append git sha
    dpk.display_name = dpk.name
    dpk.scope = "project"
    dpk.codebase = None

    # Publish dpk and install app
    dpk = project.dpks.publish(dpk=dpk)
    app = project.apps.install(dpk=dpk)
    return dpk, app


def get_installed_app_model(project: dl.Project, app: dl.App) -> [dl.Model]:
    filters = dl.Filters(resource=dl.FiltersResource.MODEL)
    filters.add(field="app.id", values=app.id)
    models = project.models.list(filters=filters)
    if isinstance(models, dl.entities.PagedEntities):
        models = list(models.all())
    return models


def create_pipeline(project: dl.Project, pipeline_template_filepath: str) -> dl.Pipeline:
    # Open pipeline template
    with open(pipeline_template_filepath, 'r') as f:
        pipeline_json = json.load(f)

    identifier = str(uuid.uuid4())[:8]
    pipeline_name = f'{pipeline_json["name"]}-{identifier}'[:35]  # TODO: append git sha

    # Update pipeline template
    pipeline_json["name"] = pipeline_name
    pipeline_json["projectId"] = project.id
    pipeline = project.pipelines.create(pipeline_json=pipeline_json)

    # TODO: identifier in order to delete test pipelines
    return pipeline


def pipeline_execution_wait(pipeline_execution: dl.PipelineExecution):
    # TODO: Validate the SDK to wait for pipeline cycle to finish
    pipeline = pipeline_execution.pipeline
    in_progress_statuses = ["pending", "in-progress"]
    while pipeline_execution.status in in_progress_statuses:
        time.sleep(5)
        pipeline_execution = pipeline.pipeline_executions.get(pipeline_execution_id=pipeline_execution.id)
    return pipeline_execution.status
