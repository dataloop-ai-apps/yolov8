import dtlpy as dl
import json
import time
import uuid
import os
import pathlib


class TestsUtils:
    def __init__(self, project: dl.Project, commit_id: str):
        self.project = project
        self.commit_id = commit_id
        self.identifier = str(uuid.uuid4())[:8]
        self.tag = f"{self.commit_id}-{self.identifier}"

    def create_dataset_with_tags(self, dataset_name: str, dataset_folder: str, upload_annotations: bool) -> dl.Dataset:
        new_dataset_name = f"{dataset_name}-{self.tag}"

        # Create dataset
        dataset: dl.Dataset = self.project.datasets.create(dataset_name=new_dataset_name)

        # Get paths
        items_path = os.path.join(dataset_folder, 'items')
        annotations_path = os.path.join(dataset_folder, 'json')
        if not os.path.exists(items_path):
            raise ValueError(f"Items folder not found in {items_path}")
        if upload_annotations and not os.path.exists(annotations_path):
            raise ValueError(f"Annotations folder not found in {items_path}")

        # Upload items without annotations and without tags
        if not upload_annotations:
            dataset.items.upload(local_path=items_path)
            return dataset

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

    def publish_dpk_and_install_app(self, dpk_name: str) -> (dl.Dpk, dl.App):
        new_dpk_name = f"{dpk_name}-{self.tag}"

        # Find dpk json
        dataloop_cfg_filepath = '.dataloop.cfg'
        with open(dataloop_cfg_filepath, 'r') as f:
            content = f.read()
        dataloop_cfg = json.loads(content)
        dpk_json = None
        dpk_json_filepath = None
        for manifest in dataloop_cfg.get("manifests", list()):
            dpk_json_filepath = manifest
            with open(dpk_json_filepath, 'r') as f:
                dpk_json = json.load(f)
            if dpk_json["name"] == dpk_name:
                break
            dpk_json = None
            dpk_json_filepath = None

        # Throw error if dpk not found
        if dpk_json is None:
            raise ValueError(f"Could not find dpk with name {dpk_name} in {dataloop_cfg_filepath}")

        # Update the dpk
        dpk = dl.Dpk.from_json(_json=dpk_json, client_api=dl.client_api, project=self.project)
        dpk.name = new_dpk_name
        dpk.display_name = dpk.name
        dpk.scope = "project"
        dpk.codebase = None

        # Set directory to dpk directory
        current_dir = os.getcwd()
        dpk_dir = os.path.join(current_dir, os.path.dirname(dpk_json_filepath))
        os.chdir(dpk_dir)

        # Publish dpk and install app
        dpk = self.project.dpks.publish(dpk=dpk)
        app = self.project.apps.install(dpk=dpk)

        # Return to original directory
        os.chdir(current_dir)
        return dpk, app

    def get_installed_app_models(self, app: dl.App) -> [dl.Model]:
        filters = dl.Filters(resource=dl.FiltersResource.MODEL)
        filters.add(field="app.id", values=app.id)
        models = self.project.models.list(filters=filters)
        if isinstance(models, dl.entities.PagedEntities):
            models = list(models.all())
        return models

    def create_pipeline(self, pipeline_template_filepath: str = None,
                        pipeline_template_dpk: dl.Dpk = None,
                        install: bool = True) -> dl.Pipeline:
        # Get pipeline template
        if pipeline_template_filepath is not None:
            # Open pipeline template
            with open(pipeline_template_filepath, 'r') as f:
                pipeline_json = json.load(f)
        elif pipeline_template_dpk is not None:
            # Get pipeline template from dpk
            pipeline_json = pipeline_template_dpk.components.pipeline_templates[0]
        else:
            raise ValueError("Either pipeline_template_filepath or pipeline_template_dpk must be provided")

        new_pipeline_name = f'{pipeline_json["name"]}-{self.tag}'[:35]

        # Update pipeline template
        pipeline_json["name"] = new_pipeline_name
        pipeline_json["projectId"] = self.project.id
        pipeline = self.project.pipelines.create(pipeline_json=pipeline_json)

        if install:
            pipeline = pipeline.install()

        # TODO: identifier in order to delete test pipelines
        return pipeline

    @staticmethod
    def update_pipeline_variable(pipeline: dl.Pipeline, variables_dict: dict) -> dl.Pipeline:
        variable_keys = list(variables_dict.keys())
        variable: dl.Variable
        for variable in pipeline.variables:
            if variable.name in variable_keys:
                variable.value = variables_dict[variable.name]
        pipeline = pipeline.update()
        return pipeline

    @staticmethod
    def pipeline_execution_wait(pipeline_execution: dl.PipelineExecution):
        pipeline = pipeline_execution.pipeline
        in_progress_statuses = ["pending", "in-progress"]
        while pipeline_execution.status in in_progress_statuses:
            time.sleep(5)
            pipeline_execution = pipeline.pipeline_executions.get(pipeline_execution_id=pipeline_execution.id)
        return pipeline_execution
