import unittest
import dtlpy as dl
import os
import json
import time
import configparser

# TODO: change when will become a part of the SDK
import sys
sys.path[0] = ""
os.chdir("../")
import test_utils as utils


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "YoloV8-E2E-Tests"
# MODEL_NAME = "yolov8"


class E2ETestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    dpk: dl.Dpk = None
    model: dl.Model = None
    test_folder: str = os.path.dirname(os.path.abspath(__file__))
    config_folder: str = os.path.join(test_folder, '..')
    test_config = None
    created_pipelines = dict()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)
        # TODO: Add dataset initialization

        # Parse config
        config = configparser.ConfigParser()
        config_path = os.path.join(cls.config_folder, 'e2e_tests.cfg')
        config.read(config_path)
        cls.config = config

        dpk_path = os.path.join(cls.config_folder, config["Common"]["dpk_path"])
        with open(dpk_path, 'r') as f:
            dpk_json = json.load(f)
        dpk = dl.Dpk.from_json(_json=dpk_json, client_api=dl.client_api, project=cls.project)
        # cls.model = cls.project.models.get(model_name=MODEL_NAME)



    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all passed pipelines
        for pipeline_data in cls.created_pipelines.values():
            if pipeline_data["status"] == dl.ExecutionStatus.SUCCESS.value:
                pipeline = pipeline_data["pipeline"]
                pipeline.delete()

        dl.logout()

    def _validate_pipeline_execution(self, pipeline_execution: dl.PipelineExecution):
        # TODO: Validate the SDK to wait for pipeline cycle to finish
        pipeline = pipeline_execution.pipeline
        in_progress_statuses = ["pending", "in-progress"]
        while pipeline_execution.status in in_progress_statuses:
            time.sleep(5)
            pipeline_execution = pipeline.pipeline_executions.get(pipeline_execution_id=pipeline_execution.id)
        # self.created_pipelines[pipeline_type]["status"] = pipeline_execution.status
        self.assertEqual(pipeline_execution.status, "success")

    # Test functions
    def test_yolov8_evaluate(self):
        # Create pipeline
        pipeline_template_filepath = os.path.join(self.test_folder, 'pipeline_template.json')
        with open(pipeline_template_filepath, 'r') as f:
            pipeline_json = json.load(f)
        pipeline = utils.create_pipeline(project=self.project, pipeline_json=pipeline_json)

        # Get filters
        filters = None
        variable: dl.Variable
        for variable in pipeline.variables:
            if variable.name == "test_filters":
                filters = dl.Filters(custom_filter=variable.value)
            if variable.name == "model":
                variable.value = self.model.id
        if filters is None:
            raise ValueError("Filters for evaluate not found in pipeline variables")
        pipeline = pipeline.update()

        # Perform execution
        pipeline.install()
        pipeline_execution = pipeline.execute(
            execution_input=[
                dl.FunctionIO(
                    type=dl.PackageInputType.MODEL,
                    value=self.model.id,
                    name="model"
                ),
                dl.FunctionIO(
                    type=dl.PackageInputType.DATASET,
                    value=self.dataset.id,
                    name="dataset"
                ),
                dl.FunctionIO(
                    type=dl.PackageInputType.JSON,
                    value=filters.prepare(),
                    name="filters"
                )
            ]
        )
        self._validate_pipeline_execution(pipeline_execution=pipeline_execution)


if __name__ == '__main__':
    unittest.main()
