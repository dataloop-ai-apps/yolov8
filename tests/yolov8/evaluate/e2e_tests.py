import unittest
import dtlpy as dl
import os
import json
import enum
import time

# TODO: change when will become a part of the SDK
import sys
sys.path[0] = ""
os.chdir("../")
import test_utils as utils


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "YoloV8-E2E-Tests"
MODEL_NAME = "yolov8"


class E2ETestCase(unittest.TestCase):
    model: dl.Model = None
    project: dl.Project = None
    dataset: dl.Dataset = None
    test_path: str = os.path.dirname(os.path.abspath(__file__))
    test_config_path: str = os.path.join(test_folder, 'e2e_tests.cfg')
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
        cls.model = cls.project.models.get(model_name=MODEL_NAME)

        # TODO: Add dataset initialization

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all passed pipelines
        for pipeline_data in cls.created_pipelines.values():
            if pipeline_data["status"] == dl.ExecutionStatus.SUCCESS.value:
                pipeline = pipeline_data["pipeline"]
                pipeline.delete()

        dl.logout()

    def _validate_pipeline_execution(self, pipeline_execution: dl.PipelineExecution, pipeline_type: TestTypes):
        # TODO: Validate the SDK to wait for pipeline cycle to finish
        pipeline = pipeline_execution.pipeline
        in_progress_statuses = ["pending", "in-progress"]
        while pipeline_execution.status in in_progress_statuses:
            time.sleep(5)
            pipeline_execution = pipeline.pipeline_executions.get(pipeline_execution_id=pipeline_execution.id)
        self.created_pipelines[pipeline_type]["status"] = pipeline_execution.status
        self.assertEqual(pipeline_execution.status, "success")

    # Test functions
    def test_yolov8_evaluate(self):
        # Create pipeline
        pipeline_template_filepath = os.path.join(self.test_path, 'pipeline_template.json')
        with open(pipeline_template_filepath, 'r') as f:
            pipeline_json = json.load(f)
        pipeline = utils.create_pipeline(project=self.project, pipeline_json=pipeline_json)

        # Get filters
        filters = None
        variable: dl.Variable
        for variable in pipeline.variables:
            if variable.name == "test_filters":
                filters = dl.Filters(custom_filter=variable.value)
        if filters is None:
            raise ValueError("Filters for evaluate not found in pipeline variables")

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
        self._validate_pipeline_execution(pipeline_execution=pipeline_execution, pipeline_type=pipeline_type)


if __name__ == '__main__':
    unittest.main()
