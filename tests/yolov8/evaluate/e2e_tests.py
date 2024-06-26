import unittest
import dtlpy as dl
import os
import json

# TODO: change when will become a part of the SDK
import sys
sys.path[0] = ""
os.chdir("../")
import test_utils as utils


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
ENV = os.environ['ENV']
DATASET_NAME = "YoloV8-E2E-Tests"
DPK_NAME = "yolov8"


class E2ETestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    dpk: dl.Dpk = None
    app: dl.App = None
    model: dl.Model = None
    test_folder: str = os.path.dirname(os.path.abspath(__file__))
    pipeline_data: dict = None

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

        cls.dpk, cls.app = utils.publish_dpk_and_install_app(project=cls.project, dpk_name=DPK_NAME)
        cls.model = utils.get_installed_app_model(project=cls.project, app=cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete the pipeline if passed
        if cls.pipeline_data is not None and cls.pipeline_data["status"] == dl.ExecutionStatus.SUCCESS.value:
            cls.pipeline_data["pipeline"].delete()
            cls.app.uninstall()
            cls.dpk.delete()

        dl.logout()

    # Test functions
    def test_yolov8_evaluate(self):
        # Create pipeline
        pipeline_template_filepath = os.path.join(self.test_folder, 'pipeline_template.json')
        pipeline = utils.create_pipeline(project=self.project, pipeline_template_filepath=pipeline_template_filepath)
        self.pipeline_data = {"pipeline": pipeline, "status": "created"}

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
        status = utils.validate_pipeline_execution(pipeline_execution=pipeline_execution)
        self.pipeline_data["status"] = status
        self.assertEqual(status, dl.ExecutionStatus.SUCCESS.value)


if __name__ == '__main__':
    unittest.main()
