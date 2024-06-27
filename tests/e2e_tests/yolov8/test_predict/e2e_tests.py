import unittest
import dtlpy as dl
import os

# TODO: change when will become a part of the SDK
from test_utils import TestsUtils

BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
ENV = os.environ['ENV']
DPK_NAME = "yolov8"


class E2ETestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    dpk: dl.Dpk = None
    app: dl.App = None
    installed_models: list = None
    model: dl.Model = None
    utils: TestsUtils = None
    pipeline_execution: dl.PipelineExecution = None
    test_folder: str = os.path.dirname(os.path.abspath(__file__))
    assets_folder: str = os.path.join('tests', 'assets', 'e2e_tests')

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv(ENV)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        cls.utils = TestsUtils(project=cls.project)

        dataset_folder = os.path.join(cls.assets_folder, 'dataset')
        cls.dataset = cls.utils.create_dataset_with_tags(
            dpk_name=DPK_NAME,
            dataset_folder=dataset_folder
        )
        cls.dpk, cls.app = cls.utils.publish_dpk_and_install_app(dpk_name=DPK_NAME)
        cls.installed_models = cls.utils.get_installed_app_model(app=cls.app)
        for model in cls.installed_models:
            if "yolov8" in model.name and "large" not in model.name:
                cls.model = model
                break

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete the pipeline if passed
        if cls.pipeline_execution is not None and cls.pipeline_execution.status == dl.ExecutionStatus.SUCCESS.value:
            cls.pipeline_execution.pipeline.delete()
            for model in cls.installed_models:
                model.delete()
            cls.app.uninstall()
            cls.dpk.delete()

        dl.logout()

    # Test functions
    def test_yolov8_predict(self):
        # Create pipeline
        pipeline_template_filepath = os.path.join(self.test_folder, 'template.json')
        pipeline = self.utils.create_pipeline(pipeline_template_filepath=pipeline_template_filepath, install=False)
        pipeline = self.utils.update_pipeline_variable(pipeline=pipeline, variable_name="dataset", variable_value=self.dataset.id)
        pipeline = self.utils.update_pipeline_variable(pipeline=pipeline, variable_name="model", variable_value=self.model.id)

        # # Get filters
        # filters = None
        # variable: dl.Variable
        # for variable in pipeline.variables:
        #     if variable.name == "predict_filters":
        #         filters = dl.Filters(custom_filter=variable.value)
        #     if variable.name == "model":
        #         variable.value = self.model.id
        # if filters is None:
        #     raise ValueError("Filters for predict not found in pipeline variables")
        # pipeline.update()

        # Perform execution
        # predict_item = self.dataset.items.list(filters=filters).all()[0]  # TODO: check why not working
        # predict_item = self.dataset.items.get(item_id="66783633b5d4e8bd5714d3e0")
        # predict_item.annotations.delete(filters=dl.Filters(resource=dl.FiltersResource.ANNOTATION))
        pipeline.install()
        # self.pipeline_execution = pipeline.execute(
        #     execution_input=[
        #         dl.FunctionIO(
        #             type=dl.PackageInputType.ITEM,
        #             value=predict_item.id,
        #             name="item"
        #         )
        #     ]
        # )
        self.pipeline_execution = pipeline.execute(execution_input=None)

        # TODO: Waiting for DAT-73101
        # self.pipeline_execution = self.pipeline_execution.wait()
        # status = self.pipeline_execution.status
        status = self.utils.pipeline_execution_wait(pipeline_execution=self.pipeline_execution)
        self.assertEqual(status, dl.ExecutionStatus.SUCCESS.value)


if __name__ == '__main__':
    unittest.main()
