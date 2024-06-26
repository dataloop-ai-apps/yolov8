import unittest
import dtlpy as dl
import os

# TODO: change when will become a part of the SDK
import test_utils as utils

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
    model: dl.Model = None
    test_folder: str = os.path.dirname(os.path.abspath(__file__))
    model_tests_folder: str = os.path.join(test_folder, '..')
    pipeline_data: dict = None

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv(ENV)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        dataset_folder = os.path.join(cls.model_tests_folder, 'dataset')
        cls.dataset = utils.create_dataset_with_tags(
            project=cls.project,
            dpk_name=DPK_NAME,
            dataset_folder=dataset_folder
        )
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
    def test_yolov8_predict(self):
        # Create pipeline
        pipeline_template_filepath = os.path.join(self.test_folder, 'pipeline_template.json')
        pipeline = utils.create_pipeline(project=self.project, pipeline_template_filepath=pipeline_template_filepath)
        self.pipeline_data = {"pipeline": pipeline, "status": "created"}

        # Get filters
        filters = None
        variable: dl.Variable
        for variable in pipeline.variables:
            if variable.name == "predict_filters":
                filters = dl.Filters(custom_filter=variable.value)
            if variable.name == "model":
                variable.value = self.model.id
        if filters is None:
            raise ValueError("Filters for predict not found in pipeline variables")
        pipeline.update()

        # Perform execution
        # predict_item = self.dataset.items.list(filters=filters).all()[0]  # TODO: check why not working
        predict_item = self.dataset.items.get(item_id="667845daa79152c0e157787d")
        predict_item.annotations.delete(filters=dl.Filters(resource=dl.FiltersResource.ANNOTATION))
        pipeline.install()
        pipeline_execution = pipeline.execute(
            execution_input=[
                dl.FunctionIO(
                    type=dl.PackageInputType.ITEM,
                    value=predict_item.id,
                    name="item"
                )
            ]
        )
        status = utils.pipeline_execution_wait(pipeline_execution=pipeline_execution)
        self.pipeline_data["status"] = status
        self.assertEqual(status, dl.ExecutionStatus.SUCCESS.value)


if __name__ == '__main__':
    unittest.main()
