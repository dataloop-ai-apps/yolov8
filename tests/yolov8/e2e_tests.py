import unittest
import dtlpy as dl
import os
import json


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = os.environ['DATASET_NAME']


class E2ETestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    model_tests_path: str = os.path.dirname(os.path.abspath(__file__))
    created_pipelines = list()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        os.chdir(cls.model_tests_path)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)

        # Update the dpk
        dataloop_json_filepath = os.path.join(cls.model_tests_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=cls.project)
        dpk.scope = "project"
        dpk.name = f'{dataloop_json["name"]}-{cls.project.id}'
        dpk.displayName = f'{dataloop_json["name"]}-{cls.project.id}'

        # Publish dpk and install app
        cls.tests_dpk = cls.project.dpks.publish(dpk=dpk)
        cls.tests_app = cls.project.apps.install(dpk=cls.tests_dpk)

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all pipelines
        for pipeline in cls.created_pipelines:
            pipeline.delete()

        # Uninstall the app and delete the dpk
        cls.tests_app.uninstall()
        cls.tests_dpk.delete()

        dl.logout()

    def create_pipeline(self, pipeline_type: str):
        pipeline = None

        # Read template from dpk
        pipeline_templates = self.tests_dpk.components.pipeline_templates
        for pipeline_template in pipeline_templates:
            if pipeline_type in pipeline_template["name"]:
                pipeline = dl.Pipeline.from_json(
                    _json=pipeline_template,
                    client_api=dl.client_api,
                    project=self.project
                )
                break

        if pipeline is None:
            raise ValueError(f"Pipeline template of type '{pipeline_type}' not found")

        # Create pipeline
        pipeline = self.project.pipelines.create(pipeline_json=pipeline)
        self.created_pipelines.append(pipeline)
        return pipeline

    # Test functions
    def test_yolov8_train(self):
        """
        Test the yolov8 train pipeline steps:
        1. Create pipeline
        2. Assume the model is already connected to an existing dataset for training
        3. Execute the pipeline
        :return:
        """
        pass

    def test_yolov8_predict(self):
        pass

    def test_yolov8_evaluate(self):
        pass


if __name__ == '__main__':
    unittest.main()
