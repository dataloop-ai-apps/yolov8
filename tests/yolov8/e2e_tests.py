import unittest
import dtlpy as dl
import os
import json
import time


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "YoloV8-E2E-Tests"


class E2ETestCase(unittest.TestCase):
    model_name = "yolov8"
    project: dl.Project = None
    dataset: dl.Dataset = None
    model_tests_path: str = os.path.dirname(os.path.abspath(__file__))
    created_pipelines = list()
    test_filters = dict()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        os.chdir(cls.model_tests_path)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)

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

        # Define filters
        predict_filters = dl.Filters(field="metadata.system.tags.predict", values=True)
        # train_filters = dl.Filters()
        # validation_filters = dl.Filters()
        evaluate_filters = dl.Filters(field="metadata.system.tags.evaluate", values=True)
        cls.test_filters["predict"] = predict_filters
        # cls.test_filters["train"] = train_filters
        # cls.test_filters["validation"] = validation_filters
        cls.test_filters["evaluate"] = evaluate_filters

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all pipelines
        for pipeline in cls.created_pipelines:
            pipeline.delete()

        # Uninstall the app and delete the dpk
        cls.tests_app.uninstall()
        cls.tests_dpk.delete()

        dl.logout()

    def create_pipeline(self, pipeline_type: str) -> dl.Pipeline:
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
    def test_yolov8_predict(self):
        """
        Test the yolov8 predict pipeline steps:
        1. Create the pipeline
        2. Assume the model is already connected to an existing dataset connected to it
        3. Use filters to get the item/s for predict (filters should be in a config)
        4. Execute the pipeline with the input: item/s
        5. Wait for the pipeline cycle to finish with status success
        """
        pipeline_type = "predict"
        pipeline = self.create_pipeline(pipeline_type=pipeline_type)
        predict_node: dl.PipelineNode = pipeline.nodes[0]
        predict_node.metadata["modelId"] = self.tests_app.models[0].id

        predict_item = self.dataset.items.list(filters=self.test_filters[pipeline_type]).all()[0]
        execution = pipeline.execute(
            execution_input=[
                dl.FunctionIO(
                    type=dl.PackageInputType.ITEM,
                    value=predict_item.id,
                    name="item"
                )
            ]
        )
        # TODO: Validate the SDK to wait for pipeline cycle to finish
        execution = execution.wait()
        self.assertEqual(execution.status, dl.ExecutionStatus.SUCCESS)

    def test_yolov8_train(self):
        """
        Test the yolov8 train pipeline steps:
        1. Create the pipeline and set the model as the input for the Train node
        2. Assume the model is already connected to an existing dataset for training
        3. Execute the pipeline with the input: model
        4. Wait for the pipeline cycle to finish with status success
        """
        pipeline_type = "train"
        pipeline = self.create_pipeline(pipeline_type=pipeline_type)

        # Update model with filters
        # model = self.project.models.get(model_name=self.model_name)

        execution = pipeline.execute(
            execution_input=[]
        )
        # TODO: Validate the SDK to wait for pipeline cycle to finish
        execution = execution.wait()
        self.assertEqual(execution.status, dl.ExecutionStatus.SUCCESS)

    def test_yolov8_evaluate(self):
        """
        Test the yolov8 evaluate pipeline steps:
        0. If possible run after train with the updated weights
        1. Create the pipeline
        2. Assume the model is already connected to an existing dataset connected to it
        3. Assume the dataset has the default evaluation metadata applied on the data.
        4. Get the model dataset and the default filters (filters should be in a config)
        5. Execute the pipeline with the input: model, dataset and filters
        6. Wait for the pipeline cycle to finish with status success
        """
        pipeline_type = "evaluate"
        pipeline = self.create_pipeline(pipeline_type=pipeline_type)

        filters = self.test_filters[pipeline_type]
        execution = pipeline.execute(
            execution_input=[
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
        # TODO: Validate the SDK to wait for pipeline cycle to finish
        execution = execution.wait()
        self.assertEqual(execution.status, dl.ExecutionStatus.SUCCESS)


if __name__ == '__main__':
    unittest.main()
