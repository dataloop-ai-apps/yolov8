import unittest
import dtlpy as dl
import os
import json
import enum


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "YoloV8-E2E-Tests"
MODEL_NAME = "yolov8"


class TestTypes(enum.Enum):
    EVALUATE = "evaluate"
    PREDICT = "predict"
    TRAIN = "train"


class E2ETestCase(unittest.TestCase):
    model: dl.Model = None
    project: dl.Project = None
    dataset: dl.Dataset = None
    model_tests_path: str = os.path.dirname(os.path.abspath(__file__))
    created_pipelines = dict()

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

    def create_pipeline(self, pipeline_type: TestTypes) -> dl.Pipeline:
        # Read template from template
        pipeline_template_filepath = os.path.join(self.model_tests_path, pipeline_type.value, 'pipeline_template.json')
        with open(pipeline_template_filepath, 'r') as f:
            pipeline_template = json.load(f)

        # Update pipeline template
        pipeline_template["name"] = f'{pipeline_template["name"]}-{self.project.id}'[:35]  # TODO: append git sha
        pipeline_template["projectId"] = self.project.id
        for variable in pipeline_template["variables"]:
            if variable["name"] == "model":
                variable["value"] = self.model.id
                break
        for node in pipeline_template["nodes"]:
            if node["type"] == "ml":
                if node["namespace"]["functionName"] in ["train", "evaluate"]:
                    node["aiLibraryId"] = self.model.package_id

        # Delete the previous pipeline and create a new one
        try:
            pipeline = self.project.pipelines.get(pipeline_name=pipeline_template["name"])
            pipeline.delete()
        except dl.exceptions.NotFound:
            pass
        pipeline = self.project.pipelines.create(pipeline_json=pipeline_template)

        # Save created pipeline
        self.created_pipelines[pipeline_type] = {
            "pipeline": pipeline,
            "status": "created"
        }
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
        # Create pipeline
        pipeline_type = TestTypes.PREDICT
        pipeline = self.create_pipeline(pipeline_type=pipeline_type)

        # Get filters
        filters = None
        variable: dl.Variable
        for variable in pipeline.variables:
            if variable.name == "predict_filters":
                filters = dl.Filters(custom_filter=variable.value)
        if filters is None:
            raise ValueError("Filters for predict not found in pipeline variables")

        # Perform execution
        # predict_item = self.dataset.items.list(filters=filters).all()[0]  # TODO: check why not working
        predict_item = self.dataset.items.get(item_id="667845daa79152c0e157787d")
        predict_item.annotations.delete(filters=dl.Filters(resource=dl.FiltersResource.ANNOTATION))
        pipeline.install()
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
        # Check the status of the pipeline execution
        execution = execution.wait()
        self.created_pipelines[pipeline_type]["status"] = execution.status
        self.assertEqual(execution.status, dl.ExecutionStatus.SUCCESS.value)

    def test_yolov8_train(self):
        """
        Test the yolov8 train pipeline steps:
        1. Create the pipeline and set the model as the input for the Train node
        2. Assume the model is already connected to an existing dataset for training
        3. Execute the pipeline with the input: model
        4. Wait for the pipeline cycle to finish with status success
        """
        # Create pipeline
        pipeline_type = TestTypes.TRAIN
        pipeline = self.create_pipeline(pipeline_type=pipeline_type)

        # Get filters
        train_filters = None
        valid_filters = None
        variable: dl.Variable
        for variable in pipeline.variables:
            if variable.name == "train_filters":
                train_filters = dl.Filters(custom_filter=variable.value)
            elif variable.name == "valid_filters":
                valid_filters = dl.Filters(custom_filter=variable.value)
        if train_filters is None:
            raise ValueError("Filters for train set not found in pipeline variables")
        if valid_filters is None:
            raise ValueError("Filters for validation set not found in pipeline variables")

        # TODO: Update model with filters and dataset
        # Update model metadata
        self.model.dataset_id = self.dataset.id
        self.model.metadata["system"]["subsets"] = dict()
        self.model.metadata["system"]["subsets"]["train"] = train_filters.prepare()
        self.model.metadata["system"]["subsets"]["valid"] = valid_filters.prepare()
        self.model.update(system_metadata=True)

        # Perform execution
        pipeline.install()
        execution = pipeline.execute(
            execution_input=[
                dl.FunctionIO(
                    type=dl.PackageInputType.MODEL,
                    value=self.model.id,
                    name="model"
                )
            ]
        )

        # TODO: Validate the SDK to wait for pipeline cycle to finish
        # Check the status of the pipeline execution
        execution = execution.wait()
        self.created_pipelines[pipeline_type]["status"] = execution.status
        self.assertEqual(execution.status, dl.ExecutionStatus.SUCCESS.value)

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
        # Create pipeline
        pipeline_type = TestTypes.EVALUATE
        pipeline = self.create_pipeline(pipeline_type=pipeline_type)

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
        execution = pipeline.execute(
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

        # TODO: Validate the SDK to wait for pipeline cycle to finish
        # Check the status of the pipeline execution
        execution = execution.wait()
        self.created_pipelines[pipeline_type]["status"] = execution.status
        self.assertEqual(execution.status, dl.ExecutionStatus.SUCCESS.value)


if __name__ == '__main__':
    unittest.main()
