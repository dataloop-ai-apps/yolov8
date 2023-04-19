from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_save
import dtlpy as dl
from PIL import Image
import cv2
import os


@dl.Package.decorators.module(description='Model Adapter for Yolov8 object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def save(self, local_path, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'model_filename': 'weights/best.pt'})

    def convert_from_dtlpy(self, data_path, **kwargs):
        train_path = os.path.join(data_path, 'train', 'json')
        validation_path = os.path.join(data_path, 'validation', 'json')
        labels = list(self.model_entity.dataset.instance_map.keys())
        labels.sort()

        #################
        # Convert Train #
        #################
        converter = dl.utilities.converter.Converter()
        converter.labels = {label: i for i, label in enumerate(labels)}
        converter.convert_directory(local_path=train_path,
                                    dataset=self.model_entity.dataset,
                                    to_format='yolo',
                                    from_format='dataloop')
        ######################
        # Convert Validation #
        ######################
        converter = dl.utilities.converter.Converter()
        converter.labels = {label: i for i, label in enumerate(labels)}
        converter.convert_directory(local_path=validation_path,
                                    dataset=self.model_entity.dataset,
                                    to_format='yolo',
                                    from_format='dataloop')

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('model_filename', 'yolov8n.pt')
        model_filepath = os.path.join(local_path, model_filename)
        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath)  # pass any model type
        else:
            model = YOLO('yolov8n.pt')  # pass any model type
        self.model = model

    def train(self, data_path, output_path, **kwargs):
        model = YOLO('yolov8n.pt')  # pass any model type
        model.model.args.update(self.configuration.get('modelArgs', dict()))
        epochs = self.configuration.get('epochs', 50)
        batch = self.configuration.get('batch', 2)
        imgsz = self.configuration.get('imgsz', 640)
        device = self.configuration.get('device', 'cpu')

        project_name = os.path.dirname(output_path)
        name = os.path.basename(output_path)

        # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories
        src_images_path_train = os.path.join(data_path, 'train', 'items')
        dst_images_path_train = os.path.join(data_path, 'train', 'images')
        src_images_path_val = os.path.join(data_path, 'validation', 'items')
        dst_images_path_val = os.path.join(data_path, 'validation', 'images')
        src_labels_path_train = os.path.join(data_path, 'train', 'yolo')
        dst_labels_path_train = os.path.join(data_path, 'train', 'labels')
        src_labels_path_val = os.path.join(data_path, 'validation', 'yolo')
        dst_labels_path_val = os.path.join(data_path, 'validation', 'labels')

        if not os.path.exists(dst_images_path_train) and os.path.exists(src_images_path_train):
            os.rename(src_images_path_train, dst_images_path_train)
        if not os.path.exists(dst_images_path_val) and os.path.exists(src_images_path_val):
            os.rename(src_images_path_val, dst_images_path_val)
        if not os.path.exists(dst_labels_path_train) and os.path.exists(src_labels_path_train):
            os.rename(src_labels_path_train, dst_labels_path_train)
        if not os.path.exists(dst_labels_path_val) and os.path.exists(src_labels_path_val):
            os.rename(src_labels_path_val, dst_labels_path_val)

        # validation_images_path
        # train_lables_path
        # train_images_path

        data = {
            'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
            'train': 'train/images',
            'val': 'validation/images',
            'names': self.model_entity.labels
        }
        data_yaml_filename = os.path.join(data_path, f'{self.model_entity.dataset_id}.yaml')
        yaml_save(file=data_yaml_filename, data=data)
        faas_callback = kwargs.get('on_epoch_end_callback')

        def on_train_epoch_end(train_obj):
            is_train = train_obj.model.training
            if is_train:
                self.current_epoch = train_obj.epoch
                metrics = train_obj.metrics
            else:
                metrics = train_obj.get_stats()
            if faas_callback is not None:
                faas_callback(self.current_epoch, epochs)
            samples = list()
            for figure, value in metrics.items():
                samples.append(dl.LogSample(figure=figure,
                                            legend='train' if is_train is True else 'validation',
                                            x=self.current_epoch,
                                            y=value))
            self.model_entity.add_log_samples(samples=samples, dataset_id=self.model_entity.dataset_id)

        model.add_callback(event='on_train_epoch_end', func=on_train_epoch_end)
        model.add_callback(event='on_val_end', func=on_train_epoch_end)
        model.train(data=data_yaml_filename,
                    exist_ok=True,  # this will override the output dir and will not create a new one
                    epochs=epochs,
                    batch=batch,
                    device=device,
                    name=name,
                    workers=0,
                    imgsz=imgsz,
                    project=project_name)

    def predict(self, batch, **kwargs):
        results = self.model.predict(source=batch, save=False, save_txt=False)  # save predictions as labels
        batch_annotations = list()
        for i_img, res in enumerate(results):  # per image
            image_annotations = dl.AnnotationCollection()
            for d in reversed(res.boxes):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                c = int(cls)
                label = res.names[c]
                xyxy = d.xyxy.squeeze()
                image_annotations.add(annotation_definition=dl.Box(left=float(xyxy[0]),
                                                                   top=float(xyxy[1]),
                                                                   right=float(xyxy[2]),
                                                                   bottom=float(xyxy[3]),
                                                                   label=label
                                                                   ),
                                      model_info={'name': self.model_entity.name,
                                                  'confidence': float(conf)})
            batch_annotations.append(image_annotations)
        return batch_annotations

    def export(self):
        model = YOLO("model.pt")
        model.fuse()
        model.info(verbose=True)  # Print model information
        model.export(format='')  # TODO:


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                          default_configuration={'weights_filename': 'yolov8n.pt',
                                                                 'epochs': 10,
                                                                 'batch': 4,
                                                                 'imgsz': 640,
                                                                 'conf_thres': 0.25,
                                                                 'iou_thres': 0.45,
                                                                 'max_det': 1000,
                                                                 'device': 'cuda:0'},
                                          output_type=dl.AnnotationType.BOX,
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name='yolov8',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop Yolo V8 implementation in pytorch',
                                    # is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/yolov8.git',
                                                            git_tag='v0.1.2'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        runner_image='ultralytics/ultralytics:latest',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json(),
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)
    return package


def model_creation(package: dl.Package):
    import ultralytics
    labels = ultralytics.YOLO().names

    model = package.models.create(model_name='pretrained-yolo-v8',
                                  description='yolo v8 arch, pretrained on ms-coco',
                                  tags=['yolov8', 'pretrained', 'ms-coco'],
                                  dataset_id=None,
                                  status='trained',
                                  # scope='public',
                                  scope='project',
                                  configuration={
                                      'weights_filename': 'yolov8n.pt',
                                      'imgz': 640,
                                      'device': 'cuda:0',
                                      'id_to_label_map': {ind: label for ind, label in enumerate(labels)}},
                                  project_id=package.project.id,
                                  labels=list(labels.values())
                                  )
    return model


def deploy():
    dl.setenv('prod')
    # project_name = 'DataloopModels'
    project_name = 'ipm-distillator-test'
    project = dl.projects.get(project_name)
    # project = dl.projects.get(project_id='0ebbf673-17a7-469c-bcb2-f00fdaedfc8b')
    package = package_creation(project=project)
    # model = model_creation(package=package)
    # model_entity = dl.models.get(model_id='640ee84307a569363353ed6a')
    # print(f'model and package deployed. package id: {package.id}, model id: {model_entity.id}')


if __name__ == "__main__":
    deploy()
    ...
    # test_predict()

    # package.artifacts.list()
    # test()

    # model_creation(package=package)
