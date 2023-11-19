from ultralytics.yolo.utils import yaml_save
from ultralytics import YOLO
from PIL import Image

import dtlpy as dl
import logging
import torch
import PIL
import os

logger = logging.getLogger('YOLOv8Adapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000


@dl.Package.decorators.module(description='Model Adapter for Yolov8 object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def save(self, local_path, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'weights_filename': 'weights/best.pt'})

    def convert_from_dtlpy(self, data_path, **kwargs):
        ##############
        # Validation #
        ##############

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError(
                'Couldnt find train set. Yolov8 requires train and validation set for training. Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError(
                'Couldnt find validation set. Yolov8 requires train and validation set for training. Add a validation set DQL filter in the dl.Model metadata')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join(field='type', values='box')
            filters.page_size = 0
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f'Could find box annotations in subset {subset}. Cannot train without annotation in the data subsets')

        #########
        # Paths #
        #########

        train_path = os.path.join(data_path, 'train', 'json')
        validation_path = os.path.join(data_path, 'validation', 'json')
        label_to_id_map = self.model_entity.label_to_id_map

        #################
        # Convert Train #
        #################
        converter = dl.utilities.converter.Converter()
        converter.labels = label_to_id_map
        converter.convert_directory(local_path=train_path,
                                    dataset=self.model_entity.dataset,
                                    to_format='yolo',
                                    from_format='dataloop')
        ######################
        # Convert Validation #
        ######################
        converter = dl.utilities.converter.Converter()
        converter.labels = label_to_id_map
        converter.convert_directory(local_path=validation_path,
                                    dataset=self.model_entity.dataset,
                                    to_format='yolo',
                                    from_format='dataloop')

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', 'yolov8n.pt')
        model_filepath = os.path.join(local_path, model_filename)
        # first load official model -https://github.com/ultralytics/ultralytics/issues/3856
        _ = YOLO('yolov8l.pt')
        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath)  # pass any model type
        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            model = YOLO(model_filename)  # pass any model type
        self.model = model

    def train(self, data_path, output_path, **kwargs):
        self.model.model.args.update(self.configuration.get('modelArgs', dict()))
        epochs = self.configuration.get('epochs', 50)
        batch_size = self.configuration.get('batch_size', 2)
        imgsz = self.configuration.get('imgsz', 640)
        device = self.configuration.get('device', None)
        augment = self.configuration.get('augment', True)
        yaml_config = self.configuration.get('yaml_config', dict())

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        project_name = os.path.dirname(output_path)
        name = os.path.basename(output_path)

        # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories
        train_name = 'train'
        val_name = 'validation'
        src_images_path_train = os.path.join(data_path, 'train', 'items')
        dst_images_path_train = os.path.join(data_path, train_name, 'images')
        src_images_path_val = os.path.join(data_path, 'validation', 'items')
        dst_images_path_val = os.path.join(data_path, val_name, 'images')
        src_labels_path_train = os.path.join(data_path, 'train', 'yolo')
        dst_labels_path_train = os.path.join(data_path, train_name, 'labels')
        src_labels_path_val = os.path.join(data_path, 'validation', 'yolo')
        dst_labels_path_val = os.path.join(data_path, val_name, 'labels')

        if not os.path.exists(dst_images_path_train) and os.path.exists(src_images_path_train):
            os.rename(src_images_path_train, dst_images_path_train)
        if not os.path.exists(dst_images_path_val) and os.path.exists(src_images_path_val):
            os.rename(src_images_path_val, dst_images_path_val)
        if not os.path.exists(dst_labels_path_train) and os.path.exists(src_labels_path_train):
            os.rename(src_labels_path_train, dst_labels_path_train)
        if not os.path.exists(dst_labels_path_val) and os.path.exists(src_labels_path_val):
            os.rename(src_labels_path_val, dst_labels_path_val)

        # yolov8 bug - if there are two directories "images" in the path it fails to get annotations
        paths = [dst_images_path_train, dst_images_path_val, dst_labels_path_train, dst_labels_path_val]
        allowed = [1, 1, 0, 0]
        for path, allow in zip(paths, allowed):
            subfolders = [x[0] for x in os.walk(path)]
            for subfolder in subfolders:
                relpath = os.path.relpath(subfolder, data_path)
                dirs = relpath.split(os.sep)
                c = 0
                for i_dir, dirname in enumerate(dirs):
                    if dirname == 'images':
                        c += 1
                        if c > allow:
                            dirs[i_dir] = 'imagesssss'
                new_subfolder = os.path.join(data_path, *dirs)
                if subfolder != new_subfolder:
                    print(new_subfolder)
                    os.rename(subfolder, new_subfolder)

        # validation_images_path
        # train_lables_path
        # train_images_path

        # check if validation exists
        if not os.path.isdir(dst_images_path_val):
            raise ValueError(
                'Couldnt find validation set. Yolov8 requires train and validation set for training. Add a validation set DQL filter in the dl.Model metadata')
        if len(self.model_entity.labels) == 0:
            raise ValueError(
                'model.labels is empty. Model entity must have labels')

        params = {'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
                  'train': train_name,
                  'val': val_name,
                  'names': list(self.model_entity.label_to_id_map.keys())
                  }

        yaml_config.update(params)
        data_yaml_filename = os.path.join(data_path, f'{self.model_entity.dataset_id}.yaml')
        yaml_save(file=data_yaml_filename, data=yaml_config)
        faas_callback = kwargs.get('on_epoch_end_callback')

        def on_epoch_end(train_obj):

            self.current_epoch = train_obj.epoch
            metrics = train_obj.metrics
            train_obj.plot_metrics()
            if faas_callback is not None:
                faas_callback(self.current_epoch, epochs)
            samples = list()
            for metric_name, value in metrics.items():
                legend, figure = metric_name.split('/')
                samples.append(dl.PlotSample(figure=figure,
                                             legend=legend,
                                             x=self.current_epoch,
                                             y=value))
            self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)
            # save model output after each epoch end
            self.save_to_model(local_path=output_path, cleanup=False)

        # self.model.add_callback(event='on_train_epoch_end', func=on_train_epoch_end)
        # self.model.add_callback(event='on_val_end', func=on_train_epoch_end)
        self.model.add_callback(event='on_fit_epoch_end', func=on_epoch_end)
        self.model.train(data=data_yaml_filename,
                         exist_ok=True,  # this will override the output dir and will not create a new one
                         resume=True,
                         epochs=epochs,
                         batch=batch_size,
                         device=device,
                         augment=augment,
                         name=name,
                         workers=0,
                         imgsz=imgsz,
                         project=project_name)

    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        image = Image.open(filename)
        # Check if the image has EXIF data
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()
            # Get the EXIF orientation tag (if available)
            if exif_data is not None:
                orientation = exif_data.get(0x0112)
                if orientation is not None:
                    # Rotate the image based on the orientation tag
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        image = image.convert('RGB')
        return image

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
                                                  'model_id': self.model_entity.id,
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
                                                                 'batch_size': 4,
                                                                 'imgsz': 640,
                                                                 'conf_thres': 0.25,
                                                                 'iou_thres': 0.45,
                                                                 'max_det': 1000},
                                          output_type=dl.AnnotationType.BOX,
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name='yolov8',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop Yolo V8 implementation in pytorch',
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/yolov8.git',
                                                            git_tag='v0.1.20'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        runner_image='ultralytics/ultralytics:8.0.183',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        preemptible=False,
                                                                        concurrency=1).to_json(),
                                        'executionTimeout': 10000 * 3600,
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
                                  scope='public',
                                  # scope='project',
                                  configuration={
                                      'weights_filename': 'yolov8n.pt',
                                      'imgz': 640,
                                      'id_to_label_map': labels},
                                  project_id=package.project.id,
                                  labels=list(labels.values()),
                                  input_type='image',
                                  output_type='box'
                                  )
    return model


def deploy():
    dl.setenv('rc')
    project_name = 'DataloopModels'
    project = dl.projects.get(project_name)
    # project = dl.projects.get(project_id='0ebbf673-17a7-469c-bcb2-f00fdaedfc8b')
    package = package_creation(project=project)
    print(f'new mode pushed. codebase: {package.codebase}')
    # model = model_creation(package=package)
    # model_entity = package.models.list().print()
    # print(f'model and package deployed. package id: {package.id}, model id: {model_entity.id}')


if __name__ == "__main__":
    # deploy()
    ...
    # test_predict()

    # package.artifacts.list()
    # test()

    # model_creation(package=package)
