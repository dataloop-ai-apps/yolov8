from ultralytics import YOLO
from PIL import Image

import dtlpy as dl
import logging
import torch
import PIL
import os
import yaml
import shutil
import numpy as np

logger = logging.getLogger('YOLOv8Adapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000

DEFAULT_WEIGHTS = ['yolov8l.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt', 'yolov8x.pt']


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
        model_filename = self.configuration.get('weights_filename', 'yolov9e.pt')
        model_filepath = os.path.join(local_path, model_filename)

        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath)  # pass any model type
        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/' + model_filename
            model = YOLO(url)  # pass any model type
        self.model = model

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

    @staticmethod
    def copy_files(src_path, dst_path):
        subfolders = [x[0] for x in os.walk(src_path)]
        os.makedirs(dst_path, exist_ok=True)

        for subfolder in subfolders:
            for filename in os.listdir(subfolder):
                file_path = os.path.join(subfolder, filename)
                if os.path.isfile(file_path):
                    # Get the relative path from the source directory
                    relative_path = os.path.relpath(subfolder, src_path)
                    # Create a new file name with the relative path included
                    new_filename = f"{relative_path.replace(os.sep, '_')}_{filename}"
                    new_file_path = os.path.join(dst_path, new_filename)
                    shutil.copy(file_path, new_file_path)

    def train(self, data_path, output_path, **kwargs):
        self.model.model.args.update(self.configuration.get('modelArgs', dict()))
        epochs = self.configuration.get('epochs', 50)
        start_epoch = self.configuration.get('start_epoch', 0)
        batch_size = self.configuration.get('batch_size', 2)
        imgsz = self.configuration.get('imgsz', 640)
        device = self.configuration.get('device', None)
        augment = self.configuration.get('augment', True)
        yaml_config = self.configuration.get('yaml_config', dict())
        lr0 = self.configuration.get('lr0', 0.01)
        auto_augment = self.configuration.get('auto_augment', 'randaugment')
        degrees = self.configuration.get('degrees', 0.0)

        resume = start_epoch > 0
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

        # copy images and labels to train and validation directories
        self.copy_files(src_images_path_train, dst_images_path_train)  # add dir to name
        self.copy_files(src_images_path_val, dst_images_path_val)
        self.copy_files(src_labels_path_train, dst_labels_path_train)
        self.copy_files(src_labels_path_val, dst_labels_path_val)

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

        data_yaml_filename = os.path.join(data_path, f'{self.model_entity.dataset_id}.yaml')
        yaml_config.update(params)
        with open(data_yaml_filename, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        faas_callback = kwargs.get('on_epoch_end_callback')

        def on_epoch_end(train_obj):

            self.current_epoch = train_obj.epoch
            metrics = train_obj.metrics
            train_obj.plot_metrics()
            if faas_callback is not None:
                faas_callback(self.current_epoch, epochs)
            samples = list()
            NaN_dict = {'box_loss': 1,
                        'cls_loss': 1,
                        'dfl_loss': 1,
                        'mAP50(B)': 0,
                        'mAP50-95(B)': 0,
                        'precision(B)': 0,
                        'recall(B)': 0}
            for metric_name, value in metrics.items():
                legend, figure = metric_name.split('/')
                logger.info(f'Updating figure {figure} with legend {legend} with value {value}')
                if not np.isfinite(value):
                    filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                    filters.add(field='modelId', values=self.model_entity.id)
                    filters.add(field='figure', values=figure)
                    filters.add(field='data.x', values=self.current_epoch - 1)
                    items = self.model_entity.metrics.list(filters=filters)

                    if items.items_count > 0:
                        value = items.items[0].y
                    else:
                        value = NaN_dict.get(figure, 0)
                    logger.warning(f'Value is not finite. For figure {figure} and legend {legend} using value {value}')
                samples.append(dl.PlotSample(figure=figure,
                                             legend=legend,
                                             x=self.current_epoch,
                                             y=value))
            self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)
            # save model output after each epoch end
            self.configuration['start_epoch'] = self.current_epoch + 1
            self.save_to_model(local_path=output_path, cleanup=False)

        self.model.add_callback(event='on_fit_epoch_end', func=on_epoch_end)
        self.model.train(data=data_yaml_filename,
                         exist_ok=True,  # this will override the output dir and will not create a new one
                         resume=resume,
                         epochs=epochs,
                         batch=batch_size,
                         device=device,
                         augment=augment,
                         name=name,
                         workers=0,
                         imgsz=imgsz,
                         project=project_name,
                         auto_augment=auto_augment,
                         degrees=degrees,
                         lr0=lr0)
