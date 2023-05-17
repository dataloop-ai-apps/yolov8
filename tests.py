from model_adapter import Adapter
import dtlpy as dl
import numpy as np
import json
from PIL import Image


def test_predict():
    adapter = Adapter(model_entity=model_entity)

    adapter.predict_items(items=[dl.items.get(item_id='6419cbcab8ab76e9d4ebc732')])
    #
    im1 = Image.open(
        r"E:\ModelsZoo\yolov8\tmp\640ee84307a569363353ed6a\datasets\64089e249ab310729aa04aa0\train\images\train\batch_1__000050.jpg")
    results = adapter.model.predict(source=im1, save=True, save_txt=True)  # save predictions as labels
    item = dl.items.get(None, '630cd62e7a00b0b71a95196c')
    a = adapter.predict_items(items=[item], with_upload=True)


def test_clone():
    to_dataset = dl.datasets.get(None, '630caf79ee6c90226a406f31')
    to_project = to_dataset.project

    m = model_entity.clone(model_name='fruits with figures exp2',
                           description='exp 2',
                           # labels=list(to_dataset.labels_flat_dict.keys()),
                           dataset=to_dataset,
                           configuration={'num_epochs': 20,
                                          'batch_size': 4},
                           project_id=to_project.id)


def test_local_train(model_entity):
    from model_adapter import Adapter
    adapter = Adapter(model_entity=model_entity)

    # adapter.prepare_data(dataset=dl.datasets.get(dataset_id='64089e249ab310729aa04aa0'),
    #                      root_path='64089e249ab310729aa04aa0')
    adapter.train_model(model=model_entity)
    # m.labels = list(m.dataset.labels_flat_dict.keys())
    # m.update()

    # split
    # items = list(m.dataset.items.list().all())
    # print(np.unique([item.dir for item in items]))
    #
    # item: dl.Item
    # for item in items:
    #     if np.random.random() > 0.8:
    #         item.move(f'/train{item.filename}')
    #     else:
    #         item.move(f'/validation{item.filename}')
    # m.dataset.metadata['system']['subsets'] = {
    #     'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
    #     'validation': json.dumps(dl.Filters(field='dir', values='/test').prepare()),
    # }
    # m.dataset.update(True)


def test_remote_train(model_entity: dl.Model):
    model_entity.train()


if __name__ == "__main__":
    dl.setenv('prod')
    model_entity = dl.models.get(None, '643d0aa66816dd593744093e')
    package = model_entity.package
    # model_entity.bucket.upload(r"C:\Users\Shabtay\Downloads\New folder")
    test_local_train(model_entity=model_entity)
