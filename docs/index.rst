matorage documentation
====================================

Efficiently store/load and manage models and training data needed for deep learning learning with matorage!

Matorage is Tensor(multidimensional matrix) Object Storage Manager with high availability distributed systems for
Deep Learning framework(Pytorch, Tensorflow V2, Keras).

Features
------

- Boilerplate data pipeline!
- remove pre-processing time on training!

**For researchers who need to focus on model training**:

- Help researchers manage training data and models easily.
- By storing data in pre-processed Tensor(multidimensional matrix), they can focus only model training.
- Reduce storage space through multiple compression methods.
- Manage data and models that occur during learning

**For AI Developer who need to focus on creating data pipeline:**

- Enables concurrency data save & load
- High availability guaranteed through backend storage.
- Easily create pipeline from user endpoints data.

Guides
------

.. toctree::
   :maxdepth: 3

   introduce

   storage

   dataset

   dataset_qa

   model

   optimizer

   model_optim_qa

Quick Start with Pytorch Example!
------------------
For an example of tensorflow, refer to the detailed document.

1. Set up Minio Server with docker
~~~~~~~~~~~~~~~~~~~~~

quick start with NAS(network access storage) using docker!

It can be managed through the web through the address http://127.0.0.1:9000/, and security is managed through ``MINIO_ACCESS_KEY`` and ``MINIO_SECRET_KEY``.

    .. code-block:: bash

        $ mkdir ~/shared # create nas storage folder
        $ docker run -it -p 9000:9000 \
            --restart always -e \
            "MINIO_ACCESS_KEY=minio" -e \
            "MINIO_SECRET_KEY=miniosecretkey" \
            -v ~/shared:/container/vol \
            minio/minio gateway nas /container/vol

2. Save pre-processed dataset
~~~~~~~~~~~~~~~~~~~~~

First, create a ``DataConfig`` by importing matorage.

This is an example of pre-processing mnist and storing it in distributed storage.
``additional`` is freely in the form of a dict, and records the shape and type of tensor to be stored in ``attributes``.

    .. code-block:: python

        from matorage import DataConfig

        traindata_config = DataConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            dataset_name='mnist',
            additional={
                "mode": "train",
                "framework" : "pytorch",
                ...
                "blah" : "blah"
            },
            attributes=[
                ('image', 'float32', (1, 28, 28)),
                ('target', 'int64', (1))
            ]
        )

Now do a simple pre-processing and save the data.

    .. code-block:: python

        from matorage import DataSaver

        traindata_saver = DataSaver(config=traindata_config)
        train_loader = DataLoader(dataset, batch_size=60, num_workers=8)
        for (image, target) in tqdm(train_loader):
            traindata_saver({
                'image': image,
                'target': target
            })
        traindata_saver.disconnect()

3. Load dataset from matorage
~~~~~~~~~~~~~~~~~~~~~

Now fetch data iteratively from storage with the same config as the saved dataset when training.

    .. code-block:: python

        from matorage.torch import Dataset

        train_dataset = Dataset(config=traindata_config, clear=True)
        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)

        for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
            image, target = image.to(device), target.to(device)

Only an index can be fetched through lazy load.

    .. code-block:: python

        train_dataset = Dataset(config=traindata_config, clear=True)
        print(train_dataset[0], len(train_dataset))

4. Save & Load Model when training
~~~~~~~~~~~~~~~~~~~~~

During training, you can save and load models of specific steps or epochs in distributed storage through inmemory.
First, make the model config the same as the dataset.

    .. code-block:: python

        from matorage import ModelConfig
        from matorage.torch import ModelManager

        model_config = ModelConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            model_name='mnist_simple_training',
            additional={
                "version" : "1.0.1",
                ...
                "blah" : "blah"
            }
        )

        model_manager = ModelManager(config=model_config)
        print(model_manager.get_metadata)
        model_manager.save(model, epoch=1)
        print(model_manager.get_metadata)

When an empty model is loaded with specific steps or epochs, the appropriate weight is filled into the model.

    .. code-block:: python

        print(model.state_dict())
        model_manager.load(model, epoch=1)
        print(model.state_dict())

5. Save & Load Optimizer when training
~~~~~~~~~~~~~~~~~~~~~

Save and load of optimizer is similar to managing model.

    .. code-block:: python

        from matorage import OptimizerConfig
        from matorage.torch import OptimizerManager

        optimizer_config = OptimizerConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            optimizer_name='adam',
            additional={
                "model" : "1.0.1",
                ...
                "blah" : "blah"
            }
        )

        optimizer_manager = OptimizerManager(config=optimizer_config)
        print(optimizer_manager.get_metadata)
        # The optimizer contains information about the step.
        optimizer_manager.save(optimizer)
        print(optimizer_manager.get_metadata)

When an empty optimizer is loaded with specific steps, the appropriate weight is filled into the optimizer.

    .. code-block:: python

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer_manager.load(optimizer, step=938)

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
