
Model and Optimizer Q & A
==========================

Both model and optimizer consist of ``Config`` and ``Manager`` classes.

You can configure the model or optimizer in the ``Config``, and save and load through the ``Manager``.

ModelConfig(OptimizerConfig)
---------------------------------------------------

With what features is the model(optimizer) managed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model(optimizer) is abstracted and managed with the following features.

- model(optimizer) name - (transformer, BERT, Adam, AdamW, ...)

- additional information - (creator, version, special information, ...)

- metadata information - (compressor)

Model(Optimizer)Saver
---------------------------------------------------

Why don't store model(optimizer) information in local storage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the model(optimizer) is attached to the device memory, to save it to the backend storage, you can save it without saving it as a file in the local storage.
If you want to save as a single file in local storage (ex, h5 format), use the save function for each framework.

In what format do you save/load the model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In both pytorch and tensorflow, the layer name is stored as key and the corresponding weight is stored as value.
Therefore, unlike the existing load method, the **weight or submodel of a specific layer** can be imported without downloading the entire large model file.

When saving a model, training information related to the model such as step or epoch can be saved as a key and reloaded through that key.

In what format do you save/load the optimizer?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimizer gains weight after training for both pytorch and tensorflow. Therefore, since it has an optimizer for a step, unlike the model, it is not necessary to separately input metadata for training when saving.

Like the model, the optimizer manages the weight of each layer as one object file. However, unlike models, the optimizer has its own parameters, and more files are created by combining model layer weights for them.

For more details, see the `torch.optim <https://pytorch.org/docs/stable/optim.html>`_ and `tf.keras.optimizers.Optimizer <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer>`_.