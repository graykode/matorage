
Dataset
==========================

The dataset is divided into a ``DataSaver`` that stores data and a ``DataLoader`` that loads the stored data when training.

Both ``DataSaver`` and ``DataLoader`` are managed through ``DataConfig``.

DataConfig
---------------------------------------------------

With what features is the data managed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data is abstracted and managed with the following features.

- dataset name - (mnist, imagenet, ...)

- additional information - (creator, training or test, version, ...)

- metadata information - (compressor, attributes)

DataSaver
---------------------------------------------------

What rules divide data in saving?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user sets ``max_object_size`` to set the maximum size that can be stored in one file. (default to 10MB)
Storing data is similar to the consumer-producer pattern. The consumer accumulates the pre-processed data
in the queue, and the producer takes the queue out of ``max_object_size`` and uploads it.
Therefore, the larger ``max_object_size``, the fewer objects are created.

How is data stored in storage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To fastly use HDF5 writes for continuous numpy data in cache, a `contiguous rows <https://www.slideshare.net/HDFEOS/caching-and-buffering-in-hdf5#25>`_ method is used.
Therefore, if ``(batch_size, 3, 224, 224)`` shape data is stored, reshape to ``(batch_size, 3 * 224 * 224)`` shape inside matorage.
When data is fetched (via iterative or index), it is decoded to original shape.

How guarantee concurrency and safety in saving?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process-independently, DataSaver stores data, creates metadata, and uploads it to the backend storage with multi threading.
Therefore, after storage, you can see that there are many json files in the metadata folder in object storage.

DataLoader
---------------------------------------------------

What is different between iterative load and index load.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterative load refers to fetching all data sequentially or by the shuffle. Therefore, it downloads all data locally before training and caches the downloaded file path.
In contrast, index load does not fetch data directly, but loads the corresponding object from memory.
Downloaded data can be cleared after training is over.

How guarantee concurrency and safety in loading?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike DataSaver, DataLoader has different implementations depending on the framework.
Pytorch guarantees concurrency through ``torch.utils.data.DataLoader`` with multi workers, while Tensorflow guarantees through ``interleave`` of ``tf.data.Dataset`` with tensorflow io.
The reason to fetch all data before training is to ensure MinIO safety.