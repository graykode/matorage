
Matorage Introduction
==========================

Why it's convenient to use pre-processed numeric matrix file for training
---------------------------------------------------

As growing in dataset size and complexity of pre-processing in deep learning, The time spent reading and preprocessing data
is more important than real learning, which increases overall learning time.

To solve this problem, we used to preprocess in advance and save it as a single file.
For example, it is a typical method to store it as a pre-processed Tensor format in `tfrecord <https://www.tensorflow.org/tutorials/load_data/tfrecord>`_ format on tensorflow, or to store
a multi-dimensional matrix in numpy format using HDF5-based `pytables <https://www.pytables.org/>`_ or `h5py <https://docs.h5py.org/en/stable/>`_ to read it during training.

If you save and use this file, you do not have to do the same pre-processing during training hyperparameter search or next epoch training,
which has the great advantage of speeding up your learning. In addition, most of these formats also support the lazy load,
which has the advantage of not having to load all the data in memory at once.

Now let's introduce matorage
---------------------------------------------------

However, both the existing tfrecord and h5 file formats are managed as one large file, which has difficulties
in sharing large data between training GPU nodes and difficulties in recovering when the file is broken. It is also difficult to add newly created data.

We overcome this disadvantage and introduce a project called Matorage (matrix + storage) that can efficiently store/load and manage data in both researching and production stages. (Pytorch, Tensorflow V2, tf.keras are supported)

1. matorage uses `MinIO <https://min.io/>`_ backend as an Object Storage with HDF5 format pytables. You can manage not only
   the training data but also the trained model and data generated during training.
2. The pre-processed matrix data is divided into several pieces and stored in Object Storage to prevent damage to one large file.
3. You can use either private MinIO Storage or public Amazon S3 as the backend storage, so you can easily share between
   the training GPU nodes or user groups that want to share data.
4. The data and model are managed as key-value pairs, and the data version, author, and detailed description as well as
   the name of the dataset can be easily managed with metadata.
5. If there is training data that is dynamically generated from the user's edge endpoint, the matorage pipeline can be used
   to store it in storage to manage the application pipeline more efficiently.
6. It supports data read and load concurrency by default, and optional in-memory option to use local storage more efficiently.
7. (Long-term plan) It supports Nvidia GPU direct storage, so it reads faster by loading it directly to GPU memory without going through CPU DMA from NVMe SSD.
