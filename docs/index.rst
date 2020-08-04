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

Examples
------------------




Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
