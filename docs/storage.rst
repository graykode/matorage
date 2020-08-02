
Storage Setting
==========================

Storage Default Settings
---------------------------------------------------

You can get started quickly through docker or basic installation. Create a MinIO Storage with network access storage (NAS)
settings with the default settings through the following command. For detailed storage settings such as distributed mode,
refer to `the corresponding document <https://docs.min.io/docs/>`_.

.. code-block:: bash

    $ mkdir ~/shared
    $ docker run -it -p 9000:9000 --restart always \
            -e "MINIO_ACCESS_KEY=minio" \
            -e "MINIO_SECRET_KEY=miniosecretkey" \
            -v ~/shared:/container/vol minio/minio \
            gateway nas /container/vol
