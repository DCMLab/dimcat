======
Manual
======

Dimcat has two main object types, :class:`~.pipeline.PipelineStep` and :class:`~.data.Data`.

Every PipelineStep comes with the method ``process_data()`` which accepts a Data object and returns a copy that includes the processed data.

.. code-block:: python

   processed_data = PipelineStep.process_data(data)
