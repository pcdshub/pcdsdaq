Basic Usage
===========
The `AmiDet` class can be used to read the mean and rms of an ami variable
over time. This can be useful for keeping track of DAQ scans in the python
layer.

This module does a best-effort approach of setting itself up automatically, but
if this isn't working for your setup you may need to call `set_pyami_proxy` and
`set_l3t_file` before everything will work.

Creating an AmiDet object
-------------------------
This code will set up an `AmiDet` object. Replace
``'AMI-NAME'`` with the name used in AMI to reference the data, and replace
``'ami_det'`` with the desired table label in the ``bluesky`` scans.

.. code-block:: python

    from pcdsdaq.ami import AmiDet

    det = AmiDet('AMI-NAME', name='ami_det')


In a Scan
---------
These `AmiDet` objects should be added to the ``dets`` list of a ``bluesky``
scan. If used as configured above, they will collect data at each scan step
while the daq is running. Below is an example scan, see the `plans_basic` page
for general information about daq scans.

.. code-block:: python

    from bluesky.plans import count

    daq.configure(events=120)
    RE(count([det, daq], 10))

.. note::
    I highly recommend you place any ami detectors BEFORE the daq object in the
    dets list, to ensure they are ready before the daq starts running. Failing
    to do this can result in missing some events.


Interactively
-------------
You can also use `AmiDet` interactively, perhaps in a second session, to snoop
on AMI information. You should use the same methods that the scan does to
control the collection of data: ``stage`` will begin collecting data,
``trigger`` will reset the data collection, and ``unstage`` will stop
collecting data. You can use ``get`` or ``read`` to check the result depending
on the level of detail you need.


Advanced Options
----------------
- Use the ``filter_string`` kwarg or set ``det.filter_string`` to filter the
  incoming data for the `AmiDet`. You can also use the `Daq.set_filter` method
  to configure the l3t trigger and set the default filtering for `AmiDet`
  instances.
- Use the ``min_duration`` kwarg or set ``det.min_duration`` to a positive
  number to specify a minimum time spent measuring data at each point. This is
  not needed if you are running pydaq and pyami in the same scan, but it may be
  useful if you are free-running the daq and need to control acquisition times
  in your scan.
- Use the ``normalize`` kwarg or set ``det.normalize`` to control normalization
  behavior (default: ``True``). If ``True``, we'll normalize with the detector
  chosen with `Daq.set_monitor` (if applicable), if a specific detector
  then we'll normalize with that detector, and if ``False`` then we'll skip
  normalization altogether.
