Environments
############

The LCLS1 Daq comes packaged with python C modules that are used to
communicate with it. These cannot be installed like normal modules
because they must always be synchronized with the live daq's current
version, adding too much environment overhead. Therefore, utilities
are provided with this module to set up the environment.

This module comes with a sourceable ``pcdsdaq_lib_setup`` script.
This works by extending your ``PYTHONPATH`` and ``LD_LIBRARY_PATH`` based on
the exported ``HUTCH`` environment variable and the operating system you are
running on (rhel6 and rhel7 are supported). If ``HUTCH`` is unset, we'll use
the latest versions. This will work out of the box if the module is installed
in a conda environment.

If not installed in a conda environment, you'll need to do the following
commands or the script will not work:

.. code-block:: bash

   $ cd pcdsdaq/pydaq_links
   $ ./linker

.. note::

   These linked modules only exist on the LCLS NFS filesystem.
