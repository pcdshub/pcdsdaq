Simulated DAQ
=============
If pydaq is unavailable or the real DAQ nodes are occupied,
you can still test your code against a simulated daq module.

Simply do:

.. code-block:: python

   from pcdsdaq.sim import set_sim_mode
   set_sim_mode(True)


.. autofunction:: pcdsdaq.sim.set_sim_mode
