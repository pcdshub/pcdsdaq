=============
Miscellaneous
=============

This page will show off some features that are not big enough to warrant their
own page.

.. ipython:: python
   :suppress:

   from pcdsdaq.daq import Daq
   from pcdsdaq.sim import set_sim_mode

   set_sim_mode(True)
   daq = Daq()

Run Number
----------

Call `Daq.run_number` to get the current run number. This will either be the
run number of the last run if we are not recording, or the current run if
we are.

.. ipython:: python

   run_num = daq.run_number()
