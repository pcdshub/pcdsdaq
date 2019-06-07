Release History
###############

v2.2.1 (2019-6-6)
=================

Bugfixes
--------
- Fix issue where the daq could rapidly cycle under specific conditions
- Fix issue where the daq class wasn't correctly reset after a disconnect
- Fix issue where deprecated ``platform`` argument was mistakenly left in at
  2.0.0
- Fix issue where module was broken on newest ``bluesky`` because ``None`` is
  no longer a valid data shape


v2.2.0 (2018-10-12)
===================

Features
--------
- Add `AmiDet` interface for interacting with ``pyami``
- Clean up and update ``pcdsdaq_lib_setup`` to work with ``pyami``
- Allow daq to be configured to run "forever, until everything else is done
  triggering" by setting ``events=0``
- Add configurable sleep time to account for the difference between the daq
  process claiming to be ready and actually being ready
  (``daq.configure(begin_sleep=0.5)``)

Bugfixes
--------
- Fix issue where package update would break the conda environment


v2.1.0 (2018-08-06)
===================

Features
--------
- Add `Daq.run_number` method to get the current run number.

v2.0.0 (2018-05-27)
===================

Features
--------
- Allow ``ctrl+c`` during a `begin` call with ``wait=True`` to stop the run.
- Add sourcable ``pcdsdaq_lib_setup`` script that will get ``pydaq`` and
  ``pycdb`` ready for your python environment.
- The `connect` method will provide more helpful error messages when it fails.
- Allow the `Daq` class to be used as a ``bluesky`` readable device.
  Once staged, runs will end on run stop documents.
  A calibcycle will be run when the `Daq` is triggered, and triggering will be
  reported as done when the `Daq` has stopped. This means it is viable to use
  the `Daq` inside normal plans like ``scan`` and ``count``.
- Add an argument to `Daq.begin`: ``end_run=True`` will end the run once the
  daq stops running, rather than leaving the run open.
- Add `Daq.begin_infinite`
- Add `Daq.config_info`
- Restore daq state after a ``bluesky`` ``plan``, e.g. disconnect if we were
  disconnected, run if we were running, etc.
- Add support for scan PVs via the `ScanVars` class. This class attaches
  itself to a ``RunEngine`` and knows when to update each PV, provided that
  the ``plan`` has reasonable metadata.

API Changes
-----------
- ``calib_cycle`` and related ``plans`` module has been removed, as using the
  `Daq` as a readable device is more intuitive and it's still early enough to
  break my API.
- ``daq_wrapper`` and ``daq_decorator`` have been move to the ``preprocessors``
  submodule, as a parallel to the ``bluesky`` structure. They have been renamed
  to `daq_during_wrapper` and `daq_during_decorator` as a parallel to the
  built-in ``fly_during_wrapper``. These are now simple preprocessors to
  run the daq at the same time as a daq-agnostic plan.
- ``complete`` no longer ends the run. This makes it more in line with the
  ``bluesky`` API.

Deprecations
------------
- The daq no longer needs to be passed a ``platform`` argument. This argument
  will be removed in a future release, and will log a warning if you pass it.

v1.2.0 (2018-05-08)
===================

Features
--------
- Add the ``record`` option to the `begin` method. This allows a user running
  interactively to concisely activate recording for single runs.

v1.1.0 (2018-03-07)
===================

Features
--------
- Add ``daq.record`` property to schedule that the next run sould be
  configured with ``record=True``

Bugfixes
--------
- Fix bug where configured record was overridden on every configure

v1.0.0 (2018-03-02)
===================

- Initial release, transferred from `<https://github.com/pcdshub/pcdsdevices>`_
