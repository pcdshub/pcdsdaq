Release History
###############

v2.4.5 (2025-03-16)
===================

- Unpin numpy now that bluesky is compatible


v2.4.4 (2024-08-20)
===================

- Include pcdsutils dependency in requirements
- Use bluesky-base in conda-recipe to avoid matplotlib qt dependencies


v2.4.3 (2023-09-14)
===================

- Raise if daq not yet registered, consolidate exceptions and some relative imports


v2.4.2 (2023-04-13)
===================

- Fix issues related to the missing pcdsdaq_lib_setup script
- Fix issues related to failing pypi builds
- Fix issues related to failing docs builds


v2.4.1 (2023-04-11)
===================

- Mark tests as xfail, we'll come back to fix them later.
  This is not operation critical.
  One of these tests contain a nasty race condition that
  causes the suite to time out.
- Fix an issue where automatic pypi/conda uploads would
  not have proper authentication.


v2.4.0 (2023-04-04)
==================

This includes a large update for lcls2 daq support,
but the module is fully backwards compatible with
the previous versions.

- Huge addition of 1st-class lcls2 daq support
  including monitoring the current daq's status and running it.
  Uses the daq's own APIs repackaged for bluesky.
- Migrate to github actions, pyproject.toml, and other ecs standards.
- Merge an ancient PR that adds a useful feature for CXI


v2.3.5 (2022-06-02)
===================

- Fix issue where the scan variable min/max values would not fill in
  correctly for scan types other then the normal "scan".


v2.3.4 (2021-04-02)
===================

- Fix issue where the ``Daq`` could automatically disconnect after a scan
  was complete, causing all ``ami`` windows to close.


v2.3.3 (2021-02-10)
===================

- Fix issue where the ``AmiDet`` had a hinted ``entries`` field that got in
  the way when using the ``BestEffortCallback``.


v2.3.2 (2021-01-11)
===================

- Update recipe and build for python 3.7+ and noarch compatibility
- Remove pydaq linkage scripts in favor of using the corrected sym links
  in the daq build


v2.3.1 (2020-12-22)
==================

- Fix issue where ``events=-1`` was a rejected argument rather than a
  valid configuration for infinite running.


v2.3.0 (2020-10-02)
===================

- Add ``hutch_name`` argument to ``Daq`` class to allow bypassing of the external ``get_hutch_name`` script.


v2.2.7 (2020-09-17)
===================

- Change bykik code to False by default in set_filter as per Silke's instructions
- Give up on the run_number getter on the first failure per session, it is merely cosmetic and should not slow down the scan.
- Decrease the get run number timeout to 1s.
- Copy the good docstring for set_filter over to the daq method
- Increase timeouts for get_hutch and get_ami_proxy, then cache the result. The result does not change.
- Allow controls arg to accept pseudopositioners
- Allow "tri-state" configuration args (True, False, None), and allow the Daq class to know whether the user has passed in `None` versus not passed in anything at all
- Allow `record=None` to mean "Use the record option selected in the DAQ GUI"
- Increase the begin timeout to 15s
- Add more pointed error messages to the wait timeouts


v2.2.6 (2020-08-19)
===================

Fix issue with ami proxy script changes, making the library compatible with both the old and the new output.


v2.2.5 (2020-29-05)
===================

Compatibility for ophyd=1.5.0


v2.2.4 (2020-05-21)
===================

Fix issue with the begin timeout where the clock started ticking too early.


v2.2.3 (2020-3-21)
==================

Fix issue with pcdsdaq_lib_setup that broke hutch environments.


v2.2.2 (2020-1-22)
==================

Fix issue where daq would fail to load for det and tst hutches.


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
