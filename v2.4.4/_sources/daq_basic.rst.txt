Basic Usage
===========
The `Daq` class allows a user to connect to the LCLS1 daq and control it.
For this to work, the daq must be up and allocated.

The most important method is `Daq.begin`. This will connect, configure,
and start running the daq all in one call. `Daq.stop` can be used to stop
acquiring data without ending a run. `Daq.begin` can be called again to
resume a stopped run. `Daq.end_run` will end the run.

`Daq.state` can be used to inspect what the daq is currently doing. It will
return one of the following possibilities:

+------------------+------------------------------------------------+
| State            | Meaning                                        |
+==================+================================================+
| ``Disconnected`` | We are not controlling the daq                 |
+------------------+------------------------------------------------+
| ``Connected``    | We are controlling the daq                     |
+------------------+------------------------------------------------+
| ``Configured``   | ``Connected``, and `configure` has been called |
+------------------+------------------------------------------------+
| ``Open``         | ``Configured``, and we are in a run            |
+------------------+------------------------------------------------+
| ``Running``      | ``Open``, and we are collecting data           |
+------------------+------------------------------------------------+

I will step through the basic options for `Daq.begin` below. You can consult
the full `api docs <./daq_api>` for more information.

Creating a Daq object
---------------------
First, I will set up the `Daq` class in simulated mode. In practice, the
`Daq` class will be set up for you in the ``hutch-python`` configuration.

.. code-block:: python

    from pcdsdaq.daq import Daq
    from pcdsdaq.sim import set_sim_mode

    set_sim_mode(True)
    daq = Daq()


.. ipython:: python
    :suppress:

    import time
    from pcdsdaq.daq import Daq
    from pcdsdaq.sim import set_sim_mode

    set_sim_mode(True)
    daq = Daq()


Running Until Stop
------------------
Calling `Daq.begin` with no arguments in the default configuration will
run the daq indefinitely, until we manually stop it.
You can also use `Daq.begin_infinite` to run the daq indefinitely with
any configuration.
Here I check `Daq.state` to verify that we've started running the daq.

.. ipython:: python

    daq.state
    start = time.time()
    daq.begin()
    daq.state
    time.sleep(1)
    daq.stop()
    print(time.time() - start)
    daq.state
    daq.end_run()
    daq.state


Running for a Fixed Number of Events
------------------------------------
Use the ``events`` argument to specify duration in number of events.
The duration of this run will depend on the current beam rate.
Optionally, we can call `Daq.wait` to pause until acquisition is complete.

.. ipython:: python

    daq.state
    start = time.time()
    daq.begin(events=240)  # 120Hz
    daq.state
    daq.wait()
    print(time.time() - start)
    daq.state
    daq.end_run()
    daq.state


Runing for a Fixed Time Duration
--------------------------------
Use the ``duration`` argument to specify duration in seconds.
We can pass ``wait=True`` to skip the `Daq.wait` call.

.. ipython:: python

    daq.state
    start = time.time()
    daq.begin(duration=1.5, wait=True)
    print(time.time() - start)
    daq.state
    daq.end_run()
    daq.state


Ending a Run
------------
As seen in the previous examples, `Daq.end_run` can be used to tell the daq
that the current run is over. You can also do with with an argument to
`Daq.begin` for a nice one-liner:

.. ipython:: python

    daq.state
    daq.begin(duration=1, wait=True, end_run=True)
    daq.state


Recording Data
--------------
You can set `Daq.record` to ``True`` to record data. This is fairly simple:

.. ipython:: python

    daq.record = True


After this call, future calls to `Daq.begin` will record data to disk.
You can undo this by simply setting:

.. ipython:: python

    daq.record = False


You can also record data for a single run using a keyword argument in
`Daq.begin`:

.. ipython:: python

   daq.record
   daq.begin(events=120, record=True)
   daq.record


Advanced Options
----------------
- ``use_l3t=True``: This will reinterpret the ``events`` argument as
                    "the number of events that pass the level 3 trigger."
- ``controls=[motor1, motor2...]``: This will post the name of each motor and
                    the current position to the daq data stream. This is
                    handled automatically with some of the ``bluesky`` tools.
- ``begin_sleep=0.25``: This configuration argument will set the empirically
                    derived sleep time needed after a call to ``begin`` that
                    ensures the daq is actually ready. If a valid argument for
                    ``time.sleep``, this will wait to end a ``begin`` call
                    until the configured sleep time elapses. This may be
                    useful if you have other devices that rely on a run to
                    actually start before doing some action.
