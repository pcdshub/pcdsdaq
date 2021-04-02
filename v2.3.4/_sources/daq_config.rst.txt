Configuration
=============
Some manual configuration is necessary to record data or to run
special ``bluesky`` plans with daq support.

You can get the current configuration from `Daq.config`.
Shown here is the default config:


.. ipython:: python
    :suppress:

    from pcdsdaq.daq import Daq
    from pcdsdaq.sim import set_sim_mode

    set_sim_mode(True)
    daq = Daq()


.. ipython:: python

    daq.config

You can also print this information nicely using `Daq.config_info`.

You can configure the daq through `Daq.configure`, which configures
the daq right away, or `Daq.preconfig`, which schedules a configuration
to be done when it is next needed. This distinction is sometimes important
because we cannot do a full configure during an open run, for example.
As an aside, `Daq.configure` has a long return value, which may make
`Daq.preconfig` the preferred method for interactive sessions.

In general, there are two kinds of configuration arguments:
those that are shared with `Daq.begin`, and those that are not.

Arguments that are shared between the two methods act as defaults.
For example, calling ``daq.configure(duration=3)`` will set the
no-arguments behavior of ``daq.begin()`` to run the daq for 3 seconds.


.. automethod:: pcdsdaq.daq.Daq.configure
   :noindex:
