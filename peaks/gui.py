"""Manage open `peaks` GUI viewers.

Examples
--------
Example usage is as follows::

    import peaks as pks

    pks.gui.viewers()      # open and suspended viewers, oldest first
    pks.gui.suspend_all()  # suspend all live inline notebook viewers
    pks.gui.close_all()    # close every peaks viewer, including native windows
"""


def viewers():
    """Return the open and suspended `peaks` viewers, oldest first.

    Each has ``backend`` (``"qt"`` or ``"notebook"``), ``status`` and ``close()``;
    inline viewers also have ``suspend()``.
    """
    from peaks.core.GUI.GUI_utils.qt_runtime import viewers

    return viewers()


def suspend_all():
    """Suspend every live inline notebook viewer; native windows are not affected.

    Suspended viewers keep a static image in the notebook and release their Qt
    resources. Re-run a cell to make its viewer interactive again.
    """
    from peaks.core.GUI.GUI_utils.qt_runtime import suspend_all

    suspend_all()


def close_all():
    """Close every `peaks` viewer: native windows and live or suspended inline ones."""
    from peaks.core.GUI.GUI_utils.qt_runtime import close_all

    close_all()
