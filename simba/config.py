
_t_diff = 1e-6


def get_t_diff():
    """The global time jitter for time-based discontinuous components.

    As the majority of ODE-solvers do not hit the evaluation times perfectly, it may happen that the ODE is evaluated
    a little too early by the ODE-solver. However, a time-based discontinuous signal should switch in such a case.
    Therefore, :math:`t_{diff}` specifies a maximal early switching time.

    Returns:
        float: The time jitter :math:`t_{diff}` in seconds.
    """
    return _t_diff


def set_t_diff(value: [float, int]):
    global _t_diff
    _t_diff = float(value)
