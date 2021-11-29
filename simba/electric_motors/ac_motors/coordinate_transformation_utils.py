import numpy as np

# transformation matrix from abc to alpha-beta representation
_t23 = 2 / 3 * np.array([
    [1, -0.5, -0.5],
    [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)]
])

# transformation matrix from alpha-beta to abc representation
_t32 = np.array([
    [1, 0],
    [-0.5, 0.5 * np.sqrt(3)],
    [-0.5, -0.5 * np.sqrt(3)]
])


def t_23(quantities):
    """
    Transformation from abc representation to alpha-beta representation
    Args:
        quantities: The properties in the abc representation like ''[u_a, u_b, u_c]''
    Returns:
        The converted quantities in the alpha-beta representation like ''[u_alpha, u_beta]''
    """
    return np.dot(_t23, quantities)


def t_32(quantities):
    """
    Transformation from alpha-beta representation to abc representation
    Args:
        quantities: The properties in the alpha-beta representation like ``[u_alpha, u_beta]``
    Returns:
        The converted quantities in the abc representation like ``[u_a, u_b, u_c]``
    """
    return np.dot(_t32, quantities)


def q(quantities, epsilon):
    """
    Transformation of the dq-representation into alpha-beta using the electrical angle
    Args:
        quantities: Array of two quantities in dq-representation. Example [i_d, i_q]
        epsilon: Current electrical angle of the motor
    Returns:
        Array of the two quantities converted to alpha-beta-representation. Example [u_alpha, u_beta]
    """
    cos = np.cos(epsilon)
    sin = np.sin(epsilon)
    return cos * quantities[0] - sin * quantities[1], sin * quantities[0] + cos * quantities[1]


def q_inv(quantities, epsilon):
    """Transformation of the alpha-beta-representation into dq using the electrical angle
    Args:
        quantities: Array of two quantities in alpha-beta-representation. Example [u_alpha, u_beta]
        epsilon: Current electrical angle of the motor
    Returns:
        Array of the two quantities converted to dq-representation. Example [u_d, u_q]
    Note:
        The transformation from alpha-beta to dq is just its inverse conversion with negated epsilon.
        So this method calls q(quantities, -epsilon).
    """
    return q(quantities, -epsilon)
