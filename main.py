import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# ----------------------------------------------------------
# Solver functions
# ----------------------------------------------------------

def nu_from_M(M):
    """
    Convert the Mach number to its respective Prandtl-Meyer angle in radians.

    :param M: Mach number
    :return: Prandtl-Meyer angle [radians]
    """
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M * M - 1))) \
           - np.arctan(np.sqrt(M * M - 1))


def mu_from_M(M):
    """
    Convert the Mach number to its respective Mach angle in radians.

    :param M: Mach number
    :return: Mach angle [radians]
    """
    return np.arcsin(1 / M)


def M_from_mu(mu):
    """
    Convert the Mach angle to the Mach number.

    :param mu: Mach angle [radians]
    :return: Mach number
    """
    return 1 / np.sin(mu)


def M_from_nu(nu):
    """
    Convert the Prandtl-Meyer angle to the Mach number.

    :param nu: Prandtl-Meyer angle [radians]
    :return: Mach number
    """
    global nu_global
    nu_global = nu
    sol = optimize.root(nu_M_function, np.array([Me]), tol=1e-8)
    return np.squeeze(sol.x)


def nu_M_function(M):
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M * M - 1))) \
           - np.arctan(np.sqrt(M * M - 1)) - nu_global


def calc_Mjet(Me, pe, pa):
    """
    Calculate the jet boundary Mach number (Mjet) from the exit and atmospheric conditions.

    :param Me: Exit Mach number
    :param pe: Exit static pressure [atm]
    :param pa: Atmospheric static pressure [atm]
    :return: Jet boundary Mach number
    """
    pt = pe * (1 + (gamma - 1) / 2 * Me * Me) ** (gamma / (gamma - 1))
    return np.sqrt(2 / (gamma - 1) * ((pt / pa) ** ((gamma - 1) / gamma) - 1))


def beta_M_function(M):
    """
    Function required for the root-finding method of the nodes immediately adjacent to the top corner. Equivalent with
    the expression nu(M) - mu(M) - beta_global - nu_a_global + phi_a_global == 0.

    :param M: Mach number to be solved for
    :return: nu(M) - mu(M) - beta_global - nu_a_global + phi_a_global
    """
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M * M - 1))) \
        - np.arctan(np.sqrt(M * M - 1)) - np.arcsin(1 / M) - beta_global - nu_a_global + phi_a_global


def alpha_M_function(M):
    """
    Function required for the root-finding method of the nodes immediately adjacent to the bottom corner. Equivalent
    with the expression nu(M) - mu(M) + alpha_global - nu_b_global + phi_b_global == 0.

    :param M: Mach number to be solved for
    :return: nu(M) - mu(M) + alpha_global - nu_b_global + phi_b_global
    """
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M * M - 1))) \
        - np.arctan(np.sqrt(M * M - 1)) - np.arcsin(1 / M) + alpha_global - nu_b_global - phi_b_global


def calc_e_jet(Me, pe, pa, phi_e):
    """
    Calculate the remaining exit conditions and jet (boundary) conditions. The total pressure is constant throughout
    the whole flow, as it is assumed to be isentropic.

    :param Me: Exit Mach number
    :param pe: Exit static pressure [atm]
    :param pa: Atmospheric static pressure [atm]
    :param phi_e: Exit flow angle [radians]
    :return: In this order: Exit Prandtl-Meyer angle [rad], exit Mach angle [rad], jet Mach number, jet Prandtl-Meyer
    angle [rad], jet Mach angle [rad], jet flow angle [rad], total pressure of the flow [atm]
    """
    Mjet = calc_Mjet(Me, pe, pa)
    nu_e = nu_from_M(Me)
    mu_e = mu_from_M(Me)
    nu_jet = nu_from_M(Mjet)
    mu_jet = mu_from_M(Mjet)
    phi_jet = nu_jet - nu_e + phi_e
    pt = pe * (1 + (gamma - 1) / 2 * Me * Me) ** (gamma / (gamma - 1))
    return nu_e, mu_e, Mjet, nu_jet, mu_jet, phi_jet, pt


def ccw(xa, ya, xb, yb, xc, yc):
    """
    Adapted from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    Check whether points A, B and C (in this order) are found in a 2D-plane in a counter-clockwise orientation. This is
    done by checking that the slope of line (A, C) is larger than that of line (A, B). Will be used to check if two line
    segments intersect.

    :param xa: x-position of point A
    :param ya: y-position of point A
    :param xb: x-position of point B
    :param yb: y-position of point B
    :param xc: x-position of point C
    :param yc: y-position of point C
    :return: True if the points are in a counter-clockwise orientation, False otherwise
    """
    return (yc - ya) * (xb - xa) > (yb - ya) * (xc - xa)


def check_intersect(xa, ya, xb, yb, xc, yc, xd, yd):
    """
    Adapted from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    Check whether segments [A, B] and [C, D] intersect, by using the ccw() function. For this to happen, A and B need to
    be separated by [C, D], and C and D by [A, B]. For A and B to be separated by [C, D], then the point sequences
    A, C, D and B, C, D should have different orientations. Similarly, for C and D to be separated by [A, B], then the
    point sequences A, B, C and A, B, D should have different orientations.

    :param xa: x-position of point A
    :param ya: y-position of point A
    :param xb: x-position of point B
    :param yb: y-position of point B
    :param xc: x-position of point C
    :param yc: y-position of point C
    :param xd: x-position of point D
    :param yd: y-position of point D
    :return:
    """
    return ccw(xa, ya, xc, yc, xd, yd) != ccw(xb, yb, xc, yc, xd, yd) and ccw(xa, ya, xb, yb, xc, yc) != \
           ccw(xa, ya, xb, yb, xd, yd)


def get_intersect(xa, ya, xb, yb, xc, yc, xd, yd):
    """
    Return the x- and y-coordinates of the intersection point between segments [A, B] and [C, D].

    :param xa: x-position of point A
    :param ya: y-position of point A
    :param xb: x-position of point B
    :param yb: y-position of point B
    :param xc: x-position of point C
    :param yc: y-position of point C
    :param xd: x-position of point D
    :param yd: y-position of point D
    :return: The x- and y-coordinates of the intersection point, in this order.
    """
    m1 = (yb - ya) / (xb - xa)
    n1 = ya - m1 * xa
    m2 = (yd - yc) / (xd - xc)
    n2 = yc - m2 * xc

    xe = (n2 - n1) / (m1 - m2)
    ye = m1 * xe + n1

    return xe, ye


def init_exit(no_init, h):
    """
    Assign the initial condition values of the nozzle exit to the global data structure.

    :param no_init: Number of nodes created on the exit nozzle initial boundary
    :param h: The height (diameter) of the nozzle exit
    :return: Returns nothing
    """
    # Cycle through all the initial boundary nodes and assign their properties and positions
    for i in range(no_init):
        x[no_init - i - 1, i] = 0
        y[no_init - i - 1, i] = h / 2 - i * h / (no_init - 1)
        phi[no_init - i - 1, i] = 0
        M[no_init - i - 1, i] = Me
        nu[no_init - i - 1, i] = nu_e
        mu[no_init - i - 1, i] = mu_e


def assign_top_corner():
    """
    Assign the values for the first set of nodes at the intersection between the first Gamma+ characteristic with the
    expansion fan of the top corner to the global data structure.

    :return: Returns nothing
    """
    # Calculate the slope angles of the last and first Gamma- characteristics of the top corner expansion fam
    beta_last = phi_jet - mu_jet
    beta_first = phi_e - mu_e
    # Create no_char (also counting the first and last) intermediate characteristics
    beta = np.linspace(beta_first, beta_last, no_char)

    # Assign needed values to nodes A and B, where A is the first node under the top corner in the initial boundary, and
    # B is the top corner
    nu_a = nu[no_init - 2, 1]
    phi_a = phi[no_init - 2, 1]
    mu_a = mu[no_init - 2, 1]
    x_a = x[no_init - 2, 1]
    y_a = y[no_init - 2, 1]
    x_b = x[no_init - 1, 0]
    y_b = y[no_init - 1, 0]

    # Cycle through all created expansion fan Gamma- characteristics
    for i, beta_i in enumerate(beta):
        # Assign jet boundary value to corner with each new characteristic, for consistency. Will be used when
        # calculating the top jet boundary condition.
        phi[no_init - 1 + i, 0] = phi_jet
        M[no_init - 1 + i, 0] = Mjet
        nu[no_init - 1 + i, 0] = nu_jet
        mu[no_init - 1 + i, 0] = mu_jet
        x[no_init - 1 + i, 0] = x[no_init - 1, 0]
        y[no_init - 1 + i, 0] = y[no_init - 1, 0]

        # Initialise global values needed for performing a root-finding method
        global beta_global, nu_a_global, phi_a_global
        # Assign values to them
        beta_global = beta_i
        nu_a_global = nu_a
        phi_a_global = phi_a

        # Calculate Mach number and other properties of C, where C is the intersection of the Gamma+ characteristic
        # from A and the current Gamma- characteristic of the expansion flow
        sol = optimize.root(beta_M_function, np.array([Me]), tol=1e-8)
        Mc = np.squeeze(sol.x)
        nu_c = nu_from_M(Mc)
        mu_c = mu_from_M(Mc)
        phi_c = beta_i + mu_c

        # Calculate position of C
        alpha_i = 0.5 * (phi_a + mu_a + phi_c + mu_c)
        x_c = (y_b - y_a + x_a * np.tan(alpha_i) - x_b * np.tan(beta_i)) / (np.tan(alpha_i) - np.tan(beta_i))
        y_c = y_a + x_c * np.tan(alpha_i) - x_a * np.tan(alpha_i)

        # Assign values of C to the global data structure matrices
        phi[no_init - 1 + i, 1] = phi_c
        M[no_init - 1 + i, 1] = Mc
        nu[no_init - 1 + i, 1] = nu_c
        mu[no_init - 1 + i, 1] = mu_c
        x[no_init - 1 + i, 1] = x_c
        y[no_init - 1 + i, 1] = y_c


def assign_bottom_corner():
    """
    Assign the values for the first set of nodes at the intersection between the first Gamma- characteristic with the
    expansion fan of the bottom corner.

    :return: Returns nothing
    """
    # Calculate the slope angles of the last and first Gamma+ characteristics of the bottom corner expansion fam
    alpha_last = - phi_jet + mu_jet
    alpha_first = - phi_e + mu_e
    # Create no_char (also counting the first and last) intermediate characteristics
    alpha = np.linspace(alpha_first, alpha_last, no_char)

    # Assign needed values to nodes A and B, where B is the first node above the bottom corner in the initial boundary,
    # and A is the bottom corner
    nu_b = nu[1, no_init - 2]
    phi_b = phi[1, no_init - 2]
    mu_b = mu[1, no_init - 2]
    x_b = x[1, no_init - 2]
    y_b = y[1, no_init - 2]
    x_a = x[0, no_init - 1]
    y_a = y[0, no_init - 1]

    # Cycle through all created expansion fan Gamma+ characteristics
    for i, alpha_i in enumerate(alpha):
        # Assign jet boundary value to corner with each new characteristic, for consistency. Will be used when
        # calculating the bottom jet boundary condition.
        phi[0, no_init - 1 + i] = - phi_jet
        M[0, no_init - 1 + i] = Mjet
        nu[0, no_init - 1 + i] = nu_jet
        mu[0, no_init - 1 + i] = mu_jet
        x[0, no_init - 1 + i] = x[0, no_init - 1]
        y[0, no_init - 1 + i] = y[0, no_init - 1]

        # Initialise global values needed for performing a root-finding method
        global alpha_global, nu_b_global, phi_b_global
        # Assign values to them
        alpha_global = alpha_i
        nu_b_global = nu_b
        phi_b_global = phi_b

        # Calculate Mach number and other properties of C, where C is the intersection of the Gamma- characteristic
        # from B and the current Gamma+ characteristic of the expansion flow
        sol = optimize.root(alpha_M_function, np.array([Me]), tol=1e-8)
        Mc = np.squeeze(sol.x)
        nu_c = nu_from_M(Mc)
        mu_c = mu_from_M(Mc)
        phi_c = alpha_i - mu_c

        # Calculate position of C
        beta_i = 0.5 * (phi_b - mu_b + phi_c - mu_c)
        x_c = (y_b - y_a + x_a * np.tan(alpha_i) - x_b * np.tan(beta_i)) / (np.tan(alpha_i) - np.tan(beta_i))
        y_c = y_a + x_c * np.tan(alpha_i) - x_a * np.tan(alpha_i)

        # Assign values of C to the global data structure matrices
        phi[1, no_init - 1 + i] = phi_c
        M[1, no_init - 1 + i] = Mc
        nu[1, no_init - 1 + i] = nu_c
        mu[1, no_init - 1 + i] = mu_c
        x[1, no_init - 1 + i] = x_c
        y[1, no_init - 1 + i] = y_c


def prop_top_BC(ia, ja, ib, jb):
    """
    Propagate to (calculate) the next node D in the top jet boundary from the nodes A (index [ia, ja]) and
    B (index [ib, jb]). Point B is the previous node in the top jet boundary, and A is the node which is on the same
    Gamma+ characteristic as node D, and on the same Gamma- characteristic as node B. Practically, a new Gamma-
    characteristic will be created with the addition of point D. After calculation, assign values to the global
    data structure.

    :param ia: Gamma- coordinate of node A
    :param ja: Gamma+ coordinate of node A
    :param ib: Gamma- coordinate of node B
    :param jb: Gamma+ coordinate of node B
    :return: Returns nothing
    """
    # At the jet boundary, the mach number is constant == Mjet. Assign these values to node D
    Md = Mjet
    nu_d = nu_jet
    mu_d = mu_jet

    # Assign needed values for calculation to nodes A and B
    nu_a = nu[ia, ja]
    phi_a = phi[ia, ja]
    mu_a = mu[ia, ja]
    x_a = x[ia, ja]
    y_a = y[ia, ja]
    phi_b = phi[ib, jb]
    x_b = x[ib, jb]
    y_b = y[ib, jb]

    # Calculate values of node D
    phi_d = nu_d - nu_a + phi_a
    alpha = 0.5 * (phi_a + mu_a + phi_d + mu_d)
    beta = 0.5 * (phi_b + phi_d)
    x_d = (y_b - y_a + x_a * np.tan(alpha) - x_b * np.tan(beta)) / (np.tan(alpha) - np.tan(beta))
    y_d = y_a + x_d * np.tan(alpha) - x_a * np.tan(alpha)

    # Add values of node D to the global data structure
    phi[ia + 1, ja] = phi_d
    M[ia + 1, ja] = Md
    nu[ia + 1, ja] = nu_d
    mu[ia + 1, ja] = mu_d
    x[ia + 1, ja] = x_d
    y[ia + 1, ja] = y_d


def prop_bottom_BC(ia, ja, ib, jb):
    """
    Propagate to (calculate) the next node D in the bottom jet boundary from the nodes A (index [ia, ja]) and
    B (index [ib, jb]). Point A is the previous node in the top jet boundary, and B is the node which is on the same
    Gamma- characteristic as node D, and on the same Gamma+ characteristic as node B. Practically, a new Gamma+
    characteristic will be created with the addition of point D. After calculation, assign values to the global
    data structure.

    :param ia: Gamma- coordinate of node A
    :param ja: Gamma+ coordinate of node A
    :param ib: Gamma- coordinate of node B
    :param jb: Gamma+ coordinate of node B
    :return: Returns nothing
    """
    # At the jet boundary, the mach number is constant == Mjet. Assign these values to node D
    Md = Mjet
    nu_d = nu_jet
    mu_d = mu_jet

    # Assign needed values for calculation to nodes A and B
    nu_b = nu[ib, jb]
    phi_b = phi[ib, jb]
    mu_b = mu[ib, jb]
    x_b = x[ib, jb]
    y_b = y[ib, jb]
    phi_a = phi[ia, ja]
    x_a = x[ia, ja]
    y_a = y[ia, ja]

    # Calculate values of node D
    phi_d = nu_b + phi_b - nu_d
    alpha = 0.5 * (phi_a + phi_d)
    beta = 0.5 * (phi_b - mu_b + phi_d - mu_d)
    x_d = (y_b - y_a + x_a * np.tan(alpha) - x_b * np.tan(beta)) / (np.tan(alpha) - np.tan(beta))
    y_d = y_a + x_d * np.tan(alpha) - x_a * np.tan(alpha)

    # Add values of node D to the global data structure
    phi[ib, jb + 1] = phi_d
    M[ib, jb + 1] = Md
    nu[ib, jb + 1] = nu_d
    mu[ib, jb + 1] = mu_d
    x[ib, jb + 1] = x_d
    y[ib, jb + 1] = y_d


def prop_normal(ia, ja, ib, jb):
    nu_a = nu[ia, ja]
    mu_a = mu[ia, ja]
    phi_a = phi[ia, ja]
    x_a = x[ia, ja]
    y_a = y[ia, ja]
    nu_b = nu[ib, jb]
    mu_b = mu[ib, jb]
    phi_b = phi[ib, jb]
    x_b = x[ib, jb]
    y_b = y[ib, jb]

    nu_p = 0.5 * (nu_b + nu_a) + 0.5 * (phi_b - phi_a)
    phi_p = 0.5 * (phi_b + phi_a) + 0.5 * (nu_b - nu_a)
    Mp = M_from_nu(nu_p)
    mu_p = mu_from_M(Mp)

    alpha = 0.5 * (phi_a + mu_a + phi_p + mu_p)
    beta = 0.5 * (phi_b - mu_b + phi_p - mu_p)
    x_p = (y_b - y_a + x_a * np.tan(alpha) - x_b * np.tan(beta)) / (np.tan(alpha) - np.tan(beta))
    y_p = y_a + x_p * np.tan(alpha) - x_a * np.tan(alpha)

    phi[ib, ja] = phi_p
    M[ib, ja] = Mp
    nu[ib, ja] = nu_p
    mu[ib, ja] = mu_p
    x[ib, ja] = x_p
    y[ib, ja] = y_p


def propagation(steps):
    global i_s
    i_shock, j_shock, k_shock = np.nan, np.nan, np.nan
    for k in range(steps + 1):
        for j in range(no_init + k):
            i = no_init - 1 + k - j

            # Propagate values according to m.o.c.
            if not np.isnan(M[i, j]):
                # Point already exists, passing...
                pass
            elif (not np.isnan(M[i - 1, j])) and (not np.isnan(M[i, j - 1])):
                # Two characteristics exist for this point's intersection, doing normal propagation
                prop_normal(i - 1, j, i, j - 1)
            elif (not np.isnan(M[i - 1, j])) and (not np.isnan(M[i - 1, j - 1])):
                # Point qualifies for top BC
                prop_top_BC(i - 1, j, i - 1, j - 1)
            elif (not np.isnan(M[i, j - 1])) and (not np.isnan(M[i - 1, j - 1])):
                # Point qualifies for bottom BC
                prop_bottom_BC(i - 1, j - 1, i, j - 1)

            # Check if shock develops by intersecting characteristics
            if check_intersect(x[i, j], y[i, j], x[i - 1, j], y[i - 1, j], x[i, j - 1], y[i, j - 1], x[i - 1, j - 1],
                               y[i - 1, j - 1]) \
                    or check_intersect(x[i, j], y[i, j], x[i, j - 1], y[i, j - 1], x[i - 1, j], y[i - 1, j],
                                       x[i - 1, j - 1], y[i - 1, j - 1]):
                i_shock = i
                j_shock = j
                k_shock = k
                break
            if not np.isnan(k_shock) and k > k_shock:
                break

            # Propagate streamline if possible
            if not np.isnan(M[i, j]) and not np.isnan(M[i, j - 1]):
                x_s1 = xs[i_s]
                y_s1 = ys[i_s]
                phi_s = phis[i_s]
                x_s2 = x_s1 + 10 * h
                y_s2 = y_s1 + 10 * h * np.tan(phi_s)
                if check_intersect(x_s1, y_s1, x_s2, y_s2, x[i, j - 1], y[i, j - 1], x[i, j], y[i, j]):
                    xe, ye = get_intersect(x_s1, y_s1, x_s2, y_s2, x[i, j - 1], y[i, j - 1], x[i, j], y[i, j])
                    i_s += 1
                    xs[i_s] = xe
                    ys[i_s] = ye
                    Ms[i_s] = M[i, j - 1]
                    phis[i_s] = phi[i, j - 1]
                    nus[i_s] = nu[i, j - 1]
                    mus[i_s] = mu[i, j - 1]

            elif not np.isnan(M[i, j]) and not np.isnan(M[i - 1, j]):
                x_s1 = xs[i_s]
                y_s1 = ys[i_s]
                phi_s = phis[i_s]
                x_s2 = x_s1 + 10 * h
                y_s2 = y_s1 + 10 * h * np.tan(phi_s)
                if check_intersect(x_s1, y_s1, x_s2, y_s2, x[i - 1, j], y[i - 1, j], x[i, j], y[i, j]):
                    xe, ye = get_intersect(x_s1, y_s1, x_s2, y_s2, x[i - 1, j], y[i - 1, j], x[i, j], y[i, j])
                    i_s += 1
                    xs[i_s] = xe
                    ys[i_s] = ye
                    Ms[i_s] = M[i - 1, j]
                    phis[i_s] = phi[i - 1, j]
                    nus[i_s] = nu[i - 1, j]
                    mus[i_s] = mu[i - 1, j]

    return i_shock, j_shock


# ----------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------

def plot_all_M(i_shock):
    xg = x[np.logical_not(np.isnan(x))]
    yg = y[np.logical_not(np.isnan(y))]
    Mg = M[np.logical_not(np.isnan(M))]
    p1 = ax1.scatter(xg, yg, s=20, c=Mg, cmap="cool")
    # p1 = ax1.tricontourf(xg, yg, Mg, levels=50, cmap="cool")
    fig1.colorbar(p1, label="Mach number")
    # ax1.hlines(0, -0.5, i_shock + 0.5, linestyles="dashdot", color="black")


def plot_char():
    for k in range(no_char):
        xt = x[no_init - 1 + k][np.logical_not(np.isnan(x[no_init - 1 + k]))]
        yt = y[no_init - 1 + k][np.logical_not(np.isnan(y[no_init - 1 + k]))]
        ax1.plot(xt, yt, color="black")

        xb = x[:, 2 * no_init - 1 + k + no_char - 2][np.logical_not(np.isnan(x[:, 2 * no_init - 1 + k + no_char - 2]))]
        yb = y[:, 2 * no_init - 1 + k + no_char - 2][np.logical_not(np.isnan(y[:, 2 * no_init - 1 + k + no_char - 2]))]
        ax1.plot(xb, yb, color="black")

        xb = x[:, no_init - 1 + k][np.logical_not(np.isnan(x[:, no_init - 1 + k]))]
        yb = y[:, no_init - 1 + k][np.logical_not(np.isnan(y[:, no_init - 1 + k]))]
        ax1.plot(xb, yb, color="black")

        xb = x[2 * no_init - 1 + k + no_char - 2][np.logical_not(np.isnan(x[:, 2 * no_init - 1 + k + no_char - 2]))]
        yb = y[2 * no_init - 1 + k + no_char - 2][np.logical_not(np.isnan(y[:, 2 * no_init - 1 + k + no_char - 2]))]
        ax1.plot(xb, yb, color="black")


def plot_jet_BC():
    xt = np.array([])
    yt = np.array([])
    xb = np.array([])
    yb = np.array([])

    i = 0
    print(x[no_init + no_char - 2 + i, i])
    while not np.isnan(x[no_init + no_char - 2 + i, i]):
        xt = np.append(xt, x[no_init + no_char - 2 + i, i])
        yt = np.append(yt, y[no_init + no_char - 2 + i, i])
        i += 1

    i = 0
    while not np.isnan(x[i, no_init + no_char - 2 + i]):
        xb = np.append(xb, x[i, no_init + no_char - 2 + i])
        yb = np.append(yb, y[i, no_init + no_char - 2 + i])
        i += 1

    ax1.plot(xt, yt, color="orange")
    ax1.plot(xb, yb, color="orange")


def plot_shock(i_shock, j_shock):
    ax1.plot(x[i_shock, j_shock], y[i_shock, j_shock], marker="X", color="red", markersize=10)


def plot_streamline():
    xg = xs[np.logical_not(np.isnan(xs))]
    yg = ys[np.logical_not(np.isnan(ys))]
    Mg = Ms[np.logical_not(np.isnan(Ms))]
    pg = pt / ((1 + (gamma - 1) / 2 * Mg * Mg) ** (gamma / (gamma - 1)))
    ax1.plot(xg, yg, color="green")
    fig2, ax2 = plt.subplots()
    ax2.plot(xg, pg, color="black", label=rf"$y_0$ = {ys[0]}")
    ax2.set_xlabel("Horizontal position x")
    ax2.set_ylabel("Static pressure p")


# Values needed for root-finding. Do not work with them outside the root-finding functions!
nu_global = np.nan
beta_global = np.nan
nu_a_global = np.nan
phi_a_global = np.nan
alpha_global = np.nan
nu_b_global = np.nan
phi_b_global = np.nan

# Initial conditions & initialisations
Me = 2.0  # Mach number at the nozzle exit
pa = 1  # Static pressure of the ambient atmosphere [atm]
pe = 2 * pa  # Static pressure at the nozzle exit
phi_e = 0  # Flow angle at the nozzle exit [rad]
gamma = 1.4  # Specific heat ratio of air
h = 1  # Height (diameter) of the nozzle exit
no_init = 31  # Number of nodes to be created at the exit of the nozzle
no_char = 31  # Number of characteristics to be created in the expansion fans at the top and bottom corners
dim = 1010  # Dimension of (each axis of) each data structure matrix. Needs to be bigger than (no_init + no_steps + 1)!!
no_steps = 500  # Maximum number of propagation steps

# Initialisation of global data structure matrices
x = np.empty((dim, dim))
x[:] = np.nan
y = np.empty((dim, dim))
y[:] = np.nan
phi = np.empty((dim, dim))
phi[:] = np.nan
M = np.empty((dim, dim))
M[:] = np.nan
nu = np.empty((dim, dim))
nu[:] = np.nan
mu = np.empty((dim, dim))
mu[:] = np.nan

# Initialisation of the streamline data structure array
xs = np.empty((2 * dim + 10))
xs[:] = np.nan
ys = np.empty((2 * dim + 10))
ys[:] = np.nan
phis = np.empty((2 * dim + 10))
phis[:] = np.nan
Ms = np.empty((2 * dim + 10))
Ms[:] = np.nan
nus = np.empty((2 * dim + 10))
nus[:] = np.nan
mus = np.empty((2 * dim + 10))
mus[:] = np.nan

# Calculation of global nozzle exit and jet boundary values needed for calculations
nu_e, mu_e, Mjet, nu_jet, mu_jet, phi_jet, pt = calc_e_jet(Me, pe, pa, phi_e)

# Values for streamline initial conditions
xs[0] = 0  # x-position
ys[0] = h / 4  # y-position
phis[0] = phi_e  # Flow ange [rad]
Ms[0] = Me  # Mach number
nus[0] = nu_e  # Prandtl-Meyer angle [rad]
mus[0] = mu_e  # Mach angle [rad]
i_s = 0  # Index of last entry in streamline arrays

# Initialise the nozzle exit boundary, the expansion fans, and perform the global propagation
init_exit(no_init, h)
assign_top_corner()
assign_bottom_corner()
i_shock, j_shock = propagation(no_steps)

# Plot Mach number distribution, characteristics, jet boundaries, chosen streamline and shock formation location
fig1, ax1 = plt.subplots()
plot_all_M(i_shock)
plot_char()
plot_jet_BC()
plot_streamline()
if not np.isnan(i_shock):
    plot_shock(i_shock, j_shock)
ax1.axis("equal")
plt.show()
