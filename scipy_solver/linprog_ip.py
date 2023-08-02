from warnings import warn

import numpy as np
import scipy.sparse as sps
from scipy.linalg import LinAlgError
from scipy.optimize._linprog_ip import (_get_delta,
                                        _get_step,
                                        _get_message,
                                        _do_step,
                                        _get_blind_start,
                                        _indicators, _display_iter)
from scipy.optimize._linprog_util import _postsolve
from scipy.optimize._optimize import OptimizeWarning, OptimizeResult, \
    _check_unknown_options

has_umfpack = True
has_cholmod = True
try:
    import sksparse
    from sksparse.cholmod import cholesky as cholmod
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:
    import scikits.umfpack  # test whether to use factorized
except ImportError:
    has_umfpack = False


def _get_rand_start(shape):
    """
    Instead of this https://github.com/scipy/scipy/blob/main/scipy/optimize/_linprog_ip.py#L436
    we use random init values

    """
    m, n = shape
    x0 = np.random.rand(n)
    y0 = np.random.rand(m)
    z0 = np.random.rand(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _get_blind_start(shape):
    """
    Return the starting point from [4] 4.4

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    m, n = shape
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq,
            sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args):

    iteration = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # first iteration is special improvement of initial point
    ip = ip if pc else False

    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol  # we might get lucky : )

    callback_outputs = []
    if disp:
        _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)
    if callback is not None:
        x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                              'con': con, 'nit': iteration, 'phase': 1,
                              'complete': False, 'status': 0,
                              'message': "", 'success': False})
        callback_outputs.append(callback(res))

    status = 0
    message = "Optimization terminated successfully."

    if sparse:
        A = sps.csc_matrix(A)
        A.T = A.transpose()  # A.T is defined for sparse matrices but is slow
        # Redefine it to avoid calculating again
        # This is fine as long as A doesn't change

    while go:

        iteration += 1

        if ip:  # initial point
            # [4] Section 4.4
            gamma = 1

            def eta(g):
                return 1
        else:
            # gamma = 0 in predictor step according to [4] 4.1
            # if predictor/corrector is off, use mean of complementarity [6]
            # 5.1 / [4] Below Figure 10-4
            gamma = 0 if pc else beta * np.mean(z * x)
            # [4] Section 4.1

            def eta(g=gamma):
                return 1 - g

        try:
            # Solve [4] 8.6 and 8.7/8.13/8.23
            d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
                A, b, c, x, y, z, tau, kappa, gamma, eta,
                sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec)

            if ip:  # initial point
                # [4] 4.4
                # Formula after 8.23 takes a full step regardless if this will
                # take it negative
                alpha = 1.0
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y,
                    d_z, d_tau, d_kappa, alpha)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                ip = False  # done with initial point
            else:
                # [4] Section 4.3
                alpha = _get_step(x, d_x, z, d_z, tau,
                                  d_tau, kappa, d_kappa, alpha0)
                # [4] Equation 8.9
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)

        except (LinAlgError, FloatingPointError,
                ValueError, ZeroDivisionError):
            # this can happen when sparse solver is used and presolve
            # is turned off. Also observed ValueError in AppVeyor Python 3.6
            # Win32 build (PR #8676). I've never seen it otherwise.
            status = 4
            message = _get_message(status)
            break

        # [4] 4.5
        rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
            A, b, c, c0, x, y, z, tau, kappa)
        go = rho_p > tol or rho_d > tol or rho_A > tol

        if disp:
            _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj)
        if callback is not None:
            x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
            res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                                  'con': con, 'nit': iteration, 'phase': 1,
                                  'complete': False, 'status': 0,
                                  'message': "", 'success': False})
            callback_outputs.append(callback(res))

        # [4] 4.5
        inf1 = (rho_p < tol and rho_d < tol and rho_g < tol and tau < tol *
                max(1, kappa))
        inf2 = rho_mu < tol and tau < tol * min(1, kappa)
        if inf1 or inf2:
            # [4] Lemma 8.4 / Theorem 8.3
            if b.transpose().dot(y) > tol:
                status = 2
            else:  # elif c.T.dot(x) < tol: ? Probably not necessary.
                status = 3
            message = _get_message(status)
            break
        elif iteration >= maxiter:
            status = 1
            message = _get_message(status)
            break

    x_hat = x / tau
    # [4] Statement after Theorem 8.2
    return x_hat, status, message, iteration, callback_outputs


def _linprog_ip(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-8,
                disp=False, alpha0=.99995, beta=0.1, sparse=False, lstsq=False,
                sym_pos=True, cholesky=None, pc=True, ip=False,
                permc_spec='MMD_AT_PLUS_A', **unknown_options):

    _check_unknown_options(unknown_options)

    # These should be warnings, not errors
    if (cholesky or cholesky is None) and sparse and not has_cholmod:
        if cholesky:
            warn("Sparse cholesky is only available with scikit-sparse. "
                 "Setting `cholesky = False`",
                 OptimizeWarning, stacklevel=3)
        cholesky = False

    if sparse and lstsq:
        warn("Option combination 'sparse':True and 'lstsq':True "
             "is not recommended.",
             OptimizeWarning, stacklevel=3)

    if lstsq and cholesky:
        warn("Invalid option combination 'lstsq':True "
             "and 'cholesky':True; option 'cholesky' has no effect when "
             "'lstsq' is set True.",
             OptimizeWarning, stacklevel=3)

    valid_permc_spec = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD')
    if permc_spec.upper() not in valid_permc_spec:
        warn("Invalid permc_spec option: '" + str(permc_spec) + "'. "
             "Acceptable values are 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', "
             "and 'COLAMD'. Reverting to default.",
             OptimizeWarning, stacklevel=3)
        permc_spec = 'MMD_AT_PLUS_A'

    # This can be an error
    if not sym_pos and cholesky:
        raise ValueError(
            "Invalid option combination 'sym_pos':False "
            "and 'cholesky':True: Cholesky decomposition is only possible "
            "for symmetric positive definite matrices.")

    cholesky = cholesky or (cholesky is None and sym_pos and not lstsq)

    x, status, message, iteration, callback_outputs = _ip_hsd(A, b, c, c0, alpha0, beta,
                                                              maxiter, disp, tol, sparse,
                                                              lstsq, sym_pos, cholesky,
                                                              pc, ip, permc_spec,
                                                              callback,
                                                              postsolve_args)

    return x, status, message, iteration, callback_outputs
