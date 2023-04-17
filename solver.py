import gzip
import pickle
from collections import namedtuple
from copy import deepcopy

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq
from scipy.optimize._linprog_util import _clean_inputs, _presolve, _get_Abc, _autoscale, \
    _postsolve

# https://github.com/scipy/scipy/blob/e574cbcabf8d25955d1aafeed02794f8b5f250cd/scipy/optimize/_linprog_util.py#L15
_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg
SMALL_EPS = 1.e-7


def mu(x, s):
    return x.dot(s) / len(x)


def _get_blind_start(A, b, c, smart_start = True):
    """

    :param A:
    :param b:
    :param c:
    :param smart_start:

    :return:
    """
    if smart_start:
        a_at_inv = cho_factor(A @ A.T)
        x_tilde = A.T @ cho_solve(a_at_inv, b)
        lambda_tilde = cho_solve(a_at_inv, A @ c)
        s_tilde = c - A.T @ lambda_tilde

        delta_x = max(0, - 1.5 * np.min(x_tilde))
        delta_s = max(0, - 1.5 * np.min(s_tilde))

        x_cap = x_tilde + delta_x
        s_cap = s_tilde + delta_s

        delta_x_cap = 0.5 * np.dot(x_cap, s_cap) / np.sum(s_cap)
        delta_s_cap = 0.5 * np.dot(x_cap, s_cap) / np.sum(x_cap)

        x0 = x_cap + delta_x_cap
        lambda0 = lambda_tilde
        s0 = s_cap + delta_s_cap
    else:
        x0 = np.ones(A.shape[1])
        lambda0 = np.zeros(A.shape[0])
        s0 = np.ones(A.shape[1])
    return x0, lambda0, s0


def ipm_chapter14(c, A_ub, b_ub, A_eq, b_eq, bounds, autoscale = False, max_iter = 100, tol = 1.e-6):
    """

    :param c:
    :param A_ub:
    :param b_ub:
    :param A_eq:
    :param b_eq:
    :param bounds:
    :param autoscale: Consider using this option if the numerical values in the constraints are separated by several orders of magnitude.
    :param max_iter:
    :param tol:

    :return:
    """
    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, None, None)

    # `_parse_linprog` contains `_check_sparse_inputs` and `_clean_inputs`
    lp = _clean_inputs(lp)

    # Keep the original arrays to calculate slack/residuals for original problem.
    lp_o = deepcopy(lp)

    rr_method = None  # Method used to identify and remove redundant rows from the equality constraint matrix after presolve.
    rr = True  # Set to False to disable automatic redundancy removal. Default: True.

    # https://github.com/scipy/scipy/blob/main/scipy/optimize/_linprog_util.py#L477
    # identify trivial infeasibilities, redundancies, and unboundedness, tighten bounds where possible, and eliminate fixed variables
    (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr, rr_method, tol)
    assert not complete

    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    A, b, c, c0, x0 = _get_Abc(lp, c0)

    if autoscale:
        A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
        postsolve_args = postsolve_args[:-2] + (C, b_scale)

    x, lambd, s = _get_blind_start(A, b, c)

    last_x = x
    solver = 'cho'

    for iteration in range(max_iter):
        try:
            _mu = mu(x, s)
            S_inv = np.diag((s + SMALL_EPS) ** -1)
            X = np.diag(x)
            XS_inv = X @ S_inv
            M = A @ XS_inv @ A.T

            if solver == 'cho':
                c_and_lower = cho_factor(M)

            # affine
            rhs = b - A @ x + A @ XS_inv @ (- A.T @ lambd + c)
            if solver == 'cho':
                grad_lambda_aff = cho_solve(c_and_lower, rhs)
            elif solver == 'lstsq':
                grad_lambda_aff = lstsq(M, rhs)[0]
            grad_s_aff = - A.T @ (lambd + grad_lambda_aff) - s + c
            grad_x_aff = -x - XS_inv @ grad_s_aff

            alpha_prime_aff = 1.
            if np.any(grad_x_aff < 0):
                alpha_prime_aff = min(1, (
                            -x[grad_x_aff < 0] / grad_x_aff[grad_x_aff < 0]).min())
            alpha_dual_aff = 1.
            if np.any(grad_s_aff < 0):
                alpha_dual_aff = min(1, (
                            -s[grad_s_aff < 0] / grad_s_aff[grad_s_aff < 0]).min())

            mu_aff = np.dot(x + alpha_prime_aff * grad_x_aff,
                            s + alpha_dual_aff * grad_s_aff) / len(x)
            sigma = (mu_aff / _mu) ** 3

            rhs = b - A @ x + A @ XS_inv @ (- A.T @ lambd + c) + A @ S_inv @ (
                        grad_s_aff * grad_x_aff - sigma * _mu)
            if solver == 'cho':
                grad_lambda = cho_solve(c_and_lower, rhs)
            elif solver == 'lstsq':
                grad_lambda = lstsq(M, rhs)[0]
            grad_s = - A.T @ (lambd + grad_lambda) - s + c
            grad_x = - S_inv @ (
                        x * s + grad_s_aff * grad_x_aff - sigma * _mu) - XS_inv @ grad_s

            alpha_prime_max = 1.
            if np.any(grad_x < 0):
                alpha_prime_max = min(1, (-x[grad_x < 0] / grad_x[grad_x < 0]).min())
            alpha_dual_max = 1.
            if np.any(grad_s < 0):
                alpha_dual_max = min(1, (-s[grad_s < 0] / grad_s[grad_s < 0]).min())

            eta = 1 - np.random.rand(1) * 0.1
            alpha_prime = min(1, eta * alpha_prime_max)
            alpha_dual = min(1, eta * alpha_dual_max)

            x = x + alpha_prime * grad_x
            lambd = lambd + alpha_dual * grad_lambda
            s = s + alpha_dual * grad_s

            if np.abs(x - last_x).max() < tol:
                break
            last_x = x
        except:
            solver = 'lstsq'

        # print(_postsolve(x, postsolve_args)[:2])

    x, fun, slack, con = _postsolve(x, postsolve_args)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'nit': iteration}
    return sol


def ipm_overleaf(c, A_ub, b_ub, A_eq, b_eq, bounds, autoscale = False, max_iter = 100, tol = 1.e-6):
    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, None, None)

    # `_parse_linprog` contains `_check_sparse_inputs` and `_clean_inputs`
    lp = _clean_inputs(lp)

    # Keep the original arrays to calculate slack/residuals for original problem.
    lp_o = deepcopy(lp)

    rr_method = None  # Method used to identify and remove redundant rows from the equality constraint matrix after presolve.
    rr = True  # Set to False to disable automatic redundancy removal. Default: True.

    # https://github.com/scipy/scipy/blob/main/scipy/optimize/_linprog_util.py#L477
    # identify trivial infeasibilities, redundancies, and unboundedness, tighten bounds where possible, and eliminate fixed variables
    (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr, rr_method, tol)
    assert not complete

    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    A, b, c, c0, x0 = _get_Abc(lp, c0)

    if autoscale:
        A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
        postsolve_args = postsolve_args[:-2] + (C, b_scale)

    x, lambd, s = _get_blind_start(A, b, c)

    last_x = x
    solver = 'cho'

    _mu = mu(x, s)
    sigma = 0.25
    gamma = 0.001

    for iteration in range(max_iter):
        try:
            S_inv = np.diag((s + SMALL_EPS) ** -1)
            s_inv = (s + SMALL_EPS) ** -1
            X = np.diag(x)
            XS_inv = X @ S_inv
            M = A @ XS_inv @ A.T

            if solver == 'cho':
                c_and_lower = cho_factor(M)

            rhs = b - A @ x + A @ XS_inv @ (- A.T @ lambd + c) - A @ s_inv * sigma * _mu
            if solver == 'cho':
                grad_lambda = cho_solve(c_and_lower, rhs)
            elif solver == 'lstsq':
                grad_lambda = lstsq(M, rhs)[0]
            AT_lambda_plut_dlambda = A.T @ (lambd + grad_lambda)
            grad_s = - AT_lambda_plut_dlambda - s + c
            grad_x = s_inv * sigma * _mu + XS_inv @ (AT_lambda_plut_dlambda - c)

            alpha = min(1., 2 ** 1.5 / len(x) * (1 - gamma) * sigma / (
                        sigma ** 2 / gamma - 2 * sigma + 1))

            x = x + alpha * grad_x
            lambd = lambd + alpha * grad_lambda
            s = s + alpha * grad_s

            _mu *= (1 - (1 - sigma) * alpha)

            if np.abs(x - last_x).max() < tol:
                break
            last_x = x
        except:
            solver = 'lstsq'

        # print(_postsolve(x, postsolve_args)[:2])

    x, fun, slack, con = _postsolve(x, postsolve_args)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'nit': iteration}
    return sol


if __name__ == '__main__':
    with gzip.open("instances/setcover/instance_1.pkl.gz", "rb") as file:
        (A, b, c) = pickle.load(file)

    c = c.numpy()
    A_ub = None
    b_ub = None
    A_eq = A.numpy()
    b_eq = b.numpy()
    bounds = None

    sol = ipm_chapter14(c, A_ub, b_ub, A_eq, b_eq, bounds)
    print(sol['x'])
