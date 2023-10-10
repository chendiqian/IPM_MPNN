from collections import namedtuple
from copy import deepcopy

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize._linprog_util import _clean_inputs, _presolve, _get_Abc, _autoscale, \
    _postsolve
from scipy.sparse import csr_matrix
from tqdm import tqdm

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


def conjugate_gradient(P, q, max_iters = 10000, tol = 1.e-5):

    y = q / P.diagonal()

    for i in range(1, max_iters + 1):
        r = q - P @ y
        if np.abs(r).max() < tol:
            break

        if i == 1:
            p = r
        else:
            beta = r.dot(P.dot(p)) / p.dot(P.dot(p))
            p = r - beta * p

        alpha = np.dot(p, r) / p.dot(P.dot(p))
        y = y + alpha * p
    return y, i


def ipm_overleaf(c,
                 A_ub,
                 b_ub,
                 A_eq,
                 b_eq,
                 bounds,
                 autoscale = False,
                 max_iter=100000,
                 tol = 1.e-9,
                 sigma=0.3):
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

    A_sparse = csr_matrix(A)

    _mu = mu(x, s)
    last_x = x
    
    intermediate_xs = []

    pbar = range(max_iter)
    for iteration in pbar:
        s_inv = (s + SMALL_EPS) ** -1
        xs_inv = x * s_inv
        A_XS_inv = csr_matrix(A * xs_inv[None])
        M = A_XS_inv @ A_sparse.T
        rhs = b - A_sparse @ x + A_XS_inv @ (- A_sparse.T @ lambd + c) - A_sparse @ s_inv * sigma * _mu

        # conjugate gradient solve M @ x = rhs
        grad_lambda, lin_system_steps = conjugate_gradient(M, rhs, max_iters = 100000, tol=1.e-5)

        AT_lambda_plut_dlambda = A_sparse.T @ (lambd + grad_lambda)
        grad_s = - AT_lambda_plut_dlambda - s + c
        grad_x = s_inv * sigma * _mu + xs_inv * (AT_lambda_plut_dlambda - c)

        alpha = 1.
        gradx_mask = grad_x < 0
        if np.any(gradx_mask):
            alpha = min(alpha, (-x[gradx_mask] / grad_x[gradx_mask]).min())
        grads_mask = grad_s < 0
        if np.any(grads_mask):
            alpha = min(alpha, (-s[grads_mask] / grad_s[grads_mask]).min())
        alpha_l = alpha_s = alpha_x = alpha

        x = x + alpha_x * grad_x
        lambd = lambd + alpha_l * grad_lambda
        s = s + alpha_s * grad_s
        _mu = mu(x, s)

        if np.abs(x - last_x).max() < tol:
            break
        last_x = x
        intermediate_xs.append(_postsolve(x, postsolve_args)[0])
        # pbar.set_postfix({'lin_system_steps': lin_system_steps})

    x, fun, slack, con = _postsolve(x, postsolve_args)
    sol = {
        'x': x,
        'xs': intermediate_xs,
        'fun': fun,
        'slack': slack,
        'con': con,
        'nit': iteration}
    return sol
