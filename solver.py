import gzip
import pickle
import warnings
from collections import namedtuple
from copy import deepcopy

from jax.ops import segment_min
import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve, lstsq
from scipy.optimize._linprog_util import _clean_inputs, _presolve, _get_Abc, _autoscale, \
    _postsolve
from scipy.sparse.linalg import cg as sp_cg
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


def conjugate_gradient_hyper(P, q, alpha, beta, max_iters = 10000, tol = 1.e-5):

    y = q / np.diag(P)
    p = np.zeros_like(q)
    last_y = y
    diag_Pinv = 1. / np.diag(P)

    for i in range(1, max_iters + 1):
        r = q - P @ y
        p = diag_Pinv * r + beta * p
        y = y + alpha * p
        if np.abs(last_y - y).max() <= tol:
            break
    return y, i


def conjugate_gradient(P, q, max_iters = 10000, tol = 1.e-5):

    y = q / np.diag(P)

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

    for iteration in tqdm(range(max_iter)):
        try:
            _mu = mu(x, s)
            s_inv = (s + SMALL_EPS) ** -1
            xs_inv = x * s_inv
            A_XS_inv = A * xs_inv[None]
            Ax = A @ x
            M = A_XS_inv @ A.T

            if solver == 'cho':
                c_and_lower = cho_factor(M)

            # affine
            rhs = b - Ax + A_XS_inv @ (- A.T @ lambd + c)
            if solver == 'cho':
                grad_lambda_aff = cho_solve(c_and_lower, rhs)
            elif solver == 'lstsq':
                grad_lambda_aff = lstsq(M, rhs)[0]
            grad_s_aff = - A.T @ (lambd + grad_lambda_aff) - s + c
            grad_x_aff = -x - xs_inv * grad_s_aff

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

            rhs = b - Ax + A_XS_inv @ (- A.T @ lambd + c) + A @ (s_inv * (
                        grad_s_aff * grad_x_aff - sigma * _mu))
            if solver == 'cho':
                grad_lambda = cho_solve(c_and_lower, rhs)
            elif solver == 'lstsq':
                grad_lambda = lstsq(M, rhs)[0]
            grad_s = - A.T @ (lambd + grad_lambda) - s + c
            grad_x = - s_inv * (
                        x * s + grad_s_aff * grad_x_aff - sigma * _mu) - xs_inv * grad_s

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


def ipm_overleaf(c,
                 A_ub,
                 b_ub,
                 A_eq,
                 b_eq,
                 bounds,
                 autoscale = False,
                 max_iter = 100000,
                 tol = 1.e-9,
                 lin_solver = 'cg',
                 step_method='line',
                 sigma=0.3,
                 gamma=0.1,
                 cg_alpha=1.e-3,
                 cg_beta=0.9):
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

    pbar = tqdm(range(max_iter))
    if step_method == 'neighbor_min':
        adj_var = (A_sparse.T @ A_sparse).tocoo()
        adj_var_row, adj_var_col = adj_var.row, adj_var.col
        # adj_cons = (A_sparse @ A_sparse.T).tocoo()
        # adj_cons_row, adj_cons_col = adj_cons.row, adj_cons.col

    for iteration in pbar:
        try:
            s_inv = (s + SMALL_EPS) ** -1
            xs_inv = x * s_inv
            A_XS_inv = A * xs_inv[None]
            M = A_XS_inv @ A.T

            rhs = b - A_sparse @ x + A_XS_inv @ (- A_sparse.T @ lambd + c) - A_sparse @ s_inv * sigma * _mu
            if lin_solver == 'cho':
                c_and_lower = cho_factor(M)
                grad_lambda = cho_solve(c_and_lower, rhs)
                lin_system_steps = 0
            elif lin_solver == 'lstsq':
                grad_lambda = lstsq(M, rhs)[0]
                lin_system_steps = 0
            elif lin_solver == 'cg':
                grad_lambda, lin_system_steps = conjugate_gradient(M, rhs, max_iters = 100000, tol=1.e-5)
            elif lin_solver == 'cg_hyper':
                grad_lambda, lin_system_steps = conjugate_gradient_hyper(M, rhs, alpha=cg_alpha, beta=cg_beta, max_iters = 100000, tol=1.e-5)
            elif lin_solver == 'scipy_cg':
                grad_lambda = sp_cg(M, rhs, tol=1.e-9)
                lin_system_steps = 0
            else:
                raise NotImplementedError

            AT_lambda_plut_dlambda = A_sparse.T @ (lambd + grad_lambda)
            grad_s = - AT_lambda_plut_dlambda - s + c
            grad_x = s_inv * sigma * _mu + xs_inv * (AT_lambda_plut_dlambda - c)

            if step_method == 'conv':
                alpha = min(1., 2 ** 1.5 / len(x) * (1 - gamma) * sigma / (sigma ** 2 / gamma - 2 * sigma + 1))
                alpha_x = alpha_l = alpha_s = alpha
            elif step_method == 'line':
                alpha = 1.
                if np.any(grad_x < 0):
                    alpha = min(alpha, (-x[grad_x < 0] / grad_x[grad_x < 0]).min())
                if np.any(grad_s < 0):
                    alpha = min(alpha, (-s[grad_s < 0] / grad_s[grad_s < 0]).min())
                alpha_l = alpha_s = alpha_x = alpha
            elif step_method == 'neighbor_min':
                # alpha_x_msg = (x / grad_x + SMALL_EPS)[adj_var_col]
                # alpha_x = np.where(grad_x[adj_var_col] < 0., -alpha_x_msg, 1.)
                # alpha_x = np.array(segment_min(alpha_x, adj_var_row))
                # alpha_x = np.minimum(alpha_x, 1.)

                alpha_x_msg = (x / grad_x + SMALL_EPS)
                alpha_x = np.where(grad_x < 0., np.abs(alpha_x_msg), 1.)
                # alpha_x = np.array(segment_min(alpha_x, adj_var_row))
                alpha_x = np.minimum(alpha_x, 1.) * 0.98

                alpha_s_msg = (s / grad_s + SMALL_EPS)
                alpha_s = np.where(grad_s < 0., np.abs(alpha_s_msg), 1.)
                # alpha_s = np.array(segment_min(alpha_s, adj_var_row))
                alpha_s = np.minimum(alpha_s, 1.) * 0.98

                alpha_l = np.minimum(np.abs(lambd / grad_lambda), 1.)
            else:
                raise ValueError

            x = x + alpha_x * grad_x
            lambd = lambd + alpha_l * grad_lambda
            s = s + alpha_s * grad_s

            if step_method != 'neighbor_min':
                _mu *= (1 - (1 - sigma) * alpha_x)
            else:
                _mu = mu(x, s)

            if np.abs(x - last_x).max() < tol:
                break
            last_x = x
            intermediate_xs.append(_postsolve(x, postsolve_args)[0])
        except (LinAlgError, FloatingPointError, ValueError, ZeroDivisionError):
            warnings.warn(f'Instability occured at iter {iteration}, turning to lstsq')
            lin_solver = 'lstsq'

        x, fun, slack, con = _postsolve(x, postsolve_args)
        pbar.set_postfix({'lin_system_steps': lin_system_steps, 'obj': fun})

    sol = {
        'x': x,
        'xs': intermediate_xs,
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

    sol = ipm_overleaf(c, A_ub, b_ub, A_eq, b_eq, bounds)
    print(sol['x'])
