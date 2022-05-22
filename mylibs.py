import numpy as np
from scipy.optimize import minimize
from sympy import E

def fun(theta_list, *args):
    n_param= len(theta_list)
    E = args[0]
    E_a = E["a"]
    E_b = E["b"]
    E_c = E["c"]
    E_d = E["d"]
    a = np.array([(1 + np.cos(theta_list[i])) / 2 for i in range(n_param)])
    b = np.array([np.sin(theta_list[i]) / 2 for i in range(n_param)])
    c = np.array([(1 - np.cos(theta_list[i])) / 2 for i in range(n_param)])

    A = 1
    for i in range(n_param):
        A *= a[i]
    a_term = A * E_a
    b_term = 0
    c_term = 0
    for i in range(n_param):
        b_term += A / a[i] * b[i] * E_b[i]
        c_term += A / a[i] * c[i] * E_c[i]
    d_term = 0
    for i in range(n_param):
        for j in range(i+1,n_param):
            d_term += A / a[i] / a[j] * b[i] * b[j] * E_d[i][j]
    energy = a_term + b_term + c_term + d_term

    return energy

def grad(theta_list, *args):
    E = args[0]
    n_param= len(theta_list)
    E_a = E["a"]
    E_b = E["b"]
    E_c = E["c"]
    E_d = E["d"]
    a = np.array([(1 + np.cos(theta_list[i])) / 2 for i in range(n_param)])
    b = np.array([np.sin(theta_list[i]) / 2 for i in range(n_param)])
    c = np.array([(1 - np.cos(theta_list[i])) / 2 for i in range(n_param)])
    A = 1
    for i in range(n_param):
        A *= a[i]
    a_derivs = np.array([-np.sin(theta_list[i]) / 2 for i in range(n_param)])
    b_derivs = np.array([np.cos(theta_list[i]) / 2 for i in range(n_param)])
    c_derivs = np.array([np.sin(theta_list[i]) / 2 for i in range(n_param)])

    a_term = np.array([A / a[i] * a_derivs[i] * E_a for i in range(n_param)])

    b_term = np.zeros(n_param)
    c_term = np.zeros(n_param)
    for m in range(n_param):
        for k in range(n_param):
            if m == k:
                b_term[m] += A / a[m] * b_derivs[m] * E_b[m]
                c_term[m] += A / a[m] * c_derivs[m] * E_c[m]
            else:
                b_term[m] += A / a[m] / a[k] * b[k] * a_derivs[m] * E_b[k]
                c_term[m] += A / a[m] / a[k] * c[k] * a_derivs[m] * E_c[k]
    
    d_term = np.zeros(n_param)
    for m in range(n_param):
        for k in range(n_param):
            for l in range(k+1,n_param):
                if m == k:
                    d_term[m] += A / a[k] / a[l] * b_derivs[k] * b[l] * E_d[k][l]
                elif m == l:
                    d_term[m] += A / a[k] / a[l] * b[k] * b_derivs[l] * E_d[k][l]
                else:
                    d_term[m] += A / a[m] / a[k] / a[l] * a_derivs[m] * b[k] * b[l] * E_d[k][l]
    
    gradient_vector = np.array([a_term[i] + b_term[i] + c_term[i] + d_term[i] for i in range(n_param)])

    return gradient_vector


def approx_optimize(E, init_theta_list, Thresh_hold):
    method = "L-BFGS-B"
    #method = "Powell"
    bounds = tuple([(-Thresh_hold, Thresh_hold) for i in range(len(init_theta_list))])
    options = {"disp": False, "maxiter": 10000, "gtol":1e-10}
    cost_history = []

    def callback(theta_list):
        print("current val",fun(theta_list, E))
        print("max theta", np.max(np.abs(theta_list)))

    opt = minimize(
        fun=fun,
        args=(E,),
        x0=init_theta_list,
        method=method,
        jac=grad,
        bounds=bounds,
        options=options,
        #callback=callback,
        tol=1e-10
    )

    return opt
