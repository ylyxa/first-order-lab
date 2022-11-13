import numpy as np
from copy import deepcopy
from scipy import optimize
from time import time

def svenn(f, x0, step=0.1):
   x=x0
   k = 0
   a, b = None, None
   left, center ,right = f(x-step), f(x) ,f(x+step)
   if center <= left and center <= right:
       return [x-step, x+step]
   elif center >= left and center >= right:
       return None
   if center >= right:
       delta, a, x, k = step, x, x+step, 1
   else:
       delta, b, x, k = -step, x, x-step, 1
   while True:
       x_next = x + 2**k * delta
       if f(x_next) < f(x):
           if delta == step:
               a = x
           else:
               b = x
           k += 1
           x = x_next
       else:
           break
   if delta == step:
       b = x_next
   else:
       a = x_next
   return a, b

def half(f, a, b, err=1e-10):
    x0 = (a + b)/2
    if b-a < err: # 7
        return x0
    yk = (a+x0)/2 # 4
    zk = (x0+b)/2

    fmid = f(x0)
    if f(zk) < fmid:
        return half(f, x0, b, err) # 6
    if f(yk) < fmid:
        return half(f, a, x0, err) # 5
    return half(f, yk, zk, err)


def rosenbrock(a, b, f0):
    return lambda x: np.sum(a*(x[1:]-x[:-1]**2)**2 + b*(1-x[:-1])**2, axis=0)+f0

def gradient_descent(f, x0, eps_1=1e-5, eps_2=1e-5, eps_grad=1e-5, n_limit=1e4):
    k = 0
    x = x0
    end = False
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)

    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)

        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k


        x_old = deepcopy(x)
        left, right = svenn(f_directed(x, -grad_f), 0)
        # print(left, right)
        a = half(f_directed(x, -grad_f), left, right)
        # a = np.max(optimize.minimize(f_directed(x, -grad_f), 1).x)
        # print(a)
        x -=  a * grad_f

        if np.linalg.norm(x-x_old) < eps_1 and np.abs(f(x)-f(x_old)) < eps_2:
            if end:
                # print(k)
                return x, k
            else:
                end = True
                k += 1
                continue
        k += 1
        end = False



def fletcher_gradient(f, x0, eps_1 = 1e-5, delta = 1e-5, eps_2=1e-3, eps_grad=1e-5, n_limit=1e4):
    n = len(x0)
    k = 0
    x = x0
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)
    a=1
    end = False
    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)


        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k


        if k == 0:
            d  = -optimize.approx_fprime(x, f, eps_grad) # 6
        else:
            if k % n != 0: # 7
                w = np.linalg.norm(grad_f)**2/np.linalg.norm(grad_f_old)**2
            else:
                w = 0
            d = - grad_f + w*d # 8


        # a = np.max(optimize.minimize(f_directed(x, d), 1).x) #9
        left, right = svenn(f_directed(x, -grad_f), 0)
        # print(left, right)
        a = half(f_directed(x, -grad_f), left, right)
        x_old = deepcopy(x) #10
        grad_f_old = deepcopy(grad_f)
        x += a*d

        if np.linalg.norm(x-x_old) < delta and np.abs(f(x)-f(x_old)) < eps_2: #11
            if end:
                # print(k)
                return x, k
            else:
                end = True
                k += 1
                continue
        k += 1
        end = False

def polak_gradient(f, x0, eps_1 = 1e-5, delta = 1e-5, eps_2=1e-3, eps_grad=1e-5, n_limit=1e4):
    n = len(x0)
    k = 0
    x = x0
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)
    a=1
    end = False
    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)


        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k


        if k == 0:
            d  = -optimize.approx_fprime(x, f, eps_grad) # 6
        else:
            if k % n != 0: # 7
                w = (grad_f @ (grad_f-grad_f_old).T)/np.linalg.norm(grad_f_old)**2
            else:
                w = 0
            d = - grad_f + w*d # 8


        # a = np.max(optimize.minimize(f_directed(x, d), 1).x) #9
        left, right = svenn(f_directed(x, -grad_f), 0)
        # print(left, right)
        a = half(f_directed(x, -grad_f), left, right)
        x_old = deepcopy(x) #10
        grad_f_old = deepcopy(grad_f)
        x += a*d

        if np.linalg.norm(x-x_old) < delta and np.abs(f(x)-f(x_old)) < eps_2: #11
            if end:
                # print(k)
                return x, k
            else:
                end = True
                k += 1
                continue
        k += 1
        end = False


def dev_fl_gradient(f, x0, eps_1 = 1e-4, delta = 1e-4, eps_2=1e-3, eps_grad=1e-6, n_limit=1e4):
    n = len(x0)
    k = 0
    x = x0
    G = np.identity(n)
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)
    a=1
    end = False
    grad_f = optimize.approx_fprime(x, f, eps_grad)
    while True:

        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k

        d = - G @ grad_f #10
        # a = np.max(optimize.minimize(f_directed(x, d), 1).x)

        left, right = svenn(f_directed(x, -grad_f), 0)
        # print(left, right)
        a = half(f_directed(x, -grad_f), left, right)

        x_old = deepcopy(x)
        v = a*d
        x += v

        if np.linalg.norm(x-x_old) < delta and np.abs(f(x)-f(x_old)) < eps_2:
            if end:
                # print(k)
                return x, k
            else:
                end = True
                k += 1
                continue
        grad_f_old = deepcopy(grad_f)
        grad_f = optimize.approx_fprime(x, f, eps_grad)
        g = (grad_f - grad_f_old).T

        G -= (G@np.outer(g, g.T)@G)/(g.T@G@g) + np.outer(v.T, v)/(g.T*v.T)

        k += 1
        end = False



def hessian_f(a, b, f0):
        return lambda x, y: np.asarray(
            [[4*a*(x**2-y)+8*a*x**2+2*b,-4*a*x],
             [-4*a*x,2*a]])

def lev_mark_gradient(f, x0, hessian ,eps_1 = 1e-5, delta = 1e-5, eps_2=1e-3, eps_grad=1e-6, n_limit=1e4):
    k = 0
    mu = 10
    n = len(x0)
    ident = np.identity(n)
    k = 0
    x = x0
    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)

        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k
        while True:
            hessian_val = hessian(*x) # 6
            d = np.linalg.inv(hessian_val + mu*ident) # 7,8

            x_old = deepcopy(x) #9, 10
            x -= d@grad_f

            if f(x) < f(x_old): #11
                mu /= 2 #12
                break
            else:
                mu *= 2 #13
                if np.linalg.norm(x-x_old) < delta and np.abs(f(x)-f(x_old)) < eps_2:
                    # print(k)
                    return x_old, k
                x = x_old
        k += 1


def main():
    f = rosenbrock(50, 2, 10)
    hess = hessian_f(50, 2, 10)


    start = time()
    res = gradient_descent(f, np.asarray([1.32, 1.2]))
    a = np.array(res[0])
    t = time() - start
    print(f'For gradient descent: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iteraions')

    start = time()
    res = fletcher_gradient(f, np.asarray([1.32, 1.2]))
    a = np.array(res[0])
    t = time() - start
    print(f'For Fletcher-Reeves: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iteraions')

    start = time()
    res = polak_gradient(f, np.asarray([1.32, 1.2]))
    a = np.array(res[0])
    t = time() - start
    print(f'For Polak-Ribiere: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iteraions')

    start = time()
    res = dev_fl_gradient(f, np.asarray([1.32, 1.2]))
    a = np.array(res[0])
    t = time() - start
    print(f'For Davidon-Fletcher-Powell: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iteraions')

    start = time()
    res = lev_mark_gradient(f, np.asarray([1.32, 1.2]), hess)
    a = np.array(res[0])
    t = time() - start
    print(f'For Levenberg-Marquardt: f({a}) = {f(a)}, {t*1000} ms elapsed, {res[1]} iteraions')


if __name__ == '__main__':
    main()
