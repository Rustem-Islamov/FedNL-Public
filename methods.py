import numpy as np
from scipy.stats import bernoulli
from utils import random_k, positive_part, random_sparsification
from utils import loss_logistic, grad
from utils import compression_dic, rand_dith
from utils import compute_bit, compute_omega
from utils import TopK, Low_Rank, PowerSgdCompression
from utils import semidef_projection, pos_projection
from utils import topK_vectors, biased_rounding
from utils import randomK_vectors, rand_dith
from utils import random_spars_matrix



################################################################
class Standard_Newton:
    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate Newton's method
        -------------------------------------------------
        '''
        self.oracle = oracle
    
    def step(self, x):
        '''
        -----------------------------------
        perform one step of Newton's method
        -----------------------------------
        input:
        x - current model weights
        
        return: 
        numpy array - next iterate of Newton's method
        '''
        lmb = self.oracle.get_reg_coef()
        d = self.oracle.get_number_of_weights()

        g = self.oracle.full_gradient(x) + lmb*x
        H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
        s = np.linalg.solve(H, g)
        return x - s 

    def find_optimum(self, x0, n_steps=10, verbose=True):
        '''
        -------------------------------------------------------------------------------------
        Implementation of Standard Newton method in order to find the solution of the problem
        -------------------------------------------------------------------------------------
        input:
        x0 - initial model weights
        n_steps - number of steps of the method 
        verbose - if True, then function values in each iteration are printed
        
        return:
        set the optimum to the problem
        '''
        iterates = []
        iterates.append(x0)
        for k in range(n_steps):
            if verbose:
                print(self.oracle.function_value(x0))
            x0 = self.step(x0)
            iterates.append(x0)
        self.oracle.set_optimum(iterates[-1])
        
 
    def method(self, x0, tol=10**(-14), max_iter=10, verbose=True):
        '''
        ----------------------------------------
        Implementation of Standard Newton method
        ----------------------------------------
        input:
        x0 - initial model weights
        tol - desired tolerance of the solution
        max_iter - maximum number of iterations of the method 
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        x_opt = self.oracle.get_optimum()
        n = self.oracle.get_number_of_nodes()
        d = self.oracle.get_number_of_weights()
        func_value = []
        iterates = []
        func_value.append(self.oracle.function_value(x0))
        iterates.append(np.linalg.norm(x0-x_opt))
        bits = []
        global_bit = 1
        bits.append(global_bit)
        n_steps = 0
        
        if verbose:
            print(func_value[-1])
            
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += 32*(d**2+d)
            bits.append(global_bit)
            
            x0 = self.step(x0)
            func_value.append(self.oracle.function_value(x0))
            iterates.append(np.linalg.norm(x0-x_opt))
            
            if verbose:
                print(func_value[-1])
            
        return np.array(func_value), np.array(iterates), np.array(bits)
    
#############################################################################   
class Newton_Star:
    def __init__(self, oracle):
        '''
        ---------------------------------------------------------
        This class is created to simulate NEWTON-STAR (NS) method 
        ---------------------------------------------------------
        '''
        self.oracle = oracle
        self.x_opt = oracle.get_optimum()
        self.H = oracle.full_Hessian(self.x_opt)+oracle.get_reg_coef()*np.eye(oracle.get_number_of_weights())
        
    def step(self, x):
        '''
        ----------------------
        perform one step of NS
        ----------------------
        input:
        x - current model weights
        
        return: 
        numpy array - next iterate of NS
        '''
        lmb = self.oracle.get_reg_coef()
        d = self.oracle.get_number_of_weights()
        g = self.oracle.full_gradient(x) + lmb*x
        
        
        return x - np.linalg.inv(self.H).dot(g) 
    
    def method(self, x0, max_iter = 10, tol=10**(-12), verbose=True):
        '''
        ---------------------------
        Implementation of NS method
        ---------------------------
        input:
        x0 - initial model weights
        max_iter - maximum number of steps of the method
        tol - desired tolerance of the solution
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        n = self.oracle.get_number_of_nodes()
        bits = []
        iterates = []
        func_value = []
        
        global_bit = 1
        bits.append(global_bit)  
        global_bit += 32*d*(d+1)//2
        bits.append(global_bit)
        func_value.append(self.oracle.function_value(x0))
        func_value.append(self.oracle.function_value(x0))
        iterates.append(np.linalg.norm(x0-x_opt))
        iterates.append(np.linalg.norm(x0-x_opt))
        
        
        
        if verbose:
            print(func_value[-1])
        n_steps = 0
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += 32*d
            bits.append(global_bit)
            x0 = self.step(x0)

            func_value.append(self.oracle.function_value(x0))
            iterates.append(x0)
            if verbose:
                print(func_value[-1])
            

        return np.array(func_value), np.array(iterates), np.array(bits)
    
###########################################################################    
class NL1:
    def __init__(self, oracle):
        '''
        -------------------------------------------------------------
        This class is created to simulate NEWTON-LEARN 1 (NL1) method
        -------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, H, max_iter=100, k=1, eta=None, tol=10**(-14), init_cost=True,\
               line_search=False, gamma=0.5, c = 0.5, verbose=True):
        '''
        -------------------------
        Implementation of NL1 method
        -------------------------
        input:
        x - initial model weightsn
        H - list of vectors h_i^0
        max_iter - maximum number of iterations of the method
        k - the parameter of Rand-K compression operator
        eta - stepsize for update of vectors h_i's
        (if eta is None, then eta is set as k/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        line_search - if True, then the method uses backtracking line search procedure
        gamma - parameter of line search procedure
        c - parameter of line search procedure
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        
        x_opt = self.oracle.get_optimum()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        d = self.oracle.get_number_of_weights()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        B = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m + j
                B += 1/N*H_old[i][j]*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))
                
        if eta is None:
            eta = k/m 
            
        f_opt = self.oracle.function_value(x_opt)    

        func_value = []
        iterates = []
        bits = []
        global_bit = 1
        bits.append(global_bit)
        
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])

        

        n_steps = 0
        
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += 32*d + k*d*32 + 32*k
            
            global_grad = self.oracle.full_gradient(x)+lmb*x
            
            for i in range(n):
                H_old[i] = H_new[i]
                h = random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = positive_part(H_old[i] + eta*h)
            
            t = 1
            D = - np.linalg.solve(B + lmb*np.eye(d), global_grad)
            der = global_grad.dot(D)
            if line_search:
                if n_steps == 1:
                    t = 0.1
             
                while self.oracle.function_value(x+t*D) > func_value[-1] + c*t*der:
                    t *= gamma
                    global_bit += 32
                         
                x = x + t*D
            else:
                x = x + D

            bits.append(global_bit)
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))
            
            if verbose:
                print(func_value[-1])

            for i in range(n):
                for j in range(m):
                    l = i*m+j
                    B += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                
        return np.array(func_value), np.array(bits), np.array(iterates)
   #############################################################################################
 
class DINGO:
    
    def __init__(self, oracle):
        '''
        ----------------------------------------------
        This class is created to simulate DINGO method
        ----------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, max_iter=200, tol=1e-15, phi=1e-6, theta=1e-4, rho=1e-4, verbose=True):
        '''
        -------------------------
        Implementation of DINGO method
        -------------------------
        
        input:
        x - initial point
        max_iter - maximum iterations of the method
        tol - desired tolerance of the solution
        phi, theta, rho - parameters of DINGO
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        '''
        x_opt = self.oracle.get_optimum()
        f_opt = self.oracle.function_value(x_opt)
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        x = x.copy()
        func_value = []
        bits = []
        global_bit = 1
        bits.append(global_bit)
        func_value.append(self.oracle.function_value(x))
        n_steps = 0

        if verbose:
            print(func_value[-1])
        delta = self.oracle.function_value(x_opt+np.ones(d)*0.1) - f_opt
        n_steps = 0
        while func_value[-1] - f_opt > tol and max_iter >= n_steps:
            
            
            global_bit += 32*d # local gradient to the master
            global_bit += 32*d # global gradient to the node
            global_bit += 3*32*d # 3 types of steps
            
            n_steps += 1
            
            g = self.oracle.full_gradient(x) + lmb*x
            g_norm = np.linalg.norm(g)**2

            H_i = []
            H_inv = []
            H_hat = []

            h_i = np.zeros(d)
            h_inv = np.zeros(d)
            h_hat = np.zeros(d)
            full_H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
            
            
            for i in range(n):
                B = self.oracle.local_Hessian(x, i)
                B += lmb*np.eye(d)

                H_i.append(B.dot(g))
                h_i += 1/n*H_i[i]
                H_inv.append(np.linalg.pinv(B).dot(g))
                h_inv += 1/n*H_inv[i]
                H = np.vstack((B, phi*np.eye(d)))
                G = np.vstack((g.reshape(d,1), np.zeros((d,1))))
                H_hat.append(np.linalg.pinv(H).dot(G))
                h_hat += 1/n*H_hat[i].squeeze()


            if h_i.dot(h_inv) >= theta*g_norm:
                p = -h_inv
            elif h_i.dot(h_hat) >= theta*g_norm:
                p = -h_hat
            else:
                p = np.zeros(d)
                for i in range(n):
                    B = self.oracle.local_Hessian(x, i)
                    B += lmb*np.eye(d)
                    H = np.vstack((B, phi*np.eye(d)))
                    G = np.vstack((g.reshape(d,1), np.zeros((d,1))))
                    if H_hat[i].squeeze().dot(h_i) >= theta*g_norm:
                        p -= 1/n*H_hat[i].squeeze()
                    else:
                        global_bit += 32*2*d
                        p = -H_hat[i].squeeze()
                        l = -g.reshape(1,d).dot(full_H).dot(np.linalg.pinv(H)).dot(G)[0].squeeze() + theta*g_norm
                        l /= g.reshape(1,d).dot(full_H).dot(np.linalg.inv(H.T.dot(H))).dot(full_H).dot(g).squeeze()

                        p -= l*np.linalg.inv(H.T.dot(H)).dot(h_i)
                        p /= n
                        
            global_bit += 32*2*d # H_t*g_t to the node and p_i to the master
            a = 1
            g_next = self.oracle.full_gradient(x+a*p)+lmb*(x+a*p)
            g_next_norm = np.linalg.norm(g_next)**2
            
            while g_next_norm > g_norm + 2*a*rho*p.dot(h_i):
                global_bit += 32*2*d
                a /= 2
                g_next = self.oracle.full_gradient(x+a*p)+lmb*(x+a*p)
                g_next_norm = np.linalg.norm(g_next)**2
               
            if n_steps < 5:
                a = min(1e-2, a)
            a = max(a, 2**(-10))
            x = x + a*p
            func_value.append(self.oracle.function_value(x))
            bits.append(global_bit)
            
            if verbose:
                print(func_value[-1])

            
        return np.array(func_value), np.array(bits)
 
    ####################################################################



def diana(X, y, w, arg, f_opt, tol=1e-15, verbose=True):
    '''
    -------------------------
    Implementation of DIANA method
    -------------------------
    X - data matrix
    y - labels vectors 
    w - initial point 
    arg - class containing all parameters of method and comressor
    f_opt - optimal function value
    tol - desired tolerance of the solution
    phi, theta, rho - parameters of DINGO
    verbose - if True, then function values in each iteration are printed

    return:
    loss - numpy array containing function value in each iteration of the method
    com_bits - numpy array containing transmitted bits by one node to the server
    '''
    alg = 'DIANA'
    dim = X.shape[1]
    
    omega = compute_omega(dim, arg)
    arg.alpha = 1 / (1 + omega)
    arg.eta = min(arg.alpha / (2 * arg.lamda), 2 / ((arg.L + arg.lamda) * (1 + 6 * omega / arg.node)))
    if verbose:
        print('algorithm ' + alg + ' starts')
        print('eta = ', arg.eta, 'compression: ', arg.comp_method)
        print('f_opt = ', f_opt)
    
    num_data = y.shape[0]
    num_data_worker = int(np.floor(num_data / arg.node))
    
    loss = []
    local_grad = np.zeros((arg.node, dim))
    
    hs = np.zeros((arg.node, dim))
    hs_mean = np.mean(hs, axis=0)
    deltas = np.zeros((arg.node, dim))
    
    loss_0 = loss_logistic(X, y, w, arg)
    if verbose:
        print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [1]
    bits = 1
    
    comp_method = compression_dic[arg.comp_method]
    com_round_bit = compute_bit(dim, arg)
    k = 0
    while k < arg.T and loss[-1] - f_opt > tol:
        k += 1
        for i in range(arg.node):
            local_grad[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                 y[i * num_data_worker:(i + 1) * num_data_worker], w, arg)
            deltas[i] = comp_method(local_grad[i] - hs[i], arg)
            hs[i] += arg.alpha * deltas[i]
        gk = np.mean(deltas, axis=0) + hs_mean
        assert gk.shape[0] == len(w)
        hs_mean += arg.alpha * np.mean(deltas, axis=0)
        assert hs_mean.shape[0] == len(w)
        w = w - arg.eta * gk
        bits += com_round_bit
        loss_k = loss_logistic(X, y, w, arg)
        loss.append(loss_k)
        com_bits.append(bits)
        if verbose:
            if k % 1000 == 0:
                print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits


def adiana(X, y, w, arg, f_opt, tol=1e-15, verbose=True):
    '''
    -------------------------
    Implementation of DIANA method
    -------------------------
    X - data matrix
    y - labels vectors 
    w - initial point 
    arg - class containing all parameters of method and comressor
    f_opt - optimal function value
    tol - desired tolerance of the solution
    phi, theta, rho - parameters of DINGO
    verbose - if True, then function values in each iteration are printed

    return:
    loss - numpy array containing function value in each iteration of the method
    com_bits - numpy array containing transmitted bits by one node to the server
    '''
    alg = 'ADIANA'
    dim = X.shape[1]
    
    omega = compute_omega(dim, arg)
    arg.alpha = 1 / (1 + omega)
    arg.theta_2 = 0.5
    if omega == 0:
        arg.prob = 1
        arg.eta = 0.5 / arg.L
    else:
        arg.prob = min(1, max(0.5 * arg.alpha, 0.5 * arg.alpha * (np.sqrt(arg.node / (32 * omega)) - 1)))
        arg.eta = min(0.5 / arg.L, arg.node / (64 * omega * arg.L * ((2 * arg.prob * (omega + 1) + 1) ** 2)))
    arg.theta_1 = min(1 / 4, np.sqrt(arg.eta * arg.lamda / arg.prob))
    arg.gamma = 0.5 * arg.eta / (arg.theta_1 + arg.eta * arg.lamda)
    arg.beta = 1 - arg.gamma * arg.lamda
    
    if verbose:
        print('algorithm ' + alg + ' starts')
        print('eta = ', arg.eta, 'compression: ', arg.comp_method)
        print('f_opt = ', f_opt)
    
    dim = X.shape[1]
    num_data = y.shape[0]
    num_data_worker = int(np.floor(num_data / arg.node))
    
    zk = w
    yk = w
    wk = w
    xk = w
    
    loss = []
    local_gradx = np.zeros((arg.node, dim))
    local_gradw = np.zeros((arg.node, dim))
    hs = np.zeros((arg.node, dim))
    hs_mean = np.mean(hs, axis=0)
    deltas = np.zeros((arg.node, dim))
    deltasw = np.zeros((arg.node, dim))
    loss_0 = loss_logistic(X, y, yk, arg)

    
    if verbose:
        print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [1]
    bits = 1
    comp_method = compression_dic[arg.comp_method]
    com_round_bit = compute_bit(dim, arg)
    k=0
    while k < arg.T and loss[-1] - f_opt > tol:
        k += 1
        xk = arg.theta_1 * zk + arg.theta_2 * wk + (1 - arg.theta_1 - arg.theta_2) * yk
        for i in range(arg.node):
            local_gradx[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                  y[i * num_data_worker:(i + 1) * num_data_worker], xk, arg)
            deltas[i] = comp_method(local_gradx[i] - hs[i], arg)
            local_gradw[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                  y[i * num_data_worker:(i + 1) * num_data_worker], wk, arg)
            deltasw[i] = comp_method(local_gradw[i] - hs[i], arg)
            hs[i] += arg.alpha * deltasw[i]
        gk = np.mean(deltas, axis=0) + hs_mean
        assert gk.shape[0] == len(w)
        hs_mean += arg.alpha * np.mean(deltasw, axis=0)
        assert hs_mean.shape[0] == len(w)
        oldyk = yk
        yk = xk - arg.eta * gk
        zk = arg.beta * zk + (1 - arg.beta) * xk + (arg.gamma / arg.eta) * (yk - xk)
        change = np.random.random()
        if bernoulli.rvs(arg.prob):
            wk = oldyk
        bits += com_round_bit
        loss_k = loss_logistic(X, y, yk, arg)
        loss.append(loss_k)
        com_bits.append(bits)
        if verbose:
            if k % 1000 == 0:
                print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits

######################################################################################################

class FedNL:
    def __init__(self, oracle):
        '''
        ----------------------------------------------
        This class is created to simulate FedNL method
        ----------------------------------------------
        '''
        self.oracle = oracle
    
    def method(self, x, hes_comp_param, hes_comp_name='LowRank', init='zero',\
               init_cost=True, option=1, upd_rule=1, lr=None,\
               max_iter=100, tol=1e-15,\
               bits_compute='OneSided', verbose=True):
        
        '''
        ------------------------------
        Implementation of FedNL method
        ------------------------------
        
        input: x - initial point
        hes_comp_param - parameter of local Hessian compression operator
        hes_comp_name - name of compression operator for Hessians
        init - if zero, then H^0_i = 0, otherwise H^0_i = \nabla^2 f_i(x^0)
        init_cost - if True, then the communication cost on initialization is included
        option - if 1, then the method uses Option 1 of FedNL; 
                 if 2, then the method uses Option 2 of FedNL
        upd_rule - if 1, then method requires Biased compressor and uses stepsize alpha=1-\sqrt{1-delta}
                   if 2, then method requires Biased compressor and uses stepsize alpha=1
                   if 3, then methods requires Unbiased compressor and uses stepsize alpha=1/(omega+1)
        lr - learning rate if the user wants to use PowerSGD where stepsize can be chosen
        max_iter - maximum number of iterations
        tol - desired tolerance of the solution
        bits_compute - if OneSided, then bits are computed in upside direction only
                       if TwoSided, then bits are computed in both directions: upside and backside
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        
        
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        
        comp_hessians = {
            'LowRank':Low_Rank,
            'TopK': TopK,
            'PowerSGD':PowerSgdCompression,
            'RandomK':random_spars_matrix
        }
        
        hes_comp_cost = {
            'LowRank':32*hes_comp_param*d,
            'TopK': 32*hes_comp_param,
            'PowerSGD': 2*32*hes_comp_param*d,
            'RandomK':32*hes_comp_param
        }

        
        x_new = x.copy()
        x_old = x.copy()
        w = x.copy()

        func_value = [] 
        bits = [] 
        iterates = []

        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x_new))
        iterates.append(np.linalg.norm(x_new-x_opt))
        global_bits = 1
        bits.append(1)
        

        H_i = []
        for i in range(n):
            if init == 'zero':
                H_i.append(np.zeros((d,d)))
            else:
                H_i.append(self.oracle.local_Hessian(x_new,i)+lmb*np.eye(d))
                
        if init_cost:
            func_value.append(self.oracle.function_value(x_new))
            global_bits = 32*d*(d+1)//2
            bits.append(global_bits)
            iterates.append(np.linalg.norm(x_new-x_opt))
            
            

        if verbose:
            print(func_value[-1])
            
        n_iters = 0
        while func_value[-1] > f_opt + tol and n_iters <= max_iter:
            n_iters += 1

            x_old = x_new


            if option == 1:
                H = np.mean(H_i, axis=0)
                H = pos_projection(H, lmb)

            if option == 2:
                H = np.mean(H_i, axis=0)
                l_i = []
                for i in range(n):
                    l_i.append(np.linalg.norm(self.oracle.local_Hessian(x_old,i)+lmb*np.eye(d) - H_i[i], 2))    
                l = np.mean(l_i)
                H += l*np.eye(d)

                
            s = np.linalg.solve(H, self.oracle.full_gradient(x_old) + lmb*x_old)
            x_new = x_old - s   
            
            iterates.append(np.linalg.norm(x_new-x_opt))
            func_value.append(self.oracle.function_value(x_new))


            if lr is None:
                for i in range(n):
                    if upd_rule == 1:
                        Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                    - H_i[i], hes_comp_param)
                        eta = 1 - np.sqrt(1-delta)
                        H_i[i] += eta*Delta
                    if upd_rule == 2:
                        Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                    - H_i[i], hes_comp_param)
                        H_i[i] += Delta
                    if upd_rule == 3:
                        Delta, omega = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                    - H_i[i], hes_comp_param)
                        eta = 1/(omega+1)
                        H_i[i] += eta*Delta
            else:
                for i in range(n):
                    if upd_rule == 1:
                        Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                    - H_i[i], hes_comp_param, delta=lr)
                        eta = 1 - np.sqrt(1-delta)
                        H_i[i] += eta*Delta
                    if upd_rule == 2:
                        Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                    - H_i[i], hes_comp_param, delta=lr)
                        H_i[i] += Delta
                    if upd_rule == 3:
                        Delta, omega = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                    - H_i[i], hes_comp_param)
                        eta = 1/(omega+1)
                        H_i[i] += eta*Delta

            global_bits += 32*d + hes_comp_cost[hes_comp_name]
            
            if bits_compute=='TwoSided':
                global_bits += 32*d

            bits.append(global_bits)
            
            if verbose:
                print(func_value[-1])

                
            
        return np.array(func_value), np.array(bits), np.array(iterates)


#############################################################################################
    
class FedNL_CR:
    
    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate FedNL-CR method
        -------------------------------------------------
        '''
        self.oracle = oracle
        
        
    def subproblem_solver(self, g, H, M, tol=10**(-15), max_iter=10000):
        '''
        -----------------------------------
        Implementation of subproblem solver
        -----------------------------------
        
        input:
        g - gradient of f at current point
        H - Hessian of f at current point
        M - cubic regularization coefficient
        tol - desired tolerance of the solution
        max_iter - maximum iterations of the solver
        
        return: 
        shift s which minimises <g,s> + 0.5*s^T*H*s + M/6*\|s\|^3
        '''
        U, LMB, V = np.linalg.svd(H)
        t = 1
        local_func = t
        d = self.oracle.get_number_of_weights()
        for i in range(d):
            local_func -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
            
        n_steps = 0
        while np.abs(local_func) > tol and n_steps <= max_iter:
            n_steps += 1
            der_value = 1
            func_value = t
            for i in range(d):
                der_value += M*(U.dot(g))[i]**2/(LMB[i] + M/2*t)**3
                func_value -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
            t = t - func_value/der_value
            
            local_func = t
            for i in range(d):
                local_func -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
                
        t = np.sqrt(t)
        s = -np.linalg.inv(H + M*t/2*np.eye(d)).dot(g)
        return s
    
    
    def get_M(self):
        '''
        -----------------------------
        Lipshitz constant for Hessian 
        -----------------------------
        
        '''
        m = self.oracle.get_number_of_local_data_points()
        n = self.oracle.get_number_of_nodes()
        c = 1/(6*np.sqrt(3))
        
        ans = []
        for i in range(n):
            temp = 0
            for j in range(m):
                temp += c/m*np.linalg.norm(self.oracle.A[i*m+j])**3
            ans.append(temp)
            
        return np.mean(ans)

        
    def method(self, x, hes_comp_param, hes_comp_name='LowRank', init='zero', init_cost=True, upd_rule=1, max_iter=100, tol=1e-15,\
               bits_compute='OneSided', verbose=True):
        
        '''
        ------------------------------
        Implementation of FedNL-CR method
        ------------------------------
        
        input: 
        x - initial point
        hes_comp_param - parameter of local Hessian compression operator
        hes_comp_name - name of compression operator for Hessians
        init - if zero, then H^0_i = 0, otherwise H^0_i = \nabla^2 f_i(x^0)
        init_cost - if True, then the communication cost on initialization is included
        upd_rule - if 1, then method requires Biased compressor and uses stepsize alpha=1-\sqrt{1-delta}
                   if 2, then method requires Biased compressor and uses stepsize alpha=1
                   if 3, then methods requires Unbiased compressor and uses stepsize alpha=1/(omega+1)
        max_iter - maximum number of iterations
        tol - desired tolerance of the solution
        bits_compute - if OneSided, then bits are computed in upside direction only
                       if TwoSided, then bits are computed in both directions: upside and backside
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        
        comp_hessians = {
            'LowRank':Low_Rank,
            'TopK': TopK,
            'PowerSGD':PowerSgdCompression,
            'RandomK':random_spars_matrix
        }
        
        hes_comp_cost = {
            'LowRank':32*hes_comp_param*d,
            'TopK': 32*hes_comp_param,
            'PowerSGD': 2*32*hes_comp_param*d,
            'RandomK':32*hes_comp_param
        }
        
        x_new = x.copy()
        x_old = x.copy()
        
        func_value = [] 
        bits = [] 
        iterates = []

        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x_new))
        global_bits = 1
        bits.append(global_bits)
        iterates.append(np.linalg.norm(x_new-x_opt))
        
        H_i = []
        for i in range(n):
            if init=='zero':
                H_i.append(np.zeros((d,d)))
            else:
                H_i.append(self.oracle.local_Hessian(x_new,i)+lmb*np.eye(d))
            
            if init_cost:
                global_bits = 32*d*(d+1)//2
                bits.append(global_bits)
                func_value.append(self.oracle.function_value(x_new))
                iterates.append(np.linalg.norm(x_new-x_opt))

        M = self.get_M()

        n_iters = 0
        
        if verbose:
            print(func_value[-1])
            
        while func_value[-1] > f_opt + tol and n_iters <= max_iter:
            
            x_old = x_new
            n_iters += 1
            
            H = np.mean(H_i, axis=0)
                
            l_i = []
            for i in range(n):
                l_i.append(np.linalg.norm(self.oracle.local_Hessian(x_old,i)+lmb*np.eye(d) - H_i[i]))    
            l = np.mean(l_i)

            
            g = self.oracle.full_gradient(x_old) + lmb*x_old

            h = self.subproblem_solver(g, H+l*np.eye(d), M)

            x_new = x_old + h

            func_value.append(self.oracle.function_value(x_new))
            iterates.append(np.linalg.norm(x_new-x_opt))
            if bits_compute == 'OneSided':
                global_bits += 32*d + hes_comp_cost[hes_comp_name] + 32
            else:
                global_bits += 32*d + 32*d + hes_comp_cost[hes_comp_name] + 32

            bits.append(global_bits)
            if upd_rule == 1:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    eta = 1 - np.sqrt(1-delta)
                    H_i[i] += eta*Delta
            if upd_rule == 2:
                Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                            - H_i[i], hes_comp_param)
                H_i[i] += Delta
            if upd_rule == 3:
                Delta, omega = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                            - H_i[i], hes_comp_param)
                eta = 1/(omega+1)
                H_i[i] += eta*Delta
            
            if verbose:
                print(func_value[-1])
            
        return np.array(func_value), np.array(bits), np.array(iterates)
        
        
####################################################################################################
class Newton_Zero:
    
    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate Newton-Zero (N0) method
        -------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, init_cost=True, line_search=False, gamma=0.5, c=0.5, tol=1e-15, max_iter=1000, verbose=True):
        '''
        -------------------------
        Implementation of N0 method
        -------------------------
        input:
        x - initial model weightsn
        init_cost - if True, then the communication cost of initalization is inclued
        line_search - if True, then the method uses backtracking line search procedure
        gamma - parameter of line search procedure
        c - parameter of line search procedure
        tol - desired tolerance of the solution
        max_iter - maximum number of iterations
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        
        
        func_value = [] 
        bits = [] 
        iterates = []
        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x))
        global_bits = 1
        bits.append(global_bits)
        iterates.append(np.linalg.norm(x-x_opt))
        
        if init_cost:
            global_bits = 32*d*(d+1)//2
            func_value.append(self.oracle.function_value(x))
            iterates.append(np.linalg.norm(x-x_opt))
            bits.append(global_bits)
            
        H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
        
        if verbose:
            print(func_value[-1])
            
        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            
            g = self.oracle.full_gradient(x)+lmb*x
            D = - np.linalg.inv(H).dot(g)
            der = D.dot(g)
            t = 1
            
            if line_search:
                while self.oracle.function_value(x+t*D) > func_value[-1] + c*t*der:
                    t *= gamma
                    global_bits += 32

            x = x + t*D

            global_bits += 32*d
            bits.append(global_bits)
            func_value.append(self.oracle.function_value(x))
            iterates.append(np.linalg.norm(x-x_opt))
            
            if verbose:
                print(func_value[-1])
                
        return np.array(func_value), np.array(bits), np.array(iterates)
        
        
#############################################################################

class Gradient_Descent:

    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate Gradient Descent (GD)
        -------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, line_search=False, gamma=0.5, c=0.5, tol=1e-15, max_iter=100000, verbose=True):
        '''
        -------------------------
        Implementation of GD method
        -------------------------
        input:
        x - initial model weightsn
        line_search - if True, then the method uses backtracking line search procedure
        gamma - parameter of line search procedure
        c - parameter of line search procedure
        tol - desired tolerance of the solution
        max_iter - maximum number of iterations
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        
        
        func_value = [] 
        bits = [] 
        iterates = []
        stepsizes = []
        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x))
        global_bits = 1
        bits.append(global_bits)
        iterates.append(np.linalg.norm(x-x_opt))
        stepsizes = []

        H = np.dot(self.oracle.A.T,self.oracle.A)/N
        temp = np.linalg.eigvalsh(H)
        L = np.abs(temp[-1])/4
        
        if verbose:
            print(func_value[-1])
            
        n_steps = 0
        
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            g = self.oracle.full_gradient(x) + lmb*x
            
            if line_search:
                D = - g
                der = D.dot(g)
                t = 1

                f_cur = self.oracle.function_value(x)
                while self.oracle.function_value(x+t*D) > f_cur + c*t*der:
                    t *= gamma
                    global_bits += 32
                    
                x = x + t*D
            else:
                x = x - 1/L*g
            
            global_bits += 32*d
            bits.append(global_bits)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))

            if verbose:
                print(func_value[-1])
            
        return np.array(func_value), np.array(bits), np.array(iterates)
    
    
    
class FedNL_LS:
    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate FedNL-LS method
        -------------------------------------------------
        '''
        self.oracle = oracle
    
    def method(self, x, hes_comp_param, hes_comp_name='LowRank', init='zero', init_cost=True, max_iter=100, tol=1e-15,\
               bits_compute='OneSided', upd_rule=1, gamma=0.5, c=0.5, verbose=True):
        
        '''
        ---------------------------------
        Implementation of FedNL-LS method
        ---------------------------------
        
        input: 
        x - initial point
        hes_comp_param - parameter of local Hessian compression operator
        hes_comp_name - name of compression operator for Hessians
        init - if zero, then H^0_i = 0, otherwise H^0_i = \nabla^2 f_i(x^0)
        init_cost - if True, then the communication cost on initialization is included
        max_iter - maximum number of iterations
        tol - desired tolerance of the solution
        bits_compute - if OneSided, then bits are computed in upside direction only
                       if TwoSided, then bits are computed in both directions: upside and backside
        upd_rule - if 1, then method requires Biased compressor and uses stepsize alpha=1-\sqrt{1-delta}
                   if 2, then method requires Biased compressor and uses stepsize alpha=1
                   if 3, then methods requires Unbiased compressor and uses stepsize alpha=1/(omega+1)
        gamma - parameter of line search procedure
        c - parameter of line search procedure
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        
        
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        
        comp_hessians = {
            'LowRank':Low_Rank,
            'TopK': TopK,
            'PowerSGD':PowerSgdCompression,
            'RandomK': random_spars_matrix
        }

        
        hes_comp_cost = {
            'LowRank':32*hes_comp_param*d,
            'TopK': 32*hes_comp_param,
            'PowerSGD': 2*32*hes_comp_param*d,
            'RandomK':32*hes_comp_param
        }
        
        x_new = x.copy()
        x_old = x.copy()
        w = x.copy()

        func_value = [] 
        bits = [] 
        iterates = []

        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x_new))
        
        global_bits = 1
        bits.append(1)
        iterates.append(np.linalg.norm(x_new-x_opt))

        H_i = []
        for i in range(n):
            if init == 'zero':
                H_i.append(np.zeros((d,d)))
            else:
                H_i.append(self.oracle.local_Hessian(x_new,i)+lmb*np.eye(d))
                
        if init_cost:
            func_value.append(self.oracle.function_value(x_new))
            global_bits = 32*d*(d+1)//2
            bits.append(global_bits)
            iterates.append(np.linalg.norm(x_new-x_opt))
            
            
        if verbose:
            print(func_value[-1])

        n_iters = 0
        while func_value[-1] > f_opt + tol and n_iters <= max_iter:
            n_iters += 1

            x_old = x_new


            H = np.mean(H_i, axis=0)
            H = pos_projection(H, lmb)

            g = self.oracle.full_gradient(x_old) + lmb*x_old
            D = -np.linalg.inv(H).dot(g)

            t = 1
            f_cur = self.oracle.function_value(x_old) 
            d_k = - np.linalg.inv(H).dot(self.oracle.full_gradient(x_old) + lmb*x_old)
            D_k = d_k.dot(self.oracle.full_gradient(x_old)+lmb*x_old)

            while self.oracle.function_value(x_old+t*d_k) > f_cur+c*t*D_k:
                t *= gamma
                global_bits += 32


            x_new = x_old + t*d_k

            iterates.append(np.linalg.norm(x_new-x_opt))
            func_value.append(self.oracle.function_value(x_new))

            for i in range(n):
                if upd_rule == 1:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    eta = 1 - np.sqrt(1-delta)
                    H_i[i] += eta*Delta
                if upd_rule == 2:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    H_i[i] += Delta
                if upd_rule == 3:
                    Delta, omega = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_old, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    eta = 0.5/(omega+1)
                    H_i[i] += eta*Delta


            global_bits += 32*d + hes_comp_cost[hes_comp_name]
            bits.append(global_bits)
            
            if verbose:
                print(func_value[-1])

        return np.array(func_value), np.array(bits), np.array(iterates)
        
        
        
##########################################################################################
class FedNL_BC:
    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate FedNL-BC method
        -------------------------------------------------
        '''
        self.oracle = oracle
    
    def method(self, x, hes_comp_param, iter_comp_param, p=0.5,\
               hes_comp_name='LowRank', iter_comp_name='Rounding',\
               init='zero', init_cost=True,\
               option=1, upd_rule=1, iter_upd_rule=1,\
               max_iter=100, tol=1e-15, verbose=True):
        
        '''
        ------------------------------
        Implementation of FedNL method
        ------------------------------
        
        input: x - initial point
        hes_comp_param - parameter of compression operator applied to Hessians
        iter_comp_param - parameter of compression operator applied to models
        p - probability with that gradients are sent to the server
        hes_comp_name - name of compression operator for Hessians
        iter_comp_name - name of compression operator for models
        init - if zero, then H^0_i = 0, otherwise H^0_i = \nabla^2 f_i(x^0)
        init_cost - if True, then the communication cost on initialization is included
        option - if 1, then the method uses Option 1 of FedNL; 
                 if 2, then the method uses Option 2 of FedNL
        upd_rule - if 1, then method requires Biased compressor and uses stepsize alpha=1-\sqrt{1-delta}
                   if 2, then method requires Biased compressor and uses stepsize alpha=1
                   if 3, then methods requires Unbiased compressor and uses stepsize alpha=1/(omega+1)
        iter_upd_rule - if 1, then method requires Biased compressor and uses stepsize eta=1-\sqrt{1-delta_M}
                        if 2, then method requires Biased compressor and uses stepsize eta=1
                        if 3, then methods requires Unbiased compressor and uses stepsize eta=1/(omega+1)
        max_iter - maximum number of iterations
        tol - desired tolerance of the solution
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        
        
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        
        comp_hessians = {
            'LowRank':Low_Rank,
            'TopK': TopK,
            'PowerSGD':PowerSgdCompression,
            'RandomK':random_spars_matrix
        }
        
        comp_iterates = {
            'TopK':topK_vectors,
            'Rounding':biased_rounding,
            'RandomK': randomK_vectors
        }
        
        hes_comp_cost = {
            'LowRank':32*hes_comp_param*d,
            'TopK': 32*hes_comp_param,
            'PowerSGD': 2*32*hes_comp_param*d,
            'RandomK':32*hes_comp_param
        }
        
        iter_comp_cost = {
            'TopK': 32*iter_comp_param,
            'Rounding': 9*d,
            'RandomK': 32*iter_comp_param
        }
        
        x_new = x.copy()
        x_old = x.copy()
        w = x.copy()
        z = x.copy()

        func_value = [] 
        bits = [] 
        iterates = []

        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x_new))
        iterates.append(np.linalg.norm(x_new-x_opt))
        global_bits = 1
        bits.append(1)
        

        H_i = []
        for i in range(n):
            if init == 'zero':
                H_i.append(np.zeros((d,d)))
            else:
                H_i.append(self.oracle.local_Hessian(x_new,i)+lmb*np.eye(d))
                
        if init_cost:
            func_value.append(self.oracle.function_value(x_new))
            global_bits = 32*d*(d+1)//2
            bits.append(global_bits)
            iterates.append(np.linalg.norm(x_new-x_opt))
            
            

        if verbose:
            print(func_value[-1])
            
        c = 1
        n_iters = 0
        while func_value[-1] > f_opt + tol and n_iters <= max_iter:
            n_iters += 1
            x_old = x_new

            if option == 1:
                H = np.mean(H_i, axis=0)
                H = pos_projection(H, lmb)

            if option == 2:
                H = np.mean(H_i, axis=0)
                l_i = []
                for i in range(n):
                    l_i.append(np.linalg.norm(self.oracle.local_Hessian(x_old,i)+lmb*np.eye(d) - H_i[i]))    
                l = np.mean(l_i)
                H += l*np.eye(d)

            if c:
                G = []
                for i in range(n):
                    G.append(self.oracle.local_gradient(w, i) + lmb*w)
                g = np.mean(G, axis=0)
                z = w
            else:
                g = H.dot(w-z) + np.mean(G, axis=0)

                
            
            s = np.linalg.solve(H, g)
            x_new = w - s

            func_value.append(self.oracle.function_value(x_new))
            iterates.append(np.linalg.norm(x_new-x_opt))

                
            for i in range(n):
                if upd_rule == 1:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(w, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    eta = 1 - np.sqrt(1-delta)
                    H_i[i] += eta*Delta
                if upd_rule == 2:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(w, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    H_i[i] += Delta
                if upd_rule == 3:
                    Delta, omega = comp_hessians[hes_comp_name](self.oracle.local_Hessian(w, i)+lmb*np.eye(d)\
                                                                - H_i[i], hes_comp_param)
                    eta = 1/(omega+1)
                    H_i[i] += eta*Delta
            
            if iter_upd_rule == 1:
                s, delta = comp_iterates[iter_comp_name](x_new-w, iter_comp_param)
                eta = 1 - np.sqrt(1-delta)
                w = w + eta*s
            if iter_upd_rule == 2:
                s, delta = comp_iterates[iter_comp_name](x_new-w, iter_comp_param)
                eta = 1 - np.sqrt(1-delta)
                w = w + s
            if iter_upd_rule == 3:
                s, omega = comp_iterates[iter_comp_name](x_new-w, iter_comp_param)
                eta = 1/(omega+1)
                w = w + eta*s

            if c:
                global_bits += 32*d + hes_comp_cost[hes_comp_name] + iter_comp_cost[iter_comp_name]
            else:
                global_bits += hes_comp_cost[hes_comp_name] + iter_comp_cost[iter_comp_name]
      
            if verbose:
                print(func_value[-1]) 

            bits.append(global_bits)
            
            c = np.random.choice([0,1], p=[1-p, p])
            
            
        return np.array(func_value), np.array(bits), np.array(iterates)
        
######################################################################################################

class FedNL_PP:
    def __init__(self, oracle):
        '''
        This class is created to simulate FedNL-PP method
        '''
        self.oracle = oracle
    
    def method(self, x, tau, hes_comp_param, hes_comp_name='LowRank', init='nonzero',
               init_cost=True, upd_rule=1, max_iter=1000, tol=1e-15,\
               bits_compute='OneSided', verbose=True):
        
        '''
        ----------------------------------
        Implementation of FedNL-PP  method
        ----------------------------------
        
        input: 
        x - initial point
        tau - number of active nodes
        hes_comp_param - parameter of local Hessian compression operator
        hes_comp_name - name of compression operator for Hessians
        init - if zero, then H^0_i = 0, otherwise H^0_i = \nabla^2 f_i(x^0)
        init_cost - if True, then the communication cost on initialization is included
        upd_rule - if 1, then method requires Biased compressor and uses stepsize alpha=1-\sqrt{1-delta}
                   if 2, then method requires Biased compressor and uses stepsize alpha=1
                   if 3, then methods requires Unbiased compressor and uses stepsize alpha=1/(omega+1)
        max_iter - maximum number of iterations
        tol - desired tolerance of the solution
        bits_compute - if OneSided, then bits are computed in upside direction only
                       if TwoSided, then bits are computed in both directions: upside and backside
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        
        
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m

        comp_hessians = {
            'LowRank':Low_Rank,
            'TopK': TopK,
            'PowerSGD':PowerSgdCompression,
            'RandomK':random_spars_matrix
        }
        
        hes_comp_cost = {
            'LowRank':32*hes_comp_param*d,
            'TopK': 32*hes_comp_param,
            'PowerSGD': 2*32*hes_comp_param*d,
            'RandomK':32*hes_comp_param
        }

        
        x_new = x.copy()
        x_old = x.copy()


        func_value = [] 
        bits = [] 
        iterates = []

        f_opt = self.oracle.function_value(x_opt)

        func_value.append(self.oracle.function_value(x_new))
        iterates.append(np.linalg.norm(x_new-x_opt))
        global_bits = 1
        bits.append(1)
        

        H_i = []
        for i in range(n):
            if init == 'zero':
                H_i.append(np.zeros((d,d)))
            else:
                H_i.append(self.oracle.local_Hessian(x_new,i)+lmb*np.eye(d))

        l_i = []
        for i in range(n):
            l_i.append(np.linalg.norm(self.oracle.local_Hessian(x_old,i)+lmb*np.eye(d) - H_i[i]))    

        w_i = []
        for i in range(n):
            w_i.append(x_new)

        g_i = []
        for i in range(n):
            g_i.append((H_i[i] + l_i[i]*np.eye(d)).dot(w_i[i]) - (self.oracle.local_gradient(w_i[i], i)+lmb*w_i[i]))
                
        if init_cost:
            func_value.append(self.oracle.function_value(x_new))
            global_bits = 32*d*(d+1)//2
            bits.append(global_bits)
            iterates.append(np.linalg.norm(x_new-x_opt))
            
            

        if verbose:
            print(func_value[-1])

        nodes = [i for i in range(n)]
            
        n_iters = 0
        while func_value[-1] > f_opt + tol and n_iters <= max_iter:
            n_iters += 1

            x_old = x_new

            H = np.mean(H_i, axis=0)
            l = np.mean(l_i)
            g = np.mean(g_i, axis=0)
            
            x_new = np.linalg.inv(H + l*np.eye(d)).dot(g)

            iterates.append(np.linalg.norm(x_new-x_opt))
            func_value.append(self.oracle.function_value(x_new))

            
            np.random.shuffle(nodes)
            for i in nodes[:tau]:
                w_i[i] = x_new

                if upd_rule == 1:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_new, i)+lmb*np.eye(d)-H_i[i],\
                                                                hes_comp_param)
                    eta = 1 - np.sqrt(1-delta)
                    H_i[i] += eta*Delta
                if upd_rule == 2:
                    Delta, delta = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_new, i)+lmb*np.eye(d) - H_i[i],\
                                                                hes_comp_param)
                    H_i[i] += Delta
                if upd_rule == 3:
                    Delta, omega = comp_hessians[hes_comp_name](self.oracle.local_Hessian(x_new, i)+lmb*np.eye(d) - H_i[i],\
                                                                hes_comp_param)
                    eta = 1/(omega+1)
                    H_i[i] += eta*Delta

                l_i[i] = np.linalg.norm(self.oracle.local_Hessian(x_new,i)+lmb*np.eye(d) - H_i[i])
                g_i[i] = (H_i[i]+l_i[i]*np.eye(d)).dot(w_i[i]) - (self.oracle.local_gradient(w_i[i], i)+lmb*w_i[i])
                
                

            global_bits += 32*d + hes_comp_cost[hes_comp_name]
            
            if bits_compute=='TwoSided':
                global_bits += 32*d

            bits.append(global_bits)
            
            if verbose:
                print(func_value[-1])
            
                
            
        return np.array(func_value), np.array(bits), np.array(iterates)
        

###############################################################################

class DORE:
    def __init__(self, oracle):
        '''
        ---------------------------------------------
        This class is created to simulate DORE method
        ---------------------------------------------
        '''
        self.oracle = oracle

    def method(self, x, verbose=True, max_iter=1e7, tol=1e-15):
        '''
        -----------------------------
        Implementation of DORE method
        -----------------------------
        input:
        x - initial model weights
        tol - desired tolerance of the solution
        max_iter - maximum number of steps of the method 
        verbose - if True, then function values in each iteration are printed

        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        x_opt = self.oracle.get_optimum()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        f_opt = self.oracle.function_value(x_opt)


        func_value = []
        bits = []
        iterates = []
        global_bits = 1

        bits.append(global_bits)
        func_value.append(self.oracle.function_value(x))
        iterates.append(np.linalg.norm(x-x_opt))

        h_i = []
        for i in range(n):
            h_i.append(np.zeros(d))


        x_hat_new = x.copy()
        x_hat_old = x.copy()
        x = x
        e = np.zeros(d)
        
        x_i = []
        for i in range(n):
            x_i.append(x)

        # define constants of the method
        c_q = 1 # variance of random dithering with s=\sqrt{d}

        a = 1/(2*(c_q+1))
        b = 1/(2*(c_q+1))
        c = 4*c_q*(c_q+1)/n

        left = (-c_q+np.sqrt(c_q**2+4*(1-(c_q+1)*b)))/(2*c_q)

        H = np.dot(self.oracle.A.T,self.oracle.A)/N
        temp = np.linalg.eigvalsh(H)
        L = np.abs(temp[-1])/4

        right = 4*lmb*L/((lmb+L)**2*(1+c*a)-4*lmb*L)
        left, right
        eta = min(left, right)

        left = eta*(lmb+L)/(2*(1+eta)*lmb*L)
        right = 2/((1+c*a)*(lmb+L))

        gamma = (left+right)/2

        n_steps = 0 

        if verbose:
            print(func_value[-1])

        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            x_hat_old = x_hat_new

            g_i = []
            Delta_i = []
            Comp_Delta_i = []
            for i in range(n):
                g_i.append(self.oracle.local_gradient(x_i[i], i)+lmb*x_i[i])
                Delta_i.append(g_i[i] - h_i[i])
                Comp_Delta_i.append(rand_dith(Delta_i[i], s=np.sqrt(d)))
                
            Comp_Delta = np.mean(Comp_Delta_i, axis=0)
            h = np.mean(h_i, axis=0)

            g = Comp_Delta + h

            x = x_hat_old - gamma*g

            q = x - x_hat_old + eta*e
            Comp_q = rand_dith(q, s=np.sqrt(d))
            e = q - Comp_q

            x_hat_new = x_hat_old + b*Comp_q

            
            for i in range(n):
                x_i[i] = x_i[i] + b*Comp_q
                h_i[i] = h_i[i] + a*Comp_Delta_i[i]


            func_value.append(self.oracle.function_value(x_hat_new))
            iterates.append(np.linalg.norm(x_hat_new-x_opt))
            global_bits += 2*(2.8 * d + 32)
            bits.append(global_bits)

            if verbose:
                print(func_value[-1])

        return np.array(func_value), np.array(bits), np.array(iterates)
    
##########################################################################

class Artemis:
    
    def __init__(self, oracle):
        '''
        ------------------------------------------------
        This class is created to simulate Artemis method
        ------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, tau, compression='OneSided', max_iter=1e7, tol=1e-15, verbose=True):
        '''
        -----------------------------
        Implementation of DORE method
        -----------------------------
        input:
        x - initial model weights
        tau - number of active nodes
        compression - if OneSided, then random dithering is applied only on uplink direction
                      otherwise random dithering is applied both on uplink and downlink directions
        tol - desired tolerance of the solution
        max_iter - maximum number of steps of the method 
        verbose - if True, then function values in each iteration are printed

        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        
        x_opt = self.oracle.get_optimum()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        f_opt = self.oracle.function_value(x_opt)
        
        Delta_i = []
        Comp_Delta_i = []
        g_i = []
        h_i = []
        for i in range(n):
            h_i.append(np.zeros(d))
            g_i.append(np.zeros(d))
            Delta_i.append(np.zeros(d))
            Comp_Delta_i.append(np.zeros(d))

        Omega_i = []
        for i in range(n):
            Omega_i.append([])

        k_i = []
        for i in range(n):
            k_i.append(-1)

        x_i = []
        for i in range(n):
            x_i.append(x)
        
        H = np.dot(self.oracle.A.T,self.oracle.A)/N
        temp = np.linalg.eigvalsh(H)
        L = np.abs(temp[-1])/4
        
        a = 0.25
        gamma = 1/(5*L)

        func_value = []
        func_value.append(self.oracle.function_value(x))

        bits = [1]
        global_bits = 1
        
        iterates = [np.linalg.norm(x-x_opt)]


        nodes = [i for i in range(n)]

        n_steps = 0
        np.random.shuffle(nodes)
        
        if verbose:
            print(func_value[-1])

        while func_value[-1] - f_opt > 1e-15 and n_steps <= max_iter:
            for i in range(n):
                if i in nodes[:tau]:
                    k_i[i] = n_steps
                    g_i[i] = self.oracle.local_gradient(x_i[i],i)+lmb*x_i[i]
                    Delta_i[i] = g_i[i] - h_i[i]
                    Comp_Delta_i[i] = rand_dith(Delta_i[i],s=np.sqrt(d))
                else:
                    g_i[i] = np.zeros(d)
                    Comp_Delta_i[i] = np.zeros(d)

            g = np.mean(h_i, axis=0) + n/tau*np.mean(Comp_Delta_i, axis=0)
            if compression == 'OneSided':
                Omega = g
            else:
                Omega = rand_dith(g, s=np.sqrt(d))

            x = x - gamma*Omega
            for i in range(n):
                Omega_i[i].append(Omega)

            np.random.shuffle(nodes)
            for i in nodes[:tau]:
                for v in Omega_i[i]:
                    x_i[i] = x_i[i] - gamma*v
                Omega_i[i] = []


            for i in range(n):
                h_i[i] = h_i[i] + a*Comp_Delta_i[i]

            n_steps += 1

            func_value.append(self.oracle.function_value(x))
            iterates.append(np.linalg.norm(x-x_opt))

            global_bits += 2.8 * d + 32
            bits.append(global_bits)
            
            if verbose:
                print(func_value[-1])
            
        return np.array(func_value), np.array(bits), np.array(iterates)

    