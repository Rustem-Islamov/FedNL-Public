import numpy as np
import math
from scipy.stats import bernoulli


def random_k(x, k):
    '''
    ----------------------------------------
    Unbiased Rand-k compressor for vectors
    ----------------------------------------
    '''
    dim = x.shape[0]
    answer = np.zeros(dim)
    positions = list(range(dim))
    np.random.shuffle(positions)
    for i in positions[:k]:
        answer[i] = x[i]*dim/k
    return answer

def random_sparsification(x, p):
    '''
    ---------------------------------------------------
    Unbiased random sparsification operator for vectors
    ---------------------------------------------------
    '''
    d = x.shape[0]
    binom = np.zeros((d+1, d+1))
    for i in range(d+1):
        binom[i,0] = 1
        binom[i,i] = 1
    for i in range(2,d+1):
        for j in range(1, i):
            binom[i,j] = binom[i-1,j]+binom[i-1,j-1]
    comp_g = np.zeros(d)
    k = 0
    for i in range(d):
        if bernoulli.rvs(p):
            comp_g[i] = x[i]/p
            k += 1
    bits = 64*k + np.log2(binom[d, k])
    bits = int(np.ceil(bits))
            
    return comp_g, bits

def positive_part(x):
    '''
    ------------------------------------------------
    Projection onto non-negative orthant for vectors
    ------------------------------------------------
    '''
    for i, c in enumerate(x):
        if c < 0:
            x[i] = 0
    return x

    

def random_sparse(x, arg):
    '''
    ---------------------------------------------------
    Unbiased random sparsification operator for vectors
    (this one is used for DIANA and ADIANA)
    ---------------------------------------------------
    '''
    r = int(arg.r)
    dim = np.shape(x)[0]
    xi_r = np.zeros(dim, dtype=int)
    loc_ones = np.random.choice(dim, r, replace=False)
    xi_r[loc_ones] = 1
    x_compress = dim / r * x * xi_r
    return x_compress



def random_dithering(x, arg):
    '''
    ----------------------------------------------
    Unbiased random dithering compression operator 
    for vectors
    ----------------------------------------------
    '''
    s = int(arg.s)
    dim = np.shape(x)[0]
    xx = np.random.uniform(0.0, 1.0, dim)
    xnorm = np.linalg.norm(x)
    if xnorm > 0:
        xsign = np.sign(x)
        x_int = np.floor(s * np.abs(x) / xnorm + xx)
        x_cpmpress = xnorm / s * xsign * x_int
    else:
        x_cpmpress = np.zeros(dim)
    return x_cpmpress
    
    
def rand_dith(x, s):
    '''
    ----------------------------------------------
    Unbiased random dithering compression operator 
    for vectors
    ----------------------------------------------
    '''
    dim = np.shape(x)[0]
    xx = np.random.uniform(0.0, 1.0, dim)
    xnorm = np.linalg.norm(x)
    if xnorm > 0:
        xsign = np.sign(x)
        x_int = np.floor(s * np.abs(x) / xnorm + xx)
        x_cpmpress = xnorm / s * xsign * x_int
    else:
        x_cpmpress = np.zeros(dim)
    return x_cpmpress



def natural_compression(x, arg):
    '''
    ----------------------------------------
    Unbiased natural compression for vectors
    (this one is used for DIANA and ADIANA)
    ----------------------------------------
    '''
    dim = np.shape(x)[0]
    xabs = np.abs(x)
    xsign = np.sign(x)
    x_compress = x
    for i in range(dim):
        if x[i] != 0.0:
            xlog = np.log2(xabs[i])
            xdown = np.exp2(np.floor(xlog))
            xup = np.exp2(np.ceil(xlog))
            p = (xup - xabs[i]) / xdown
            if np.random.uniform(0.0, 1.0) <= p:
                x_compress[i] = xsign[i] * xdown
            else:
                x_compress[i] = xsign[i] * xup
    return x_compress


def no_compression(x, arg):
    '''
    ------------------------------------------
    Identical compression operator for vectors
    ------------------------------------------
    '''
    return x


def loss_logistic(X, y, w, arg):
    '''
    ---------------------------------------
    Logistic regression loss
    (this one is used for DIANA and ADIANA)
    ---------------------------------------
    '''
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    z = y * np.dot(X, w)
    tmp = np.minimum(z, 0)
    loss = np.log((np.exp(tmp) + np.exp(tmp - z)) / np.exp(tmp))
    loss_sum = np.sum(loss) / len(y)
    reg = (np.linalg.norm(w) ** 2) * arg.lamda / 2

    return loss_sum + reg

def sigmoid(z):
    '''
    ----------------
    Sigmoid function
    ----------------
    '''
    return 1 / (1 + math.exp(-z))

def regularizer2(w, lamda):
    '''
    -----------------
    L2 regularization
    -----------------
    '''
    res = np.linalg.norm(w) ** 2
    return res * lamda / 2

def prox2(x, eta):
    '''
    -----------------------------
    Proximal operator 
    (this one is used for ADIANA)
    -----------------------------
    '''
    newx = x / (1 + eta)
    return newx

def grad(X, y, w, arg):
    '''
    --------------------------------
    Gradient for logistic regression
    --------------------------------
    '''
    
    m = len(y)
    z = y * np.dot(X, w)
    tmp0 = np.minimum(z, 0)
    tmp1 = np.exp(tmp0 - z) / ((np.exp(tmp0) + np.exp(tmp0 - z)))
    tmp2 = - tmp1 * y
    res = np.dot(X.T, tmp2) / m + arg.lamda * w
    return res

def compute_bit(dim, arg):
    '''
    -------------------------------
    Number of transmitted bits used 
    by each compression operator
    -------------------------------
    '''
    if arg.comp_method == 'rand_sparse':
        bit = 32 * arg.r
    elif arg.comp_method == 'rand_dithering':
        bit = 2.8 * dim + 32
    elif arg.comp_method == 'natural_comp':
        bit = 9 * dim
    else:
        bit = 32 * dim
    return bit


def compute_omega(dim, arg):
    '''
    -------------------------
    Variance parameter for 
    each compression operator
    -------------------------
    '''
    if arg.comp_method == 'rand_sparse':
        omega = dim / arg.r - 1
    elif arg.comp_method == 'rand_dithering':
        omega = 1  # level s=sqrt(dim)
    elif arg.comp_method == 'natural_comp':
        omega = 1 / 8
    else:
        omega = 0
    return omega


def topK_vectors(x, k):
    '''
    ---------------------------------------------
    Biased Top-K compression operator for vectors
    ---------------------------------------------
    '''
    d = x.shape[0]
    ans = np.zeros(d)
    values = []
    for i in range(d):
        values.append((abs(x[i]), i))
    
    values.sort()
    values = values[::-1]
    
    delta = k/d

    for v in values[:k]:
        c, i = v
        ans[i] = x[i]
    
    return ans, delta

def randomK_vectors(x, k):
    '''
    ----------------------------------------
    Unbiased Rand-k compressor for vectors
    ----------------------------------------
    '''
    dim = x.shape[0]
    answer = np.zeros(dim)
    positions = list(range(dim))
    np.random.shuffle(positions)
    for i in positions[:k]:
        answer[i] = x[i]*dim/k
    omega = dim/k - 1
    return answer, omega

def biased_rounding(x, b=2):
    '''
    ------------------------------------------------
    Biased rounding compression operator for vectors
    ------------------------------------------------
    '''
    d = x.shape[0]
    ans = np.zeros(d)
    
    for i in range(d):
        if x[i] == 0:
            ans[i] = 0
        else:
            k = np.floor(math.log(abs(x[i]), b))
            if abs(b**k - abs(x[i])) <= abs(b**(k+1)-abs(x[i])):
                ans[i] = np.sign(x[i])*b**k
            else:
                ans[i] = np.sign(x[i])*b**(k+1)
                
    delta = 4*b/(b+1)**2
    
    return ans, delta


def semidef_projection(X):
    '''
    --------------------------------------------
    Projection of symmetric matrix onto the cone
    of positive semidefinite matrices
    --------------------------------------------
    '''
    d = X.shape[0]
    ans = np.zeros((d,d))
    S, U = np.linalg.eig(X)
    for i in range(d):
        if S[i] > 0:
            ans += (S[i]*U.T[i].reshape(d,1).dot(U.T[i].reshape(1,d))).astype(float)
    
    return ans

def pos_projection(X, mu):
    '''
    ----------------------------------------------
    Projection of symmetric matrix onto the cone
    of positive definite matrices with constant mu
    ----------------------------------------------
    '''
    d = X.shape[0]
    ans = semidef_projection(X-mu*np.eye(d))
    ans += mu*np.eye(d)
    return ans

def TopK(X, k):
    '''
    ----------------------------------------------
    Biased Top-K compression operator for matrices
    ----------------------------------------------
    '''
    d = X.shape[0]
    ans = np.zeros((d,d))
    values = []
    for i in range(d):
        for j in range(i, d):
            values.append((abs(X[i,j]),i,j))

    values.sort()
    values = values[::-1]
    
    delta = 2*k/(d*(d+1))

    for v in values[:k]:
        c, i, j = v
        ans[i,j] = X[i,j]
        ans[j,i] = X[j,i]
    
    return ans, delta


def Low_Rank(X, k):
    '''
    -----------------------------------------------
    Biased Rank-R compression operator for matrices
    -----------------------------------------------
    '''
    V, S, U = np.linalg.svd(X,full_matrices=True)
    d = X.shape[0]
    ans = np.zeros((d,d))
    delta = 0
    for i in range(k):
        ans += (S[i]*(V[:,i].reshape(-1,1).dot(U[i].reshape(-1,1).T))).astype(float)
        delta += S[i]**2
        
    delta = delta/(sum((S**2))+1e-15)
    return ans, delta


def PowerSgdCompression(X, k, delta=0.1):
    '''
    -------------------------------------------------
    Biased PowerSGD compression operator for matrices
    -------------------------------------------------
    '''
    d = X.shape[0]
    mu = np.zeros(d)
    sigma = np.eye(d)
    Q = np.random.multivariate_normal(mu, sigma, size=k).T
    P = X.dot(Q)
    for i in range(P.shape[1]):
        P[:,i] /= (np.linalg.norm(P[:,i])+1e-10)
    Q = X.T.dot(P)
    ans = P.dot(Q.T)
    

    return ans, delta

    
    
def random_spars_matrix(X, k):
    '''
    ---------------------------------
    Unbiased Rand-K sparsification 
    compression operator for matrices
    ---------------------------------
    '''
    d = X.shape[0]
    ans = np.zeros((d,d))
    indexes = []
    for i in range(d):
        for j in range(i,d):
            indexes.append((i,j))
    
    
    omega = d*(d+1)/(2*k)-1
    scale = d*(d+1)/(2*k)

    np.random.shuffle(indexes)

    for i,j in indexes[:k]:
        ans[i,j] = scale*X[i,j]
        ans[j,i] = scale*X[j,i]
    return ans, omega
    
    
    
def generate_synthetic(alpha, beta, iid, d, n, m):
    '''
    ----------------------------------
    synthetic data generation function
    ----------------------------------
    input:
    alpha, beta - parameters of data
    iid - if 1, then the data distribution over nodes is iid
        - if 0, then the data distribution over nodes is non-iid
    d - dimension of the problem
    n - number of nodes
    m - size of local data
    
    output:
    numpy arrays A (features) and b (labels)
    '''
    
    NUM_USER = n
    dimension = d
    NUM_CLASS = 1
    N = n*m
    
    samples_per_user = [m for i in range(NUM_USER)]
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    #print(mean_b)
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    if iid == 1:
        W_global = np.random.normal(0, 1, dimension)
        b_global = np.random.normal(0, 1, NUM_CLASS)


    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, dimension)
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)


        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            p = sigmoid(tmp[0])
            yy[j] = np.random.choice([-1,1], p=[p,1-p])

        X_split[i] = xx
        y_split[i] = yy

    A = np.zeros((N, d))
    b =  np.zeros(N)
    for j in range(NUM_USER):
        A[j*m:(j+1)*m] = X_split[j]
        b[j*m:(j+1)*m] = y_split[j]

    return A, b


# dictionary of compression operators for vectors
compression_dic = {
    'rand_sparse': random_sparse,
    'rand_dithering': random_dithering,
    'natural_comp': natural_compression,
    'no_comp': no_compression
}

# default data set parameters for each data set
default_dataset_parameters = {
    'a1a': {
        'N':1600,
        'n':16,
        'm':100,
        'd':123
    },
    'a9a': {
        'N':32560, 
        'n':80,
        'm':407,
        'd':123
    },
    'w7a': {
        'N':24600, 
        'n':50,
        'm':492,
        'd':300
    },
    'w8a': {
        'N':49700, 
        'n':142,
        'm':350,
        'd':300
    },
    'phishing': {
        'N':11000, 
        'n':100,
        'm':110,
        'd':68    
    }
}

    
    
def read_data(dataset_path, N, n, m, d, lmb,
             labels=['+1', '-1']):
    '''
    -------------------------
    Function for reading data
    -------------------------
    '''
    b = np.zeros(N)
    A = np.zeros((N, d))
    
    f = open(dataset_path, 'r')
    for i, line in enumerate(f):
        line = line.split()
        if i < N:
            for c in line:
                # labels of classes depend on the data set
                # look carefully what they exactly are
                # for a1a, a9a, w8a, w7a they are {+1, -1}
                # for phishing they are {1, 0}
                if c == labels[0]: 
                    b[i] = 1
                elif c == labels[1]:
                    b[i] = -1
                elif c == '\n':
                    continue
                else:
                    c = c.split(':')
                    A[i][int(c[0]) - 1] = float(c[1])     

    f.close()
    return A, b