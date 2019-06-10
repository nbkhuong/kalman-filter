import numpy as np
import matplotlib.pyplot as plt

## Hyper parameters
x0 = 1
v0 = 0
alpha = .15
dt = 0.2
var_perturb = 0.01
var_pos = 400
var_vel = 4
N = 100 # Total number of time steps

F = np.array([[1, dt], 
              [0, 1]])
B = np.array([0.5*dt**2, dt, 0])
H = np.array([1, 0]).reshape(1,2)
P = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
R = np.array([var_perturb]).reshape(1,1)
Q = np.array([[var_pos, 0],
              [0, var_vel]])

F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[var_pos, 0, 0.0], [0, var_vel, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)

## Compute motion
def motion_calc(x0, v0, alpha, dt, var_perturb, N):
    
    eta = np.random.normal(0, np.sqrt(var_perturb), 1)
    X_predict = []
    X_last_predict = []
    pos_predict = [] 
    vel_predict = []
        
    for i in range(N):
        if (i == 0):
            X_last_predict = np.array([x0, v0, 0])
        else:
            X_last_predict = np.array([X_predict[len(X_predict) - 1][0], X_predict[len(X_predict) - 1][1], 0])
            
        predict = np.dot(F, X_last_predict) + np.dot(B, alpha) + eta
        
        pos_predict.append(predict[0])
        vel_predict.append(predict[1])
        
    return pos_predict, vel_predict
    
## Adding noise
def add_noise(inp, var_p, var_v):
    noise_pos = np.random.normal(0, np.sqrt(var_p), 1)
    noise_vel = np.random.normal(0, np.sqrt(var_v), 1)
    
    zk = inp[0] + noise_pos
    Vk = inp[1] + noise_vel
    
    return zk, Vk

## main()
estimations = []
pos_mea = []
pos_pre = []
x_k = x0
v_k = v0

for i in range(N):
    xk, vk = motion_calc(x_k, v_k, alpha, dt, var_perturb, 1)
    X = np.array([xk[0], vk[0], 0])
    measurement = add_noise(X, var_pos, var_vel)[0]
    pos_mea.append(measurement)
    pos_pre.append(xk)
    print(xk)
    P = np.dot(np.dot(F, P), F.T) + Q
    tmp = np.matmul(H, X)
    estimations.append(tmp[0])
    
    y = measurement - np.dot(H, X)
    tmp = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(np.dot(P, H.T), np.linalg.inv(tmp))
    X = X + np.dot(K, y)

    x_k = X[0]
    v_k = X[1]
    I = np.eye(F.shape[1])
    P = np.dot(np.dot(I - np.dot(K, H), P), (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)

plt.figure()
plt.plot(range(len(pos_mea)), np.asanyarray(pos_mea), label = 'Measurements', color='green')
plt.plot(range(len(pos_pre)), np.asanyarray(pos_pre), label = 'Ground Truth', color='red')
plt.plot(range(len(estimations)), np.asanyarray(estimations), label = 'Kalman Filter Estimates', color='blue')
plt.legend()
plt.grid()
plt.show()
    
plt.show()



