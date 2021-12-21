import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

 #Generating a synthetic data set, with Gaussian noise
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e   

#Plotting raw data and true function
plt.figure()
# Plot the data
plt.scatter(X, y, 12, marker='o')           
# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

#Regression with Polynomial Basis Functions
# X float(n, ): univariate data
# d int: degree of polynomial  

def polynomial_transform(X, d):
    V_matrix=[]#initializing matrix
    for xi in X:
        row=[]#initializing row
        for i in range(0,d+1):
            row.append(xi**i)
        V_matrix.append(row)
    return np.array(V_matrix)

# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    Phi_tran=np.transpose(Phi)
    w1=np.matmul(Phi_tran,Phi)
    w2=np.matmul(Phi_tran,y)
    w1_inv=np.linalg.inv(w1)
    w=np.matmul(w1_inv,w2)
    return w

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    n=len(y)
    w_tran=np.transpose(w)
    h=np.matmul(Phi,w_tran)
    for i in range(n):
        error_sum=np.sum(np.power(np.subtract(y,h),2))
    error_mse=error_sum/n
    return error_mse

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

#Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], bbox_to_anchor=(1, 1),fontsize=16)
plt.axis([2, 25, 15, 60])

#Visualizing each learned model
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)),bbox_to_anchor=(1, 1))
plt.axis([-8, 8, -15, 15])

#Regression with Radial Basis Functions

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    RB_kernel=[]
    for xi in X:
        row=[]
        for idx in range(len(B)):
            row.append(np.exp(-1*gamma*pow((xi-B[idx]),2)))
        RB_kernel.append(row)
    return RB_kernel

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(Phi, y, lam):
    Phi_tran=np.transpose(Phi)
    w1=np.matmul(Phi_tran,Phi)+(lam*np.identity(len(y)))
    w2=np.matmul(Phi_tran,y)
    w1_inv=np.linalg.inv(w1)
    w=np.matmul(w1_inv,w2)
    return w

##model building and Analysis
w_rb = {}               # Dictionary to store all the trained models
validationErr_rb = {}   # Validation error of the models
testErr_rb = {}  
lam=[0.001,0.01,0.1,1,10,100,1000]
for idx,l in enumerate(lam):
    Phi_trn_rb=radial_basis_transform(X_trn,X_trn,0.1)#Transforming the training data to higher dimensions
    w_rb[idx]=train_ridge_model(Phi_trn_rb,y_trn,l)#Learbing the model on training data
    
    Phi_val_rb = radial_basis_transform(X_val,X_trn,0.1)#Transforming the val;idation data to higher dimensions
    validationErr_rb[idx] = evaluate_model(Phi_val_rb, y_val, w_rb[idx])#Evaluating the model on validation data
    
    Phi_tst_rb = radial_basis_transform(X_tst,X_trn, 0.1)#Transfroming the test data to higher dimensions
    testErr_rb[idx] = evaluate_model(Phi_tst_rb, y_tst, w_rb[idx])#evaluating the model on test data
    
    
#Visualalizing the Validation and test errors for every lamda values  
x = [i for i in range(0, len(lam))]
plt.figure()
plt.plot(validationErr_rb.keys(), validationErr_rb.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr_rb.keys(), testErr_rb.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lamda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(x, ('0.001', '0.01', '0.1', '1', '10','100','1000'),fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)


#visualizing each learned model
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
lamda=[0.001,0.01,0.1,1,10,100,1000]

for idx,lam in enumerate(lamda):
    X_l = radial_basis_transform(x_true, X_trn,0.1)
    y_l = np.matmul(X_l,w_rb[idx])
    plt.plot(x_true, y_l, linewidth=2)

plt.legend(['true'] + lamda,bbox_to_anchor=(1, 1))

