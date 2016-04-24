import numpy as np
import pandas as pd

#first we use the adaBoost() function to get the classifier that we need
#adaBoost() funtion contains some sub functions which are train() and classify()
#after getting the final classifiers, then we use them to predict new data
# the function we are using in the final stage is called agg_class

def train(X,w,y):
    n = np.shape(X)[0]
    p = np.shape(X)[1]
    mode = np.repeat(0,p)
    theta = np.repeat(0,p)
    loss = np.repeat(0,p)
# find optimal theta for every dimension j
    for j in range(p):
        #get the index of the order
        indx = sorted(range(len(X[:,j])), key=lambda k: X[:,j][k])
        x_j = X[indx,j]
        # using a cumulative sum, count the weight when progressively shifting the threshold to the right
        w_cum = np.cumsum(np.cross(w[indx],y[indx]))
        # handle multiple occurrences of same x_j value: threshold
        # point must not lie between elements of same value

        ######################
        #w_cum[duplicates(List, x_j) == 1] = NA
        # find the optimum threshold and classify accordingl
        #m < - max(abs(w_cum), na.rm = TRUE)
        #########################

        m = max(abs(w_cum))
        #guarantee that the type of input is array:
        maxIndx = min([i for i in range(len(w_cum)) if abs(w_cum[i]) == m])
        mode[j] = (w_cum[maxIndx] < 0) * 2 - 1
        theta[j] = x_j[maxIndx]
        c = ((x_j > theta[j]) * 2 - 1) * mode[j]
        loss[j] = np.dot((c!= y), w.transpose)

    # determine optimum dimension, threshold and comparison
    m = min(loss)
    j_star = min([i for i in range(len(loss)) if loss[i] == m])
    #pars < - dict(j=j_star, theta=theta[j_star], mode=mode[j_star])
    pars = [j_star, theta[j_star], mode[j_star]]
    return pars


def classify(X,pars):
    label = (2*(X[,pars[1]] > pars[2]) - 1)*pars[3]
return(label)



def adaBoost(X,y,B): #program does nothing as written
    n = np.shape(X)[1]
    w = np.repeat(1/n,n)
    alpha = np.repeat(0,B)
    #allPars < - rep(list(list()), B)
    allPars = pd.DataFrame(index=B, columns=3)

    # boost base classifiers
    for b in range(1: B):
        # step a) train base classifier
        allPars.iloc[b,] = train(X, w, y)

        # step b) compute error
        missClass = (y != classify(X, allPars.iloc[b,]))
        e = (np.dot(w, missClass.transpose)/sum(w))[1]

        # step c) compute voting weight
        alpha[b] = log((1 - e)/e)

        # step d) recompute weights
        w = np.cross(w, exp(np.cross(alpha[b], missClass)))
    return allPars
    return alpha


def agg_class(X,alpha,allPars):
    n = np.shape(X)[0]
    B = np.shape(alpha)
    Labels = [[0 for i in range(B)] for i in range(n)]

    for b in range(B):
        Labels[,b] = classify(X,allPars.iloc[b,])
        # weight classifier response with respective alpha coefficient
        Labels = np.dot(Labels, alpha.transpose)
        c_hat = np.sign(Labels)
    return(c_hat)
