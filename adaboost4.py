def adaBoost(X,y,B): #program does nothing as written
    def train(X, w, y):
        n = np.shape(X)[0]
        p = np.shape(X)[1]
        mode = np.repeat(0.0, p)
        theta = np.repeat(0.0, p)
        loss = np.repeat(0.0, p)

        # find optimal theta for every dimension j
        for j in range(p):
            # get the index of the order
            indx = sorted(range(len(X.iloc[:, j])), key=lambda k: X.iloc[:, j][k])
            x_j = X.iloc[indx, j]
            # using a cumulative sum, count the weight when progressively shifting the threshold to the right
            w_cum = np.cumsum(np.multiply(w[indx], y[indx]))
            # handle multiple occurrences of same x_j value: threshold
            # point must not lie between elements of same value
            ######################
            # w_cum[duplicates(List, x_j) == 1] = NA
            # find the optimum threshold and classify accordingl
            # m < - max(abs(w_cum), na.rm = TRUE)
            #########################
            m = max(abs(w_cum))
            # guarantee that the type of input is array:
            maxIndx = min([i for i in range(len(w_cum)) if abs(w_cum[i]) == m])
            mode[j] = (w_cum[maxIndx] < 0) * 2 - 1
            theta[j] = x_j[maxIndx]
            c = ((X.iloc[:, j] > theta[j]) * 2 - 1) * mode[j]
            loss[j] = np.dot(np.array(np.array(c.values) != y, dtype=float), w)
            # determine optimum dimension, threshold and comparison
            # for test only:
            #print c
        m = min(loss)
        j_star = min([i for i in range(len(loss)) if loss[i] == m])
        # pars < - dict(j=j_star, theta=theta[j_star], mode=mode[j_star])
        pars = [j_star, theta[j_star], mode[j_star]]
        return pars

    def classify(X, pars):
        label = (2 * (X.iloc[:, pars[0]] > pars[1]) - 1) * pars[2]
        return label

    n = np.shape(X)[0]
    w = np.ones(n)/n
    alpha = np.repeat(0.0,B)
    #allPars < - rep(list(list()), B)
    allPars = pd.DataFrame(index=range(B), columns=range(3))
    # boost base classifiers
    for b in range(B):
        # step a) train base classifier„
        allPars.iloc[b,:] = train(X, w, y)
        # step b) compute error
        missClass = np.array(np.array(classify(X, allPars.iloc[b, :])) != y, dtype=float)
        e = np.dot(w, missClass)/sum(w)
        # step c) compute voting weight
        if e==0:
            alpha[b] = 10
        else:
        #   alpha[b] = np.log((1 - e)/e)
            alpha[b] = np.log((1 - e)/e)
        # step d) recompute weights
        w = np.multiply(w, np.exp(np.multiply(alpha[b], missClass)))

    return [allPars, alpha]



def agg_class(X,alpha,allPars):
    n = np.shape(X)[0]
    B = len(alpha)
    Labels = pd.DataFrame(0.0,index=range(n), columns=range(B))

    for b in range(B):
        Labels.iloc[:, b] = np.copy(classify(X, allPars.iloc[b, :]).values)
        # weight classifier response with respective alpha coefficient
    Labels = np.dot(Labels, alpha)
    c_hat = np.sign(Labels)
    return(c_hat)



co_app = pd.read_csv('book_1_co_occurrence.csv',index_col=0)
label = pd.read_csv('WeasleyLabel1.csv',index_col=0)
#calculate the number of pairs;
num = len(co_app)*(len(co_app)-1)
X = pd.DataFrame(index = range(num), columns=range(3))
y = np.repeat(0,num)
n = np.shape(co_app)[0]
for i in range(len(co_app.index)-1):
    for j in range(i+1,n):
        X.iloc[i*(2*n-i-1)/2+(j-i)-1,0] = co_app.iloc[i,j]
co_app = pd.read_csv('book_1_polarity.csv',index_col=0)
for i in range(len(co_app.index) - 1):
    for j in range(i + 1, n):
        X.iloc[i * (2 * n - i - 1) / 2 + (j - i) - 1, 1] = co_app.iloc[i, j]
co_app = pd.read_csv('book_1_subjectivity.csv', index_col=0)
for i in range(len(co_app.index) - 1):
    for j in range(i + 1, n):
        X.iloc[i * (2 * n - i - 1) / 2 + (j - i) - 1, 2] = co_app.iloc[i, j]
for i in range(len(label.index) - 1):
    for j in range(i + 1, n):
        y[i * (2 * n - i - 1) / 2 + (j - i) - 1] = label.iloc[i, j]

B = 1
allPars = adaBoost(X,y,B)[0]
alpha = adaBoost(X,y,B)[1]
result = agg_class(X,alpha,allPars)

#calculate the percentage of correctness:
(len(y) - sum(abs(result - y)))/len(y)



#split
train = X.iloc[range(0,500),:]
test  = X.iloc[range(700,800),:]

ytrain = y[range(0,500)]
ytest  = y[range(700,800)]

B = 10

allPars = adaBoost(train,ytrain,B)[0]
alpha = adaBoost(train,ytrain,B)[1]
result = agg_class(test,alpha,allPars)

#calculate the percentage of correctness:
(len(ytest) - sum(abs(result - ytest)))/len(ytest)






# there's some problem with the code we are using, since all the results are the same
#here we change the X and y to make the third dimension the best classifier:
X = pd.DataFrame(np.random.randn(10,12),index=range(10),columns=range(12))
X.iloc[0,:] = [9,8,1,7,1,2,4,2,5,3,3,5]
X.iloc[1,:] = [2,3,1,2,-2,5,3,2,4,5,3,3]
X.iloc[2,:] = [8,7,1,8,-2,4,52,3,-4,3,22,3]
X.iloc[3,:] = [1,2,9,1,5,2,3,-4,-2,-1,-3,6]
X.iloc[4,:] = [10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.5,-3,100]
X.iloc[5,:] = [-1,1,1,1,1,100,20,3.2,-5.4,22,0,0.001]
X.iloc[6,:] = [1,3,-1,-4,-2,1,2,23,4.5,22,3,0.001]
X.iloc[7,:] = [2.8,1,23,1,-5,22,3.5,-44,2,-1,-3.6,6.3]
X.iloc[8,:] = [-0.0001,23,39,1,54,-0.2,3,-14,-23,-0.001,-33,0.96]
X.iloc[9,:] = [13,24,-0.9,1,52,20,3,-0.4,0.002,-2,4,-2]
y = np.array([1,1,-1,-1,1,1,-1,-1,-1,1])






