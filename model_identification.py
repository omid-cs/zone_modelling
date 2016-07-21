import numpy as np
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as nparam
import statsmodels.api as sm

#from statsmodels.nonparametric.api import KernelReg
#import statsmodels.sandbox.nonparametric.dgp_examples as dgp
import warnings
from loaddata import *
#from kernel_extras import *

from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear

if __name__ == '__main__':

    smap_retriever = SMAPDataRetriever()
    data = smap_retriever.get_data('371','BLD_X','09/04/2015','09/05/2015')

    y = []
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    t = []
    time_offset = -1
    previtem = None


    for d in sorted(data):
        if time_offset == -1:
            time_offset = d
        t.append(((d-time_offset)/600000)%144)
        y.append(data[d]["temp"])
        if previtem == None:
            previtem = data[d]["temp"]
        x0.append(previtem)
        previtem = data[d]["temp"]
        x1.append(data[d]["reheat"])
        x2.append(data[d]["outtemp"])
        x3.append(data[d]["flow"])


    # choose the best kernel bandwidth using least square and leave-one-out cross validation
    # use Nadaraya-Watson kernel regression to compute conditional expectations
    # of each dependent/independent variable given the time index ()
    model1 = nparam.KernelReg(endog=[x1],
                             exog=[t], reg_type='lc',
                             var_type='c', bw='cv_ls')

    bw_x1= model1.bw
    mean_x1, mtx_x1 = model1.fit()

    model2 = nparam.KernelReg(endog=[x2],
                             exog=[t], reg_type='lc',
                             var_type='c', bw='cv_ls')

    bw_x2= model2.bw
    mean_x2, mtx_x2 = model2.fit()

    model3 = nparam.KernelReg(endog=[x3],
                             exog=[t], reg_type='lc',
                             var_type='c', bw='cv_ls')

    bw_x3= model3.bw
    mean_x3, mtx_x3 = model3.fit()

    model_y = nparam.KernelReg(endog=[y],
                             exog=[t], reg_type='lc',
                             var_type='c', bw='cv_ls')

    bw_y= model_y.bw
    mean_y, mfx_y = model_y.fit()

    # must be the same as time-shiffted mean_y
    mean_x0 = [mean_y[0]]
    mean_x0.extend(mean_y[:-1])
    mean_x0 = np.array(mean_x0)

    X = np.vstack([x0-mean_x0, x1-mean_x1, x2-mean_x2, x3-mean_x3]).T
    est = sm.OLS(y-mean_y, X).fit()
    beta_hat = est.params
    print beta_hat

    plt.plot(t, est.params[0] * (x0-mean_x0) + est.params[1] * (x1-mean_x1) + est.params[2] * (x2-mean_x2) + est.params[3] * (x3-mean_x3), 'r')
    plt.plot(t, y-mean_y, 'b')

    # use Nadaraya-Watson kernel regression to estimate the nonparametric terms
    bx = np.dot(np.vstack([x0, x1, x2, x3]).T,beta_hat)
    newy = y - bx

    model_final = nparam.KernelReg(endog=[newy],
                             exog=[t], reg_type='lc',
                             var_type='c', bw='cv_ls')

    bw_final= model_final.bw
    mean_final, mtx_final = model_final.fit()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(t,x2,'o', alpha=0.5)
    ax.plot(t[:144],mean_x2[:144],lw=2)
    #ax.plot(t[:144],mean_final[:144],lw=2)
    plt.show()
