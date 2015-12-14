import numpy as np
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as nparam

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
        x1.append(data[d]["reheat"])
        x2.append(data[d]["outtemp"])
        x3.append(data[d]["flow"])
        previtem = data[d]["temp"]

    # 1441602000000 1441177200000 1441177800000
    # find the best kernel bandwidth using leave-one-out cross validation
    # use NW regression to compute coefficients of the parametric part of the model
    model1 = nparam.KernelReg(endog=[x1],
                             exog=[t], reg_type='lc',
                             var_type='u', bw='cv_ls')

    bw_x1= model1.bw
    mean_x1, mtx_x1 = model1.fit()

    model2 = nparam.KernelReg(endog=[x2],
                             exog=[t], reg_type='lc',
                             var_type='u', bw='cv_ls')

    bw_x2= model2.bw
    mean_x2, mtx_x2 = model2.fit()

    model3 = nparam.KernelReg(endog=[x3],
                             exog=[t], reg_type='lc',
                             var_type='u', bw='cv_ls')

    bw_x3= model3.bw
    mean_x3, mtx_x3 = model3.fit()

    model_y = nparam.KernelReg(endog=[y],
                             exog=[t], reg_type='lc',
                             var_type='u', bw='cv_ls')

    bw_y= model_y.bw
    mean_y, mfx_y = model_y.fit()

    # must be the same as time-shiffted mean_y
    mean_x0 = [mean_y[0]]
    mean_x0.extend(mean_y[:-1])
    mean_x0 = np.array(mean_x0)

    # solve linear regression using OLS y-meany = b_0*(x0-mean_x0)+b_1*(x1-mean_x1)+b_2*(x2-mean_x2)+b_3*(x3-mean_x3)+eps
    X = np.vstack([x0-mean_x0, x1-mean_x1, x2-mean_x2, x3-mean_x3]).T
    beta_hat = np.linalg.lstsq(X,y-mean_y)[0]
    print beta_hat

    # use NW regression to estimate the nonparametric part
    bx = np.dot(np.vstack([x0, x1, x2, x3]).T,beta_hat)
    newy = y - bx

    model_final = nparam.KernelReg(endog=[newy],
                             exog=[t], reg_type='lc',
                             var_type='u', bw='cv_ls')

    bw_final= model_final.bw
    mean_final, mtx_final = model_final.fit()

    # print sm_mean

    # model = SemiLinear(endog=[y], exog=[X], exog_nonparametric=[t], var_type='c', k_linear=3)
    # print(model.bw)
    # print(model.b)
    #
    # # use NW regression to estimate the nonparametric part
    # mean, mfx = model.fit()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.plot(t,x3,'o', alpha=0.5)
    #ax.plot(t[:144],mean_x3[:144],lw=2)
    ax.plot(t[:144],mean_final[:144],lw=2)
    plt.show()


    # fig, ax1 = plt.subplots()
    # ax1.plot(t,y,'r',label='temp')
    # ax2 = ax1.twinx()
    # ax2.plot(t,x1,'g',label='reheat')
    # ax2.plot(t,x2,'b',label='outtemp')
    # plt.show()
    # plt.legend()
