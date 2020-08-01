import statsmodels.tsa.arima_model as arima_model
import statsmodels.tsa.arima_process as arima_process
import numpy as np
import matplotlib.pylab as plt
import pandas as pd


class Super_ARIMA:

    def __init__(self, seed= 12345, round = True):
        np.random.seed(seed)
        self.model = None
        self.round = round

    def generate_test_arima(self,
                            test_N = 250,
                            arparms = [0.75, -0.25],
                            maparms = [0.65, 0.35]):
        arp = np.array(arparms)
        map = np.array(maparms)
        ar = np.r_[1, -arp] # add zero-lag and negate
        ma = np.r_[1, map] # add zero-lag
        return arima_process.arma_generate_sample(ar, ma, test_N)
        #model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)


    def fit(self,X, y,parms = (2,0,2)):
        '''
        similar to standard fit predict of normal sklearn models
        but eXog not necessary
        :param y:
        :param eXog:
        :return:
        '''
        eXog = X
        if len(parms) == 3:
            model  = arima_model.ARIMA(y, parms,exog = eXog).fit(trend='nc', disp=0)
        elif len(parms) == 2:
            model = arima_model.ARMA(y, parms, exog=eXog).fit(trend='nc', disp=0)
        self.model = model

    def predict(self, X, steps=10):
        '''
        predict the fited model above for N steps
        :param y:
        :param eXog: Must (steps X Nexog features)
        :return:
        '''
        #assertion checks
        eXog = X
        assert steps >= 1
        if eXog is not None:
            assert steps == np.shape(eXog)[0]

        #predictions
        predictions = self.model.forecast(steps, eXog, alpha= 0.05)[0]

        if self.round is True:
            predictions = np.round(predictions)

        return predictions


    def evaluate_performance(self, y, eXog=None, parms = (2,0,2),
                             verbose = False,Nsteps = 10,Ntests = 10):
        '''
        go back in time and refit the model comparing with reality
        :return:
        '''
        Ny = len(y)

        #initialise the output results dataframe
        output = {}
        output['t'] = np.arange(Ny)
        output['y'] = y
        output = pd.DataFrame(output)
        output = output.set_index('t')
        output_truths = {}
        output_times = {}
        output_pred = {}

        #run the cross validation
        Ntestsmax = min(Ny/2 - Nsteps,Ntests)
        for i in range(Ntestsmax):
            if verbose is True:
                print('running test '+str(i+1)+' of '+str(Ntestsmax))
            lab = 'test '+str(i)
            idxlo = Ny - Nsteps - i
            ttest = np.arange(idxlo,Ny)
            #prep the y variable
            yin = y[:idxlo]
            ytrue = y[idxlo:idxlo+Nsteps]
            output_times[lab] = ttest[:Nsteps]
            output_truths[lab] = ytrue

            #prep the exogenious variables
            if eXog is not None:
                exin = eXog[:idxlo,:]
                extest = eXog[idxlo:,:]
            else:
                exin = None
                extest = None

            cltemp = Super_ARIMA(round=self.round)
            cltemp.fit(exin, yin,parms = parms)
            y_pred = cltemp.predict(extest, Ny - idxlo)
            output_pred[lab] = y_pred[:Nsteps]
            output[lab] = np.nan
            output.loc[ttest,lab] = y_pred
        self.evaluation_output = output
        self.evaluation_residue = output.sub(output['y'], axis='rows').iloc[:,1:]
        self.evaluation_truths = pd.DataFrame(output_truths)
        self.evaluation_times = pd.DataFrame(output_times)
        self.evaluation_pred = pd.DataFrame(output_pred)

        residual_summary = pd.DataFrame({})
        for i in range(Nsteps):
            res = pd.DataFrame(self.evaluation_residue.apply(lambda column: column.dropna().values[i])).transpose()
            residual_summary = residual_summary.append(res)
        residual_summary.index = np.arange(Nsteps)
        self.residue_summary = residual_summary



def evaluate_performance(X, y, model=Super_ARIMA(round=True),
                         kwargs_for_model={},
                         kwargs_for_fit={'parms':(2,0,2)},
                         kwargs_for_predict={'steps':10},
                         verbose = False,Nsteps = 10,Ntests = 10):
    '''
    go back in time and refit the model comparing with reality
    all input models should have a 'steps' argument in their predict functions
    This Must be the same as the 'Nsteps' input argument here example default arguments
    correct for the superarima code above

    :return:
    '''
    kwargs_for_predict['steps'] = Nsteps
    eXog = X
    Ny = len(y)
    #initialise the output results dataframe
    output = {}
    output['t'] = np.arange(Ny)
    output['y'] = y
    output = pd.DataFrame(output)
    output = output.set_index('t')
    output_truths = {}
    output_times = {}
    output_pred = {}
    #run the cross validation
    Ntestsmax = min(Ny/2 - Nsteps,Ntests)
    for i in range(Ntestsmax):
        if verbose is True:
            print('running test '+str(i+1)+' of '+str(Ntestsmax))
        lab = 'test '+str(i)
        idxlo = Ny - Nsteps - i
        ttest = np.arange(idxlo,Ny)
        #prep the y variable
        yin = y[:idxlo]
        ytrue = y[idxlo:idxlo+Nsteps]
        output_times[lab] = ttest[:Nsteps]
        output_truths[lab] = ytrue
        #prep the exogenious variables
        if eXog is not None:
            exin = eXog[:idxlo,:]
            extest = eXog[idxlo:,:]
        else:
            exin = None
            extest = None

        cltemp = model(**kwargs_for_model)
        cltemp.fit(exin, yin,**kwargs_for_fit)
        y_pred = cltemp.predict(extest, **kwargs_for_predict)
        #cltemp = model
        #cltemp.fit(yin, eXog=exin, parms=parms)
        #y_pred = cltemp.predict(Ny - idxlo, eXog=extest)
        output_pred[lab] = y_pred[:Nsteps]
        output[lab] = np.nan
        output.loc[ttest,lab] = y_pred

    evaluation = {}
    evaluation['output'] = output
    evaluation['residue'] =  output.sub(output['y'], axis='rows').iloc[:,1:]
    evaluation['truths'] = pd.DataFrame(output_truths)
    evaluation['times'] = pd.DataFrame(output_times)
    evaluation['pred'] = pd.DataFrame(output_pred)
    return evaluation


class plot_performance_evaluator:
    def __init__(self, truths, pred):
        self.truths = truths
        self.pred = pred

    def calculate_correlations(self):
        self.correlations = self.truths.corrwith(self.pred, axis=1)

    def make_plots(self,
                   step_plots = [0,1,9],
                   step_plot_str_add = 1,
                   figure = 'performance_evaluator.pdf'):
        # make diagnostic plots
        truths = self.truths
        pred = self.pred
        self.calculate_correlations()
        correlations = self.correlations
        residue = pred - truths

        Nplots = len(step_plots) + 2
        ncols = 2
        nrows = int(np.ceil(Nplots / ncols))

        # make the abs_residual plot
        absres = residue.abs()
        meanres = absres.mean(axis=1)
        stdres = absres.std(axis=1)
        fig = plt.figure()
        t = np.arange(len(meanres)) + step_plot_str_add
        x = meanres.values
        sig = stdres.values
        ax1 = fig.add_subplot(nrows, ncols, 1)
        ax1.set_xlabel('forecast step')
        ax1.set_ylabel('absolute\nresidue')
        ax1.plot(t, x, color='b')
        ax1.fill_between(t, x - sig, x + sig, alpha=0.4, color='b')

        # make the correlation plot
        x = correlations.values
        ax1 = fig.add_subplot(nrows, ncols, 2)
        ax1.set_xlabel('forecast step')
        ax1.set_ylabel('correlation\ncoefficient')
        ax1.plot(t, x)

        # make the correlation plots
        iplot = 3
        for step in step_plots:
            x = pred.iloc[step, :]
            y = truths.iloc[step, :]
            r = np.corrcoef(x, y)[0, 1]
            ax1 = fig.add_subplot(nrows, ncols, iplot)
            ax1.set_xlabel('predicted')
            ax1.set_ylabel('true')
            ax1.scatter(x, y)
            ax1.set_title(str(step+step_plot_str_add) + '-step, r = ' + str(np.round(r, 2)))
            iplot += 1

        plt.tight_layout()
        plt.savefig(figure)



if __name__ == '__main__':
    '''
    test the new arima model code
    '''
    Ntest = 1000
    Nforecast = 20

    #instantiate class
    cl = Super_ARIMA(seed = 12345)
    cl.test_arparms = [0.75, -0.25]
    cl.test_maparms = [0.65, 0.35]

    #generate arma part of test timeseries
    y_test_arima = cl.generate_test_arima(test_N = Ntest+Nforecast,
                            arparms = [0.75, -0.25],
                            maparms = [0.65, 0.35])

    #simulate exogenious variables using polynomial
    exoP = [0.1,0.05]
    xex = np.arange(Ntest + Nforecast)
    yex = np.zeros(Ntest + Nforecast)
    eXog_test = np.zeros((Ntest + Nforecast, len(exoP)))
    for i in range(len(exoP)):
        eXog_test[:,i] = xex**i
        yex += exoP[i]*eXog_test[:,i]
    y_test = y_test_arima + yex


    #now try to fit the test time series
    cl2 = Super_ARIMA(seed=12345)
    cl2.fit(y_test[:Ntest],eXog=eXog_test[:Ntest,:])
    y_pred = cl2.predict(steps = Nforecast, eXog=eXog_test[Ntest:, :])


    #plot the result predictions and test
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(xex,y_test_arima,label='arima process')
    ax1.set_xlabel('time')
    ax1.set_ylabel('arima')

    ax1 = fig.add_subplot(312)
    ax1.plot(xex, yex, label='exogenious time series')
    ax1.set_xlabel('time')
    ax1.set_ylabel('exog')

    ax1 = fig.add_subplot(313)
    ax1.plot(xex, y_test, label='combined time series')
    ax1.plot(xex[Ntest:], y_pred, label='predicted')
    ax1.set_xlabel('time')
    ax1.set_ylabel('All')

    plt.savefig('arima_test.pdf')
    plt.close()


    #perform cross validation to evaluate model performance
    cl2.evaluate_performance(y_test,
                             eXog=eXog_test,
                             verbose = True,
                             parms=(2, 2),
                             Nsteps=10,
                             Ntests=50)
    op = cl2.evaluation_output
    residue_summary = cl2.residue_summary

    evaluation_times = cl2.evaluation_times
    evaluation_residue = cl2.evaluation_residue
    evaluation_truths = cl2.evaluation_truths
    evaluation_pred = cl2.evaluation_pred

    #make diagnostic plots
    truths = evaluation_truths
    pred = evaluation_pred
    correlations = truths.corrwith(pred, axis=1)
    residue = pred - truths
    plt.close()
    plot_performance = plot_performance_evaluator(truths=truths, pred=pred)
    plot_performance.make_plots(step_plots=[0,4,9], figure='test_eval_plot.pdf')
