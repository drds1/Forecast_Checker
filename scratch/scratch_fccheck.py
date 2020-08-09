import Forecast_Checker as fc
import matplotlib.pylab as plt
import numpy as np



if __name__ == '__main__':
    '''
    Generate synthetic light curve based on arima process with polynomial 
    exogoneous variables
    '''
    Nepochs = 1000
    forecast_step = 200
    synth = fc.generate_synthetic_data(polynomial_exog_coef = [0.0,0.005],
                            Nepochs = Nepochs,
                            forecast_step = forecast_step,
                            synthetic_class = fc.Custom_ARIMA(seed=12345),
                            synthetic_kwargs = {'arparms':[0.75, -0.25],
                                                'maparms':[0.65, 0.35]})
    y_test = synth['y_full']
    y_test_arima = synth['y_arima']
    yex = synth['y_eXog']
    eXog_test = synth['eXog_features']



    '''
    now try the time series
    we chop-off the end and try a blind forecast
    The eXogenous variables must be known so only chose variables
    for which you will know the future values 
    '''
    cl2 = fc.Custom_ARIMA(seed=12345)
    cl2.fit(y_test[:Nepochs], eXog=eXog_test[:Nepochs, :])
    y_pred = cl2.predict(steps=forecast_step, eXog=eXog_test[Nepochs:, :])


    '''
    Lets plot the results of our prediction
    '''
    # plot the result predictions and test
    t = np.arange(len(yex))
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(t, y_test_arima, label='arima process')
    ax1.set_xlabel('time')
    ax1.set_ylabel('arima')

    ax1 = fig.add_subplot(312)
    ax1.plot(t, yex, label='exogenious time series')
    ax1.set_xlabel('time')
    ax1.set_ylabel('exog')

    ax1 = fig.add_subplot(313)
    ax1.plot(t, y_test, label='combined time series')
    ax1.plot(t[Nepochs:], y_pred, label='predicted')
    ax1.set_xlabel('time')
    ax1.set_ylabel('All')
    ax1.set_xlim([990,1100])

    plt.savefig('arima_test.png')
    plt.close()



    '''
    Evaluate the performance
    as correlation coefficient vs step size
    '''
    ep = fc.evaluate_performance(eXog_test, y_test, model=fc.Custom_ARIMA,
                 kwargs_for_model={'round':False},
                 kwargs_for_fit={'parms': (2, 0, 2)},
                 kwargs_for_predict={'steps': 10},
                 verbose=False, Nsteps=10, Ntests=10)
    ep.evaluate()
    ep.make_performance_plot(file='test_eval_plot.png')


    #evaluations = evaluate_performance(eXog_test, y_test, model=Custom_ARIMA,
    #                     kwargs_for_model={'round':False},
    #                     kwargs_for_fit={'parms': (2, 0, 2)},
    #                     kwargs_for_predict={'steps': 10},
    #                     verbose=False, Nsteps=10, Ntests=10)
    #evaluation_times = evaluations['times']
    #evaluation_truths = evaluations['truths']
    #evaluation_pred = evaluations['pred']
    #correlations = evaluations['correlations']
#
    ## make diagnostic plots
    #truths = evaluation_truths
    #pred = evaluation_pred
    #correlations = truths.corrwith(pred, axis=1)
    #residue = pred - truths
    #plt.close()
    #plot_performance = plot_performance_evaluator(truths=truths, pred=pred)
    #plot_performance.make_plots(step_plots=[0, 4, 9], figure='test_eval_plot.png')
#
#
#
