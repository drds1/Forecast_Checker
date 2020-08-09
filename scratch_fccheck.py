from arima_exog import *





if __name__ == '__main__':
    '''
    Generate synthetic light curve based on arima process with polynomial 
    exogoneous variables
    '''
    Nepochs = 1000
    forecast_step = 200
    synth = generate_synthetic_data(polynomial_exog_coef = [0.0,0.005],
                            Nepochs = Nepochs,
                            forecast_step = forecast_step,
                            synthetic_class = Custom_ARIMA(seed=12345),
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
    cl2 = Custom_ARIMA(seed=12345)
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



