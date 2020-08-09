import unittest
import os
import Forecast_Checker as fc




class Test_Forecast_Checker(unittest.TestCase):


    def test_Forecast_Checker(self):

        '''
        gen synthetic light curves
        '''
        finished = False
        Nepochs = 1000
        forecast_step = 200
        synth = fc.generate_synthetic_data(polynomial_exog_coef=[0.0, 0.005],
                                           Nepochs=Nepochs,
                                           forecast_step=forecast_step,
                                           synthetic_class=fc.Custom_ARIMA(seed=12345),
                                           synthetic_kwargs={'arparms': [0.75, -0.25],
                                                             'maparms': [0.65, 0.35]})
        y_test = synth['y_full']
        y_test_arima = synth['y_arima']
        yex = synth['y_eXog']
        eXog_test = synth['eXog_features']


        '''
        try a fit
        '''
        cl2 = fc.Custom_ARIMA(seed=12345)
        cl2.fit(y_test[:Nepochs], eXog=eXog_test[:Nepochs, :])
        y_pred = cl2.predict(steps=forecast_step, eXog=eXog_test[Nepochs:, :])


        '''
        Evaluate the performance
        '''
        ep = fc.evaluate_performance(eXog_test, y_test, model=fc.Custom_ARIMA,
                                     kwargs_for_model={'round': False},
                                     kwargs_for_fit={'parms': (2, 0, 2)},
                                     kwargs_for_predict={'steps': 10},
                                     verbose=False, Nsteps=10, Ntests=4)
        ep.evaluate()

        finished = True
        assert finished == True
        #ep.make_performance_plot(file='test_eval_plot.png')


        self.assertEqual(finished,True)


if __name__ == '__main__':
    unittest.main()