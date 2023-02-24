import statsmodels.api as sm
import itertools
import pandas as pd


# def hypertuning_sarima(df, target_column):
#     """
#     grid search(hypertunning sarima) cv for finding the best values of p,q,d,s
#
#     p:- indicate number of autoregressive terms
#     q:- indicate number of moving average terms (lags of the forecast errors)
#     d:- indicate differencing that must be done to stationarize series indicate differencing that must be done to stationarize series
#     s:- indicates seasonal length in the data
#     """
#     p = range(1, 3)
#     d = range(1, 2)
#     q = range(1, 3)
#
#     pdq = list(itertools.product(p, d, q))
#
#     pdqs = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
#
#     def sarimax_gridsearch(ts, pdq, pdqs):
#         ans = []
#         for comb in pdq:
#             for combs in pdqs:
#                 try:
#                     mod = sm.tsa.statespace.SARIMAX(ts,
#                                                     order=comb,
#                                                     seasonal_order=combs,
#                                                     enforce_stationarity=False,
#                                                     enforce_invertibility=False)
#
#                     output = mod.fit()
#                     ans.append([comb, combs, output.bic])
#                     print('SARIMAX {} x {}12 : BIC Calculated ={}'.format(comb, combs, output.bic))
#                 except:
#                     continue
#
#         # Convert into dataframe
#         ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'bic'])
#
#         # Sort and return top 5 combinations
#         ans_df = ans_df.sort_values(by=['bic'], ascending=True)[0:5]
#         print(ans_df)
#
#         return ans_df
#
#     gridsearch_output_df = sarimax_gridsearch(df[target_column], pdq, pdqs)
#     sarima_p = gridsearch_output_df.iloc[1, :][0][0]
#     sarima_d = gridsearch_output_df.iloc[1, :][0][1]
#     sarima_q = gridsearch_output_df.iloc[1, :][0][2]
#     sarima_s = gridsearch_output_df.iloc[1, 1][3]
#     best_sarima_par_list = list([sarima_p, sarima_d, sarima_q, sarima_s])
#     return best_sarima_par_list, target_column


def model_building(train_df, test_df, best_sarima_par_list,target_column):
    """
        building model by using best p,d,q,s parameters found by grid search cv (hypertuning)

    """
    p, d, q, s = best_sarima_par_list

    model = sm.tsa.statespace.SARIMAX(train_df[target_column], order=(p, d, q), seasonal_order=(p, d, q, s))
    results = model.fit()

    # assignning start and end time of prediction
    start = len(train_df)
    end = len(train_df) + len(test_df) - 1
    predicted_values = results.predict(start=start, end=end, dynamic=True)
    return predicted_values