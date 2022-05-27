import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import time

# import pickle

warnings.filterwarnings("ignore")

filename = '../data/arima_data.xls'
# forrecastnum = 5
data = pd.read_excel(filename, index_col=u'日期')
print(len(data))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# data.plot()
# plt.title('Time Series')
# plt.show()
# plot_acf(data)
# plt.show()
# print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
D_data = data.diff(periods=1).dropna()
# D_data.columns = [u'销量差分']
# D_data.plot()
# plt.show()
# plot_acf(D_data).show()
# plot_pacf(D_data).show()
# print(u'1阶差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))
#
# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))
#
# data[u'销量'] = data[u'销量'].astype(float)
pmax = min(int(len(D_data) / 10), 20)
qmax = min(int(len(D_data) / 10), 20)
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            print("=" * 40)
            print(f"get p:{p} and n: {q}")
            model = ARIMA(data, (p, 1, q)).fit()
            save_path = f"../data/model/0511arima{p}_{q}.pkl"
            model.save(save_path)
            # with open(save_path, 'wb') as f:
            #     pickle.dumps(model, f)
            tmp.append(model.bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix = pd.DataFrame(bic_matrix)
bic_matrix.to_csv('../data/0511_arima_data.csv')
print(bic_matrix)
p, q = bic_matrix.stack().idxmin()
print(u'bic最小的P值和q值为：%s、%s' % (p, q))
# for _ in range(20):
start_time = time.time()
model = ARIMA(data, (p, 1, q)).fit()
model.summary2()
end_time = time.time()
cost = end_time - start_time
print(f"cost time:{cost} ms")
forecast = model.forecast(7)  # 预测未来5天的销售额
print("forecast result: \n")
print("\n".join([str(i) for i in forecast[0]]))
