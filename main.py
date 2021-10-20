import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import metrics
import statistics as stats

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

from sklearn.model_selection import train_test_split

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/home/hduser/Descargas/Id3703.csv", sep=";")
df1 = pd.read_csv("/home/hduser/Descargas/Id3704.csv", sep=";")
df2 = pd.read_csv("/home/hduser/Descargas/Id3705.csv", sep=";")
df3 = pd.read_csv("/home/hduser/Descargas/Id3706.csv", sep=";")
df4 = pd.read_csv("/home/hduser/Descargas/Id3492.csv", sep=";")
df5 = pd.read_csv("/home/hduser/Descargas/Id3491.csv", sep=";")

df6 = pd.read_csv("/home/hduser/Descargas/Id3491febrero.csv", sep=";")
df7 = pd.read_csv("/home/hduser/Descargas/Id3491marzo.csv", sep=";")
df9 = pd.read_csv("/home/hduser/Descargas/Id3491abrl.csv", sep=";")
df10 = pd.read_csv("/home/hduser/Descargas/Id3491mayo.csv", sep=";")
df11 = pd.read_csv("/home/hduser/Descargas/Id3491junio.csv", sep=";")

print(df1)

print()
print("Vmed de 3703: ", stats.mean(df.vmed))
print("Vmed de 3704: ", stats.mean(df1.vmed))
print("Vmed de 3705: ", stats.mean(df2.vmed))
print("Vmed de 3706: ", stats.mean(df3.vmed))
print("Vmed de 3492: ", stats.mean(df4.vmed))
print("Vmed de 3491: ", stats.mean(df5.vmed))
print()
print("Carga de 3703: ", stats.mean(df.carga))
print("Carga de 3704: ", stats.mean(df1.carga))
print("Carga de 3705: ", stats.mean(df2.carga))
print("Carga de 3706: ", stats.mean(df3.carga))
print("Carga de 3492: ", stats.mean(df4.carga))
print("Carga de 3491: ", stats.mean(df5.carga))
print("Carga de 3491 en febrero: ", stats.mean(df6.carga))
print("Carga de 3491 en marzo: ", stats.mean(df7.carga))
print("Carga de 3491 en abril: ", stats.mean(df9.carga))
print("Carga de 3491 en mayo: ", stats.mean(df10.carga))
print("Carga de 3491 en junio: ", stats.mean(df11.carga))

febrero = stats.mean(df6.carga)
marzo = stats.mean(df7.carga)
abril = stats.mean(df9.carga)
mayo = stats.mean(df10.carga)
junio = stats.mean(df11.carga)

meses = febrero, marzo, abril, mayo, junio
mes = "febrero", "marzo", "abril", "mayo", "junio"

plt.plot(mes, meses)
plt.xlabel('2020')
plt.ylabel('Carga')
plt.show()


print()
print("PRUEBA CON LINEAR REGRESSION")

X_multiple = df4[['vmed_1','vmed_2','vmed_3','vmed_4','vmed_5','vmed_6','vmed_7','vmed_8']]

y_multiple = df4['vmed']

X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.2)

lr_multiple = LinearRegression()

lr_multiple.fit(X_train, y_train)

y_pred_multiple = lr_multiple.predict(X_test)

lr_dif = pd.DataFrame({'Actual': y_test, 'PredecidoLinealRegression': y_pred_multiple})
print(lr_dif.head(25))

print("Pesos de los atributos")
print(lr_multiple.coef_)

print()

print("SCORE: ", lr_multiple.score(X_train,y_train))
print('Error cuadratico medio (MSE): ', np.sqrt(metrics.mean_squared_error(y_test, y_pred_multiple)))
print('Error absoluto medio (MAE): ', metrics.mean_absolute_error(y_test, y_pred_multiple))
print("R2",metrics.r2_score(y_test,y_pred_multiple))
#print("Mean absolute Percentaje Error", sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred_multiple))


print()
print("PRUEBA CON RANDOM FOREST")

regressor = RandomForestRegressor()

regressor.fit(X_train, y_train.values.ravel())

#values para los valores del array (shape(n,1))
#ravel para convertir el array shape (n,)

y_predR = regressor.predict(X_test)

rf_dif = pd.DataFrame({'Actual': y_test, 'PredecidoRF': y_predR})
print(rf_dif.head(25))

print()
print("SCORE: ", regressor.score(X_train,y_train))
print('Error cuadrático medio (MSE): ', np.sqrt(metrics.mean_squared_error(y_test, y_predR)))
print('Error abosluto medio (MAE): ', metrics.mean_absolute_error(y_test, y_predR))
print("R2",metrics.r2_score(y_test,y_predR))
#print("Mean absolute Percentaje Error", sklearn.metrics.mean_absolute_percentage_error(y_test, y_predR))

print()
print("PRUEBA CON SVR")

SVR = SVR()

SVR.fit(X_train, y_train.values.ravel())

y_predSVR = SVR.predict(X_test)

svr_dif = pd.DataFrame({'Actual': y_test, 'PredecidoSVR': y_predSVR})
print(svr_dif.head(25))

print()
print("SCORE: ", SVR.score(X_train,y_train))
print('Error cuadrático medio (MSE): ', np.sqrt(metrics.mean_squared_error(y_test, y_predSVR)))
print('Error absoluto medio (MAE): ', metrics.mean_absolute_error(y_test, y_predSVR))
print("R2",metrics.r2_score(y_test,y_predSVR))
#print("Mean absolute Percentaje Error", sklearn.metrics.mean_absolute_percentage_error(y_test, y_predSVR))


print()
print("PRUEBA CON GRADIENT BOOST REGRESSOR")

GBR = GradientBoostingRegressor()

GBR.fit(X_train, y_train)

y_predGBR = GBR.predict(X_test)

gbr_dif = pd.DataFrame({'Actual': y_test, 'PredecidoGBR': y_predGBR})
print(gbr_dif.head(25))

print()
print("SCORE: ", GBR.score(X_train,y_train))
print('Error cuadrático medio (MSE): ', np.sqrt(metrics.mean_squared_error(y_test, y_predGBR)))
print('Error absoluto medio (MAE): ', metrics.mean_absolute_error(y_test, y_predGBR))
print("R2",metrics.r2_score(y_test,y_predGBR))
#print("Mean absolute Percentaje Error", sklearn.metrics.mean_absolute_percentage_error(y_test, y_predGBR))

print()

B = df2[['vmed', 'vmed_1','vmed_2', 'vmed_3', 'vmed_4', 'vmed_5', 'vmed_6', 'vmed_7', 'vmed_8',]]


def tidy_corr_matrix(corr_mat):

    #Función para convertir una matriz de correlación de pandas en formato tidy

    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('variable_1', ascending=True)

    return(corr_mat)

corr_matrix = B.select_dtypes(include=['float64', 'int']) \
    .corr(method='pearson')
print(tidy_corr_matrix(corr_matrix).head(25))

print()


ax = sns.heatmap(
    corr_matrix,
    vmin=0, vmax=1, center=0.5,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#heat_map = sb.heatmap(corr_matrix, annot=True)
#plt.show()

