import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

filename = 'lab2_2_DS.csv'

df = pd.read_csv(filename)

if len(df.columns) >= 2:
    col_x = df.columns[0]
    col_y = df.columns[1]

    if len(df.columns) > 2:
        col_x = input(f'столбец для X (по умолчанию {df.columns[0]}): ') or df.columns[0]
        col_y = input(f'столбец для Y (по умолчанию {df.columns[1]}): ') or df.columns[1]
else:
    print('В файле должно быть минимум 2 столбца')
    exit()

x = df[[col_x]].values
y = df[col_y].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

results = []
models_predictions = {}

degrees = [1, 2, 3, 4]

for d in degrees:
    if d == 1:
        model_name = f'Полином степени {d} (Линейная регрессия)'
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        models_predictions[model_name] = {'model': model, 
                                          'poly': None, 
                                          'color': 'black', 
                                          'linestyle': '--'}
    else:
        model_name = f'Полином степени {d}'
        poly = PolynomialFeatures(degree=d)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        
        model = LinearRegression()
        model.fit(x_train_poly, y_train)
        y_pred = model.predict(x_test_poly)
        
        colors = {2: 'blue', 3: 'green', 4: 'purple'}
        models_predictions[model_name] = {'model': model, 
                                          'poly': poly, 
                                          'color': colors.get(d, 'red'), 
                                          'linestyle': '-'}

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results.append({'Модель': model_name,
                    'MSE': mse,
                    'R2': r2,
                    'MAPE': mape})


df_results = pd.DataFrame(results)
print(df_results.to_string(index=False, float_format=lambda x: '%.4f' % x))


best_r2_row = df_results.loc[df_results['R2'].idxmax()]
best_mse_row = df_results.loc[df_results['MSE'].idxmin()]
best_mape_row = df_results.loc[df_results['MAPE'].idxmin()]

print(f'Лучшая модель по метрике R2: {best_r2_row['Модель']}')
print(f'Лучшая модель по метрике MSE: {best_mse_row['Модель']}')
print(f'Лучшая модель по метрике MAPE: {best_mape_row['Модель']}')

plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='gray', label='Тестовые данные', alpha=0.8)


sort_idx = x_test.squeeze().argsort()
x_sorted = x_test[sort_idx]

for name, info in models_predictions.items():
    if info['poly'] is None:
        y_line = info['model'].predict(x_sorted)
    else:
        x_sorted_poly = info['poly'].transform(x_sorted)
        y_line = info['model'].predict(x_sorted_poly)
        
    plt.plot(
        x_sorted, y_line, 
        color=info['color'], 
        linestyle=info['linestyle'], 
        linewidth=2, 
        label=f'{name} (R2={r2_score(y_test, info['model'].predict(info['poly'].transform(x_test) if info['poly'] else x_test)):.2f})')

plt.xlabel(col_x)
plt.ylabel(col_y)
plt.title('Сравнение моделей полиномиальной и линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()