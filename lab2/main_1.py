import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split


filename = 'lab2_1_DS.csv'

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

models = {'Простая линейная регрессия': LinearRegression(),
          'Ridge регрессия': Ridge(alpha=1.0),
          'LASSO регрессия': Lasso(alpha=0.1),
          'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)}
results = []


for name, model in models.items():
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  mape = mean_absolute_percentage_error(y_test, y_pred)

  results.append({'Модель': name,
                  'MAE': mae,
                  'R2': r2,
                  'MAPE': mape,
                  'Intercept': model.intercept_,
                  'Coefficient': model.coef_[0]})


df_results = pd.DataFrame(results)
print(df_results[['Модель', 'MAE', 'R2', 'MAPE']].to_string(index=False, float_format=lambda x: '%.4f' % x))

best_r2_row = df_results.loc[df_results['R2'].idxmax()]
best_mse_row = df_results.loc[df_results['MAE'].idxmin()]
best_mape_row = df_results.loc[df_results['MAPE'].idxmin()]

print(f'Лучшая модель по метрике R2: {best_r2_row['Модель']}')
print(f'Лучшая модель по метрике MSE: {best_mse_row['Модель']}')
print(f'Лучшая модель по метрике MAPE: {best_mape_row['Модель']}')

plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='black', label='Тестовые данные', alpha=0.7)


sort_idx = x_test.squeeze().argsort()
x_sorted = x_test[sort_idx]

colors = {'Простая линейная регрессия': 'red',
          'Ridge регрессия': 'blue',
          'LASSO регрессия': 'green',
          'Elastic Net': 'purple'}

for name, model in models.items():
  y_line = model.predict(x_sorted)
  plt.plot(x_sorted, y_line, color=colors[name], linewidth=2, 
           label=f'{name} (R2={r2_score(y_test, model.predict(x_test)):.2f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение моделей линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()




