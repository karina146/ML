import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = 'lab1_DS.csv'
if not os.path.exists(filename):
    print(f'Файл {filename} не найден')
    exit()

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

x = df[col_x].values
y = df[col_y].values

print('\nСтатистическая информация')
for col in [col_x, col_y]:
    print(f'Столбец {col}:')
    print(f'  Количество: {df[col].count()}')
    print(f'  Минимум:  {df[col].min()}')
    print(f'  Максимум: {df[col].max()}')
    print(f'  Среднее: {df[col].mean():.4f}')

n = len(x)
mean_x = np.mean(x)
mean_y = np.mean(y)

a = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
b = mean_y - a * mean_x

print(f'\nПараметры регрессионной прямой (МНК): y = {a:.4f}*x + {b:.4f}')

y_pred = a * x + b


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# График 1
axes[0].scatter(x, y, color='blue', label='Исходные точки')
axes[0].set_title('1. Исходные точки')
axes[0].set_xlabel(col_x)
axes[0].set_ylabel(col_y)
axes[0].grid(True)
axes[0].legend()

# График 2
axes[1].scatter(x, y, color='blue', label='Исходные точки')
axes[1].plot(x, y_pred, color='red', linewidth=2, label=f'МНК: y = {a:.2f}x + {b:.2f}')
axes[1].set_title('2. Регрессионная прямая')
axes[1].set_xlabel(col_x)
axes[1].set_ylabel(col_y)
axes[1].grid(True)
axes[1].legend()

# График 3
axes[2].scatter(x, y, color='blue', label='Исходные точки', zorder=5)
axes[2].plot(x, y_pred, color='red', linewidth=2, label='МНК прямая', zorder=4)

for i in range(n):
    x_i = x[i]
    y_i = y[i]
    y_hat = y_pred[i]
    err = y_i - y_hat
    
    # немного меняю масштаб оси, что график выглядел поприличнее
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    scale_factor = x_range/y_range if y_range != 0 else 1
    
    wigth_in_x = abs(err)*scale_factor
    x_start = x_i if err >= 0 else x_i - wigth_in_x
    
    square_x = [x_i, x_i + wigth_in_x, x_i + wigth_in_x, x_i]
    square_y = [y_i, y_i, y_hat, y_hat]
    
    axes[2].fill(square_x, square_y, color='orange', alpha=0.3, hatch='//', edgecolor='orange')

axes[2].fill([], [], color='orange', alpha=0.3, hatch='//', label='Квадраты ошибок')
axes[2].set_title('3. Квадраты ошибок')
axes[2].set_xlabel(col_x)
axes[2].set_ylabel(col_y)
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
