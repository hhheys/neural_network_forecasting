# Установка библиотек:
# pip install ucimlrepo pandas numpy scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. Загрузка реального набора данных UCI
# ---------------------------------------------------------

dataset = fetch_ucirepo(id=849)

features = dataset.data.features.copy()
targets = dataset.data.targets.copy()

# Объединяем дату и целевой показатель первой зоны
df = pd.DataFrame({
    "DateTime": features["DateTime"],
    "Power": targets["Zone 1 Power Consumption"]
})

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    format="%m/%d/%Y %H:%M"
)

df = df.sort_values("DateTime")
df = df.set_index("DateTime")


# ---------------------------------------------------------
# 2. Переход от 10-минутных данных к почасовым
# ---------------------------------------------------------

# В исходном наборе наблюдения записаны каждые 10 минут.
# Для примера рассчитываем среднее значение за каждый час.

hourly = df["Power"].resample("1h").mean().dropna()

print("Количество почасовых наблюдений:", len(hourly))
print(hourly.head())


# ---------------------------------------------------------
# 3. Формирование скользящего окна
# ---------------------------------------------------------

WINDOW = 24  # предыдущие 24 часа


def create_windows(series, window):
    X = []
    y = []

    values = series.to_numpy()

    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i])

    return np.array(X), np.array(y)


X, y = create_windows(hourly, WINDOW)

# Даты, которым соответствуют прогнозируемые значения
dates = hourly.index[WINDOW:]


# ---------------------------------------------------------
# 4. Разделение на обучающую и тестовую выборки
# ---------------------------------------------------------

# Первые 80 % наблюдений — обучение,
# последние 20 % — проверка.

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

dates_test = dates[split:]

print("Обучающих примеров:", len(X_train))
print("Тестовых примеров:", len(X_test))


# ---------------------------------------------------------
# 5. Масштабирование
# ---------------------------------------------------------

# Параметры масштабирования вычисляются только
# по обучающей выборке.

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(
    y_train.reshape(-1, 1)
).ravel()


# ---------------------------------------------------------
# 6. Создание и обучение многослойного перцептрона
# ---------------------------------------------------------

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True
)

model.fit(X_train_scaled, y_train_scaled)


# ---------------------------------------------------------
# 7. Прогноз
# ---------------------------------------------------------

prediction_scaled = model.predict(X_test_scaled)

prediction = scaler_y.inverse_transform(
    prediction_scaled.reshape(-1, 1)
).ravel()


# ---------------------------------------------------------
# 8. Оценка качества
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))

print()
print("Результаты MLP")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")


# ---------------------------------------------------------
# 9. График №1. Исходный ряд
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

# Для читаемости показываем первые две недели
hours_to_show = 24 * 14

plt.plot(
    hourly.index[:hours_to_show],
    hourly.iloc[:hours_to_show]
)

plt.title("Почасовое потребление электроэнергии")
plt.xlabel("Дата и время")
plt.ylabel("Потребление")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("01_hourly_power.png", dpi=200)
plt.show()


# ---------------------------------------------------------
# 10. График №2. Факт и прогноз MLP
# ---------------------------------------------------------

# Показываем первую неделю тестового периода
n = 24 * 7

plt.figure(figsize=(12, 5))

plt.plot(
    dates_test[:n],
    y_test[:n],
    label="Фактическое значение"
)

plt.plot(
    dates_test[:n],
    prediction[:n],
    label="Прогноз MLP"
)

plt.title("Фактическое и прогнозируемое энергопотребление")
plt.xlabel("Дата и время")
plt.ylabel("Потребление")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("02_mlp_forecast.png", dpi=200)
plt.show()


# ---------------------------------------------------------
# 11. График №3. Ошибка прогноза
# ---------------------------------------------------------

errors = y_test - prediction

plt.figure(figsize=(12, 4))

plt.plot(
    dates_test[:n],
    errors[:n]
)

plt.axhline(0, linewidth=1)

plt.title("Ошибка прогноза MLP")
plt.xlabel("Дата и время")
plt.ylabel("Факт − прогноз")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("03_mlp_errors.png", dpi=200)
plt.show()