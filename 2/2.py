# pip install ucimlrepo pandas numpy scikit-learn matplotlib torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------

dataset = fetch_ucirepo(id=849)

features = dataset.data.features.copy()
targets = dataset.data.targets.copy()

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
# 2. Переход к почасовым данным
# ---------------------------------------------------------

hourly = df["Power"].resample("1h").mean().dropna()

print("Количество почасовых наблюдений:", len(hourly))
print("\nПервые значения:")
print(hourly.head())


# ---------------------------------------------------------
# 3. Формирование окон по 24 часа
# ---------------------------------------------------------

WINDOW = 24


def create_windows(series, window):
    X = []
    y = []

    values = series.to_numpy()

    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])

    return np.array(X), np.array(y)


X, y = create_windows(hourly, WINDOW)

dates = hourly.index[WINDOW:]


# ---------------------------------------------------------
# 4. Разделение данных на обучение и тест
# ---------------------------------------------------------

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

dates_test = dates[split:]

print("\nОбучающих примеров:", len(X_train))
print("Тестовых примеров:", len(X_test))


# ---------------------------------------------------------
# 5. Масштабирование
# ---------------------------------------------------------

scaler = StandardScaler()

# Масштабирование строится только по обучающим данным
scaler.fit(X_train.reshape(-1, 1))

X_train_scaled = scaler.transform(
    X_train.reshape(-1, 1)
).reshape(X_train.shape)

X_test_scaled = scaler.transform(
    X_test.reshape(-1, 1)
).reshape(X_test.shape)

y_train_scaled = scaler.transform(
    y_train.reshape(-1, 1)
).ravel()


# ---------------------------------------------------------
# 6. Преобразование в tensor
# ---------------------------------------------------------

X_train_tensor = torch.tensor(
    X_train_scaled[:, :, None],
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled[:, :, None],
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False
)


# ---------------------------------------------------------
# 7. Модель transformer
# ---------------------------------------------------------

class TimeSeriesTransformer(nn.Module):

    def __init__(
        self,
        seq_length=24,
        d_model=32,
        nhead=4,
        num_layers=2
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            1,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                seq_length,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        # Используем последний элемент последовательности
        x = x[:, -1, :]

        return self.output_layer(x).squeeze(-1)


model = TimeSeriesTransformer()


# ---------------------------------------------------------
# 8. Обучение
# ---------------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 50

loss_history = []

print("\nНачало обучения\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        prediction_batch = model(X_batch)

        loss = criterion(
            prediction_batch,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {avg_loss:.6f}"
        )


# ---------------------------------------------------------
# 9. Прогноз на тестовой выборке
# ---------------------------------------------------------

model.eval()

with torch.no_grad():

    prediction_scaled = model(
        X_test_tensor
    ).numpy()


prediction = scaler.inverse_transform(
    prediction_scaled.reshape(-1, 1)
).ravel()


# ---------------------------------------------------------
# 10. Метрики
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

print("\nРезультаты Transformer")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")


# ---------------------------------------------------------
# 11. Прогноз будущих значений
# ---------------------------------------------------------

# При окне 24 часа самый ранний прогноз можно сделать
# только начиная с 02.01.2017 00:00

forecast_dates = pd.to_datetime([
    "2017-01-02 00:00:00",
    "2017-01-02 01:00:00",
    "2017-01-02 02:00:00"
])

rows = []

for forecast_date in forecast_dates:

    # Находим положение прогнозируемого часа
    position = hourly.index.get_loc(forecast_date)

    # Берем предыдущие 24 часа
    window = hourly.iloc[
        position - WINDOW:position
    ].values

    # Масштабирование
    window_scaled = scaler.transform(
        window.reshape(-1, 1)
    ).reshape(1, WINDOW, 1)

    window_tensor = torch.tensor(
        window_scaled,
        dtype=torch.float32
    )

    # Прогноз
    model.eval()

    with torch.no_grad():
        pred_scaled = model(window_tensor).item()

    # Возврат к исходным значениям
    pred = scaler.inverse_transform(
        np.array([[pred_scaled]])
    )[0, 0]

    actual = hourly.loc[forecast_date]

    error = actual - pred

    rows.append({
        "Дата и время": forecast_date,
        "Фактическое значение": actual,
        "Прогноз Transformer": pred,
        "Ошибка": error
    })


result_3 = pd.DataFrame(rows)

print("\nТри наглядных прогноза:\n")

print(
    result_3.to_string(
        index=False,
        formatters={
            "Фактическое значение": "{:.2f}".format,
            "Прогноз Transformer": "{:.2f}".format,
            "Ошибка": "{:.2f}".format
        }
    )
)


# ---------------------------------------------------------
# 12. График трёх прогнозов
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    result_3["Дата и время"],
    result_3["Фактическое значение"],
    marker="o",
    linewidth=2,
    label="Фактическое значение"
)

plt.plot(
    result_3["Дата и время"],
    result_3["Прогноз Transformer"],
    marker="o",
    linewidth=2,
    label="Прогноз Transformer"
)

plt.title(
    "Прогноз энергопотребления на три последовательных часа"
)

plt.xlabel("Дата и время")
plt.ylabel("Потребление")

plt.legend()
plt.grid(alpha=0.3)

plt.xticks(
    result_3["Дата и время"],
    [
        "02.01 00:00",
        "02.01 01:00",
        "02.01 02:00"
    ]
)

plt.tight_layout()

plt.savefig(
    "transformer_3_predictions.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# 13. Общий график факт/прогноз
# ---------------------------------------------------------

N = 24 * 7

plt.figure(figsize=(13, 5))

plt.plot(
    dates_test[:N],
    y_test[:N],
    label="Фактическое значение"
)

plt.plot(
    dates_test[:N],
    prediction[:N],
    label="Прогноз Transformer"
)

plt.title(
    "Фактическое и прогнозируемое энергопотребление"
)

plt.xlabel("Дата и время")
plt.ylabel("Потребление")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "transformer_forecast.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# 14. График обучения
# ---------------------------------------------------------

plt.figure(figsize=(8, 4))

plt.plot(
    range(1, EPOCHS + 1),
    loss_history
)

plt.title(
    "Изменение ошибки Transformer при обучении"
)

plt.xlabel("Эпоха")
plt.ylabel("MSE")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "transformer_training.png",
    dpi=200
)

plt.show()