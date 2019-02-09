import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
plt.style.use('seaborn-whitegrid')

# Обозначаем координаты города и его центра
box = (-75, -72.9, 40, 41.8)
city_center = (-74.0063889, 40.7141667)
# Функиця для определения, находится ли точка в городе
def select_within_box(df, box):
    return (df.pickup_longitude >= box[0]) & (df.pickup_longitude <= box[1]) & \
           (df.pickup_latitude >= box[2]) & (df.pickup_latitude <= box[3]) & \
           (df.dropoff_longitude >= box[0]) & (df.dropoff_longitude <= box[1]) & \
           (df.dropoff_latitude >= box[2]) & (df.dropoff_latitude <= box[3])
            
# Функция с https://stackoverflow.com/questions/27928/
# На вход принимает координаты точек, возвращает расстояние между ними в милях
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

# В качестве модели выбираем xgboost, опытным путем определили, что при 1000 деревьев получается хороший результат
model = XGBRegressor(n_estimators=1000)

# Считываем тренировочные и тестовые данные в датафрейм pandas
# Тренировочные данные не пометсятся в память целиком - nrows - количество строк, которые читаем; chunksize - количество строк, с которыми работаем одновременно
df_test =  pd.read_csv('test.csv', parse_dates=["pickup_datetime"])
df_train =  pd.read_csv('train.csv', nrows=10000000, chunksize=50000, parse_dates=["pickup_datetime"])

# Единовременно работаем с одним "чанком" (в нашем случае 50000)
for chunk in df_train:
    # Добавляем параметр "расстояние в милях"
    chunk['distance_miles'] = distance(chunk.pickup_latitude, chunk.pickup_longitude, \
                                        chunk.dropoff_latitude, chunk.dropoff_longitude)
    # Добавляем расстояние от точки отправления до центра города
    chunk['distance_to_center'] = distance(city_center[1], city_center[0], chunk.pickup_latitude, chunk.pickup_longitude)
    # Добавляем прарметры "Год" и "Час"
    chunk['year'] = chunk.pickup_datetime.apply(lambda t: t.year)
    chunk['hour'] = chunk.pickup_datetime.apply(lambda t: t.hour)
    # Добавляем параметр "День недели" (0-понедельник...6-воскресенье)
    chunk['weekday'] = chunk.pickup_datetime.apply(lambda t: t.weekday())

    # Убираем из данных поездки с отрицательной и нулевой стоимостью
    chunk = chunk[chunk.fare_amount>0]
    # Убираем строки с отсутствующими данными
    chunk = chunk.dropna(how = 'any', axis = 'rows')
    # Убираем из данных поездки без пассажиров
    chunk = chunk[chunk.passenger_count > 0]
    # Оставляем только поездки внутри города
    chunk = chunk[select_within_box(chunk, box)]
    # Убираем поездки с нулевым расстоянием
    chunk = chunk[chunk.distance_miles > 0]

    # Выбираем цель и параметры для обучения
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                'passenger_count', 'distance_miles', 'distance_to_center', 'year', 'weekday', 'hour']
    X = chunk[features].values
    y = chunk['fare_amount'].values

    # Обучаем модель
    model.fit(X, y, eval_metric="rmse")

# Все те же столбцы добавляем в тестовые данные
df_test['distance_miles'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                                     df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['distance_to_center'] = distance(city_center[1], city_center[0], df_test.pickup_latitude, df_test.pickup_longitude)
df_test['year'] = df_test.pickup_datetime.apply(lambda t: t.year)
df_test['hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)
df_test['weekday'] = df_test.pickup_datetime.apply(lambda t: t.weekday())

# Переводим тестовые данные в массив np
XTEST = df_test[features].values

# Предсказываем значения на тестовых данных
y_pred_final = model.predict(XTEST)

# Генерируем csv-файл
submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)