# Импорт библиотек
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
df = pd.read_csv("train.csv")
print("Первые строки данных:")
print(df.head())

# Количество пропущенных значений
print("Пропущенные значения:")
print(df.isnull().sum())

# Заполнение пропущенных значений
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.isnull().sum())

# Нормализация данных
scaler = MinMaxScaler()
df["Age"] = scaler.fit_transform(df[["Age"]])

# Преобразование категориальных данных
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Сохранение обработанных данных
df.to_csv("processed_titanic.csv", index=False)
print("Обработанные данные сохранены в processed_titanic.csv")