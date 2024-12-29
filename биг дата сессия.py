#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Келесі болып өзімнің деректерімді оқимын.
#Менің деректер жинағым студенттердің депрессиясына қатысты ақпараттарды қамтиды. 
#Бұл жинақта жас, академиялық қысым, қаржылық стресс, ұйқы ұзақтығы, оқу үлгерімі (CGPA) сияқты факторлар бар. 
#Мен осы деректерді талдау арқылы депрессия деңгейіне қандай факторлар әсер ететінін анықтағым келеді. 


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "aa.csv"
data = pd.read_csv(file_path)
# Бұл қадамда біз әрбір бағандағы жетіспейтін мәндерді анықтаймыз және олардың жалпы санын көрсетеміз.
# Деректер жайлы жалпы ақпарат
print("Размер данных:", data.shape)
print("\nПропущенные значения:\n", data.isnull().sum())
print("\nТипы данных:\n", data.dtypes)
display(data.head())


# In[2]:


# Проверка пропущенных значений
print("Пропущенные значения до обработки:\n", data.isnull().sum())

# Заполнение пропусков для числовых данных медианой
numeric_data = data.select_dtypes(include=['float64', 'int64'])
for column in numeric_data.columns:
    data[column].fillna(data[column].median(), inplace=True)

# Заполнение пропусков для категориальных данных самым частым значением
categorical_data = data.select_dtypes(include=['object'])
for column in categorical_data.columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

print("Пропущенные значения после обработки:\n", data.isnull().sum())


# In[3]:


# Удаление дубликатов
initial_shape = data.shape
data.drop_duplicates(inplace=True)
final_shape = data.shape

print(f"Количество удаленных дубликатов: {initial_shape[0] - final_shape[0]}")


# In[6]:


data['Age'] = data['Age'].astype(int)  # Приведение возраста к целому числу

def convert_sleep_duration(duration):
    if 'hour' in duration:
        if 'Less' in duration:
            return 4  # Меньше 5 часов
        elif 'More' in duration:
            return 9  # Больше 8 часов
        else:
            return int(duration.split('-')[0])  # Берем нижнюю границу диапазона
    return np.nan

data['Sleep Duration'] = data['Sleep Duration'].apply(convert_sleep_duration)


# In[7]:


# Преобразование бинарных категорий в числовой формат
binary_columns = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})


# In[12]:


data['Gender'] = data['Gender'].astype('category')
data['City'] = data['City'].astype('category')
data['Profession'] = data['Profession'].astype('category')
data['Dietary Habits'] = data['Dietary Habits'].astype('category')
data['Degree'] = data['Degree'].astype('category')


# In[13]:


print("\nОбновленные типы данных:\n", data.dtypes)


# In[8]:


numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Расчет количества строк и столбцов
n_cols = 5  # Максимальное количество графиков в строке
n_rows = int(np.ceil(len(numeric_columns) / n_cols))  # Вычисление необходимого количества строк

plt.figure(figsize=(15, 5 * n_rows))

for i, column in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(data[column])
    plt.title(column)

plt.tight_layout()
plt.show()


# In[9]:


# Функция для нахождения выбросов
def find_outliers(df):
    outliers = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    return outliers

# Проверка на выбросы
outliers = find_outliers(data)

# Выводим столбцы с выбросами
print("Столбцы с выбросами:")
for col, indices in outliers.items():
    if indices:
        print(f"{col}: индексы выбросов {indices}")

# Удаление выбросов
cleaned_data = data.drop(index=[index for indices in outliers.values() for index in indices])

# Выводим очищенные данные
print("\nОчищенные данные:")
print(cleaned_data.head())

# Визуализация данных до и после удаления выбросов
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График до очистки
sns.boxplot(data=data, ax=axes[0])
axes[0].set_title("До удаления выбросов")

# График после очистки
sns.boxplot(data=cleaned_data, ax=axes[1])
axes[1].set_title("После удаления выбросов")

plt.tight_layout()
plt.show()


# In[14]:


# 2. Исследовательский анализ данных (EDA)
# Основная статистика
print(data.describe(include='all'))


# In[15]:


# Описательная статистика для числовых признаков
descriptive_stats = data[numeric_columns].agg(['mean', 'median', lambda x: x.mode()[0], 'std']).rename(index={'<lambda_0>': 'mode'})
print("\nОписательная статистика для числовых признаков:")
print(descriptive_stats)


# In[59]:


# Построение гистограмм и графиков плотности распределения
numerical_columns = data.select_dtypes(include=[np.number]).columns
n_cols = 3  # Количество столбцов в сетке
n_rows = int(np.ceil(len(numerical_columns) / n_cols))  # Количество строк в сетке

plt.figure(figsize=(5 * n_cols, 5 * n_rows))  # Размер фигуры зависит от числа строк и столбцов

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)  # Динамическое определение сетки
    sns.histplot(data[col], kde=True, bins=30, color='blue', stat='density')
    plt.title(f'Гистограмма и плотность: {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')

plt.tight_layout()
plt.show()




# In[17]:


# Выделение категориальных столбцов
categorical_columns = data.select_dtypes(include=['category']).columns

# 1. Частотный анализ
for col in categorical_columns:
    print(f"\nЧастотный анализ для {col}:")
    print(data[col].value_counts())

# 2. Построение столбчатых диаграмм
plt.figure(figsize=(15, 10))

for i, col in enumerate(categorical_columns, 1):
    plt.subplot(len(categorical_columns), 1, i)  # Создаем подграфик для каждого столбца
    sns.countplot(data=data, x=col, palette='viridis', order=data[col].value_counts().index)
    plt.title(f'Распределение категорий: {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.xticks(rotation=45)  # Поворот меток, если они длинные

plt.tight_layout()
plt.show()


# In[47]:


# Анализ категориальных переменных с использованием groupby()
categorical_group = data.groupby('Gender')['Depression'].mean()
print("Средний уровень депрессии по полу:")
print(categorical_group)
# Анализ депрессии по полу
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Depression', data=data, palette='Set2')
plt.title('Средний уровень депрессии по полу')
plt.show()


# In[34]:


# Столбчатая диаграмма для переменной 'Gender'
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=data, palette='Set1')
plt.title('Частотная диаграмма для переменной "Gender"')
plt.show()

# Диаграмма распределения для переменной 'Gender'
gender_counts = data['Gender'].value_counts()
plt.figure(figsize=(8, 8))
gender_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set1', len(gender_counts)), startangle=90, legend=False)
plt.title('Распределение по полю "Gender"')
plt.ylabel('')
plt.show()




# In[19]:


# 1. Исключим категориальные переменные и оставим только числовые
numerical_columns = data.select_dtypes(include=['float64', 'int64','int32']).columns

# 2. Вычислим коэффициенты корреляции только для числовых переменных
correlation_matrix = data[numerical_columns].corr()

# Выводим корреляционную матрицу
print("\nКорреляция между числовыми признаками:")
print(correlation_matrix)

# 3. Построение тепловой карты для корреляции
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Тепловая карта корреляции')
plt.show()

# 4. Поиск мультиколлинеарности
# Рассмотрим корреляцию зависимой переменной (Depression) с независимыми переменными
depression_corr = correlation_matrix['Depression'].drop('Depression')  # Убираем саму депрессию из анализа

# Выводим только те переменные, которые сильно коррелируют с депрессией
print("\nКорреляция депрессии с другими признаками:")
print(depression_corr)

# Находим признаки с высокой корреляцией (порог можно выбрать, например, > 0.7 или < -0.7)
high_corr = depression_corr[depression_corr > 0.7]
print("\nПризнаки с высокой корреляцией с депрессией:")
print(high_corr)

# Найдем высокую корреляцию среди независимых переменных
# Для этого исключим зависимую переменную и ищем корреляции между независимыми переменными
independent_variables = correlation_matrix.drop('Depression', axis=1)  # Убираем зависимую переменную
high_corr_independent = independent_variables[abs(independent_variables) > 0.7]

print("\nВысокая корреляция между независимыми переменными:")
print(high_corr_independent)


# In[22]:


#вычисление VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = independent_data_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(independent_data_with_const.values, i) for i in range(independent_data_with_const.shape[1])]

#Выводим VIF
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

#Фильтруем переменные с высоким VIF (например, VIF > 5 или VIF > 10)
high_vif = vif_data[vif_data["VIF"] > 5]
print("\nПеременные с высоким VIF (мультиколлинеарность):")
print(high_vif)


# In[48]:


# Частотный анализ категориальных переменных
print(data['Gender'].value_counts())
print(data['City'].value_counts().head(10))  # Топ-10 городов


# In[45]:


# 1. Создание бинарного индикатора высокого академического давления
data['High Academic Pressure'] = (data['Academic Pressure'] > 3).astype(int)

# 2. Создание нового столбца "Общий стресс" как суммы академического и финансового давления
data['Total Stress'] = data['Academic Pressure'] + data['Financial Stress']

# 3. Преобразование категориальных данных (например, Gender) в числовой формат (label encoding)
data['Gender_Num'] = data['Gender'].astype('category').cat.codes

# Просмотр первых 5 строк с новыми столбцами
data[['Academic Pressure', 'Financial Stress', 'Total Stress', 'Gender', 'Gender_Num']].head()



# In[46]:


# 1. Фильтрация студентов с высоким уровнем стресса
high_stress_students = data[data['Total Stress'] > 6]
print(f"Число студентов с высоким уровнем стресса: {high_stress_students.shape[0]}")

# 2. Сортировка по уровню депрессии
sorted_data = data.sort_values(by='Depression', ascending=False)
print(sorted_data[['Depression', 'Academic Pressure', 'Financial Stress']].head())

# 3. Фильтрация студентов по категориям
students_with_suicidal_thoughts = data[data['Have you ever had suicidal thoughts ?'] == 1]
print(f"Число студентов с суицидальными мыслями: {students_with_suicidal_thoughts.shape[0]}")



# In[55]:


# Риск-фактор депрессии
data['Depression_Risk'] = (data['Academic Pressure'] * 0.4 + 
                           data['Financial Stress'] * 0.3 + 
                           data['Have you ever had suicidal thoughts ?'] * 0.3)
print(data[['Academic Pressure', 'Financial Stress', 'Have you ever had suicidal thoughts ?', 'Depression_Risk']].head())


# In[56]:


# Определим пользовательскую функцию для вычисления уровня стресса
def stress_level(row):
    if row['Academic Pressure'] > 4 and row['Financial Stress'] > 4:
        return 'High Stress'
    elif row['Academic Pressure'] > 3 or row['Financial Stress'] > 3:
        return 'Moderate Stress'
    else:
        return 'Low Stress'

# Применяем эту функцию к данным
data['Stress Level'] = data.apply(stress_level, axis=1)

# Выводим первые несколько строк
print(data[['Academic Pressure', 'Financial Stress', 'Stress Level']].head())


# In[58]:


city_grouped = data.groupby('Gender')['Academic Pressure'].mean()
print(city_grouped)


# In[57]:


# Создание сводной таблицы
pivot_table = data.pivot_table(values='Academic Pressure', index='City', columns='Gender', aggfunc='mean')

# Печатаем сводную таблицу
print(pivot_table)


# In[ ]:





# In[ ]:




