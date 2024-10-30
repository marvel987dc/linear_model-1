import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#path
path = r"C:\Users\barre\Documents\Semester 3 (Ongoing)\Introduction to AI\JuanBarrero_COMP237assignment_3\Exercise#2_JuanBarrero\Data"
filename = 'BikeSharingData.csv'
fullpath = os.path.join(path, filename)

df_JuanBarrero = pd.read_csv(fullpath)

print("First 5 records:")
print(df_JuanBarrero.head())

print("Column names")
print(df_JuanBarrero.columns)

print("Shape of the data")
print(df_JuanBarrero.shape)

print("Types of data")
print(df_JuanBarrero.dtypes)

missing_values = df_JuanBarrero.isnull().sum()

if missing_values.sum() == 0:
    print("No missing values")
else:
    print("Missing values per column: ")
    print(missing_values)

list_of_dummies = pd.get_dummies(df_JuanBarrero, drop_first=True)
df_JuanBarrero = pd.get_dummies(df_JuanBarrero, drop_first=True)

print("Category to numerical values: ")
print(df_JuanBarrero.head())

categorical_columns = df_JuanBarrero.select_dtypes(include=['object']).columns

df_JuanBarrero = df_JuanBarrero.drop(categorical_columns, axis=1)
df_JuanBarrero = df_JuanBarrero.drop(columns=['instant'], axis=1)

print("Columns after dropping: ")
print(df_JuanBarrero.columns)

df = df_JuanBarrero

def normalize_dataframe(data):
  #Normalize the dataframe
    newDf = data.values

    print('Type: ', type(newDf))

    df_JuanBarrero_normalized = (newDf - newDf.min() ) / (newDf.max() - newDf.min())
    print("Normalized DataFrame:")

    newNormalized = pd.DataFrame(df_JuanBarrero_normalized, columns = df.columns)
    print(newNormalized.head())

    plt.figure(figsize=(9, 10))
    plt.boxplot(newNormalized["weekday"], vert=False, patch_artist=True)
    plt.show()
normalize_dataframe(df)

#Draw the scatter matrix

scatter_matrix_vars = ['temp', 'atemp','hum','windspeed','cnt']

data = pd.DataFrame(df, columns = scatter_matrix_vars)

print(data)
pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(13, 15), diagonal='kde')
plt.show()

#Build a model

# df_JuanBarrero, df_test = train_test_split(df, test_size=0.2, random_state=100)
# print(df_JuanBarrero.shape, df_test.shape)
newList = list_of_dummies.drop(columns=['instant'], axis=1)
print(newList.columns.tolist())

x = df[['temp', 'atemp', 'hum'] + newList.columns.tolist()]
y = df_JuanBarrero['cnt']

set_seed = 53

x_train_juan, x_test_juan, y_train_juan, y_test_juan = train_test_split(x, y, test_size=0.2, random_state=set_seed)

reg = LinearRegression().fit(x_train_juan, y_train_juan)

coefficients = pd.DataFrame(reg.coef_, x.columns, columns=['Coefficient'])
print('Coefficients: \n', coefficients)
print("Score: ", reg.score(x_train_juan, y_train_juan))


newX = df[['temp', 'atemp', 'hum', 'windspeed'] + newList.columns.tolist()]
newY = df_JuanBarrero['cnt']
newX_train_juan, newX_test_juan, newY_train_juan, newY_test_juan = train_test_split(newX, newY, test_size=0.2, random_state=set_seed)
reg_with_windspeed = LinearRegression().fit(newX_train_juan, newY_train_juan)

coefficients_with_windspeed = pd.DataFrame(reg_with_windspeed.coef_, newX.columns, columns=['Coefficient'])
print('Coefficients with windspeed: \n', coefficients_with_windspeed)
print("Score with windspeed: ", reg_with_windspeed.score(newX_train_juan, newY_train_juan))

print("Test Score (without windspeed): \n", reg.score(x_test_juan, y_test_juan))
print("Test Score (with windspeed): \n", reg_with_windspeed.score(newX_test_juan, newY_test_juan))



