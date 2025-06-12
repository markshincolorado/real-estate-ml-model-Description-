from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
    'sqft': [1400, 1600, 1800, 2000, 2200],
    'bedrooms': [3, 3, 4, 4, 5],
    'price': [350000, 370000, 400000, 420000, 450000]
}
df = pd.DataFrame(data)

X = df[['sqft', 'bedrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print("Model Coefficients:", model.coef_)
