import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

pd.set_option("display.max_columns",None)

data = pd.read_csv("House.csv")

encoder = LabelEncoder()
print(data.shape)
categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])
    
x = data.drop("price",axis=1)
y = data["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size = 0.2,
    random_state = 50
)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

score = r2_score(y_test, predictions)

print("\nR^2 Score: ", score)
#print(predictions[:5])
print("Model Trained Successfully!")


print("\nX_train: ", x_train.shape)
print("X_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
print()
#for i in range(5):
    #print("Actual: ", y_test.iloc[i], "Predicted: ",predictions[i])

yes_no = {
    "yes":1,
    "no":0
}
status = {
    "furnished":0,
    "semi-furnished":1,
    "unfurnished":2
}
area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
stories = int(input("Enter Stories: "))
mainroad = input("Main Road (yes/no): ").lower()
mainroad = yes_no[mainroad]
guestroom = input("Guestroom (yes/no): ").lower()
guestroom = yes_no[guestroom]
basement = input("Basement (yes/no): ").lower()
basement = yes_no[basement]
hotwaterheating = input("HotWaterHeating (yes/no): ").lower()
hotwaterheating = yes_no[hotwaterheating]
airconditioning = input("AirConditioning (yes/no): ").lower()
airconditioning = yes_no[airconditioning]
parking = int(input("Enter Parking: "))
prefarea = input("PrefArea (yes/no): ").lower()
prefarea = yes_no[prefarea]
furnishingstatus = input("FurnishingStatus (furnished/semi-furnished/unfurnished): ").lower()
furnishingstatus = status[furnishingstatus]

new_house = [[
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus
]]

predicted_price = model.predict(new_house)
print(predicted_price)
print(f"\nPredicted House Price: ₹{predicted_price[0]:,.2f}")