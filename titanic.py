import pandas as pd
df=pd.read_csv("train.csv")
s = (df["Age"].median(numeric_only=True))
print("Median Age : ",s)
#1
df = df.fillna({"Age":s})
#print(df.to_string())
#2
survived = (df["Survived"].sum(numeric_only=True))
total = 891

total_survivalrate = (survived/total)*100
print("Total Survival Rate : ",total_survivalrate)

#3 class wise
class_ = df.groupby("class")
print(class_["Survived"].sum())
class1 = 136
class3 = 119

survival_rate1 = (class1/total)*100
survival_rate3 = (class3/total)*100
print("Class 1:",survival_rate1)
print("Class 3:",survival_rate3)
print("Survival rate of Class1 vs Class3 = ",survival_rate1/survival_rate3)


#4 survival by gender
gender_survial = df.groupby("Sex")
print(gender_survial["Survived"].sum())
