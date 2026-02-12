import pandas as pd
df = pd.read_csv("Eco.csv")
print(df.head(10))#1 to print the desirable no. of rows from the top
print(df.tail(10))#2 from bottom
print(df.dtypes)#3 datatype of each columns (it's not a method)
print(df.isnull())#4 to check whether anything is null or not(Boolean value)
print(df.info())#5 gives total information
print(df["Purchase Price"].max())
print(df["Purchase Price"].min())
print(df[df["Language"]=="fr"].count())#6 to count anytype of info
print(df[df["Job"].str.contains("Scientist")].count()) #7 finds all the string which has Scientist as the only or a sub-word
print(df[df["IP Address"]=="132.207.160.22"]["Email"])
print((df["CC Exp Date"].apply(lambda x:x[3:]=="20")))