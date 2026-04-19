
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel

df1=pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')
df3=pd.merge(df1,df2,how='outer')
#print(df3)
#print(df3.info())
#print(df3.describe())
#print(df3.isnull().sum())

#filling missing age values with the median of age
df3['Age'].fillna(df3['Age'].median(),inplace=True)
#filling missing Embarked values with first mode
df3['Embarked'].fillna(df3['Embarked'].mode()[0],inplace=True)
#Filling missing fare value with median of fare
df3['Fare'].fillna(df3['Fare'].median(),inplace=True)
#print(df3.isnull().sum())


#Dropping duplicates
df3.drop_duplicates(inplace=True)
#print(df3)

#Converting data types
df3['PassengerId'] = df3['PassengerId'].astype(str)
df3['Survived']=df3['Survived'].astype(bool)


#Displaying survival rate by genders as a group based insights by using mean
(df3.groupby('Sex')['Survived'].mean())

#dispalying survival rate visually
colors=['Blue','Pink']
sns.barplot(x='Sex',y='Survived',data=df3,palette=colors)
plt.title('Survival Rate by Gender')
xlabel('Gender')
ylabel('Survival rate')
plt.show()

#Displaying survival by passenger class
colors2=['Gold','Red','Grey']
#Assigned first class with gold, second class with red, third class grey
sns.barplot(x='Pclass',y='Survived',data=df3,palette=colors2)
plt.title('Survival Rate by Passenger Class')
xlabel('Passenger class')
ylabel('Survival rate')
plt.show()

#To view the distribution of each insight
df3.hist()
plt.show()


















