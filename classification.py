import pandas as pd                                 #to read csv file
from sklearn.preprocessing import LabelEncoder      #to encode the string , transform str to int_value
from sklearn.tree import DecisionTreeClassifier     #to use classifier 

#read csv file and store it in df variable
df = pd.read_csv("D:\dataclassification.csv")
#axis=col 
inputs = df.drop('more_4000', axis=1)
target = df['more_4000']
# add cols and encode it 
le = LabelEncoder()
inputs['names_n'] = le.fit_transform(inputs['names'])
inputs['prices_n'] = le.fit_transform(inputs['prices'])
inputs['review_n'] = le.fit_transform(inputs['reviews'])
# drop the cols that doesn't encoded 😁
inputs_n = inputs.drop(['names', 'prices', 'reviews'], axis=1)
print(inputs_n)
# uses classfier that have gini ,best,
model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)
model.fit(inputs_n, target)
model.score(inputs_n, target)
predicate = model.predict([[9, 0, 2]])
print(predicate)