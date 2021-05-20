import pandas as pd 

x = pd.read_csv('c:/Users/JQZhang/Desktop/MCTS-IDL/examples/ionosphere.csv', encoding='utf-8', header=0)

col = x.columns.tolist()
values = set(x[col[-1]].tolist())
count = 0
for val in values:
    x.loc[x[col[-1]]==val, col[-1]] = count 
    count = count + 1
y = x.pop(col[-1])
y = pd.to_numeric(y)
#y = y.values
label = col[-1]
del col[-1]
str_attr = []
for c in col:
    if x[c].dtype == 'object':
        str_attr.append(c)
x = pd.get_dummies(x, prefix=str_attr)
#x = x.values 
x[label] = y
x.to_csv('c:/Users/JQZhang/Desktop/MCTS-IDL/examples/ionosphere1.csv')