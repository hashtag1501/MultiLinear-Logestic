import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TOEFL_train,TOEFL_test, result_train,result_test = train_test_split(result,TOEFL,test_size = 0.25,random_state = 0)

df = pd.read_csv("Admission_Predict.csv")

TOEFL = df["TOEFL Score"].tolist()
result = df["GRE Score"].tolist()

sc_x = StandardScaler()
TOEFL_train = sc_x.fit_transform(TOEFL_train)
TOEFL_test = sc_x.fit_transform(TOEFL_test)


colors = []
for data in result:
  if data==1:
    colors.append("green")
  else:
    colors.append("red")

fig = go.Figure(data = go.Scatter(
    x = TOEFL , 
    y = result, 
    mode = 'markers',
    marker = dict(color = colors)
 
))

fig.show()




