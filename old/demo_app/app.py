import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


### Config
st.set_page_config(
    page_title="Welcome to Beander",
    page_icon=" ☕",
    layout="wide"
)

DATA = ('C:/Users/romai/Documents/Data_Science_Courses/Fullstack/Jedha_Fullstack_Data_Science/Project/Final/coffee/Coffee_dataset_cleaned_column_and_rows.csv')

### App
st.title('Welcome to Beander ☕')
st.header('Beander, the coffee matchmaker.')
st.markdown('''Brought to you by Pietro, Isa & Romain 🌱''')

with st.expander("🤓 Here is a fun video to understand coffee tasting:"):
    st.video("https://www.youtube.com/watch?v=IkssYHTSpH4")

st.markdown("---")


# Use `st.cache` when loading data is extremly useful
# because it will cache your data so that your app 
# won't have to reload it each time you refresh your app
@st.cache_data
def load_data():
    data = pd.read_csv(DATA)
    return data

st.header("🪄 The backstages")
st.subheader('''Find all our coffee references here: ''')

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run


## Run the below code if the box is checked ✅
if st.checkbox('👀 Check the box to see the complete database'):
    st.subheader('The complete coffee dataset')
    st.write(data)    

## Highlighting basic infos

st.header('Here are some basic infos you may need')
st.subheader('''🏆 Your roasting machine deserves better than just a bag of quakers
''')
st.markdown('Beander got you covered with only top-notch coffee beans.')
data_cup = data.groupby('Variety')['Total.Cup.Points'].mean()
fig_cup = px.bar(data_cup)
fig_cup.update_layout(
    title='Average Total Cup Points by Variety',
    xaxis_title='Variety',
    yaxis_title='Average Total Cup Points',
    showlegend = False)
fig_cup.add_hline(y=80, line_width=3, line_dash='solid', line_color='red')
st.plotly_chart(fig_cup)

## DATA SEGMENTATION
## visualizing the clusters

st.header('🤖 How it works')
st.subheader('📈 PCA visualisation based on a K-Means with 8 clusters')
st.markdown('''These clusters are defined by 5 common coffee descriptors : Aroma, Aftertaste, Acidity, Body, Sweetness. ''')

## Kmeans pipeline
numeric_cols =data[['Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness']] #Profile aromatique
scaler = StandardScaler()
features_scaled = scaler.fit_transform(numeric_cols)
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(features_scaled)
data['Cluster'] = kmeans.labels_

## PCA For dataviz
pca = PCA()
df_pca = pca.fit_transform(features_scaled)
df_final = pd.DataFrame(df_pca)
df_final = df_final.dot(pca.components_.T)
df_final = pd.concat([df_final, data["Cluster"]], axis=1)
new_names = {0:'PC1', 1:'PC2', 2:'PC3', 3:'PC4', 4:'PC5'}
df_final = df_final.rename(columns=new_names)

fig = px.scatter_3d(data_frame=df_final, x='PC1', y='PC2', z='PC3', color='Cluster', color_continuous_scale='Inferno')
st.plotly_chart(fig)

## Input sliders
st.header('👾 How do you like your coffee?')

# Aroma
st.subheader('🍫 Aroma')
values = st.slider('Aroma', 0.00, 10.00, label_visibility='collapsed')
st.write('Aroma', values)

# Aftertaste
st.subheader('⏱ Aftertaste')
values = st.slider('Aftertaste', 0.00, 10.00, label_visibility='collapsed')
st.write('Aftertaste', values)

# Acidity
st.subheader('🍋 Acidity')
values = st.slider('Acidity', 0.00, 10.00, label_visibility='collapsed', help='Should it tickle a little ?')
st.write('Acidity', values)

# Body
st.subheader('💪 Body')
values = st.slider('Body', 0.00, 10.00, label_visibility='collapsed')
st.write('Body', values)

# Sweetness
st.subheader('🧁 Sweetness')
values = st.slider('Sweetness', 0.00, 10.00, label_visibility='collapsed')
st.write('Sweetness', values)
