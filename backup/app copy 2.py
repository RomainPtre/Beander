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
    data = pd.read_csv(DATA, index_col=0)
    return data

st.header("🪄 The backstages")
st.subheader('''Find all our coffee references here: ''')

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run


## Run the below code if the box is checked ✅
if st.checkbox('👀 Check the box to see the complete dataset'):
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

# Replace with a joblib plus tard
## Kmeans pipeline
numeric_cols = data[['Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness']] # Profile aromatique
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

## Input flavors
st.header('👾 How do you like your coffee?')
st.markdown('''
Here you can create your own coffee tasting profile. We will then predict which farm/producer you should get in touch with to drink it!
            ''')


# # Create a new empty row

## Input sliders
# Aroma
st.subheader('🍫 Aroma')
aroma_values = st.slider('Aroma', 0.00, 10.00, label_visibility='collapsed')

# Aftertaste
st.subheader('⏱ Aftertaste')
aftertaste_values = st.slider('Aftertaste', 0.00, 10.00, label_visibility='collapsed')

# Acidity
st.subheader('🍋 Acidity')
acidity_values = st.slider('Acidity', 0.00, 10.00, label_visibility='collapsed', help='Should it tickle a little ?')

# Body
st.subheader('💪 Body')
body_values = st.slider('Body', 0.00, 10.00, label_visibility='collapsed')

# Sweetness
st.subheader('🧁 Sweetness')
sweetness_values = st.slider('Sweetness', 0.00, 10.00, label_visibility='collapsed')

df_coffee_reco = data

# Initialize session state for df_coffee_reco to persist the data across user interactions
if 'df_coffee_reco' not in st.session_state:
    st.session_state.df_coffee_reco = None

### Predicting to which cluster user data belongs to
## Submit button to create new row
if st.button('Go fetch, Beander!'):
    st.header('🎉 Here are some coffee we recommend checking-out to make your deam come true')
    user_row = {'Aroma': aroma_values,
                'Aftertaste': aftertaste_values,
                'Acidity': acidity_values,
                'Body': body_values,
                'Sweetness': sweetness_values}
    
    # adding the user_values to the features dataframe
    user_row = pd.DataFrame([user_row])
    user_row_scaled = scaler.transform(user_row)
    user_pred = kmeans.predict(user_row_scaled)


    ## Showing the correct cluster in a 3D PCA
    # backing up
    df_pca = df_final

    ## Splitting dataframe based on user_cluster or not
    mask_user = df_pca['Cluster']==user_pred[0]
    df_user_pca = df_pca[mask_user]
    df_rest_pca = df_pca[~mask_user]

    # Showing 3D PCA with cluster of interest
    # Defining custom colorscale so that the two traces use the same scale for cluster_colors
    # Creating two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f'''***Your coffee belongs to the cluster*** n. **:red[{user_pred[0]}]**''')
        color_scale = 'Inferno'
        color_range = [df_pca['Cluster'].min(), df_pca['Cluster'].max()]

        fig_pca = go.Figure(go.Scatter3d(x=df_user_pca['PC1'], y=df_user_pca['PC2'], z=df_user_pca['PC3'], mode='markers', marker=dict(color=df_user_pca['Cluster'], colorscale=color_scale, cmin=color_range[0], cmax=color_range[1], opacity=1)))
        fig_pca.add_trace(go.Scatter3d(x=df_rest_pca['PC1'], y=df_rest_pca['PC2'], z=df_rest_pca['PC3'], mode='markers', marker=dict(color=df_rest_pca['Cluster'], colorscale=color_scale, cmin=color_range[0], cmax=color_range[1],opacity=0.1)))
        fig_pca.update_layout(showlegend=False, scene=dict(xaxis_title='PC1', yaxis_title='Pc2', zaxis_title='PC3'))
        st.plotly_chart(fig_pca)

    with col2:
        st.subheader(f'''Here is a small overview of the results''')
        ## Returns the list of all coffees from this cluster
        # Preparing the dataframe
        # Filtering only coffee from user's cluster
        recommend = df_coffee_reco['Cluster']==user_pred[0]
        st.session_state.df_coffee_reco = df_coffee_reco[recommend]

        # Keeping only columns of interest
        columns_to_keep = ['Species', 'Country.of.Origin', 'Farm.Name', 'Region', 'In.Country.Partner', 'Owner.1', 'Variety', 'Processing.Method', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Total.Cup.Points', 'Moisture', 'Color', 'altitude_mean_meters']
        st.session_state.df_coffee_reco = st.session_state.df_coffee_reco[columns_to_keep]

        # Fill a column with all values from tasting profile for fancy display of tasting profile
        st.session_state.df_coffee_reco['Tasting profile'] = np.nan
        st.session_state.df_coffee_reco['Tasting profile'] = st.session_state.df_coffee_reco.apply(lambda x: [x['Aroma'], x['Aftertaste'], x['Acidity'], x['Body'], x['Sweetness']], axis=1)

        # Editing the dataframe display with streamlit functions
        df_edited = st.data_editor(
            st.session_state.df_coffee_reco,
            column_config={
                'Total.Cup.Points': st.column_config.ProgressColumn(
                    'Coffee rating',
                    help='How much the coffee was rated by certified tasters',
                    format='⭐ %.1f',
                    min_value=0,
                    max_value=100
                ),
                'Tasting profile': st.column_config.BarChartColumn(
                    '  🍫  ⏱  🍋   💪   🧁',
                    help='Each bar corresponds to the sliders above (same order)',
                    y_min=0,
                    y_max=10.00
                )
            },
            hide_index=True
                    )

if st.session_state.df_coffee_reco is not None:
    # Creating a list of variety for a dropdown menu
    variety_dropdown = st.session_state.df_coffee_reco['Variety'].unique().tolist()

    # Creating filters as dropdown menus
    st.subheader('''Filters ''')
    # varieties
    variety_filter = st.multiselect('Select the coffee varieties', variety_dropdown, default=variety_dropdown)
    filtered_df = st.session_state.df_coffee_reco[st.session_state.df_coffee_reco['Variety'].isin(variety_filter)]

    broader_df = st.data_editor(
        filtered_df,
            column_config={
                'Total.Cup.Points': st.column_config.ProgressColumn(
                    'Coffee rating',
                    help='How much the coffee was rated by certified tasters',
                    format='⭐ %.1f',
                    min_value=0,
                    max_value=100
                ),
                'Tasting profile': st.column_config.BarChartColumn(
                    '  🍫  ⏱  🍋   💪   🧁',
                    help='Each bar corresponds to the sliders above (same order)',
                    y_min=0,
                    y_max=10.00
                )
            },
            hide_index=True,
            key = 'broader_df'
                    )
else:
    st.write('Press the button to see recommendations')