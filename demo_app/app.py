import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

### Config
st.set_page_config(
    page_title="Welcome to Beander",
    page_icon=" ☕",
    layout="wide"
)

DATA = ('./src/Coffee_dataset_cleaned_column_and_rows.csv')

### App
st.title('Welcome to Beander ☕')
st.header('Beander, the coffee matchmaker.')
st.markdown('''Brought to you by Pietro, Isa & Romain 🌱''')

with st.expander("🤓 Here is a fun video to understand coffee tasting:"):
    st.video("https://www.youtube.com/watch?v=IkssYHTSpH4")

st.markdown("---")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA, index_col=0)
    return data

st.header("🪄 The backstages")
st.subheader('''Find all our coffee references here: ''')

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("")  # change text from "Loading data..." to "" once the the load_data function has run

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
    showlegend=False)
fig_cup.add_hline(y=80, line_width=3, line_dash='solid', line_color='red')
st.plotly_chart(fig_cup)

## DATA SEGMENTATION
## visualizing the clusters
st.header('🤖 How it works')
st.subheader('📈 PCA visualisation based on a K-Means with 8 clusters')
st.markdown('''These clusters are defined by 5 common coffee descriptors : Aroma, Aftertaste, Acidity, Body, Sweetness. ''')

# Replace with a joblib plus tard
## Kmeans pipeline
numeric_cols = data[['Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness']]  # Profile aromatique
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
new_names = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'}
df_final = df_final.rename(columns=new_names)

fig = px.scatter_3d(data_frame=df_final, x='PC1', y='PC2', z='PC3', color='Cluster', color_continuous_scale='Inferno')
st.plotly_chart(fig)

## Input flavors
st.header('👾 How do you like your coffee?')
st.markdown('''
Here you can create your own coffee tasting profile. We will then predict which farm/producer you should get in touch with to drink it!
            ''')

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

# Initialize session state variables to persist data across interactions
if 'df_coffee_reco' not in st.session_state:
    st.session_state.df_coffee_reco = None
if 'user_pred' not in st.session_state:
    st.session_state.user_pred = None

### Predicting to which cluster user data belongs to
## Submit button to create new row
if st.button('Go fetch, Beander!'):
    user_row = {'Aroma': aroma_values,
                'Aftertaste': aftertaste_values,
                'Acidity': acidity_values,
                'Body': body_values,
                'Sweetness': sweetness_values}
    
    # Adding the user_values to the features dataframe
    user_row = pd.DataFrame([user_row])
    user_row_scaled = scaler.transform(user_row)
    st.session_state.user_pred = kmeans.predict(user_row_scaled)[0]

    ## Showing the correct cluster in a 3D PCA
    #backing up data
    df_pca = df_final

    ## Splitting dataframe based on user_cluster or not
    mask_user = df_pca['Cluster'] == st.session_state.user_pred
    df_user_pca = df_pca[mask_user]
    df_rest_pca = df_pca[~mask_user]

    # Storing the dataframe and PCA results in session state
    st.session_state.df_coffee_reco = data[data['Cluster'] == st.session_state.user_pred]

    # Keeping only columns of interest
    columns_to_keep = ['Species', 'Country.of.Origin', 'Farm.Name', 'Region', 'In.Country.Partner', 'Owner.1', 'Variety', 'Processing.Method', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Total.Cup.Points', 'Moisture', 'Color', 'altitude_mean_meters']
    st.session_state.df_coffee_reco = st.session_state.df_coffee_reco[columns_to_keep]

    # Fill a column with all values from tasting profile for fancy display of tasting profile
    st.session_state.df_coffee_reco['Tasting profile'] = st.session_state.df_coffee_reco.apply(lambda x: [x['Aroma'], x['Aftertaste'], x['Acidity'], x['Body'], x['Sweetness']], axis=1)

    # Displaying the 3D PCA
    color_range = [df_pca['Cluster'].min(), df_pca['Cluster'].max()]

    st.session_state.fig_pca = go.Figure(go.Scatter3d(x=df_user_pca['PC1'], y=df_user_pca['PC2'], z=df_user_pca['PC3'], mode='markers', marker=dict(color=df_user_pca['Cluster'], colorscale='Inferno', cmin=color_range[0], cmax=color_range[1], opacity=1)))
    st.session_state.fig_pca.add_trace(go.Scatter3d(x=df_rest_pca['PC1'], y=df_rest_pca['PC2'], z=df_rest_pca['PC3'], mode='markers', marker=dict(color=df_rest_pca['Cluster'], colorscale='Inferno', cmin=color_range[0], cmax=color_range[1], opacity=0.1)))
    st.session_state.fig_pca.update_layout(showlegend=False, scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

# OUTPUT
# Checking if previous results in cache or not
if st.session_state.df_coffee_reco is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f'''***Your coffee belongs to the cluster*** n. **:red[{st.session_state.user_pred}]**''')
        st.plotly_chart(st.session_state.fig_pca)

    with col2:
        st.subheader(f'''Here is a small overview of the results''')
        columns_order = ['Owner.1', 'Tasting profile', 'Total.Cup.Points', 'Variety', 'Country.of.Origin', 'Processing.Method', 'altitude_mean_meters', 'Species', 'Farm.Name', 'Region', 'In.Country.Partner', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Moisture', 'Color']

        ## Filters
        # Initialize dataframe
        filtered_df = st.session_state.df_coffee_reco

        # Variety
        activate_variety_filter = st.checkbox('Enable variety filter')
        # Updating the dropdown and filtered dataframe
        if activate_variety_filter:
            variety_dropdown = st.session_state.df_coffee_reco['Variety'].unique().tolist()
            variety_filter = st.selectbox('Select the coffee varieties', variety_dropdown, index=None)
            filtered_df = filtered_df[filtered_df['Variety']==variety_filter]
        # else:
        #     filtered_df = st.session_state.df_coffee_reco

        # Process
        activate_process_filter = st.checkbox('Enable process filter')
        # Updating the dropdown and filtered dataframe
        if activate_process_filter:
            process_dropdown = st.session_state.df_coffee_reco['Processing.Method'].unique().tolist()
            process_filter = st.selectbox('Select the processing method', process_dropdown, index=None)
            filtered_df = filtered_df[filtered_df['Processing.Method']==process_filter]
        # else:
        #     filtered_df = st.session_state.df_coffee_reco

        ## OUTPUT FILTERED (or not) DATAFRAME
        # Displaying the filtered dataframe with custom columns
        df_edited = st.data_editor(
            filtered_df,
            column_config={
                'Owner.1': 'Exploitation name',
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
                ),
                'Country.of.Origin': 'Country',
                'Processing.Method' : 'Process',
                'altitude_mean_meters': 'Altitude',
                'Farm.Name':None,
                'In.Country.Partner':None,
                'Aroma':None,
                'Aftertaste':None,
                'Acidity':None,
                'Body':None,
                'Sweetness':None,
                'Moisture':st.column_config.NumberColumn(
                    format='%.2f%%'
                )
            },
            hide_index=True, key='broader_df', column_order=columns_order
        )

else:
    st.write('Press the button to see recommendations')
