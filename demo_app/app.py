import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import joblib as jb
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


### Config
st.set_page_config(
    page_title="Welcome to Beander",
    page_icon=" ☕",
    layout="wide"
)

DATA = ('./src/Coffee_dataset_cleaned_column_and_rows.csv')

### Headers
st.title('Welcome to Beander ☕')
st.header('Beander, the coffee matchmaker.')
st.markdown('''***Brought to you by Pietro, Isa & Romain*** 🌱''')

st.markdown('---')


### Loading Dataset in cache
@st.cache_data
def load_data(DATA):
    data = pd.read_csv(DATA, index_col=0)
    return data

data = load_data(DATA)

# To call the original table without the cluster columns, to display in the 'backstage' section
data_orig = data
numeric_cols = data[['Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness']]  # Profile aromatique

## DATA SEGMENTATION
## Kmeans pipeline
scaler_kmeans = StandardScaler()
features_scaled_kmeans = scaler_kmeans.fit_transform(numeric_cols)
kmeans = jb.load('./src/models/Kmeans.pkl')

## Agglomerative clustering pipeline
# HAS TO BE AFTER USER INPUT
data_agg = data
numeric_cols_agg = data_agg[['Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness']]  # Profile aromatique
# numeric_cols_agg.iloc[-1] = user_row
scaler_agg = StandardScaler()
features_scaled_agg = scaler_agg.fit_transform(numeric_cols_agg)
knn = jb.load('./src/models/knn.pkl')

## Euclidean distance pipeline
# Normalisation de la donnée
scaler_dist = StandardScaler()
features_scaled_dist = scaler_dist.fit_transform(numeric_cols)

## PCA For dataviz
## Kmeans
pca = PCA()
df_pca = pca.fit_transform(features_scaled_kmeans)
df_final = pd.DataFrame(df_pca)
df_final = df_final.dot(pca.components_.T)
df_final = pd.concat([df_final, data['ClusterKmeans']], axis=1)
new_names = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'}
df_final = df_final.rename(columns=new_names)


######################################################
## Input flavors
st.header('👾 How do you like your coffee?')
st.markdown('''
Here you can create your own coffee tasting profile. We will then predict which farm/producer you should get in touch with to drink it!
            ''')

## Input sliders
# Aroma
st.subheader('🍫 Aroma')
aroma_values = st.slider('Aroma', data['Aroma'].min(), data['Aroma'].max(), label_visibility='collapsed', format ='')

# Aftertaste
st.subheader('⏱ Aftertaste')
aftertaste_values = st.slider('Aftertaste', data['Aftertaste'].min(), data['Aftertaste'].max(), label_visibility='collapsed', format='')

# Acidity
st.subheader('🍋 Acidity')
acidity_values = st.slider('Acidity', data['Acidity'].min(), data['Acidity'].max(), label_visibility='collapsed', help='Should it tickle a little ?', format='')

# Body
st.subheader('💪 Body')
body_values = st.slider('Body', data['Body'].min(), data['Body'].max(), label_visibility='collapsed', format='')

# Sweetness
st.subheader('🧁 Sweetness')
sweetness_values = st.slider('Sweetness', data['Sweetness'].min(), data['Sweetness'].max(), label_visibility='collapsed', format='')

# Initialize session state variables to persist data across interactions
if 'df_coffee_reco_kmeans' not in st.session_state:
    st.session_state.df_coffee_reco_kmeans = None
if 'user_pred_kmeans' not in st.session_state:
    st.session_state.user_pred_kmeans = None

if 'df_coffee_reco_dist' not in st.session_state:
    st.session_state.df_coffee_reco_dist = None

if 'user_pred_dist' not in st.session_state:
    st.session_state.user_pred_dist = None

if 'df_coffee_reco_agg' not in st.session_state:
    st.session_state.df_coffee_reco_agg = None

if 'user_pred_agg' not in st.session_state:
    st.session_state.user_pred_agg = None

#################################################################################
### Predicting to which cluster user data belongs to
## Submit button to create new row
if st.button('Go fetch, Beander!'):
    user_row = {'Aroma': aroma_values,
                'Aftertaste': aftertaste_values,
                'Acidity': acidity_values,
                'Body': body_values,
                'Sweetness': sweetness_values}


    # transforming user_row as dataframew to concatenate it to data in the different pipelines
    user_row = pd.DataFrame([user_row])

    ######################
    ## KMEANS
    user_row_scaled = scaler_kmeans.transform(user_row)
    st.session_state.user_pred_kmeans = kmeans.predict(user_row_scaled)[0]

    ## Showing the correct cluster in a 3D PCA
    #backing up data
    df_pca = df_final

    ## Splitting dataframe based on user_cluster or not
    mask_user = df_pca['ClusterKmeans'] == st.session_state.user_pred_kmeans
    df_user_pca = df_pca[mask_user]
    df_rest_pca = df_pca[~mask_user]

    # Storing the dataframe in session state
    st.session_state.df_coffee_reco_kmeans = data[data['ClusterKmeans'] == st.session_state.user_pred_kmeans]

    # Keeping only columns of interest
    columns_to_keep = ['Species', 'Country.of.Origin', 'Farm.Name', 'Region', 'In.Country.Partner', 'Owner.1', 'Variety', 'Processing.Method', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Total.Cup.Points', 'Moisture', 'Color', 'altitude_mean_meters']
    st.session_state.df_coffee_reco_kmeans = st.session_state.df_coffee_reco_kmeans[columns_to_keep]

    # Fill a column with all values from tasting profile for fancy display of tasting profile
    st.session_state.df_coffee_reco_kmeans['Tasting profile'] = st.session_state.df_coffee_reco_kmeans.apply(lambda x: [x['Aroma'], x['Aftertaste'], x['Acidity'], x['Body'], x['Sweetness']], axis=1)

    # Displaying the 3D PCA
    color_range = [df_pca['ClusterKmeans'].min(), df_pca['ClusterKmeans'].max()]

    st.session_state.fig_pca_kmeans = go.Figure(go.Scatter3d(x=df_user_pca['PC1'], y=df_user_pca['PC2'], z=df_user_pca['PC3'], mode='markers', marker=dict(color=df_user_pca['ClusterKmeans'], colorscale='Inferno', cmin=color_range[0], cmax=color_range[1], opacity=1)))
    st.session_state.fig_pca_kmeans.add_trace(go.Scatter3d(x=df_rest_pca['PC1'], y=df_rest_pca['PC2'], z=df_rest_pca['PC3'], mode='markers', marker=dict(color=df_rest_pca['ClusterKmeans'], colorscale='Inferno', cmin=color_range[0], cmax=color_range[1], opacity=0.1)))
    st.session_state.fig_pca_kmeans.update_layout(showlegend=False, scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

    ##########################
    # AGGLOMERATIVE CLUSTERING
    ## Model and table
    # numeric_cols_agg.iloc[-1] = user_row
    # agg = AgglomerativeClustering(n_clusters=11, metric='euclidean', linkage='ward')
    # agg.fit(features_scaled_agg)
    # data_agg['ClusterAgg'] = agg.labels_

    user_row_scaled_agg = scaler_agg.transform(user_row)
    user_pred_agg = knn.predict(user_row)   # To fix: not working properly when using user_row_scaled_agg
    user_pred_agg = int(user_pred_agg[0])
    st.session_state.user_pred_agg = user_pred_agg

    df_coffee_reco_agg = data_agg[data_agg['ClusterAgg'] == st.session_state.user_pred_agg]
    # df_coffee_reco_agg = df_coffee_reco_agg.iloc[:-1]
    # user_pred_agg = data_agg['ClusterAgg'].iloc[-1]

    st.session_state.df_coffee_reco_agg = df_coffee_reco_agg

    ## PCA
    pca = PCA()
    df_pca_agg = pca.fit_transform(features_scaled_agg)
    df_final_agg = pd.DataFrame(df_pca_agg)
    df_final_agg = df_final_agg.dot(pca.components_.T)
    df_final_agg = pd.concat([df_final_agg, data['ClusterAgg']], axis=1)
    new_names = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'}
    df_final_agg = df_final_agg.rename(columns=new_names)

    df_pca_agg = df_final_agg

    ## Splitting dataframe based on user_cluster or not
    mask_user = df_pca_agg['ClusterAgg'] == user_pred_agg
    df_user_pca_agg = df_pca_agg[mask_user]
    df_rest_pca_agg = df_pca_agg[~mask_user]

    # Storing the dataframe in session state
    st.session_state.df_coffee_reco_agg = data_agg[data_agg['ClusterAgg'] == st.session_state.user_pred_agg]

    # Keeping only columns of interest
    columns_to_keep = ['Species', 'Country.of.Origin', 'Farm.Name', 'Region', 'In.Country.Partner', 'Owner.1', 'Variety', 'Processing.Method', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Total.Cup.Points', 'Moisture', 'Color', 'altitude_mean_meters']
    st.session_state.df_coffee_reco_agg = st.session_state.df_coffee_reco_agg[columns_to_keep]

    # Fill a column with all values from tasting profile for fancy display of tasting profile
    st.session_state.df_coffee_reco_agg['Tasting profile'] = st.session_state.df_coffee_reco_agg.apply(lambda x: [x['Aroma'], x['Aftertaste'], x['Acidity'], x['Body'], x['Sweetness']], axis=1)


    # Displaying the 3D PCA
    color_range_agg = [df_pca_agg['ClusterAgg'].min(), df_pca_agg['ClusterAgg'].max()]

    st.session_state.fig_pca_agg = go.Figure(go.Scatter3d(x=df_user_pca_agg['PC1'], y=df_user_pca_agg['PC2'], z=df_user_pca_agg['PC3'], mode='markers', marker=dict(color=df_user_pca_agg['ClusterAgg'], colorscale='Inferno', cmin=color_range_agg[0], cmax=color_range_agg[1], opacity=1)))
    st.session_state.fig_pca_agg.add_trace(go.Scatter3d(x=df_rest_pca_agg['PC1'], y=df_rest_pca_agg['PC2'], z=df_rest_pca_agg['PC3'], mode='markers', marker=dict(color=df_rest_pca_agg['ClusterAgg'], colorscale='Inferno', cmin=color_range_agg[0], cmax=color_range_agg[1], opacity=0.1)))
    st.session_state.fig_pca_agg.update_layout(showlegend=False, scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

    #####################################
    # EUCLIDEAN DISTANCES
    user_input_scaled = scaler_dist.transform(user_row)
    distances = euclidean_distances(features_scaled_dist, user_input_scaled)
    st.session_state.user_pred_dist = distances
    data['ClusterDist'] = distances
    # Storing the reco basedo n distances in session state
    st.session_state.df_coffee_reco_dist = data#['ClusterDist']

    # Fill a column with all values from tasting profile for fancy display of tasting profile
    st.session_state.df_coffee_reco_dist['Tasting profile'] = st.session_state.df_coffee_reco_dist.apply(lambda x: [x['Aroma'], x['Aftertaste'], x['Acidity'], x['Body'], x['Sweetness']], axis=1)

    

# OUTPUT
# Checking if previous results in cache or not
tab1, tab2, tab3 = st.tabs(['**Euclidean Distances**', '**K-Means**', '**Agglomerative Clustering**'])

## K-Means output
with tab2:
    if st.session_state.df_coffee_reco_kmeans is not None:
        col1, col2 = st.columns([0.4, 0.6], gap='large')
        with col1:
            st.subheader(f'''***Your coffee belongs to the cluster*** n. **:red[{st.session_state.user_pred_kmeans+1}]**''')
            st.plotly_chart(st.session_state.fig_pca_kmeans)

        with col2:
            st.subheader('''Here are some recommendations''')
            st.markdown('''🤓 Feel free to play with the filters and column sorting for more results! ''')
            columns_order = ['Owner.1', 'Tasting profile', 'Total.Cup.Points', 'Variety', 'Country.of.Origin', 'Processing.Method', 'altitude_mean_meters', 'Species', 'Farm.Name', 'Region', 'In.Country.Partner', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Moisture', 'Color']

            ## Filters
            # Initialize dataframe
            filtered_df = st.session_state.df_coffee_reco_kmeans

            # Variety
            activate_variety_filter = st.checkbox('Filter by variety 🌱')
            # Updating the dropdown and filtered dataframe
            if activate_variety_filter:
                variety_dropdown = st.session_state.df_coffee_reco_kmeans['Variety'].unique().tolist()
                variety_filter = st.selectbox('Select the coffee varieties', variety_dropdown)
                filtered_df = filtered_df[filtered_df['Variety']==variety_filter]

            # Process
            activate_process_filter = st.checkbox('Filter by process 🧪')
            # Updating the dropdown and filtered dataframe
            if activate_process_filter:
                process_dropdown = st.session_state.df_coffee_reco_kmeans['Processing.Method'].unique().tolist()
                process_filter = st.selectbox('Select the processing method', process_dropdown)
                filtered_df = filtered_df[filtered_df['Processing.Method']==process_filter]

            # Altitude
            activate_altitude_filter = st.checkbox('Filter by altitude 🗻')
            # Updating the dropdown and filtered dataframe
            if activate_altitude_filter:
                altitude_min = st.session_state.df_coffee_reco_kmeans['altitude_mean_meters'].min()
                altitude_max = st.session_state.df_coffee_reco_kmeans['altitude_mean_meters'].max()
                altitude_filter = st.select_slider('Select the altitude range', value=[altitude_min, altitude_max], 
                                                options=range(int(altitude_min), int(altitude_max)+1), 
                                                help='Missing altitudes are labelled 0')
                filtered_df = filtered_df[(filtered_df['altitude_mean_meters'] >= altitude_filter[0]) & 
                                                    (filtered_df['altitude_mean_meters'] <= altitude_filter[1])]

            ## OUTPUT FILTERED (or not) DATAFRAME
            # Displaying the filtered dataframe with custom columns
            df_edited_knn = st.data_editor(
                filtered_df.head(10),
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
                hide_index=True, key='broader_df_knn', column_order=columns_order, disabled=True
            )
    else:
        st.write('Press the button to see recommendations')

## Agglomerative output
with tab3:
    if st.session_state.df_coffee_reco_agg is not None:
        col1, col2 = st.columns([0.4, 0.6], gap='large')
        with col1:
            st.subheader(f'''***Your coffee belongs to the cluster*** n. **:red[{st.session_state.user_pred_agg+1}]**''')
            st.plotly_chart(st.session_state.fig_pca_agg)

        with col2:
            st.subheader('''Here are some recommendations''')
            st.markdown('''🤓 Feel free to play with the filters and column sorting for more results! ''')
            columns_order = ['Owner.1', 'Tasting profile', 'Total.Cup.Points', 'Variety', 'Country.of.Origin', 'Processing.Method', 'altitude_mean_meters', 'Species', 'Farm.Name', 'Region', 'In.Country.Partner', 'Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness', 'Moisture', 'Color']
            # st.data_editor(st.session_state.df_coffee_reco_agg.head(10))

            ## Filters
            # Initialize dataframe
            filtered_df_agg = st.session_state.df_coffee_reco_agg

            # Variety
            activate_variety_filter = st.checkbox('Filter by variety 🌱', key='variety_agg')
            # Updating the dropdown and filtered dataframe
            if activate_variety_filter:
                variety_dropdown = st.session_state.df_coffee_reco_agg['Variety'].unique().tolist()
                variety_filter = st.selectbox('Select the coffee varieties', variety_dropdown)
                filtered_df_agg = filtered_df_agg[filtered_df_agg['Variety']==variety_filter]

            # Process
            activate_process_filter = st.checkbox('Filter by process 🧪', key='process_agg')
            # Updating the dropdown and filtered dataframe
            if activate_process_filter:
                process_dropdown = st.session_state.df_coffee_reco_agg['Processing.Method'].unique().tolist()
                process_filter = st.selectbox('Select the processing method', process_dropdown)
                filtered_df_agg = filtered_df_agg[filtered_df_agg['Processing.Method']==process_filter]

            # Altitude
            activate_altitude_filter = st.checkbox('Filter by altitude 🗻', key='process_alt')
            # Updating the dropdown and filtered dataframe
            if activate_altitude_filter:
                altitude_min = st.session_state.df_coffee_reco_agg['altitude_mean_meters'].min()
                altitude_max = st.session_state.df_coffee_reco_agg['altitude_mean_meters'].max()
                altitude_filter = st.select_slider('Select the altitude range', value=[altitude_min, altitude_max], 
                                                options=range(int(altitude_min), int(altitude_max)+1), 
                                                help='Missing altitudes are labelled 0')
                filtered_df_agg = filtered_df_agg[(filtered_df_agg['altitude_mean_meters'] >= altitude_filter[0]) & 
                                                    (filtered_df_agg['altitude_mean_meters'] <= altitude_filter[1])]

            df_edited_agg = st.data_editor(
            filtered_df_agg.head(10),
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
        hide_index=True, key='df_agg', column_order=columns_order, disabled=True
    )

    else :
        st.write('Press the button to see recommendations')

## Euclidean distances output
with tab1:
    if st.session_state.df_coffee_reco_dist is not None:
        st.subheader(f'''***Here are the :red[10] closest coffees***''')
        st.markdown('''🤓 Feel free to play with the filters and column sorting for more results! ''')

        data['Distance'] = st.session_state.user_pred_dist
        filtered_df_dist = st.session_state.df_coffee_reco_dist.sort_values(by='Distance')
        ## Filters
        # # Initialize dataframe
        # filtered_df_dist = st.session_state.df_coffee_reco_dist

        # Variety
        activate_variety_filter = st.checkbox('Filter by variety 🌱', key= 'variety_dist')
        # Updating the dropdown and filtered dataframe
        if activate_variety_filter:
            variety_dropdown = st.session_state.df_coffee_reco_dist['Variety'].unique().tolist()
            variety_filter = st.selectbox('Select the coffee varieties', variety_dropdown)
            filtered_df_dist = filtered_df_dist[filtered_df_dist['Variety']==variety_filter]

        # Process
        activate_process_filter = st.checkbox('Filter by process 🧪', key= 'process_dist')
        # Updating the dropdown and filtered dataframe
        if activate_process_filter:
            process_dropdown = st.session_state.df_coffee_reco_dist['Processing.Method'].unique().tolist()
            process_filter = st.selectbox('Select the processing method', process_dropdown)
            filtered_df_dist = filtered_df_dist[filtered_df_dist['Processing.Method']==process_filter]

        # Altitude
        activate_altitude_filter = st.checkbox('Filter by altitude 🗻', key= 'altitude_dist')
        # Updating the dropdown and filtered dataframe
        if activate_altitude_filter:
            altitude_min = st.session_state.df_coffee_reco_dist['altitude_mean_meters'].min()
            altitude_max = st.session_state.df_coffee_reco_dist['altitude_mean_meters'].max()
            altitude_filter = st.select_slider('Select the altitude range', value=[altitude_min, altitude_max], 
                                            options=range(int(altitude_min), int(altitude_max)+1), 
                                            help='Missing altitudes are labelled 0')
            filtered_df_dist = filtered_df_dist[(filtered_df_dist['altitude_mean_meters'] >= altitude_filter[0]) & 
                                                (filtered_df_dist['altitude_mean_meters'] <= altitude_filter[1])]

        # st.data_editor(df_sorted_dist.head(10))
        df_edited_dist = st.data_editor(
        filtered_df_dist.head(10),
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
    hide_index=True, key='df_dist', column_order=columns_order, disabled=True
)

    else:
        st.write('Press the button to see recommendations')    
st.markdown('---')



############################################################
## Highlighting basic infos
st.header('Here are some basic infos you may need')

col1, col2 = st.columns([0.5, 0.5], gap='large')
with col1:
    st.subheader('''🏆 Your roaster deserves better than just a bag of quakers''')
    st.markdown('***Beander got you covered with only top-notch coffee beans.***')
    data_cup = data.groupby('Variety')['Total.Cup.Points'].mean()
    fig_cup = px.bar(data_cup)
    fig_cup.update_layout(xaxis_title='Varieties', yaxis_title='Average Score', showlegend=False)
    fig_cup.add_hline(y=80, line_width=3, line_dash='solid', line_color='red')
    st.plotly_chart(fig_cup)

with col2:
    st.subheader('''📊 Characterization by aromatic profile''')
    st.markdown('''***Here are the 5 coffee descriptors Beander uses.***''')
    box = px.box(data, y=['Aroma', 'Aftertaste', 'Acidity', 'Body', 'Sweetness'])
    box.update_layout(yaxis_title='Distribution')
    st.plotly_chart(box)


st.markdown('---')

if st.checkbox('👀 Click here to see how **Beander** works'):
    st.header('🪄 Welcome to the backstages')
    ## visualizing the clusters
    st.subheader('📈 PCA visualisation based on a K-Means with 8 clusters')
    st.markdown('''These clusters are defined by 5 common coffee descriptors : Aroma, Aftertaste, Acidity, Body, Sweetness. ''')
    fig = px.scatter_3d(data_frame=df_final, x='PC1', y='PC2', z='PC3', color='ClusterKmeans', color_continuous_scale='Inferno')
    st.plotly_chart(fig)

    st.subheader('The complete coffee dataset')
    st.dataframe(data_orig)