import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_URL = "data/new_york_taxi_clusters.csv"
LOGO_URL = "images/new-york-pickups.png"

### Config
st.set_page_config(
    page_title="New York Pickups",
    page_icon="🚕",
    layout="wide"
)

### Dictionaries for mapping
day_of_week_dict = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

month_dict = {
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September"
}

knn_cluster_mapping = {
        0: "Manhattan",
        1: "East Queens",
        2: "Brooklyn",
        3: "New Jersey",
        4: "Bronx & West Queens",
    }

dbscan_cluster_mapping = {
        0: "Manhattan",
        1: "John F. Kennedy Airport",
        2: "Newark Liberty Airport",
        3: "LaGuardia Airport",
        4: "Newark Liberty Airport",
        5: "LaGuardia Airport",
        6: "John F. Kennedy Airport",
        7: "Brooklyn",
        8: "Manhattan Bridge View",
        9: "LaGuardia Airport",
        10: "Brooklyn",
        11: "Brooklyn",
        12: "Brooklyn",
        13: "Center Blvd",
        14: "Atlantic Terminal",
        15: "East Williamsburg",
        16: "Pioneer Works Center",
        17: "Brooklyn",
        18: "Manhattan",
        19: "Queens Plaza",
        -1: "Outliers"
    }

poi_list = ["John F. Kennedy Airport", 
             "Newark Liberty Airport", 
             "LaGuardia Airport", 
             "Manhattan Bridge View", 
             "Center Blvd", 
             "Atlantic Terminal", 
             "East Williamsburg", 
             "Pioneer Works Center"]

### Data
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y-%m-%d %H:%M:%S')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['kms'] = data['kms'].map(knn_cluster_mapping)
    data['dbscan'] = data['dbscan'].map(dbscan_cluster_mapping)
    return data

data = load_data()

### Streamlit pages

### Title page
def title_page():
    st.title("New York Pickups")
    st.image(LOGO_URL, width="stretch")

### Dataset page
def dataset_page():
    st.title("The dataset")
    st.write(f"The dataset  :red[**is a sample corresponding to 1.8% of the full dataset**] and have **{data.shape[0]} rows** and **{data.shape[1]} columns**, below is a summary:")
    
    meta_data = pd.DataFrame({
        "features": [
            "Date/Time",
            "Lat",
            "Lon",
            "Month",
            "Date",
            "Hour",
            "DayofWeek",
            "kms",
            "dbscan"
            ],
             
        "description": [
            "Date and time of pickup",
            "Latitude of pickup location",
            "Longitude of pickup location",
            "Month of the ride",
            "Date of the ride",
            "Hour of the ride",
            "Day of week for the ride",
            "Cluster label from KMeans",
            "Cluster label from DBSCAN"
            ]
    })
    with st.expander("Show metadata"):
        st.dataframe(meta_data, hide_index=True, width='stretch')

    with st.expander("Show raw data preview"):
        st.dataframe(data.head(), hide_index=True, width='stretch')

    with st.expander("Show data types"):
        data_types = data.dtypes
        st.dataframe(pd.DataFrame(data_types).T, hide_index=True, width='stretch')

    with st.expander("Show descriptive statistics"):
        st.write(data.describe())

    with st.expander("Show missing values"):
        null_table = data.isnull().sum()
        st.dataframe(pd.DataFrame(null_table).T, hide_index=True, width='stretch')

    with st.expander("Show KNN clusters mapping"):
        st.write(knn_cluster_mapping)

    with st.expander("Show DBSCAN clusters mapping"):
        st.write(dbscan_cluster_mapping)

def ride_over_time_page():
    st.title("Number of Rides over Time")
    ride_per_date = data.groupby('Date').size().reset_index(name='Number of Rides')

    fig = px.line(ride_per_date,
                  x="Date", 
                  y="Number of Rides", 
                  title="Number of Rides over Time")

    st.plotly_chart(fig, width='content')

def ride_per_month_page():
    st.title("Number of Rides per Month")
    fig = px.bar(data['Month'].value_counts().sort_index(),
                 labels={'index': 'Month', 'value': 'Number of Rides'}, 
                 title='Number of Rides per Month')
    
    fig.update_layout(showlegend=False)   
    
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(month_dict.keys()),
        ticktext=list(month_dict.values())
    )

    st.plotly_chart(fig, width='content')

def ride_per_dow_page():
    st.title("Rides per Day of Week")
    fig = px.bar(data['DayofWeek'].value_counts().sort_index(),
                 labels={'index': 'Day of Week', 'value': 'Number of Rides'}, 
                 title='Number of Rides per Day of Week')
    
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(day_of_week_dict.keys()),
        ticktext=list(day_of_week_dict.values())
    )
    
    fig.update_layout(showlegend=False) 

    st.plotly_chart(fig, width='content')

def ride_per_hour_page():
    st.title("Rides per Hour")
    fig = px.bar(data['Hour'].value_counts().sort_index(), 
                 labels={'index': 'Hour', 'value': 'Number of Rides'}, 
                 title='Number of Rides per Hour')
    
    fig.update_layout(showlegend=False)

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(0,24)),
        ticktext=[str(i) for i in range(0,24)]
    )

    st.plotly_chart(fig, width='content')

def knn_page():
    st.title("KNN Clustering")

    fig = px.scatter_map(data, 
               lat="Lat", 
               lon="Lon", 
               color="kms",
               color_continuous_scale=px.colors.sequential.Viridis
               )
    
    fig.update_coloraxes(
        colorbar_title_text="Cluster",
        colorbar_title_side="right",
        colorbar_tickmode='array',
        colorbar_tickvals=list(knn_cluster_mapping.keys()),
        colorbar_ticktext=list(knn_cluster_mapping.values())
    )
    
    st.plotly_chart(fig, width='content')

def dbscan_page():
    st.title("DBSCAN Clustering")

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Show POI clusters only", value=True, key="only_poi")

    with col2:
        st.checkbox("Show outliers", value=False, key="show_outliers")

    if st.session_state.only_poi:
        # POI clusters is True
        if st.session_state.show_outliers:
            df_plot = data[(data['dbscan'].isin(poi_list)) | (data['dbscan'] == "Outliers")]
        else:
            df_plot = data[data['dbscan'].isin(poi_list)]
    
    else:
        # POI clusters is False
        if st.session_state.show_outliers:
            df_plot = data
        else:
            df_plot = data[data['dbscan'] != "Outliers"]

    fig = px.scatter_map(df_plot, 
               lat="Lat", 
               lon="Lon", 
               color="dbscan",
               color_continuous_scale=px.colors.sequential.Viridis
            )
    
    # Managing the map zoom level depending on the filters
    if st.session_state.show_outliers:
        fig.update_layout(map=dict(zoom=7))
    else:
        if st.session_state.only_poi:
            fig.update_layout(map=dict(zoom=10))
        else:
            fig.update_layout(map=dict(zoom=9))
    
    st.plotly_chart(fig, width='content')

### Pages layout
pages = {
    "Context": [
    st.Page(title_page, title="Welcome", icon="👋"),
    st.Page(dataset_page, title="Dataset", icon="📜")
    ],
    "Dataset Exploration": [
    st.Page(ride_over_time_page, title="Rides over Time", icon="📈"),
    st.Page(ride_per_month_page, title="Rides per Month", icon="🍂"),
    st.Page(ride_per_dow_page, title="Rides per Day of Week", icon="📆"),
    st.Page(ride_per_hour_page, title="Rides per Hour", icon="🕔")
    ],
    "Clusters Analysis": [
    st.Page(knn_page, title="KNN Clustering", icon="🗺️"),
    st.Page(dbscan_page, title="DBSCAN Clustering", icon="🗺️"),
    ]
    }

pg = st.navigation(pages)

pg.run()