import numpy as np
import pandas as pd
import gmaps
import gmaps.datasets
import googlemaps
import gmaps

API_KEY = 'AIzaSyB3ERaF6vtAQOmKZs-_0SpvgsMlLFBjCcU'

df = pd.read_csv('dc-residential-properties/DC_Properties.csv')
df.dropna(subset=['LATITUDE'],inplace= True)
gm = googlemaps.Client(key=API_KEY)
gmaps.configure(api_key=API_KEY)  # Your Google API key
locations = df[['LATITUDE', 'LONGITUDE']]  # put latitide and longitude as a variable name 'locations'
val = df['PRICE']  # put the weight into variable name 'val'
df.head()
center_lat = 38.895
center_lng = -77.0366

def drawHeatMap(location, val, zoom, intensity, radius):
    # setting the data and parameters
    heatmap_layer = gmaps.heatmap_layer(locations, val, dissipating = True)
    heatmap_layer.max_intensity = intensity
    heatmap_layer.point_radius = radius
    # draw the heatmap into a figure
    fig = gmaps.figure()
    fig = gmaps.figure(center = [center_lat,center_lng], zoom_level=zoom)
    fig.add_layer(heatmap_layer)
    return fig

# set up parameters
zoom=10
intensity=5
radius=15

# call the function to draw the heatmap
#f=drawHeatMap(locations, val, zoom, intensity, radius)

heatmap_layer = gmaps.heatmap_layer(locations, val, dissipating = True)
heatmap_layer.max_intensity = intensity
heatmap_layer.point_radius = radius
# draw the heatmap into a figure
fig = gmaps.figure()
fig = gmaps.figure(center = [center_lat,center_lng], zoom_level=zoom)
fig.add_layer(heatmap_layer)
