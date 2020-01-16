# import libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

# import street map
street_map = gpd.read_file("dc-residential-properties/Census_Tracts_in_2010.shp")
df = pd.read_csv('dc-residential-properties/DC_Properties.csv')
# designate coordinate system
crs = {'init': 'espc:4326'}
# zip x and y coordinates into single feature
geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
# create GeoPandas dataframe
geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

fig, ax = plt.subplots(figsize=(15, 15))
# add .shp mapfile to axes
street_map.plot(ax=ax, alpha=0.4, color='grey')
# add geodataframe to axes
# assign 'price' variable to represent coordinates on graph
# add legend
# make datapoints transparent using alpha
# assign size of points using markersize
geo_df.plot(column='PRICE', ax=ax, alpha=0.5, legend=True, markersize=10)
# add title to graph
plt.title('Rental Prices in DC', fontsize=15, fontweight='bold')
# set latitiude and longitude boundaries for map display
plt.xlim(-77.11390873, -77.11390873)
plt.ylim(38.99553969, 38.99553969)
# show map
plt.show()
