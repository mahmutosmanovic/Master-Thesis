from settings import *

def transform_to_epsg32636(df, lat_col='lat', lon_col='lon'):
    """
    Transforms latitude and longitude coordinates in a DataFrame to EPSG:32636 (UTM Zone 36N).
    
    Parameters:
    df (pd.DataFrame): DataFrame containing latitude and longitude columns.
    lat_col (str): Name of the latitude column (default: 'latitude').
    lon_col (str): Name of the longitude column (default: 'longitude').
    
    Returns:
    pd.DataFrame: DataFrame with 'x' and 'y' columns containing the transformed coordinates.
    """
    # Create geometry column from lat/lon
    df['geometry'] = df.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
    
    # Create GeoDataFrame with WGS84 CRS
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = 'EPSG:4326'
    
    # Transform to EPSG:32636
    gdf_utm = gdf.to_crs('EPSG:32636')
    
    # Extract x and y coordinates
    gdf_utm['x'] = gdf_utm.geometry.x
    gdf_utm['y'] = gdf_utm.geometry.y
    
    # Return DataFrame with transformed coordinates
    return gdf_utm[['x', 'y']]
