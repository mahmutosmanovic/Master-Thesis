import pandas as pd
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

def pick_utm_epsg(s_lat, n_lat, w_lon, e_lon, datum_name = "WGS 84"):
    utm_list = query_utm_crs_info(
        datum_name=datum_name,
        area_of_interest=AreaOfInterest(
            west_lon_degree=w_lon,
            south_lat_degree=s_lat,
            east_lon_degree=e_lon,
            north_lat_degree=n_lat,
        ),
    )
    if not utm_list:
        raise ValueError(f"No UTM CRS found for lat=[s:{s_lat} - n:{n_lat}], lon=[w:{w_lon} - e:{e_lon}]")
    return f"{utm_list[0].auth_name}:{utm_list[0].code}"

def transform_df_to_utm(df, lat_col = "location-lat", lon_col = "location-long", src_crs = "EPSG:4326"):
    utm_epsg = pick_utm_epsg(
        s_lat=df[lat_col].min(),
        n_lat=df[lat_col].max(),
        w_lon=df[lon_col].min(),
        e_lon=df[lon_col].max(),
    )

    transformer = Transformer.from_crs(src_crs, utm_epsg, always_xy=True)
    x, y = transformer.transform(
        df[lon_col].to_numpy(),
        df[lat_col].to_numpy(),
    )

    out = df.copy()
    out["x"] = x
    out["y"] = y
    out["utm_epsg"] = utm_epsg
    return out

def dt_from_datetime(df, datetime_col="timestamp", id_col="tag-local-identifier"):
    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")

    if id_col is None:
        out = out.sort_values(datetime_col, kind="stable")
        out["dt"] = out[datetime_col].diff()
    else:
        out = out.sort_values([id_col, datetime_col], kind="stable")
        out["dt"] = out.groupby(id_col)[datetime_col].diff()

    out["dt_seconds"] = out["dt"].dt.total_seconds()
    return out

def dt_from_t(df, t_col="t", id_col="tag-local-identifier"): # Assumes t to be in seconds!!
    out = df.copy()
    if id_col is None:
        out = out.sort_values(t_col, kind="stable")
        out["dt_seconds"] = out[t_col].diff()
    else:
        out = out.sort_values([id_col, t_col], kind="stable")
        out["dt_seconds"] = out.groupby(id_col)[t_col].diff()
    return out

def segments_from_dt(df, dt_col="dt_seconds", id_col="tag-local-identifier", max_gap=20, ):
    out = df.copy()
    if id_col is None:
        out["segment"] = out[dt_col].gt(max_gap).cumsum()
    else:
        out["segment_local"] = (
            out[dt_col].gt(max_gap)
            .groupby(out[id_col])
            .cumsum()
            .astype(int)
        )

        out["segment"] = (
            out.groupby([id_col, "segment_local"], sort=False)
               .ngroup()
               .astype(int)
        )

        out = out.drop(columns="segment_local")
    return out

def reset_first_dt_in_segment(df, segment_col="segment", dt_col="dt", dt_seconds_col="dt_seconds"):
    out = df.copy()
    first = out.groupby(segment_col).cumcount() == 0

    if dt_col in out.columns:
        out.loc[first, dt_col] = pd.Timedelta(0)
    if dt_seconds_col in out.columns:
        out.loc[first, dt_seconds_col] = 0.0

    return out

def filter_min_segment_points(df, min_points=2, segment_col="segment"):
    out = df.copy()
    out["segment_n"] = out.groupby(segment_col)[segment_col].transform("size")
    out = out[out["segment_n"] >= min_points].copy()
    return out

def filter_min_segment_time(df, min_time=120, dt_col="dt_seconds", segment_col="segment"):
    out = df.copy()
    out["segment_duration"] = out.groupby(segment_col)[dt_col].transform("sum")
    return out[out["segment_duration"] >= min_time].copy()

def prepare_peruvian_boobies(input_path="data/peruvian_boobies/peruvian_boobies.csv", output_path="track_segments/peruvian_boobies"):
    print("--- Preparing gps tracks for peruvian boobies ---")

    df = pd.read_csv(input_path, na_values="NA", usecols=["location-lat", "location-long", "tag-local-identifier", "timestamp"])
    print("NA Values:")
    print(df.isna().any())
    print()

    df = transform_df_to_utm(df)
    df = dt_from_datetime(df)
    df = segments_from_dt(df, max_gap=120)
    df = reset_first_dt_in_segment(df)
    df = filter_min_segment_points(df, min_points=10)
    df = filter_min_segment_time(df)
    print()
    print(df)
    print(pd.unique(df["segment_n"]))

def prepare_jackals(input_path="data/jackals/jackal_data.csv", output_path="track_segments/jackals"):
    print("--- Preparing gps tracks for jackals ---")

    df = pd.read_csv(input_path, sep=",", na_values=["NA"], usecols=["lat", "lon", "TAG", "dateTime_local"])
    
    print("NA Values:")
    print(df.isna().any())
    print()
    df
    df = transform_df_to_utm(df, lat_col="lat", lon_col="lon")
    df = dt_from_datetime(df, datetime_col="dateTime_local", id_col="TAG")
    df = segments_from_dt(df, max_gap=20, id_col="TAG")
    df = reset_first_dt_in_segment(df)
    df = filter_min_segment_points(df, min_points=100)
    df = filter_min_segment_time(df)
    print()
    print(df)
    print(pd.unique(df["segment_n"]))

def prepare_spur_winged_lapwings(input_path="data/spur_winged_lapwings/spur_winged_lapwings1.csv", output_path="track_segments/peruvian_boobies"):
    print("--- Preparing gps tracks for spur winged lapwings ---")

    df = pd.read_csv(input_path, na_values="NA", usecols=["location-lat", "location-long", "tag-local-identifier", "timestamp"])
    print("NA Values:")
    print(df.isna().any())
    print()

    df = transform_df_to_utm(df)
    df = dt_from_datetime(df)
    df = segments_from_dt(df, max_gap=20)
    df = reset_first_dt_in_segment(df)
    df = filter_min_segment_points(df, min_points=100)
    df = filter_min_segment_time(df)
    print()
    print(df)
    print(pd.unique(df["segment_n"]))

if __name__ == "__main__":
    prepare_jackals()
    prepare_spur_winged_lapwings()