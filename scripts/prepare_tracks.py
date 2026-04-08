import os
import numpy as np
import pandas as pd
from pathlib import Path
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

def segment_t_from_dt(df, segment_col="segment", dt_seconds_col="dt_seconds"):
    out = df.copy()
    out[dt_seconds_col] = out[dt_seconds_col].fillna(0)
    out["segment_t"] = out.groupby(segment_col)[dt_seconds_col].cumsum()
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

def filter_min_segment_extent(df, min_extent=1.0, segment_col="segment", x_col="x", y_col="y"):
    out = df.copy()

    x_span = out.groupby(segment_col)[x_col].transform("max") - out.groupby(segment_col)[x_col].transform("min")
    y_span = out.groupby(segment_col)[y_col].transform("max") - out.groupby(segment_col)[y_col].transform("min")

    out["segment_x_span"] = x_span
    out["segment_y_span"] = y_span
    out["segment_extent"] = np.hypot(x_span, y_span)

    return out[out["segment_extent"] >= min_extent].copy()

def center_segment_coordinates(df, segment_col="segment", x_col="x", y_col="y"):
    out = df.copy()

    out[x_col] = out[x_col] - out.groupby(segment_col)[x_col].transform("mean")
    out[y_col] = out[y_col] - out.groupby(segment_col)[y_col].transform("mean")

    return out

def add_step_metrics(df, x_col="x", y_col="y", dt_col="dt_seconds", group_cols=None):
    out = df.copy()

    if group_cols is None:
        dx = out[x_col].diff()
        dy = out[y_col].diff()
    else:
        dx = out.groupby(group_cols)[x_col].diff()
        dy = out.groupby(group_cols)[y_col].diff()

    out["step_dist"] = np.hypot(dx, dy)
    out["speed_mps"] = out["step_dist"] / out[dt_col]

    bad_dt = out[dt_col].isna() | (out[dt_col] <= 0)
    out.loc[bad_dt, ["step_dist", "speed_mps"]] = np.nan
    return out

def split_segments_on_motion(
    df,
    id_col="TAG",
    segment_col="segment",
    dt_col="dt_seconds",
    speed_col="speed_mps",
    step_col="step_dist",
    max_speed=None,
    max_step=None,
):
    out = df.copy()

    bad = out[dt_col].isna() | (out[dt_col] <= 0)

    if max_speed is not None:
        bad |= out[speed_col] > max_speed
    if max_step is not None:
        bad |= out[step_col] > max_step

    # current row starts a new segment if the step into it is bad
    out["_motion_break"] = bad

    out["_subseg"] = (
        out["_motion_break"]
        .groupby([out[id_col], out[segment_col]])
        .cumsum()
        .astype(int)
    )

    out[segment_col] = (
        out.groupby([id_col, segment_col, "_subseg"], sort=False)
           .ngroup()
           .astype(int)
    )

    return out.drop(columns=["_motion_break", "_subseg"])

def save_segments_from_df(df, out_dir, segment_col="segment", t_col="segment_t", x_col="x", y_col="y", id_col="tag-local-identifier", time_col="timestamp"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for segment_id, g in df.groupby(segment_col, sort=False):
        t = g[t_col].to_numpy(dtype=np.float32)
        x = g[x_col].to_numpy(dtype=np.float32)
        y = g[y_col].to_numpy(dtype=np.float32)

        path = out_dir / f"{int(segment_id):08d}.npz"
        np.savez_compressed(path, t=t, x=x, y=y)

        row = {
            "segment": int(segment_id),
            "path": path.name,
            "n_points": int(len(g)),
            "duration_s": float(g["segment_duration"].iloc[0]) if "segment_duration" in g.columns else float(t[-1]),
            "xmin": float(np.min(x)),
            "xmax": float(np.max(x)),
            "ymin": float(np.min(y)),
            "ymax": float(np.max(y)),
        }

        if id_col in g.columns:
            row[id_col] = g[id_col].iloc[0]

        if time_col in g.columns:
            row["start_time"] = g[time_col].iloc[0]
            row["end_time"] = g[time_col].iloc[-1]

        if "segment_n" in g.columns:
            row["segment_n"] = int(g["segment_n"].iloc[0])

        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)

    return manifest

def plot_speed_binned(df, plot_dir):
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Keep finite, non-negative speeds
    speeds = df["speed_mps"].replace([np.inf, -np.inf], np.nan).dropna()
    speeds = speeds[speeds >= 0]

    # Optional: inspect binned counts
    bin_width = 5  # m/s
    max_speed = np.ceil(speeds.max() / bin_width) * bin_width
    bins = np.arange(0, max_speed + bin_width, bin_width)

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(speeds, bins=bins)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.title("Histogram of step speeds")
    plt.tight_layout()
    plt.savefig(plot_dir / "speed_histogram.png", dpi=200)
    plt.close()

def rolling_median_xy(
        df,
        x_col="x",
        y_col="y",
        segment_col="segment",
        id_col="tag-local-identifier",
        window=3,
    ):
        out = df.copy()
        group_cols = [segment_col] if id_col is None else [id_col, segment_col]

        out[x_col] = (
            out.groupby(group_cols, sort=False)[x_col]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
        )
        out[y_col] = (
            out.groupby(group_cols, sort=False)[y_col]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
        )
        return out

def kalman_filter_xy_segment(
        g,
        x_col="x",
        y_col="y",
        dt_col="dt_seconds",
        meas_var=25.0,
        accel_var=4.0,
    ):
        g = g.copy()

        xs = g[x_col].to_numpy(dtype=float)
        ys = g[y_col].to_numpy(dtype=float)
        dts = g[dt_col].fillna(0.0).to_numpy(dtype=float)

        n = len(g)
        if n == 0:
            return g

        # state = [x, y, vx, vy]
        state = np.array([xs[0], ys[0], 0.0, 0.0], dtype=float)
        P = np.diag([meas_var, meas_var, 25.0, 25.0]).astype(float)

        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        R = np.diag([meas_var, meas_var]).astype(float)

        xf = np.zeros(n, dtype=float)
        yf = np.zeros(n, dtype=float)

        xf[0] = xs[0]
        yf[0] = ys[0]

        for i in range(1, n):
            dt = float(dts[i])
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0

            F = np.array([
                [1.0, 0.0, dt,  0.0],
                [0.0, 1.0, 0.0, dt ],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float)

            q = accel_var
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt2 * dt2

            Q1 = np.array([
                [dt4 / 4.0, dt3 / 2.0],
                [dt3 / 2.0, dt2],
            ], dtype=float) * q

            Q = np.array([
                [Q1[0, 0], 0.0,      Q1[0, 1], 0.0     ],
                [0.0,      Q1[0, 0], 0.0,      Q1[0, 1]],
                [Q1[1, 0], 0.0,      Q1[1, 1], 0.0     ],
                [0.0,      Q1[1, 0], 0.0,      Q1[1, 1]],
            ], dtype=float)

            # predict
            state = F @ state
            P = F @ P @ F.T + Q

            # update
            z = np.array([xs[i], ys[i]], dtype=float)
            y = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            state = state + K @ y
            P = (np.eye(4) - K @ H) @ P

            xf[i] = state[0]
            yf[i] = state[1]

        g[x_col] = xf
        g[y_col] = yf
        return g

def kalman_filter_xy(
        df,
        x_col="x",
        y_col="y",
        dt_col="dt_seconds",
        segment_col="segment",
        id_col="tag-local-identifier",
        meas_var=25.0,
        accel_var=4.0,
    ):
        out = df.copy()
        group_cols = [segment_col] if id_col is None else [id_col, segment_col]

        parts = []
        for _, g in out.groupby(group_cols, sort=False):
            parts.append(
                kalman_filter_xy_segment(
                    g,
                    x_col=x_col,
                    y_col=y_col,
                    dt_col=dt_col,
                    meas_var=meas_var,
                    accel_var=accel_var,
                )
            )

        return pd.concat(parts, axis=0).sort_index()

def prepare_jackals(input_path="data/jackals/jackal_data.csv", out_dir="track_segments/jackals"):
    print("--- Preparing gps tracks for jackals ---")

    df = pd.read_csv(input_path, sep=",", na_values=["NA"], usecols=["lat", "lon", "TAG", "dateTime_local"])
    
    print("NA Values:")
    print(df.isna().any())
    print()
    df["dateTime_local"] = pd.to_datetime(df["dateTime_local"], errors="coerce")
    df = df.dropna(subset=["TAG", "dateTime_local", "lat", "lon"])

    df = transform_df_to_utm(df, lat_col="lat", lon_col="lon")
    df = dt_from_datetime(df, datetime_col="dateTime_local", id_col="TAG")
    df = segments_from_dt(df, max_gap=20, id_col="TAG")

    df = rolling_median_xy(df, x_col="x", y_col="y", segment_col="segment", id_col="TAG", window=3)
    df = kalman_filter_xy(df, x_col="x", y_col="y", dt_col="dt_seconds", segment_col="segment", id_col="TAG", meas_var=50.0, accel_var=3.0)

    df = add_step_metrics(df, group_cols=["TAG", "segment"])
    df = split_segments_on_motion(df, id_col="TAG", segment_col="segment", max_speed=20)

    df = reset_first_dt_in_segment(df)
    df = add_step_metrics(df, group_cols=["TAG", "segment"])

    df = filter_min_segment_points(df, min_points=100)
    df = filter_min_segment_time(df)
    df = filter_min_segment_extent(df, min_extent=100)
    df = segment_t_from_dt(df)

    report_dir = Path(out_dir).parent / "report" / Path(out_dir).name
    plot_speed_binned(df, report_dir)

    print(f"--- Saving segments to {out_dir} ---")
    save_segments_from_df(df, out_dir, id_col="TAG")
    print("--- Finished preparation for jackals ---")

def prepare_spur_winged_lapwings(input_path="data/spur_winged_lapwings/spur_winged_lapwings1.csv", out_dir="track_segments/spur_winged_lapwings"):
    print("--- Preparing gps tracks for spur winged lapwings ---")

    df = pd.read_csv(input_path, na_values="NA", usecols=["location-lat", "location-long", "tag-local-identifier", "timestamp"])
    print("NA Values:")
    print(df.isna().any())
    print()

    df = transform_df_to_utm(df)
    df = dt_from_datetime(df)
    df = segments_from_dt(df, max_gap=20)

    df = rolling_median_xy(df, window=3)
    df = kalman_filter_xy(df, dt_col="dt_seconds", meas_var=50.0, accel_var=8.0)

    df = add_step_metrics(df, group_cols=["tag-local-identifier", "segment"])
    df = split_segments_on_motion(df, id_col="tag-local-identifier", segment_col="segment", max_speed=100)

    df = reset_first_dt_in_segment(df)
    df = add_step_metrics(df, group_cols=["tag-local-identifier", "segment"])

    df = filter_min_segment_points(df, min_points=100)
    df = filter_min_segment_time(df)
    df = filter_min_segment_extent(df, min_extent=250)
    df = segment_t_from_dt(df)
    
    report_dir = Path(out_dir).parent / "report" / Path(out_dir).name
    plot_speed_binned(df, report_dir)

    print(f"--- Saving segments to {out_dir} ---")
    save_segments_from_df(df, out_dir)
    print("--- Finished preparation for spur winged lapwings ---")

def prepare_pigeons(input_path="data/pigeons", out_dir="track_segments/pigeons"):
    print("--- Preparing gps tracks for pigeons ---")

    dfs = []
    for i, path in enumerate(os.listdir(input_path)):
        df = pd.read_csv(os.path.join(input_path, path), na_values="NA", usecols=["lat", "lon", "t"])
        df["file_id"] = i
        dfs.append(df)
    df = pd.concat(dfs)

    print("NA Values:")
    df = df.dropna()
    print(df.isna().any())
    print()

    df = transform_df_to_utm(df, lat_col="lat", lon_col="lon")
    df = dt_from_t(df, id_col="file_id")
    df = segments_from_dt(df, max_gap=20, id_col="file_id")

    df = add_step_metrics(df, group_cols=["file_id", "segment"])
    df = split_segments_on_motion(df, id_col="file_id", segment_col="segment", max_speed=100) # pigeons have speeds matching their stated max

    df = reset_first_dt_in_segment(df)
    df = add_step_metrics(df, group_cols=["file_id", "segment"])

    df = filter_min_segment_points(df, min_points=100)
    df = filter_min_segment_time(df)
    df = filter_min_segment_extent(df, min_extent=250)
    df = segment_t_from_dt(df)
    
    report_dir = Path(out_dir).parent / "report" / Path(out_dir).name
    plot_speed_binned(df, report_dir)

    print(f"--- Saving segments to {out_dir} ---")
    save_segments_from_df(df, out_dir)
    print("--- Finished preparation for pigeons ---")

if __name__ == "__main__":
    prepare_pigeons()
    prepare_jackals()
    prepare_spur_winged_lapwings()