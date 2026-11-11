"""Evaluate a trained policy on high-rate drone zebra tracks (or a GPS CSV)."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .prepare_movebank import read_fixes


def summarize(rows, horizon_steps, dt):
    """Report actual exposure separately from reward per configured horizon."""
    reward = np.array([r['reward'] for r in rows], dtype=float)
    return {
        'steps': len(rows), 'simulated_seconds': len(rows)*dt,
        'total_reward': float(reward.sum()),
        'mean_reward_per_step': float(reward.mean()),
        'reward_per_horizon_step': float(reward.sum()/horizon_steps),
        'visible_fraction': float(np.mean([r['target_visible_fraction'] for r in rows])),
        'mean_distance_m': float(np.mean([r['mean_distance_m'] for r in rows])),
        'mean_disturbance': float(np.mean([r['disturbance'] for r in rows])),
        'max_disturbance': float(np.max([r['disturbance'] for r in rows])),
    }


def read_run_config(run_dir, animal_label='Zebra'):
    """Use the trained dimensions; do not load unrelated synthetic behavior files."""
    with open(run_dir/'config.yaml') as f:
        cfg = yaml.safe_load(f)
    if cfg['animal']['env']['count'] != 1:
        raise ValueError(f'{animal_label} evaluation requires a run trained with one animal; '
                         'changing the animal count would change checkpoint dimensions.')
    if cfg['model']['space'].get('animal_features') != 12:
        raise ValueError('This environment requires 12 animal features, but the run uses '
                         'an older observation layout. Use a compatible checkpoint/environment; '
                         'do not simply change the saved feature count.')
    cfg['run_dir'] = str(Path(run_dir).resolve())
    return cfg


def check_fix_capacity(fixes, id_col, episodes, steps, dt):
    """Count full time windows without joining different individuals."""
    if episodes < 1 or steps < 1:
        raise ValueError('episodes and steps must be positive')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    horizon = steps * dt
    available = sum(
        int(np.floor((animal['timestamp'].iloc[-1] -
                      animal['timestamp'].iloc[0]).total_seconds() / horizon + 1e-9))
        for _, animal in fixes.groupby(id_col, sort=False) if len(animal) >= 2)
    if episodes > available:
        raise ValueError(
            f'Not enough recorded duration: requested {episodes} episodes of '
            f'{horizon:g} seconds, but only {available} complete episodes fit. '
            'Capacity uses timestamps within continuous tracks; separate animals or gaps are never joined.')
    return available


def prepare_sequential(fixes, id_col, out_dir, episodes, steps, dt):
    """Interpolate consecutive time windows at dt, preserving GPS timing."""
    available = check_fix_capacity(fixes, id_col, episodes, steps, dt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    horizon = steps * dt
    for identifier, animal in fixes.groupby(id_col, sort=False):
        if len(animal) < 2:
            continue
        times = (animal['timestamp'] - animal['timestamp'].iloc[0]).dt.total_seconds().to_numpy()
        if {'x_m', 'y_m'}.issubset(animal.columns):
            source_x, source_y = animal['x_m'].to_numpy(), animal['y_m'].to_numpy()
            epsg = None
        else:
            from pyproj import Transformer
            lon, lat = animal['location-long'].to_numpy(), animal['location-lat'].to_numpy()
            zone = min(60, max(1, int((np.median(lon) + 180) // 6) + 1))
            epsg = (32600 if np.median(lat) >= 0 else 32700) + zone
            transformer = Transformer.from_crs(4326, epsg, always_xy=True)
            source_x, source_y = transformer.transform(lon, lat)
        source_x, source_y = source_x - source_x[0], source_y - source_y[0]
        count = int(np.floor(times[-1] / horizon + 1e-9))
        for window in range(count):
            start = window * horizon
            elapsed = np.arange(steps + 1, dtype=float) * dt
            query = start + elapsed
            x = np.interp(query, times, source_x)
            y = np.interp(query, times, source_y)
            x, y = x - x[0], y - y[0]
            path = (out_dir / f'episode_{len(rows):03d}.npz').resolve()
            np.savez_compressed(path, t=elapsed, x=x, y=y)
            rows.append({
                'path': str(path), id_col: identifier, 'n_points': steps + 1,
                'duration_s': horizon,
                'individual': str(animal['individual'].iloc[0]) if 'individual' in animal else str(identifier),
                'source_start_seconds': (float(animal['source_time_s'].iloc[0]) + start
                                         if 'source_time_s' in animal else start),
                'source_end_seconds': (float(animal['source_time_s'].iloc[0]) + start + horizon
                                       if 'source_time_s' in animal else start + horizon),
                'start_time': animal['timestamp'].iloc[0] + pd.to_timedelta(start, unit='s'),
                'end_time': animal['timestamp'].iloc[0] + pd.to_timedelta(start + horizon, unit='s'),
                'source_start_index': max(0, int(np.searchsorted(times, start, side='right')) - 1),
                'source_end_index': min(len(times) - 1, int(np.searchsorted(times, start + horizon))),
                'utm_epsg': epsg, 'xmin': float(x.min()), 'xmax': float(x.max()),
                'ymin': float(y.min()), 'ymax': float(y.max()),
            })
            if len(rows) == episodes:
                manifest = pd.DataFrame(rows)
                manifest.to_parquet(out_dir/'manifest.parquet', index=False)
                return manifest, available


def write_plots(steps, episodes, out, animal_label='Zebra'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, title in zip(axes, ['mean_reward_per_step', 'visible_fraction', 'mean_disturbance'],
                             ['Mean reward / executed step', 'Target visible fraction', 'Model disturbance']):
        ax.bar(episodes['episode'], episodes[col])
        ax.set(title=title, xlabel='Episode')
    fig.tight_layout()
    fig.savefig(out/'performance.png', dpi=160)
    plt.close(fig)
    first = steps[steps['episode'] == 0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(first['animal_x'], first['animal_y'], label=f'{animal_label} replay')
    for drone in range(int(first['drone_count'].iloc[0])):
        ax.plot(first[f'drone_{drone}_x'], first[f'drone_{drone}_y'], label=f'Drone {drone}')
    ax.set(xlabel='Local east (m)', ylabel='Local north (m)', title='First evaluated trajectory')
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'drone_trajectory.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--csv', help='Optional legacy Movebank GPS CSV')
    sources.add_argument('--tracks-file', help='Local cleaned SLEAP HDF5; default downloads pinned drone dataset')
    parser.add_argument('--body-length-m', type=float, default=2.0,
                        help='Assumed median head-to-tailbase length in metres (default 2.0)')
    parser.add_argument('--max-track-gap', type=float, default=0.5,
                        help='Split drone tracks across missing-data gaps longer than this many seconds')
    parser.add_argument('--run', help="Trained folder under runs/, or 'latest'")
    parser.add_argument('--weights', default='best')
    parser.add_argument('--episodes', type=int, default=1, help='Number of consecutive full time windows to evaluate')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--individual', help='Zebra track ID (e.g. track_25), or individual ID for GPS CSV')
    parser.add_argument('--out', help='New or empty output directory')
    parser.add_argument('--prepare-only', action='store_true', help='Inspect usable fixes without loading a policy')
    parser.add_argument('--record', action='store_true', help='Also export video and snapshots of first episode (requires ffmpeg)')
    args = parser.parse_args()
    if not args.prepare_only and not args.run:
        parser.error('--run is required unless --prepare-only is used')
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    out = Path(args.out or ('evals/zebras/'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f'Output directory must be empty: {out}')
    try:
        if args.csv:
            source = Path(args.csv).resolve()
            if not source.is_file():
                parser.error(f'CSV not found: {source}')
            fixes, id_col, input_rows = read_fixes(source, args.individual)
            source_info = {'dataset': 'Movebank GPS CSV', 'input_rows': input_rows,
                           'usable_rows': len(fixes), 'individuals': int(fixes[id_col].nunique()),
                           'replay_timing': 'gps_timestamp_aligned_linear_interpolation'}
        else:
            from .zebra_drone_data import fetch_source, read_drone_tracks
            source = Path(args.tracks_file).resolve() if args.tracks_file else fetch_source()
            fixes, id_col, source_info = read_drone_tracks(
                source, args.individual, args.body_length_m, args.max_track_gap)
            source_info['replay_timing'] = 'video_frame_aligned_linear_interpolation'
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(source_info, indent=2))
    if args.prepare_only:
        print('Dataset inspected; no model loaded.')
        return

    from .run_utils import resolve_run_dir
    run_dir = resolve_run_dir(args.run)
    cfg = read_run_config(run_dir)
    steps, dt = int(cfg['max_episode_steps']), float(cfg['dt'])
    try:
        available = check_fix_capacity(fixes, id_col, args.episodes, steps, dt)
    except ValueError as exc:
        parser.error(str(exc))
    # Capacity is checked before creating outputs, preparing tracks or loading weights.
    manifest, available = prepare_sequential(
        fixes, id_col, out/'tracks', args.episodes, steps, dt)
    print(f'{len(fixes)} usable points; {available} complete episodes available.')
    (out/'source.json').write_text(json.dumps(source_info, indent=2) + '\n')
    print(f'Sequential timed replay at dt={dt:g} seconds; no time compression. '
          'Replay animals do not react to drones. '
          'Drone-derived metric scale is an explicit body-length assumption.')

    import torch
    from box import Box
    from environment import Env, MovementDim
    from environment.behaviors import REPLAY_CFG
    from .eval_models import init_agent, choose_action

    cfg['animal']['init']['movement_dim'] = MovementDim.TWO_D
    cfg['animal']['init']['behavior'] = REPLAY_CFG(str(out/'tracks/manifest.parquet'))
    config = Box(cfg)
    agent, agent_type = init_agent(config, run_dir, args.weights)
    episode_rows, all_rows = [], []
    for episode in range(args.episodes):
        # Each block is fixed in advance, including after early termination.
        index = episode
        chosen = manifest.iloc[[episode]].copy()
        # Pin this block even if the environment retries its reset.
        selected_manifest = out/f'episode_{episode:03d}_manifest.parquet'
        chosen.to_parquet(selected_manifest, index=False)
        ep_cfg = deepcopy(config)
        ep_cfg.animal.init.behavior = REPLAY_CFG(str(selected_manifest), selection='cycle')
        env = Env(ep_cfg, seed=args.seed+episode)
        env.enable_step_logging = True
        if args.record and episode == 0:
            from .play import PaperViewer
            env.viewer = PaperViewer(ep_cfg, run_name=run_dir.name, seed=args.seed)
            env.viewer.set_output_dir(out/'recording')
            env.render_mode = 'human'
        obs, _ = env.reset(seed=args.seed+episode)
        rows = []
        limit = min(int(config.max_episode_steps), max(1, int(np.ceil(float(chosen['duration_s'].iloc[0])/float(config.dt)))))
        for step in range(1, limit+1):
            with torch.no_grad():
                action = choose_action(agent, obs, agent_type)
            obs, reward, terminated, truncated, info = env.step(action)
            animal = env.animals[0]
            distances = [float(np.linalg.norm(d.pos.to_numpy()-animal.pos.to_numpy())) for d in env.drones]
            row = {'episode': episode, 'window_index': int(index), 'step': step,
                   'time_s': step*float(config.dt), 'reward': float(reward),
                   'target_visible_fraction': float(info['target_visible_fraction']),
                   'disturbance': float(animal.disturbance),
                   'mean_distance_m': float(np.mean(distances)), 'drone_count': len(env.drones),
                   'animal_x': float(animal.pos.x), 'animal_y': float(animal.pos.y),
                   'hard_safety_violation': bool(info['hard_safety_violation'])}
            for i, d in enumerate(env.drones):
                row.update({f'drone_{i}_x': float(d.pos.x), f'drone_{i}_y': float(d.pos.y),
                            f'drone_{i}_z': float(d.pos.z), f'drone_{i}_distance_m': distances[i]})
            rows.append(row)
            if args.record and episode == 0:
                env.viewer.log_step_metrics(reward, info)
            if terminated or truncated:
                break
        stats = summarize(rows, int(config.max_episode_steps), float(config.dt))
        loss = (env.lost_track_counter >= env.track_loss_grace or
                np.any(env.lost_track_counters >= env.track_loss_grace))
        stats.update({'episode': episode, 'window_index': int(index), 'seed': args.seed+episode,
                      'source_file': str(source), 'source_start_time': str(chosen['start_time'].iloc[0]),
                      'source_end_time': str(chosen['end_time'].iloc[0]),
                      'individual': str(chosen['individual'].iloc[0]),
                      'source_start_seconds': float(chosen['source_start_seconds'].iloc[0]),
                      'source_end_seconds': float(chosen['source_end_seconds'].iloc[0]),
                      'replay_timing': source_info['replay_timing'],
                      'source_start_index': int(chosen['source_start_index'].iloc[0]),
                      'source_end_index': int(chosen['source_end_index'].iloc[0]),
                      'track_lost': bool(loss),
                      'hard_safety_violation': bool(info['hard_safety_violation']),
                      'stop_reason': 'track_lost' if loss else 'safety' if info['hard_safety_violation'] else 'window_or_horizon_end'})
        episode_rows.append(stats)
        all_rows.extend(rows)
        # Save after every episode so completed results survive interruption.
        pd.DataFrame(episode_rows).to_csv(out/'episodes.csv', index=False)
        pd.DataFrame(all_rows).to_csv(out/'steps.csv', index=False)
        if args.record and episode == 0:
            env.viewer.close()
        print(f"Episode {episode+1}/{args.episodes}: visibility={stats['visible_fraction']:.3f}, "
              f"distance={stats['mean_distance_m']:.1f}m, reward/step={stats['mean_reward_per_step']:.4f}")

    episodes = pd.DataFrame(episode_rows)
    summary = {'run': str(run_dir), 'weights': args.weights, 'source_file': str(source),
               'episodes': len(episodes), 'replay_timing': source_info['replay_timing'],
               'track_loss_episodes': int(episodes['track_lost'].sum()),
               'mean_episode_reward_per_step': float(episodes['mean_reward_per_step'].mean()),
               'step_weighted_visible_fraction': float(np.mean([r['target_visible_fraction'] for r in all_rows])),
               'mean_distance_m': float(np.mean([r['mean_distance_m'] for r in all_rows])),
               'mean_disturbance': float(np.mean([r['disturbance'] for r in all_rows])),
               'interpolation': 'linear at recorded timestamps/frame times; no time compression',
               'available_episodes': available, 'usable_points': len(fixes), 'source': source_info,
               'disturbance': 'model-based; replay animals do not react'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    snapshot = deepcopy(cfg)
    snapshot['animal']['init']['movement_dim'] = 'TWO_D'
    snapshot['animal']['init']['behavior'] = 'REPLAY_CFG'
    snapshot['replay_manifest'] = str(out/'tracks/manifest.parquet')
    (out/'config.yaml').write_text(yaml.safe_dump(snapshot))
    write_plots(pd.DataFrame(all_rows), episodes, out)
    print(json.dumps(summary, indent=2))
    print(f'Results saved to: {out}')


if __name__ == '__main__':
    main()

