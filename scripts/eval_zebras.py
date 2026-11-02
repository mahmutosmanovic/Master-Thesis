"""Evaluate a trained drone policy on a local zebra GPS CSV, without a baseline."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .prepare_movebank import prepare


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


def read_run_config(run_dir):
    """Use the trained dimensions; do not load unrelated synthetic behavior files."""
    with open(run_dir/'config.yaml') as f:
        cfg = yaml.safe_load(f)
    if cfg['animal']['env']['count'] != 1:
        raise ValueError('Zebra CSV evaluation requires a run trained with one animal; '
                         'changing the animal count would change checkpoint dimensions.')
    if cfg['model']['space'].get('animal_features') != 12:
        raise ValueError('This environment requires 12 animal features, but the run uses '
                         'an older observation layout. Use a compatible checkpoint/environment; '
                         'do not simply change the saved feature count.')
    cfg['run_dir'] = str(Path(run_dir).resolve())
    return cfg


def write_plots(steps, episodes, out):
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
    ax.plot(first['animal_x'], first['animal_y'], label='Zebra replay')
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
    parser.add_argument('--csv', default='data/zebras/Z3864.csv')
    parser.add_argument('--run', help="Trained folder under runs/, or 'latest'")
    parser.add_argument('--weights', default='best')
    parser.add_argument('--episodes', type=int, default=10, help='Maximum distinct windows to evaluate')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--individual', help='Optional individual ID if CSV contains multiple animals')
    parser.add_argument('--max-gap', type=float, default=7200, help='Original GPS seconds before splitting a track')
    parser.add_argument('--time-scale', type=float, default=1, help='Recorded seconds per simulated second')
    parser.add_argument('--out', help='New or empty output directory')
    parser.add_argument('--prepare-only', action='store_true', help='Inspect/prepare CSV without loading a policy')
    parser.add_argument('--record', action='store_true', help='Also export video and snapshots of first episode (requires ffmpeg)')
    args = parser.parse_args()
    if not args.prepare_only and not args.run:
        parser.error('--run is required unless --prepare-only is used')
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    source = Path(args.csv).resolve()
    if not source.is_file():
        parser.error(f'CSV not found: {source}')
    out = Path(args.out or ('evals/zebras/'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f'Output directory must be empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cfg = None
    if not args.prepare_only:
        from .run_utils import resolve_run_dir
        run_dir = resolve_run_dir(args.run)
        cfg = read_run_config(run_dir)
    duration = float(cfg['dt'])*int(cfg['max_episode_steps']) if cfg else 204.8
    report = prepare(source, out/'tracks', individual=args.individual,
                     max_gap=args.max_gap, episode_seconds=duration, time_scale=args.time_scale)
    print(json.dumps(report, indent=2))
    print('GPS movement is linearly interpolated. Replay does not model zebra reactions to drones.')
    if args.time_scale != 1:
        print(f'TIME COMPRESSION: animal speeds multiplied by {args.time_scale:g}.')
    if args.prepare_only:
        print(f'Prepared tracks and sampling/trajectory report: {out}')
        return

    import torch
    from box import Box
    from environment import Env, MovementDim
    from environment.behaviors import REPLAY_CFG
    from .eval_models import init_agent, choose_action

    manifest = pd.read_parquet(out/'tracks/manifest.parquet')
    indices = np.linspace(0, len(manifest)-1, min(args.episodes, len(manifest)), dtype=int)
    cfg['animal']['init']['movement_dim'] = MovementDim.TWO_D
    cfg['animal']['init']['behavior'] = REPLAY_CFG(str(out/'tracks/manifest.parquet'))
    config = Box(cfg)
    agent, agent_type = init_agent(config, run_dir, args.weights)
    episode_rows, all_rows = [], []
    for episode, index in enumerate(indices):
        # A one-row manifest pins the chosen window even if reset retries occur.
        chosen = manifest.iloc[[index]].copy()
        chosen['path'] = chosen['path'].map(lambda p: str((out/'tracks'/p).resolve()))
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
                      'source_csv': str(source), 'source_start_time': str(chosen['start_time'].iloc[0]),
                      'source_end_time': str(chosen['end_time'].iloc[0]),
                      'time_scale': args.time_scale, 'track_lost': bool(loss),
                      'hard_safety_violation': bool(info['hard_safety_violation']),
                      'stop_reason': 'track_lost' if loss else 'safety' if info['hard_safety_violation'] else 'window_or_horizon_end'})
        episode_rows.append(stats)
        all_rows.extend(rows)
        # Save after every episode so completed results survive interruption.
        pd.DataFrame(episode_rows).to_csv(out/'episodes.csv', index=False)
        pd.DataFrame(all_rows).to_csv(out/'steps.csv', index=False)
        if args.record and episode == 0:
            env.viewer.close()
        print(f"Episode {episode+1}/{len(indices)}: visibility={stats['visible_fraction']:.3f}, "
              f"distance={stats['mean_distance_m']:.1f}m, reward/step={stats['mean_reward_per_step']:.4f}")

    episodes = pd.DataFrame(episode_rows)
    summary = {'run': str(run_dir), 'weights': args.weights, 'source_csv': str(source),
               'episodes': len(episodes), 'time_scale': args.time_scale,
               'track_loss_episodes': int(episodes['track_lost'].sum()),
               'mean_episode_reward_per_step': float(episodes['mean_reward_per_step'].mean()),
               'step_weighted_visible_fraction': float(np.mean([r['target_visible_fraction'] for r in all_rows])),
               'mean_distance_m': float(np.mean([r['mean_distance_m'] for r in all_rows])),
               'mean_disturbance': float(np.mean([r['disturbance'] for r in all_rows])),
               'interpolation': 'linear between GPS fixes; disturbance is model-based'}
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
