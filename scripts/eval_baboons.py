"""Evaluate a trained policy on the first 2,048 GPS fixes of one baboon."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .baboon_data import DEFAULT_CSV, select_fixes
from .eval_zebras import summarize, read_run_config, check_fix_capacity, prepare_sequential, write_plots


def resolve_model(run=None, model_path=None, weights='best'):
    """Resolve a run directory or exact checkpoint without altering its config."""
    if model_path:
        checkpoint = Path(model_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        run_dir = checkpoint.parent
    else:
        direct = Path(run).expanduser()
        if direct.is_dir():
            run_dir = direct.resolve()
        else:
            from .run_utils import resolve_run_dir
            run_dir = resolve_run_dir(run).resolve()
        with open(run_dir/'config.yaml') as f:
            cfg = yaml.safe_load(f)
        checkpoint = run_dir / f"{cfg['agent_type']}_{weights}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    if not (run_dir/'config.yaml').is_file():
        raise FileNotFoundError(f'Config not found: {run_dir / "config.yaml"}')
    return run_dir, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=str(DEFAULT_CSV), help='Movebank baboon CSV')
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--run', help='Trained run directory, folder under runs/, or latest')
    models.add_argument('--model-path', help='Exact policy checkpoint; config.yaml must be alongside it')
    parser.add_argument('--max-gap', type=float, default=2.0,
                        help='Reject gaps above this many seconds in the first 2048 fixes (default 2)')
    parser.add_argument('--weights', default='best')
    parser.add_argument('--episodes', type=int, default=1, help='Consecutive windows within the selected 2048 fixes (default 1)')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--individual', help='Baboon local identifier; otherwise choose the first eligible ID in file order')
    parser.add_argument('--out', help='New or empty output directory')
    parser.add_argument('--prepare-only', action='store_true', help='Export the selected 2048 fixes and metadata without loading a policy')
    parser.add_argument('--record', action='store_true', help='Also export video and snapshots of first episode (requires ffmpeg)')
    args = parser.parse_args()
    if not args.prepare_only and not (args.run or args.model_path):
        parser.error('--run or --model-path is required unless --prepare-only is used')
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    out = Path(args.out or ('evals/baboons/'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f'Output directory must be empty: {out}')
    try:
        source = Path(args.csv).expanduser().resolve()
        fixes, id_col, source_info = select_fixes(source, args.individual, args.max_gap)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(source_info, indent=2))
    if args.prepare_only:
        out.mkdir(parents=True, exist_ok=True)
        fixes.to_csv(out/'selected_fixes.csv', index=False)
        (out/'source.json').write_text(json.dumps(source_info, indent=2) + '\n')
        print(f'Selected fixes saved to: {out}; no model loaded.')
        return

    try:
        run_dir, checkpoint = resolve_model(args.run, args.model_path, args.weights)
        cfg = read_run_config(run_dir, animal_label='Baboon')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    steps, dt = int(cfg['max_episode_steps']), float(cfg['dt'])
    try:
        available = check_fix_capacity(fixes, id_col, args.episodes, steps, dt)
    except ValueError as exc:
        parser.error(str(exc))
    # Capacity is checked before creating outputs, preparing tracks or loading weights.
    manifest, available = prepare_sequential(
        fixes, id_col, out/'tracks', args.episodes, steps, dt)
    fixes.to_csv(out/'selected_fixes.csv', index=False)
    print(f'{len(fixes)} usable points; {available} complete episodes available.')
    (out/'source.json').write_text(json.dumps(source_info, indent=2) + '\n')
    print(f'Sequential timed replay at dt={dt:g} seconds; no time compression. '
          'Replay animals do not react to drones. '
          'Coordinates are projected to local UTM metres.')

    import torch
    from box import Box
    from environment import Env, MovementDim
    from environment.behaviors import REPLAY_CFG
    from .eval_models import init_agent_actor_only, choose_action

    cfg['animal']['init']['movement_dim'] = MovementDim.TWO_D
    cfg['animal']['init']['behavior'] = REPLAY_CFG(str(out/'tracks/manifest.parquet'))
    config = Box(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent, agent_type, _ = init_agent_actor_only(config, run_dir, checkpoint.name)
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
    summary = {'run': str(run_dir), 'weights': str(checkpoint), 'source_file': str(source),
               'episodes': len(episodes), 'replay_timing': source_info['replay_timing'],
               'track_loss_episodes': int(episodes['track_lost'].sum()),
               'mean_episode_reward_per_step': float(episodes['mean_reward_per_step'].mean()),
               'step_weighted_visible_fraction': float(np.mean([r['target_visible_fraction'] for r in all_rows])),
               'mean_distance_m': float(np.mean([r['mean_distance_m'] for r in all_rows])),
               'mean_disturbance': float(np.mean([r['disturbance'] for r in all_rows])),
               'interpolation': 'linear at recorded timestamps/frame times; no time compression',
               'available_episodes': available, 'usable_points': len(fixes), 'source': source_info,
               'configured_horizon_steps': steps, 'dt': dt,
               'disturbance': 'model-based; replay animals do not react'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    snapshot = deepcopy(cfg)
    snapshot['animal']['init']['movement_dim'] = 'TWO_D'
    snapshot['animal']['init']['behavior'] = 'REPLAY_CFG'
    snapshot['replay_manifest'] = str(out/'tracks/manifest.parquet')
    (out/'config.yaml').write_text(yaml.safe_dump(snapshot))
    write_plots(pd.DataFrame(all_rows), episodes, out, animal_label='Baboon')
    print(json.dumps(summary, indent=2))
    print(f'Results saved to: {out}')


if __name__ == '__main__':
    main()

