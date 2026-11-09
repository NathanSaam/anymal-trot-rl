"""
PPO Training Script for Quadruped Locomotion

Stable-Baselines3 PPO training, paired policy/normalization checkpoints, and
deterministic evaluation of speed, contact timing, slip, and vertical motion.
"""

import os
import argparse
import json
import time
from collections import defaultdict
from typing import Callable

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("Please install stable-baselines3: pip install stable-baselines3")
    exit(1)

from anymal_env_improved import AnymalWalkingEnvImproved


class CustomLogCallback(BaseCallback):
    """
    Clean, minimal logging callback.
    Prints: Episode count, Average Distance, Average Reward, Average Episode Length, Loss
    """

    def __init__(self, print_every_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every_episodes
        
        # Episode buffers
        self.ep_rewards = []
        self.ep_rewards_raw = []
        self.ep_distances = []
        self.ep_lengths = []
        self.ep_max_velocities = []

        # -- ADDED: Full history for plotting --
        self.all_ep_rewards = []
        self.all_ep_rewards_raw = []
        self.all_ep_lengths = []
        self.all_ep_velocities = []
        self.all_ep_distances = []
        
        # Current episode accumulators (per env, but we track env 0)
        self.current_ep_reward = 0.0
        self.current_ep_reward_raw = 0.0
        self.current_ep_x_vels = []
        self.current_ep_length = 0
        
        self.num_episodes = 0
        self.total_timesteps = 0

    def _on_step(self) -> bool:
        self.total_timesteps += self.training_env.num_envs
        
        # Track env 0 for logging
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        r = float(rewards[0])
        raw_r = r
        if hasattr(self.training_env, "get_original_reward"):
            raw_r = float(self.training_env.get_original_reward()[0])
        done = bool(dones[0])
        info = infos[0]

        self.current_ep_reward += r
        self.current_ep_reward_raw += raw_r
        self.current_ep_x_vels.append(float(info.get("x_vel", 0.0)))
        self.current_ep_length += 1

        if done:
            self.num_episodes += 1
            
            # Store episode stats
            self.ep_rewards.append(self.current_ep_reward)
            self.ep_rewards_raw.append(self.current_ep_reward_raw)
            self.ep_lengths.append(self.current_ep_length)
            
            # Actual world-x displacement from reset; no assumed control rate.
            dist = float(info.get("x_pos", 0.0))
            self.ep_distances.append(dist)

            # -- ADDED: Calculate average velocity for tracking accuracy --
            avg_vel = sum(self.current_ep_x_vels) / len(self.current_ep_x_vels) if self.current_ep_x_vels else 0.0
            
            # Track max velocity achieved
            max_vel = 0.0
            if self.current_ep_x_vels:
                max_vel = max(self.current_ep_x_vels)
                self.ep_max_velocities.append(max_vel)
            
            # -- ADDED: Save to history for plotting --
            self.all_ep_rewards.append(self.current_ep_reward)
            self.all_ep_rewards_raw.append(self.current_ep_reward_raw)
            self.all_ep_lengths.append(self.current_ep_length)
            self.all_ep_velocities.append(avg_vel)
            self.all_ep_distances.append(dist)
            
            # Print stats periodically
            if self.num_episodes % self.print_every == 0:
                self._print_stats()

            # Reset accumulators
            self.current_ep_reward = 0.0
            self.current_ep_reward_raw = 0.0
            self.current_ep_x_vels = []
            self.current_ep_length = 0

        return True

    def _print_stats(self):
        k = min(self.print_every, len(self.ep_rewards))
        if k == 0:
            return
            
        mean_rew_raw = sum(self.ep_rewards_raw[-k:]) / k
        mean_dist = sum(self.ep_distances[-k:]) / k
        mean_len = sum(self.ep_lengths[-k:]) / k
        mean_max_vel = sum(self.ep_max_velocities[-k:]) / k if self.ep_max_velocities else 0.0
        
        # Get loss from logger if available
        loss_str = "N/A"
        if self.logger and hasattr(self.logger, 'name_to_value'):
            nv = self.logger.name_to_value
            if "train/loss" in nv:
                loss_str = f"{nv['train/loss']:.4f}"
            elif "train/policy_gradient_loss" in nv:
                loss_str = f"{nv['train/policy_gradient_loss']:.4f}"
        
        # Format timesteps nicely
        ts_str = f"{self.total_timesteps:,}".replace(",", "_")
        
        print(f"Ep {self.num_episodes:5d} | "
              f"avg_dist: {mean_dist:6.2f}m | "
              f"R: {mean_rew_raw:8.2f} | "
              f"ep_len: {mean_len:6.1f} | "
              f"loss: {loss_str}")

    def save_plots(self, log_dir):
        """Generates and saves dot-plot graphs of training history."""
        if not self.all_ep_rewards:
            print("No episodes recorded, skipping plots.")
            return

        episodes = range(1, len(self.all_ep_rewards) + 1)

        # 2x2 grid, smaller figure
        fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        (ax1, ax2), (ax3, ax4) = axs

        # 1. Rewards (blue)
        ax1.plot(episodes, self.all_ep_rewards_raw,
                'o', markersize=2.5, alpha=0.6, color='blue')
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Rewards per Episode")
        ax1.grid(True, alpha=0.3)

        # 2. Episode Length (green + red line)
        ax2.plot(episodes, self.all_ep_lengths,
                'o', markersize=2.5, alpha=0.6, color='green')
        ax2.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label="Max (1000)")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Length (Survival)")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Speed Tracking (purple + orange target)
        ax3.plot(episodes, self.all_ep_velocities,
                'o', markersize=2.5, alpha=0.6, color='purple')
        ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5,
                    label="Target (0.5 m/s)")
        ax3.set_ylabel("Avg Speed (m/s)")
        ax3.set_title("Speed Tracking Accuracy")
        ax3.legend(loc="lower right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Distance (teal)
        ax4.plot(episodes, self.all_ep_distances,
                'o', markersize=2.5, alpha=0.6, color='teal')
        ax4.set_ylabel("Distance (m)")
        ax4.set_xlabel("Episode")
        ax4.set_title("Total Distance Traveled")
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        save_path = os.path.join(log_dir, "training_metrics.png")
        fig.savefig(save_path, dpi=100)

        plt.close(fig)

        print(f"\n[Graph] Training metrics saved to: {save_path}")


class GaitMetricsCallback(BaseCallback):
    """Report physical gait metrics and weighted reward terms across every env."""
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)

    def _on_step(self):
        for info in self.locals["infos"]:
            for key in ("x_vel", "gait_match", "flight", "diagonal_support", "foot_slip"):
                if key in info:
                    self.metrics[f"gait/{key}"].append(info[key])
            if "z_vel" in info:
                self.metrics["gait/vertical_velocity_squared"].append(info["z_vel"] ** 2)
            for key, value in info.get("reward_terms", {}).items():
                self.metrics[f"rewards/{key}"].append(value)
        return True

    def _on_rollout_end(self):
        for key, values in self.metrics.items():
            self.logger.record(key, float(np.mean(values)))
        if self.metrics["gait/gait_match"]:
            print(f"Steps {self.num_timesteps:,} | "
                  f"speed {np.mean(self.metrics['gait/x_vel']):.2f} m/s | "
                  f"gait match {np.mean(self.metrics['gait/gait_match']):.1%} | "
                  f"flight {np.mean(self.metrics['gait/flight']):.1%}")
        self.metrics.clear()


def make_env(model_path: str, rank: int = 0, seed: int = 0,
             gait: str = "trot") -> Callable[[], gym.Env]:
    """Create environment factory function."""
    def _init():
        env = AnymalWalkingEnvImproved(model_path, render_mode=None, gait=gait)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def train(args):
    """Train PPO agent on quadruped walking task."""
    
    os.makedirs(args.log_dir, exist_ok=True)
    config_path = os.path.join(args.log_dir, "run_config.json")
    if os.path.exists(config_path) or os.path.exists(os.path.join(args.log_dir, "ppo_anymal.zip")):
        raise FileExistsError("Choose a fresh --log-dir to preserve the previous run")
    with open(config_path, "w") as f:
        json.dump(dict(vars(args), env_version=AnymalWalkingEnvImproved.ENV_VERSION), f, indent=2)
    import torch
    torch.set_num_threads(1)
    
    print("\n" + "=" * 60)
    print("  Anymal Quadruped Training")
    print("=" * 60)
    print(f"  Log dir:     {args.log_dir}")
    print(f"  Num envs:    {args.num_envs}")
    print(f"  Timesteps:   {args.total_timesteps:,}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size}")
    print("=" * 60 + "\n")
    
    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(args.model, i, args.seed, args.gait)
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([make_env(args.model, 0, args.seed, args.gait)])
    env.seed(args.seed)
    
    # Normalize observations and rewards - CRITICAL for stable training
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,  # Disable default logging
        tensorboard_log=args.log_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=get_activation_fn("elu"),
            log_std_init=-0.5,
        ),
    )
    
    # Callbacks
    custom_log = CustomLogCallback(print_every_episodes=args.log_every)
    
    # Train!
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[custom_log, GaitMetricsCallback(), CheckpointCallback(
                save_freq=max(250_000 // args.num_envs, 1),
                save_path=os.path.join(args.log_dir, "checkpoints"),
                name_prefix="anymal", save_vecnormalize=True,
            )],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save model (same structure as original)
    model_path = os.path.join(args.log_dir, "ppo_anymal")
    vecnorm_path = os.path.join(args.log_dir, "vec_normalize.pkl")
    model.save(model_path)
    env.save(vecnorm_path)
    print(f"\nSaved model to: {model_path}")
    
    # -- ADDED: Save Plots --
    custom_log.save_plots(args.log_dir)
    
    print("Done.")
    
    env.close()


def linear_schedule(initial_value: float):
    """Linear schedule for learning rate or clip range."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def get_activation_fn(name: str):
    """Get activation function by name."""
    import torch.nn as nn
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    return activations.get(name.lower(), nn.ELU)


def load_policy_env(model_xml, checkpoint, render_mode=None):
    """Load the gait configuration and matching frozen normalization statistics."""
    from pathlib import Path
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise ValueError("Missing run_config.json. Old 50-observation policies must be retrained; "
                         "use a checkpoint from the new gait environment.")
    config = json.loads(config_path.read_text())
    if config.get("env_version") != AnymalWalkingEnvImproved.ENV_VERSION:
        raise ValueError("Checkpoint environment version does not match; train a fresh policy")
    norm_path = (checkpoint.with_name(checkpoint.stem.replace("anymal_", "anymal_vecnormalize_", 1)
                                     + ".pkl") if checkpoint.parent.name == "checkpoints"
                 else run_dir / "vec_normalize.pkl")
    if not norm_path.exists():
        raise FileNotFoundError(f"Matching normalization statistics missing: {norm_path}")
    raw_env = AnymalWalkingEnvImproved(model_xml, render_mode=render_mode, gait=config["gait"])
    vec = DummyVecEnv([lambda: raw_env])
    try:
        vec = VecNormalize.load(str(norm_path), vec)
        vec.training, vec.norm_reward = False, False
        policy = PPO.load(str(checkpoint), env=vec, device="cpu")
    except Exception:
        vec.close()
        raise
    return policy, vec, raw_env


def play(args):
    """Deterministic playback or headless evaluation with contact traces."""
    headless = args.command == "evaluate"
    policy, env, raw_env = load_policy_env(args.model, args.checkpoint,
                                          None if headless else "human")
    env.seed(args.seed)
    records, summaries = [], []
    try:
        obs = env.reset()
        if not headless:
            import mujoco
            raw_env._render_frame()
            raw_env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            raw_env.viewer.cam.trackbodyid = raw_env.base_body_id
            raw_env.viewer.cam.distance = 3.0
        for episode in range(args.num_episodes):
            episode_records, episode_reward = [], 0.0
            while True:
                start = time.monotonic()
                action, _ = policy.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                info = infos[0]
                if "error" in info:
                    raise RuntimeError(f"Simulation failed during evaluation: {info['error']}")
                episode_reward += float(rewards[0])
                row = {k: info[k] for k in ("x_vel", "z_vel", "gait_match", "flight",
                                            "diagonal_support", "foot_slip")}
                row.update(episode=episode, time=info["episode_length"] * raw_env.control_dt,
                           contacts=info["foot_contacts"].tolist(),
                           clearance=info["foot_clearance"].tolist())
                episode_records.append(row)
                if not headless:
                    time.sleep(max(0, raw_env.control_dt - (time.monotonic() - start)))
                if dones[0]:
                    break  # VecEnv already reset; keep its next observation.
            summary = {key: float(np.mean([r[key] for r in episode_records]))
                       for key in ("x_vel", "gait_match", "flight", "diagonal_support", "foot_slip")}
            summary.update(episode=episode + 1, reward=episode_reward,
                           distance=info["x_pos"], steps=len(episode_records),
                           vertical_velocity_rms=float(np.sqrt(np.mean(
                               [r["z_vel"] ** 2 for r in episode_records]))),
                           completed=bool(info.get("TimeLimit.truncated", False)))
            summaries.append(summary)
            records.extend(episode_records)
            print(f"Episode {episode + 1}: {summary['x_vel']:.2f} m/s, "
                  f"distance {summary['distance']:.2f} m, steps {summary['steps']}, "
                  f"gait match {summary['gait_match']:.1%}, flight {summary['flight']:.1%}, "
                  f"vertical RMS {summary['vertical_velocity_rms']:.3f} m/s")
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        env.close()
    if headless and records:
        from pathlib import Path
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(
            dict(checkpoint=args.checkpoint, gait=raw_env.gait, seed=args.seed,
                 episodes=summaries, trace=records), indent=2))
        first = [r for r in records if r["episode"] == 0]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        t = [r["time"] for r in first]
        axes[0].imshow(np.array([r["contacts"] for r in first]).T, aspect="auto",
                       interpolation="nearest", origin="lower", vmin=0, vmax=1,
                       extent=[0, t[-1], -0.5, 3.5], cmap="Greys")
        axes[0].set_yticks(range(4), raw_env.FOOT_ORDER)
        axes[0].set_title("Foot contact (black = stance), first evaluation episode")
        axes[1].plot(t, [r["clearance"] for r in first])
        axes[1].legend(raw_env.FOOT_ORDER, ncol=4)
        axes[1].set_ylabel("Sole height (m)")
        axes[2].plot(t, [r["x_vel"] for r in first], label="Forward")
        axes[2].plot(t, [r["z_vel"] for r in first], label="Vertical")
        axes[2].axhline(raw_env.command_lin_vel_x, ls="--", color="gray", label="Target")
        axes[2].set_ylabel("Velocity (m/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(output.with_suffix(".png"), dpi=150)
        plt.close(fig)
        print(f"Saved {output.with_suffix('.json')} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Play PPO on Quadruped Walking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--model", type=str, default="anybotics_anymal_c/scene.xml",
                             help="Path to MuJoCo model XML")
    train_parser.add_argument("--log-dir", type=str, default="./anymal_gait_logs",
                             help="Directory for logs and checkpoints")
    train_parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                             help="Total training timesteps")
    train_parser.add_argument("--num-envs", type=int, default=8,
                             help="Number of parallel environments")
    train_parser.add_argument("--n-steps", type=int, default=2048,
                             help="Steps per environment per update")
    train_parser.add_argument("--batch-size", type=int, default=256,
                             help="Minibatch size")
    train_parser.add_argument("--n-epochs", type=int, default=5,
                             help="Number of epochs per update")
    train_parser.add_argument("--lr", type=float, default=3e-4,
                             help="Learning rate")
    train_parser.add_argument("--log-every", type=int, default=10,
                             help="Print stats every N episodes")
    train_parser.add_argument("--seed", type=int, default=42,
                             help="Random seed")
    
    train_parser.add_argument("--gait", choices=["trot", "walk"], default="trot")

    # Play arguments
    play_parser = subparsers.add_parser("play", help="Play with trained model")
    play_parser.add_argument("--model", type=str, default="anybotics_anymal_c/scene.xml",
                            help="Path to MuJoCo model XML")
    play_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to model checkpoint")
    play_parser.add_argument("--num-episodes", type=int, default=5,
                            help="Number of episodes to play")
    
    play_parser.add_argument("--seed", type=int, default=123)
    eval_parser = subparsers.add_parser("evaluate", help="Headless deterministic gait evaluation")
    eval_parser.add_argument("--model", default="anybotics_anymal_c/scene.xml")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--num-episodes", type=int, default=5)
    eval_parser.add_argument("--seed", type=int, default=123)
    eval_parser.add_argument("--output", default="anymal_gait_evaluation")
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command in ("play", "evaluate"):
        play(args)
