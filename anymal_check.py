import os
import argparse
import time
import mujoco

from train_anymal_improved import load_policy_env


# --- SETTINGS ---
SLOW_MO = 1.0       # 1.0 = Real Time, 2.0 = Half Speed (slower)
AUTO_TRACK = True   # Make camera follow the robot automatically


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_xml", type=str, default="anybotics_anymal_c/scene.xml"
    )
    parser.add_argument("--log_dir", type=str, default="./anymal_gait_logs")
    parser.add_argument("--model_name", type=str, default="ppo_anymal")
    args = parser.parse_args()

    model_path = os.path.join(args.log_dir, args.model_name)
    model, env, original_env = load_policy_env(args.model_xml, model_path, "human")

    dt = original_env.model.opt.timestep * original_env.frame_skip
    target_dt = dt * SLOW_MO

    # 2. Reset
    obs = env.reset()
    
    # --- CAMERA SETUP ---
    # We need to render once to initialize the viewer
    original_env._render_frame()
    if AUTO_TRACK and original_env.viewer is not None:
        # Lock camera to the base body (ID 1)
        original_env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        original_env.viewer.cam.trackbodyid = original_env.base_body_id
        original_env.viewer.cam.distance = 3.0  # Zoom out slightly
        original_env.viewer.cam.elevation = -20 # Look down slightly

    try:
        while True:
            start_time = time.time()
        
            # Predict
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            if dones[0]:
                # VecEnv already reset and returned the new initial observation.
                # Re-lock camera on reset if needed
                if AUTO_TRACK and original_env.viewer:
                    original_env.viewer.cam.trackbodyid = original_env.base_body_id

            # Sync with Real Time
            # We calculate how much time the computation took, and sleep for the remainder
            elapsed = time.time() - start_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
