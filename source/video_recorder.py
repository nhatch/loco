import time
import os
from gym.monitoring.video_recorder import VideoRecorder

def video_recorder(env, save_dir):
    n = int(time.time())
    path = os.path.join(save_dir, 'video{}'.format(n))
    v = VideoRecorder(
            env=env,
            base_path=path,
            enabled=True)
    return v
