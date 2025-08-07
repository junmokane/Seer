from calvin_env.envs.play_table_env import get_env
from pathlib import Path


val_folder = Path("calvin/dataset/task_ABC_D") / "validation"
print(val_folder)
env = get_env(val_folder, show_gui=False)
exit()