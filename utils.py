import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

# import IPython
# e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        chunk_size = 100  # You can also pass this as a class argument

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id:04d}.h5")

        is_pad = None
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs.get('sim', False)
            actions = root['actions'][:]
            episode_len = actions.shape[0]

            # --- Choose a start time ---
            if episode_len >= chunk_size:
                is_pad = np.zeros(chunk_size, dtype=bool)
                start_ts = np.random.randint(0, episode_len - chunk_size + 1)
                action = actions[start_ts:start_ts + chunk_size]
            else:
                # print(f"Episode {episode_id} is shorter than chunk size {chunk_size}. Padding with last action.")
                start_ts = 0
                pad_len = chunk_size - episode_len
                pad = np.repeat(actions[-1][None, :], pad_len, axis=0)
                action = np.concatenate([actions, pad], axis=0)
                is_pad = np.zeros(chunk_size, dtype=bool)
                is_pad[episode_len:] = True  # Mark padded steps as True

            # --- Observation at start_ts ---
            pos = root['ee_positions'][start_ts]
            ori = root['ee_orientations'][start_ts]
            grip = root['gripper_opening'][start_ts]
            qpos = np.concatenate([pos, ori, [grip]], axis=0)

            # --- RGB image at start_ts from all cameras ---
            image_dict = {
                cam_name: root[f'{cam_name}_images'][start_ts]
                for cam_name in self.camera_names
            }
            all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)

        self.is_sim = is_sim

        # --- Convert to tensors ---
        image_data = torch.from_numpy(all_cam_images).float() / 255.0
        image_data = image_data.permute(0, 3, 1, 2)  # (K, C, H, W)

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action).float()

        # --- Normalize ---
        qpos_data = (qpos_data - self.norm_stats["ee_pos_ori_grip_mean"]) / self.norm_stats["ee_pos_ori_grip_std"]
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return image_data, qpos_data, action_data, is_pad



# def get_norm_stats(dataset_dir, num_episodes):
#     all_qpos_data = []
#     all_action_data = []
#     for episode_idx in range(num_episodes):
#         print(f"Modified utils.py for loading xArm dataset")
#         dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx:04d}.h5")
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][()] 
#             qvel = root['/observations/qvel'][()]
#             action = root['/action'][()]
#         all_qpos_data.append(torch.from_numpy(qpos))
#         all_action_data.append(torch.from_numpy(action))
#     all_qpos_data = torch.stack(all_qpos_data)
#     all_action_data = torch.stack(all_action_data)
#     all_action_data = all_action_data

#     # normalize action data
#     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
#     action_std = all_action_data.std(dim=[0, 1], keepdim=True)
#     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

#     # normalize qpos data
#     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
#     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

#     stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
#              "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
#              "example_qpos": qpos}

#     return stats



def get_norm_stats(dataset_dir, num_episodes):
    all_qpos = []
    all_action = []

    print(f"Modified utils.py for EE pose loading")

    for i in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{i:04d}.h5")
        if not os.path.exists(path): continue

        with h5py.File(path, "r") as f:
            pos = f["ee_positions"][:]
            ori = f["ee_orientations"][:]
            grip = f["gripper_opening"][:].reshape(-1, 1)
            qpos = np.concatenate([pos, ori, grip], axis=1)
            action = f["actions"][:]

        all_qpos.append(torch.from_numpy(qpos))
        all_action.append(torch.from_numpy(action))

    all_qpos = torch.cat(all_qpos, dim=0)
    all_action = torch.cat(all_action, dim=0)

    return {
        "ee_pos_ori_grip_mean": all_qpos.mean(dim=0).numpy(),
        "ee_pos_ori_grip_std": torch.clamp(all_qpos.std(dim=0), min=1e-2).numpy(),
        "action_mean": all_action.mean(dim=0).numpy(),
        "action_std": torch.clamp(all_action.std(dim=0), min=1e-2).numpy()
    }
    
def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
