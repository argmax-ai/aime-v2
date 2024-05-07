import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from threading import Event, Thread
from typing import *

import numpy as np
import torch
from h5py import File
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler


class ArrayDict(dict):
    def vmap_(self, fn, rewrite=True) -> "ArrayDict":
        for k, v in self.items():
            result = fn(v)
            if rewrite:
                self[k] = result
        return self

    def expand_dim_equal_(
        self, black_list=["image", "frontview_image", "agentview_image"]
    ) -> "ArrayDict":
        # TODO: logic is wrong if there is image data in the dict
        considered_keys = [k for k in self.keys() if k not in black_list]
        if len(considered_keys) == 0:
            return self
        max_length = max([len(self[k].shape) for k in considered_keys])
        for k, v in self.items():
            if k in black_list:
                continue
            if len(v.shape) < max_length:
                for _ in range(max_length - len(v.shape)):
                    v = v[..., None]
                self[k] = v
        return self

    def __len__(self) -> int:
        lengths = [len(v) for v in self.values()]
        assert np.all([l == lengths[0] for l in lengths]), {
            k: v.shape for k, v in self.items()
        }
        return lengths[0]

    def __getitem__(self, index: Union[int, str]):
        if isinstance(index, str):
            return dict.__getitem__(self, index)
        else:
            return ArrayDict({k: v[index] for k, v in self.items()})

    def to(self, target: Union[str, torch.Tensor], *args, **kwargs) -> "ArrayDict":
        return self.vmap_(lambda v: v.to(target, *args, **kwargs))

    def to_torch(self) -> "ArrayDict":
        return self.vmap_(lambda v: torch.tensor(v))

    def to_numpy(self) -> "ArrayDict":
        return self.vmap_(lambda v: v.detach().cpu().numpy())

    def to_float_torch(self) -> "ArrayDict":
        return self.vmap_(lambda v: v.float())

    def detach(self) -> "ArrayDict":
        return self.vmap_(lambda v: v.detach())

    def get_type(self) -> "ArrayDict":
        return type(list(self.values())[0])

    @classmethod
    def merge_list(cls, array_dicts: List["ArrayDict"], merge_fn) -> "ArrayDict":
        keys = array_dicts[0].keys()
        return ArrayDict(
            {k: merge_fn([array_dict[k] for array_dict in array_dicts]) for k in keys}
        )

    @classmethod
    def stack(cls, array_dicts: List["ArrayDict"], dim: int) -> "ArrayDict":
        if array_dicts[0].get_type() is torch.Tensor:
            merge_fn = partial(torch.stack, dim=dim)
        else:
            merge_fn = partial(np.stack, axis=dim)
        return cls.merge_list(array_dicts, merge_fn)

    @classmethod
    def cat(cls, array_dicts: List["ArrayDict"], dim: int) -> "ArrayDict":
        if array_dicts[0].get_type() is torch.Tensor:
            merge_fn = partial(torch.cat, dim=dim)
        else:
            merge_fn = partial(np.concatenate, axis=dim)
        return cls.merge_list(array_dicts, merge_fn)

    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()
        return self


class Trajectory:
    def __init__(
        self,
        file_name,
        selected_keys: Optional[List[str]] = None,
        lazy: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.file_name = file_name
        self.selected_keys = selected_keys
        self.lazy = lazy
        self.args = args
        self.kwargs = kwargs
        self._data = None
        self._extra_data = ArrayDict()
        self._load()

    def _load(self):
        """load trajectories"""
        raise NotImplementedError

    def __len__(self):
        """return the length of the trajectory"""
        raise NotImplementedError

    def get_trajectory(self):
        """get the full trajectory"""
        return self.get_clip(0, len(self))

    def _get_clip(self, start, end):
        """get a clip of the trajectory, [start, end)"""
        raise NotImplementedError

    def get_clip(self, start, end):
        """get a clip of the trajectory, [start, end), including extra data"""
        data = self._get_clip(start, end)
        if len(self._extra_data.keys()) > 0:
            data.update(self._extra_data[start:end])
        return data

    def update(self, data: ArrayDict):
        """update self._data with data. Due to the format convergence, be careful when calling this!"""
        raise NotImplementedError

    def _save(self):
        """save self._data to local disk, normally be called after update"""
        raise NotImplementedError

    def label(self, label: ArrayDict):
        """
        add extra information to the trajectory, assume to be on the same length.
        this is useful for adding new reward functions to an existing dataset
        """
        assert len(self) == len(label)
        self._extra_data.update(label)

    def _convert_to_output(self, data: ArrayDict):
        data.expand_dim_equal_()
        data.to_torch()
        data.to_float_torch()
        for k, v in data.items():
            if len(v.shape) == 4:
                data[k] = v.permute(0, 3, 1, 2).contiguous()
                if data[k].shape[1] == 3:
                    data[k] = data[k] / 255
        return data


class NPZTrajectory(Trajectory):
    def _load(self):
        data = ArrayDict(np.load(self.file_name))
        if self.selected_keys is not None:
            data = ArrayDict({k: v for k, v in data.items() if k in self.selected_keys})
        self._data = data

    def __len__(self):
        return len(self._data)

    def _get_clip(self, start, end):
        return self._convert_to_output(self._data[start:end])

    def update(self, data: ArrayDict):
        self._data = data
        self._save()

    def _save(self):
        from aimev2.utils import savenpz

        data = deepcopy(self._data).to_numpy()
        savenpz(data, self.file_name)


class HDF5Trajectory(Trajectory):
    def _load(self):
        self.f = File(self.file_name)
        if not self.lazy:
            self._data = ArrayDict(
                {
                    k: v[:]
                    for k, v in self.f.items()
                    if self.selected_keys is None or k in self.selected_keys
                }
            )

    def __len__(self):
        if self.lazy:
            some_key = list(self.f.keys())[0]
            return self.f[some_key].shape[0]
        else:
            return len(self._data)

    def _get_clip(self, start, end) -> ArrayDict:
        # NOTE: there could be some problem here, if you remove the ArrayDict in the lazy load part, it will think data is a dict when lazy=False.
        if self.lazy:
            data = ArrayDict(
                {
                    k: v[start:end]
                    for k, v in self.f.items()
                    if self.selected_keys is None or k in self.selected_keys
                }
            )
        else:
            data = self._data[start:end]
        return self._convert_to_output(data)

    def update(self, data: ArrayDict):
        if self.f is not None:
            self.f.close()
        self._save(data)
        self._load()

    def _save(self, data):
        from aimev2.utils import savehdf5

        data = deepcopy(data).to_numpy()
        savehdf5(data, self.file_name)


def get_trajectory_class(file_name: str):
    if file_name.endswith(".npz"):
        return NPZTrajectory
    elif file_name.endswith(".hdf5"):
        return HDF5Trajectory
    else:
        raise NotImplementedError


class SequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        horizon: int,
        overlap: bool,
        max_capacity: Optional[int] = None,
        selected_keys: Optional[List[str]] = None,
        lazy: bool = True,
        use_label: bool = False,
        label_shift: int = 0,
        black_list: List[str] = [],
    ) -> None:
        super().__init__()
        self.root = root
        self.horizon = horizon
        self.overlap = overlap
        self.max_capacity = max_capacity
        self.selected_keys = selected_keys
        self.lazy = lazy
        self.use_label = use_label
        self.label_shift = label_shift
        self.black_list = black_list
        self.loaded_file = []
        self.black_list_file = black_list.copy()  # files that won't be loaded again
        self.trajectories = []
        self.index_lookup = []

        self.update()  # call update to do the initialization

    def update(self):
        self._update_trajectories()
        self._update_index_map()

    def _update_trajectories(self):
        file_list = self.sort(
            [file for file in os.listdir(self.root) if file not in self.black_list_file]
        )
        if self.max_capacity is not None:
            file_list = file_list[: self.max_capacity]
        for file in file_list:
            if file in self.loaded_file:
                continue
            self.loaded_file.append(file)
            traj_cls = get_trajectory_class(file)
            file = os.path.join(self.root, file)
            self.trajectories.append(traj_cls(file, self.selected_keys, self.lazy))

    def _update_index_map(self):
        trajectory_index = (
            0
            if len(self.index_lookup) == 0
            else max([pair[0] for pair in self.index_lookup]) + 1
        )
        while trajectory_index < len(self.trajectories):
            trajectory = self.trajectories[trajectory_index]

            length = len(trajectory)

            total_clip = max(
                1,
                length // self.horizon
                if not self.overlap
                else max(length - self.horizon + 1, 0),
            )  # at least one clip for each trajectory

            for i in range(total_clip):
                time_index = i * self.horizon if not self.overlap else i
                self.index_lookup.append((trajectory_index, time_index))

            trajectory_index += 1

    def keep(self, num_trajectories: int, mode: str = "forward"):
        """
        keep a subset of the dataset
        mode can be selected from `forward`, `backward`, `random`
        """
        if num_trajectories >= len(self.trajectories):
            return
        index = np.arange(len(self.trajectories))
        if mode == "forward":
            remain_index = index[num_trajectories:]
            selected_index = index[:num_trajectories]
        elif mode == "backward":
            remain_index = index[:-num_trajectories]
            selected_index = index[-num_trajectories:]
        elif mode == "random":
            np.random.shuffle(index)
            remain_index = index[num_trajectories:]
            selected_index = index[:num_trajectories]
        elif mode == "uniform":
            selected_index = np.linspace(
                0, len(self.trajectories) - 1, num=num_trajectories, dtype=int
            )
            remain_index = np.array([i for i in index if i not in selected_index])
        selected_index = list(selected_index)
        self.trajectories = [self.trajectories[index] for index in selected_index]
        self.black_list_file = self.black_list_file + [
            self.loaded_file[index] for index in remain_index
        ]
        self.loaded_file = [self.loaded_file[index] for index in selected_index]
        self.index_lookup = []
        self._update_index_map()

    def __len__(self):
        return len(self.index_lookup)

    def __getitem__(self, index):
        data, trajectory_index = self.collect_clip(index)
        if len(data) < self.horizon:
            # when the trajectory is short, meaning early termination, copy the late step
            # NOTE: this stratergy is temporay and may not be correct for some environment, for example bonus final reward?
            extend_data = data[-1:]
            extend_data.vmap_(
                lambda v: torch.repeat_interleave(
                    v, repeats=self.horizon - len(data), dim=0
                )
            )
            data = ArrayDict.cat([data, extend_data], dim=0)
        if self.use_label:
            data["label"] = (
                torch.ones(len(data), dtype=int) * trajectory_index + self.label_shift
            )
        return data

    def sample(self, batch_size):
        indexes = np.random.randint(len(self), size=batch_size)
        data = ArrayDict.stack([self[index] for index in indexes], dim=1)
        return data

    @property
    def num_trajectories(self):
        return len(self.trajectories)

    def collect_clip(self, index):
        trajectory_index, time_index = self.index_lookup[index]
        return self.trajectories[trajectory_index].get_clip(
            time_index, time_index + self.horizon
        ), trajectory_index

    def get_trajectory(self, index):
        return self.trajectories[index].get_trajectory()

    def update_trajectory(self, index: int, data: ArrayDict):
        """update the content of an trajectory, assuming the length does not change"""
        self.trajectories[index].update(data)

    def split_train_and_val(self, train_ratio: float):
        """split the dataset to two with the ratio based on the number of trajectories, mainly for train and validation for a fix dataset."""
        total_trajectories = self.num_trajectories
        train_trajectories = int(total_trajectories * train_ratio)
        indexes = np.arange(total_trajectories)
        train_indexes = np.random.choice(
            indexes, size=train_trajectories, replace=False
        )
        train_files = [self.loaded_file[index] for index in train_indexes.tolist()]
        val_files = [file for file in self.loaded_file if file not in train_files]
        train_dataset = SequenceDataset(
            self.root,
            self.horizon,
            self.overlap,
            self.max_capacity,
            self.selected_keys,
            self.lazy,
            self.use_label,
            self.label_shift,
            self.black_list + val_files,
        )
        val_dataset = SequenceDataset(
            self.root,
            self.horizon,
            self.overlap,
            self.max_capacity,
            self.selected_keys,
            self.lazy,
            self.use_label,
            self.label_shift,
            self.black_list + train_files,
        )
        return train_dataset, val_dataset

    @classmethod
    def sort(cls, file_list):
        return sorted(file_list, key=lambda file_name: int(file_name.split(".")[0]))

    def summary(self):
        rewards = []
        successes = []
        lengths = []
        for traj in self.trajectories:
            traj = traj.get_trajectory()
            rewards.append(traj["reward"].sum().item())
            lengths.append(len(traj["reward"]))
            if "success" in traj.keys():
                successes.append(traj["success"][-1].sum().item())

        summary = {
            "length_mean": np.mean(lengths),
            "length_std": np.std(lengths),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
        }

        if len(successes) > 0:
            summary["success_rate"] = np.mean(successes)

        return summary


class MultiDataset(Dataset):
    # NOTE: We assume only the last dataset in the sequence is allowed to change the size
    def __init__(
        self,
        datasets: List[SequenceDataset],
        use_label: bool = False,
        label_shift: int = 0,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.use_label = use_label
        self.label_shift = label_shift
        self.update()

    def update(self):
        for dataset in self.datasets:
            dataset.update()
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.accumulate_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.accumulate_lengths[-1]

    def __getitem__(self, index) -> Any:
        dataset_index = np.where(index < self.accumulate_lengths)[0][0]
        shift_index = (
            0 if dataset_index == 0 else self.accumulate_lengths[dataset_index - 1]
        )
        data = self.datasets[dataset_index][index - shift_index]
        if self.use_label:
            if "label" in data.keys():
                # the inner datasets are already using labels
                local_label_shift = (
                    0
                    if dataset_index == 0
                    else self.accumulate_lengths[dataset_index - 1]
                )
                data["label"] = data["label"] + local_label_shift + self.label_shift
            else:
                data["label"] = (
                    torch.ones(len(data), dtype=int) * dataset_index + self.label_shift
                )
        return data

    @property
    def num_trajectories(self):
        return sum([dataset.num_trajectories for dataset in self.datasets])

    @property
    def num_sub_datasets(self):
        return len(self.datasets)


class MultiFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        horizon: int,
        overlap: bool,
        max_capacity: Optional[int] = None,
        selected_keys: Optional[List[str]] = None,
        lazy: bool = True,
        use_label: bool = False,
    ) -> None:
        super().__init__()

        self.root = root
        self.datasets = [
            SequenceDataset(
                os.path.join(root, folder),
                horizon,
                overlap,
                max_capacity=max_capacity,
                selected_keys=selected_keys,
                lazy=lazy,
            )
            for folder in sorted(os.listdir(root))
        ]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.accumulate_lengths = np.cumsum(self.lengths)
        self.lazy = lazy
        self.use_label = use_label

    def __len__(self):
        return self.accumulate_lengths[-1]

    def __getitem__(self, index) -> Any:
        dataset_index = np.where(index < self.accumulate_lengths)[0][0]
        shift_index = (
            0 if dataset_index == 0 else self.accumulate_lengths[dataset_index - 1]
        )
        data = self.datasets[dataset_index][index - shift_index]
        if self.use_label:
            data["label"] = torch.ones(len(data), dtype=int) * dataset_index
        return data

    @property
    def num_trajectories(self):
        return sum([dataset.num_trajectories for dataset in self.datasets])

    @property
    def num_sub_datasets(self):
        return len(self.datasets)

    def update(self):
        for dataset in self.datasets:
            dataset.update()

    def split_train_and_val(self, train_ratio: float):
        train_datasets = []
        val_datasets = []
        for dataset in self.datasets:
            _train, _val = dataset.split_train_and_val(train_ratio)
            train_datasets.append(_train)
            val_datasets.append(_val)
        return MultiDataset(train_datasets, self.use_label), MultiDataset(
            val_datasets, self.use_label
        )


class StatefullWorker:
    def __init__(
        self,
        dataset: SequenceDataset,
        rng,
        random_start: bool = True,
        lazy: bool = False,
    ) -> None:
        self.dataset = dataset
        self.rng = rng
        self.random_start = random_start
        self.lazy = lazy
        self.horizon = dataset.horizon
        self.traj = self.get_new_trajectory()

    def get_next(self):
        data = []
        size = 0
        while size < self.horizon:
            if self.lazy:
                data_to_add = self.traj.get_clip(
                    self.start_index, self.start_index + self.horizon - size
                )
                self.start_index += len(data_to_add)
            else:
                data_to_add = self.traj[: self.horizon - size]
                self.traj = self.traj[self.horizon - size :]
            if self.first:
                data_to_add["is_first"][0] = 1.0
                self.first = False
            data.append(data_to_add)
            size += len(data_to_add)
            if (self.lazy and self.start_index >= len(self.traj)) or (
                not self.lazy and len(self.traj) == 0
            ):
                self.traj = self.get_new_trajectory()
        data = ArrayDict.cat(data, dim=0)
        return data

    def get_new_trajectory(self):
        traj = self.rng.choice(self.dataset.trajectories)
        if self.random_start:
            total_length = len(traj)
            self.start_index = self.rng.integers(
                max(total_length - self.horizon, 0) + 1
            )
        else:
            self.start_index = 0
        if not self.lazy:
            traj = traj.get_clip(self.start_index, len(traj))
        self.first = True
        return traj


class StatefullLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        max_iter=None,
        collect_fn=partial(ArrayDict.stack, dim=1),
        num_workers=8,
        random_start=True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.collect_fn = collect_fn
        self.num_workers = num_workers
        self.random_start = random_start
        self.data_queue = deque()
        self.event = Event()

        def loader_helper(max_prefetch=8):
            executor = ThreadPoolExecutor(self.num_workers)
            workers = [
                StatefullWorker(
                    self.dataset, np.random.default_rng(seed), self.random_start
                )
                for seed in np.random.randint(123456789, size=(self.batch_size))
            ]
            while not self.event.is_set():
                if len(self.data_queue) < max_prefetch:
                    tasks = [executor.submit(worker.get_next) for worker in workers]
                    data = self.collect_fn([t.result() for t in tasks])
                    self.data_queue.append(data)
                else:
                    time.sleep(0.005)

        self.collector = Thread(target=loader_helper)
        self.collector.daemon = True
        self.collector.start()
        self.count = 0

    def __iter__(self):
        while self.max_iter is None or self.count < self.max_iter:
            if len(self.data_queue) > 0:
                self.count += 1
                yield self.data_queue.popleft()
            else:
                time.sleep(0.005)
        self.stop()

    def stop(self):
        self.event.set()
        self.collector.join()

    def __del__(self):
        self.stop()


def get_epoch_loader(
    dataset: SequenceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        collate_fn=partial(ArrayDict.stack, dim=1),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_sample_loader(
    dataset: SequenceDataset,
    batch_size: int,
    batchs: int,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    return DataLoader(
        dataset,
        batch_size,
        collate_fn=partial(ArrayDict.stack, dim=1),
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=RandomSampler(
            dataset, replacement=True, num_samples=batchs * batch_size
        ),
    )
