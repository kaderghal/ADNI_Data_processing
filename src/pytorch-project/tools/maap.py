import torch.nn as nn
from fastai import basic_train, data_block
import numpy as np
from torch import Tensor
import torch
import torch.optim
import os
import time
 
from pathlib import Path
 
 
class MemMapItemList(data_block.ItemList):
    def __init__(self, items, path, data_shape, dtype = np.float32, **kwargs):
        super().__init__(items, path, **kwargs)
        self.data_shape = data_shape
        self.copy_new.append("data_shape")
        self._file_process_dict = {}  # Deleting this structure might cause grief when the main thread is killing the workers
        self._dtype = dtype
 
    def get(self, i):
        pid = os.getpid()
        mem_file = self._file_process_dict.get(pid, None)  # each process owns its handler.
        if mem_file is None:
            mem_file = np.memmap(self.path, self._dtype, mode='r+', shape=self.data_shape)
            self._file_process_dict[pid] = mem_file
        idx = self.items[i]
        item_data = np.copy(mem_file[idx, :])
        if self._dtype == np.float32:
            item = data_block.FloatItem(item_data)
        else:
            item = data_block.Category(item_data, item_data)
        return item
 
    def reconstruct(self, t: Tensor, x: Tensor = None):
        return data_block.FloatItem(t.cpu().numpy())
 
    def labels_from_memmap(self, npy_memfile, data_shape, dtype=np.float32, **kwargs):
        y = MemMapItemList(self.items, npy_memfile, data_shape, dtype=dtype)
        res = self._label_list(x=self, y=y)
        return res
 
    @classmethod
    def from_memfile(cls, path, data_shape):
        "Constructs a MemMapItemList from a numpy mem mapped file"
        items = np.arange(0, data_shape[0])
        return MemMapItemList(items, path, data_shape)
 
 
def gen_some_data_for_io(folder, N, lx, ly):
    feat = np.random.rand(N, lx)
    feat[:, 0] = np.arange(N)
    target = np.random.rand(N, ly)
    target[:, 0] = np.arange(N)
 
    fx = folder / "x.npy"
    fy = folder / "y.npy"
 
    npfx = np.memmap(fx, np.float32, "w+", shape=feat.shape)
    npfx[:] = feat[:]
    npfx.flush()
 
    npfy = np.memmap(fy, np.float32, "w+", shape=target.shape)
    npfy[:] = target[:]
    npfy.flush()
 
    del npfx
    del npfy
 
 
class Validation_Net(nn.Module):
    "Dummy learner. It passes the first feature from input to the output"
 
    def __init__(self, input_size=5, output_size=3):
        super().__init__()
        self.last = nn.Linear(input_size, output_size)
 
    def forward(self, x):
        out = self.last(x)
        out[:, 0] = x[:, 0]
        return out
 
 
class Validation_Loss(torch.nn.Module):
    "Just makes sure that the first column from the input is identical with the target"
 
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        diff = x[:, 0] - y[:, 0]
        abs_diff = torch.abs(diff)
        abs_sum = torch.sum(abs_diff)
        if abs_sum > 0.000001:
            raise Exception("Input and lables are misalligned. Maybe the batch reading is wrong")
        dls = x - y
        dls = torch.sum(torch.pow(dls, 2))
        return dls
 
 
def train_network(folder, N, lx, ly):
    train_data_shape = (N, lx)
    test_data_shape = (N, ly)
 
    item_list = MemMapItemList.from_memfile(folder / "x.npy", data_shape=train_data_shape)
    splitted = item_list.random_split_by_pct(valid_pct=0.1)
    labeled = splitted.labels_from_memmap(folder / "y.npy", data_shape=test_data_shape)
    data_bunch = labeled.databunch(bs=512, num_workers=4)  # Test few values to see what's best for your hw+data stack
 
    model = Validation_Net()
    learner = basic_train.Learner(data=data_bunch, model=model, true_wd=True, wd=0.0001,
                                  loss_func=Validation_Loss(), path=folder)
 
    learner.fit(3, lr=0.001)
    t0 = time.time()
    learner.fit(3, lr=0.001)
    t1 = time.time()
    print("Time {}".format(t1 - t0))
 
 
if __name__ == "__main__":
    N = 100000
    lx = 5
    ly = 3
    folder = Path(".")
    gen_some_data_for_io(folder, N, lx, ly)
    train_network(folder, N, lx, ly)