import math
import random
import numpy as np
from collections import deque
from typing import Optional

import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from torch import Generator
from torch.nn import functional as F


# Modification of DistributedSampler that distributes samples per gpu in a batchwise fashion.
# This allows us to do true sequential sampling of a dataset, when shuffle is set to false.
class MyDistributedSampler(DistributedSampler):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        super().__init__(dataset=dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=shuffle,
                         seed=seed,
                         drop_last=drop_last)
        self.batch_size = batch_size

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        db_size = len(self.dataset) // (
            batch_size * self.num_replicas) * batch_size * self.num_replicas
        if self.drop_last and db_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (db_size - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                db_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(
                self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (
                    indices *
                    math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        if self.batch_size == 1:
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
        else:
            batches = [
                indices[i:i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]
            batches = batches[self.rank:len(batches):self.num_replicas]
            indices = [i for b in batches for i in b]
            assert len(
                indices
            ) == self.num_samples, f"{len(indices)} {self.num_samples}"

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class ResumableDistributedSampler(Sampler):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = None,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False,
                 n_seq_samples: int = -1) -> None:
        self.sampler = MyDistributedSampler(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.n_seq_samples = n_seq_samples
        self.start_idx = 0
        self.num_replicas = self.sampler.num_replicas
        self.rank = self.sampler.rank

    def __iter__(self):
        indices = list(self.sampler)
        if self.n_seq_samples > 0:
            start_inds = torch.tensor(
                [i for i in indices if i % self.n_seq_samples == 0]).cuda()
            start_inds_all = gather(start_inds, distributed=True)
            start_inds_all = torch.cat(start_inds_all).cpu()

            num_start = math.ceil(
                (len(start_inds_all) - self.sampler.num_replicas) /
                self.sampler.num_replicas)
            total_num_start = self.sampler.num_replicas * num_start
            start_inds_all = start_inds_all[:total_num_start]
            start_inds = start_inds_all[self.sampler.rank::self.sampler.
                                        num_replicas]
            print('creating inds')
            indices = np.arange(self.n_seq_samples)[np.newaxis, :].repeat(
                len(start_inds), axis=0)
            indices = indices + start_inds.cpu().numpy()[:, np.newaxis].repeat(
                self.n_seq_samples, axis=1)
            indices = indices.reshape(-1)
            print(len(indices))

        return iter(indices[self.start_idx:])

    def __len__(self) -> int:
        return len(self.sampler) - self.start_idx

    def set_epoch(self, epoch: int, instance: int = 0) -> None:
        self.sampler.set_epoch(epoch)
        world_size = torch.distributed.get_world_size()
        self.start_idx = instance // world_size


class LMBatchSampler(Sampler):
    def __init__(self, memory_size: int, repeat: int, sampler: Sampler,
                 batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.memory_size = memory_size
        self.repeat = repeat

        self.seed_base = 93823982
        self.epoch = 0
        assert drop_last

        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if self.distributed else 0

        self.memory = deque(maxlen=self.memory_size)
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = deque(maxlen=128)
        self.init_from_ckpt = False

    def state_dict(self):
        batch_history = gather(torch.tensor(self.batch_history),
                               self.distributed)
        memory = gather_memory(self.memory, self.distributed)
        return {
            'memory': memory,
            'db_head': self.db_head,
            'num_batches_seen': self.num_batches_seen,
            'num_batches_yielded': self.num_batches_yielded,
            'batch_history': batch_history
        }

    def load_state_dict(self, state_dict):
        self.memory = deque(reverse_tensorized_memory(state_dict['memory'],
                                                      self.rank),
                            maxlen=self.memory_size)
        self.db_head = state_dict['db_head']
        self.num_batches_seen = state_dict['num_batches_seen']
        self.num_batches_yielded = state_dict['num_batches_yielded']
        self.init_from_ckpt = True

        batch_history = state_dict['batch_history'][self.rank]
        batch_history = deque([b.tolist() for b in batch_history], maxlen=128)
        self.batch_history = batch_history

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def sample_k(self, q, k):
        # import random
        if k < len(q):
            return random.sample(q, k=k)
        elif k == len(q):
            return q
        else:
            return random.choices(q, k=k)

    def update_sample_stats(self, sample_info):
        db2buff = {b['idx']: i for i, b in enumerate(self.memory)}
        sample_index = sample_info['index'].detach()
        sample_loss = sample_info['loss'].detach().cpu()
        for i in range(self.batch_size):
            db_idx = sample_index[i].item()
            if db_idx in db2buff:
                b = self.memory[db2buff[db_idx]]
                b['loss'] = sample_loss[i]
                b['seen'] = True
                b['num_seen'] += 1
        samples = [
            self.memory[db2buff[idx]] for idx in sample_index.tolist()
            if idx in db2buff
        ]
        if not samples:
            return {}
        else:
            return tensorize_memory(samples)

    def __iter__(self):
        from collections import deque
        self.generator = Generator()
        self.generator.manual_seed(self.seed_base + self.epoch)
        random.seed(self.seed_base + self.epoch)

        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)

        # Resubmit batches not seen by the model
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0,
                       -1):
            yield self.batch_history[-i]

        all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self):
            if self.db_head < len(all_indices):
                indices = all_indices[self.db_head:self.db_head +
                                      self.batch_size]
                self.memory += [{
                    'idx': idx,
                    'lifespan': 0,
                    'loss': None,
                    'seen': False,
                    'num_seen': 0
                } for idx in indices]
                self.db_head += len(indices)
                if len(indices) > 0 and len(self.memory) < self.memory_size:
                    continue
            for j in range(self.repeat):
                batch = self.sample_k(self.memory, self.batch_size)
                batch_idx = [b['idx'] for b in batch]
                self.batch_history += [batch_idx]
                self.num_batches_yielded += 1
                yield batch_idx

        self.init_from_ckpt = False

    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch=epoch)


def tensorize_memory(memory):
    memory_tensor = {}
    for k in memory[0]:
        tens_list = [s[k] for s in memory]
        if all(t is None for t in tens_list):
            continue
        dummy = [t for t in tens_list if t is not None][0] * 0.
        tens_list = [t if t is not None else dummy for t in tens_list]
        try:
            if isinstance(tens_list[0], torch.Tensor):
                tens = torch.stack(tens_list)
            elif isinstance(tens_list[0], (int, bool, float)):
                tens = torch.tensor(tens_list)
            else:
                tens = torch.tensor(tens_list)
            memory_tensor[k] = tens
        except Exception as e:
            print(tens_list)
            print(e)
    return memory_tensor


def reverse_tensorized_memory(memory_tensor, rank=0):
    memory = []
    keys = list(memory_tensor.keys())
    siz = memory_tensor[keys[0]][rank].shape[0]
    for i in range(siz):
        memory += [{
            k: memory_tensor[k][rank][i].item() if k in {
                'idx', 'lifespan', 'seen', 'num_seen'
            } else memory_tensor[k][rank][i].cpu()
            for k in keys
        }]
    return memory


def gather(tensor, distributed=False):
    if not distributed:
        return [tensor]
    else:
        world_size = torch.distributed.get_world_size()
        size = tuple(tensor.shape)
        size_all = [size for _ in range(world_size)]
        torch.distributed.all_gather_object(size_all, size)

        tensor = tensor.cuda()
        max_sz = max([sz[0] for sz in size_all])
        expand_sz = tuple([max_sz] + list(size)[1:])
        tensor_all = [
            torch.zeros(size=expand_sz, dtype=tensor.dtype).cuda()
            for _ in range(world_size)
        ]
        if tensor.shape[0] < max_sz:
            pad = [0] * (2 * len(size))
            pad[-1] = max_sz - tensor.shape[0]
            tensor = F.pad(tensor, pad=pad)
        torch.distributed.all_gather(tensor_all, tensor)
        return [
            tensor_all[r][:size_all[r][0]].cpu() for r in range(world_size)
        ]


def gather_memory(memory, distributed=False):
    memory_tensor = tensorize_memory(memory)
    for k in memory_tensor:
        memory_tensor[k] = gather(memory_tensor[k], distributed)
    return memory_tensor


class DMBatchSampler(Sampler):
    def __init__(self,
                 memory_size: int,
                 repeat: int,
                 sampler: Sampler,
                 batch_size: int,
                 limit_num_seen_coeff: int = -1,
                 drop_last: bool = True,
                 rank: int = None) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.memory_size = memory_size
        self.repeat = repeat
        self.gamma = 0.5  # polyak average coeff
        self.feat_dim = 2048
        self.limit_num_seen_coeff = limit_num_seen_coeff

        self.seed_base = 93823982
        self.epoch = 0
        assert drop_last

        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        if rank is None:
            rank = torch.distributed.get_rank() if self.distributed else 0
        self.rank = rank

        # Init memory
        self.memory = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False

    def state_dict(self):
        batch_history = gather(torch.tensor(self.batch_history),
                               self.distributed)
        memory = gather_memory(self.memory, self.distributed)
        state_dict = {
            'memory': memory,
            'db_head': self.db_head,
            'num_batches_seen': self.num_batches_seen,
            'num_batches_yielded': self.num_batches_yielded,
            'batch_history': batch_history,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.memory = reverse_tensorized_memory(state_dict['memory'],
                                                self.rank)
        if torch.distributed.is_initialized():
            for b in self.memory:
                b['feature'] = b['feature'].cuda()
                # b['similarity'] = b['similarity'].cpu()

        self.db_head = state_dict['db_head']
        self.num_batches_seen = state_dict['num_batches_seen']
        self.num_batches_yielded = state_dict['num_batches_yielded']
        self.init_from_ckpt = True

        batch_history = state_dict['batch_history'][self.rank]
        batch_history = deque([b.tolist() for b in batch_history], maxlen=128)
        self.batch_history = batch_history

        keys2reset = [
            k for k in self.memory[0]
            if k not in {'idx', 'lifespan', 'seen', 'num_seen'}
        ]
        for b in self.memory:
            if not b['seen']:
                for k in keys2reset:
                    b[k] = None

        # If saved at end of epoch, signal that next epoch should start from the top.
        if self.num_batches_yielded == len(self):
            self.init_from_ckpt = False

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def sample_k(self, q, k):
        if k <= len(q):
            return random.sample(q, k=k)
        else:
            return random.choices(q, k=k)

    def add_to_memory(self, n):
        if self.db_head >= len(self.all_indices):
            return True

        # Add indices to memory
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.memory += [{
                'idx': idx,
                'lifespan': 0,
                'loss': None,
                'neighbor_similarity': None,
                'feature': None,
                'num_seen': 0,
                'seen': False,
            }]
        self.db_head += len(indices_to_add)

        # Increase lifespan count
        for b in self.memory:
            b['lifespan'] += 1

        return False

    def resize_memory(self, n):
        n2rm = len(self.memory) - n
        if n2rm <= 0:
            return

        def max_coverage_reduction(x, n2rm):
            # removes samples 1 by 1 that are most similar to currently selected.
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
            sim.fill_diagonal_(-10.)
            idx2rm = []
            for i in range(n2rm):
                neig_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neig_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)
            return idx2rm

        # Only remove samples that have already been evaluated
        memory = [(b, i) for i, b in enumerate(self.memory) if b['seen']]
        if len(memory) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.memory]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()

        else:
            feats = torch.stack([b['feature'] for b, i in memory], 0)
            idx2rm = max_coverage_reduction(feats, n2rm)
            idx2rm = [memory[i][1] for i in idx2rm]

        # Remove samples from memory
        idx2rm = set(idx2rm)
        self.memory = [b for i, b in enumerate(self.memory) if i not in idx2rm]

        # Recompute nearest neighbor similarity for tracking
        if any(b['seen'] for b in self.memory):
            feats = torch.stack(
                [b['feature'] for b in self.memory if b['seen']], 0)
            feats = feats.cuda() if torch.cuda.is_available() else feats
            if feats.shape[0] > 1:
                feats_sim = torch.einsum('ad,bd->ab', feats, feats)
                neig_sim = torch.topk(feats_sim, k=2, dim=-1,
                                    sorted=False)[0][:, 1:].mean(dim=1).cpu()
                i = 0
                for b in self.memory:
                    if b['seen']:
                        b['neighbor_similarity'] = neig_sim[i]
                        i += 1

    def update_sample_stats(self, sample_info):
        db2buff = {b['idx']: i for i, b in enumerate(self.memory)}
        sample_loss = sample_info['loss'].detach().cpu()
        sample_index = sample_info['index'].detach().cpu()
        sample_features = F.normalize(sample_info['feature'].detach(), p=2, dim=-1)

        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg

        for i in range(self.batch_size):
            db_idx = sample_index[i].item()
            if db_idx in db2buff:
                b = self.memory[db2buff[db_idx]]
                if not b['seen']:
                    b['loss'] = sample_loss[i]
                    b['feature'] = sample_features[i]
                else:
                    b['loss'] = polyak_avg(b['loss'], sample_loss[i],
                                           self.gamma)
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i], self.gamma),
                                               p=2,
                                               dim=-1)
                b['num_seen'] += 1
                b['seen'] = True

        if self.limit_num_seen_coeff > 0:
            max_n_seen = self.limit_num_seen_coeff * self.repeat
            self.memory = [
                b for b in self.memory if b['num_seen'] < max_n_seen
            ]
            db2buff = {b['idx']: i for i, b in enumerate(self.memory)}

        samples = [
            self.memory[db2buff[idx]] for idx in sample_index.tolist()
            if idx in db2buff
        ]
        if not samples:
            return {}
        else:
            return tensorize_memory(samples)

    def __iter__(self):
        random.seed(self.seed_base + self.rank * 1000 + self.epoch)

        self.all_indices = list(self.sampler)
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)

        # Resubmit batches not seen by the model
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0,
                       -1):
            yield self.batch_history[-i]

        assert self.memory_size <= len(self.all_indices)
        while self.num_batches_yielded < len(self):
            done = self.add_to_memory(self.batch_size)
            if not done and len(self.memory) < self.memory_size:
                continue  # keep adding until memory is full

            self.resize_memory(self.memory_size)
            for j in range(self.repeat):
                batch = self.sample_k(self.memory, self.batch_size)
                batch_idx = [b['idx'] for b in batch]
                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]
                yield batch_idx

        self.init_from_ckpt = False

    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch=epoch)


class SMBatchSampler(Sampler):
    def __init__(self,
                 memory_size: int,
                 repeat: int,
                 sampler: Sampler,
                 batch_size: int,
                 top_k_entropy: int,
                 limit_num_seen_coeff: int = -1,
                 drop_last: bool = True,
                 rank: int = None) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.memory_size = memory_size
        self.repeat = repeat
        self.gamma = 0.5  # polyak average coeff
        self.feat_dim = 2048
        self.limit_num_seen_coeff = limit_num_seen_coeff

        self.top_k_entropy = top_k_entropy

        self.seed_base = 93823982
        self.epoch = 0
        assert drop_last

        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        if rank is None:
            rank = torch.distributed.get_rank() if self.distributed else 0
        self.rank = rank

        # Init memory
        self.memory = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False

    def state_dict(self):
        batch_history = gather(torch.tensor(self.batch_history),
                               self.distributed)
        memory = gather_memory(self.memory, self.distributed)
        state_dict = {
            'memory': memory,
            'db_head': self.db_head,
            'num_batches_seen': self.num_batches_seen,
            'num_batches_yielded': self.num_batches_yielded,
            'batch_history': batch_history,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.memory = reverse_tensorized_memory(state_dict['memory'],
                                                self.rank)
        if torch.distributed.is_initialized():
            for b in self.memory:
                b['feature'] = b['feature'].cuda()
                # b['similarity'] = b['similarity'].cpu()

        self.db_head = state_dict['db_head']
        self.num_batches_seen = state_dict['num_batches_seen']
        self.num_batches_yielded = state_dict['num_batches_yielded']
        self.init_from_ckpt = True

        batch_history = state_dict['batch_history'][self.rank]
        batch_history = deque([b.tolist() for b in batch_history], maxlen=128)
        self.batch_history = batch_history

        keys2reset = [
            k for k in self.memory[0]
            if k not in {'idx', 'lifespan', 'seen', 'num_seen'}
        ]
        for b in self.memory:
            if not b['seen']:
                for k in keys2reset:
                    b[k] = None

        # If saved at end of epoch, signal that next epoch should start from the top.
        if self.num_batches_yielded == len(self):
            self.init_from_ckpt = False

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def sample_k(self, q, k):
        if k <= len(q):
            return random.sample(q, k=k)
        else:
            return random.choices(q, k=k)

    def add_to_memory(self, n):
        if self.db_head >= len(self.all_indices):
            return True

        # Add indices to memory
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.memory += [{
                'idx': idx,
                'lifespan': 0,
                'loss': None,
                # 'similarity': None,
                'neighbor_similarity': None,
                'feature': None,
                'entropy': None,
                'num_seen': 0,
                'seen': False,
            }]
        self.db_head += len(indices_to_add)

        # Increase lifespan count
        for b in self.memory:
            b['lifespan'] += 1

        return False

    def resize_memory(self, n):
        n2rm = len(self.memory) - n
        if n2rm <= 0:
            return

        def max_coverage_reduction(x, n2rm):
            # removes samples 1 by 1 that are most similar to currently selected.
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
            sim.fill_diagonal_(-10.)
            idx2rm = []
            for i in range(n2rm):
                neig_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neig_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)
            return idx2rm

        # Only remove samples that have already been evaluated
        memory = [(b, i) for i, b in enumerate(self.memory) if b['seen']]
        
        # Only remove samples from the ones under top N//k entropy
        if len(memory) >= self.top_k_entropy:
            entropies = torch.stack([b['entropy'] for b, i in memory], 0)
            top_k_n = int(len(memory)//(self.top_k_entropy))
            top_n_entropies = torch.topk(entropies, k=top_k_n, largest=True).indices
            memory = [buff for index, buff in enumerate(memory) if index not in top_n_entropies]
            
        if len(memory) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.memory]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()

        else:
            # Compute top 5 neighbor average similarity
            feats = torch.stack([b['feature'] for b, i in memory], 0)
            idx2rm = max_coverage_reduction(feats, n2rm)
            # idx2rm = neig_sim.argsort(descending=True)[:n2rm]
            idx2rm = [memory[i][1] for i in idx2rm]

        # Remove samples from memory
        idx2rm = set(idx2rm)
        self.memory = [b for i, b in enumerate(self.memory) if i not in idx2rm]

        # Recompute nearest neighbor similarity for tracking
        if any(b['seen'] for b in self.memory):
            feats = torch.stack(
                [b['feature'] for b in self.memory if b['seen']], 0)
            feats = feats.cuda() if torch.cuda.is_available() else feats
            if feats.shape[0] > 1:
                feats_sim = torch.einsum('ad,bd->ab', feats, feats)
                neig_sim = torch.topk(feats_sim, k=2, dim=-1,
                                    sorted=False)[0][:, 1:].mean(dim=1).cpu()
                i = 0
                for b in self.memory:
                    if b['seen']:
                        b['neighbor_similarity'] = neig_sim[i]
                        i += 1

    def update_sample_stats(self, sample_info):
        # device = sample_info['loss'].device
        db2buff = {b['idx']: i for i, b in enumerate(self.memory)}
        sample_loss = sample_info['loss'].detach().cpu()
        sample_index = sample_info['index'].detach().cpu()
        sample_entropy = sample_info['entropy'].detach().cpu()
        sample_features = F.normalize(sample_info['feature'].detach(), p=2, dim=-1)

        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg

        for i in range(self.batch_size):
            db_idx = sample_index[i].item()
            if db_idx in db2buff:
                b = self.memory[db2buff[db_idx]]
                if not b['seen']:
                    b['loss'] = sample_loss[i]
                    b['feature'] = sample_features[i]
                    b['entropy'] = sample_entropy[i]
                else:
                    b['loss'] = polyak_avg(b['loss'], sample_loss[i],
                                           self.gamma)
                    b['entropy'] = polyak_avg(b['entropy'], sample_entropy[i],
                                           self.gamma)
                    
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i], self.gamma),
                                               p=2,
                                               dim=-1)

                b['num_seen'] += 1
                b['seen'] = True

        if self.limit_num_seen_coeff > 0:
            max_n_seen = self.limit_num_seen_coeff * self.repeat
            self.memory = [
                b for b in self.memory if b['num_seen'] < max_n_seen
            ]
            db2buff = {b['idx']: i for i, b in enumerate(self.memory)}

        samples = [
            self.memory[db2buff[idx]] for idx in sample_index.tolist()
            if idx in db2buff
        ]
        if not samples:
            return {}
        else:
            return tensorize_memory(samples)

    def __iter__(self):
        random.seed(self.seed_base + self.rank * 1000 + self.epoch)

        self.all_indices = list(self.sampler)
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)

        # Resubmit batches not seen by the model
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0,
                       -1):
            yield self.batch_history[-i]

        assert self.memory_size <= len(self.all_indices)
        while self.num_batches_yielded < len(self):
            done = self.add_to_memory(self.batch_size)
            if not done and len(self.memory) < self.memory_size:
                continue  # keep adding until memory is full

            self.resize_memory(self.memory_size)
            for j in range(self.repeat):
                batch = self.sample_k(self.memory, self.batch_size)
                batch_idx = [b['idx'] for b in batch]
                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]
                yield batch_idx

        self.init_from_ckpt = False

    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch=epoch)
