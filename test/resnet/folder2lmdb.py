# credits: https://github.com/xunge/pytorch_lmdb_imagenet/blob/master/folder2lmdb.py



import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = None  # Don't open the LMDB here

        # Open the LMDB once just to fetch length and keys
        with lmdb.open(db_path, subdir=osp.isdir(db_path),
                       readonly=True, lock=False,
                       readahead=False, meminit=False).begin(write=False) as txn:
            self.length = self.loads_data(txn.get(b'__len__'))
            self.keys = self.loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def _ensure_opened(self):
        if self.env is None:
            self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False)

    def __getitem__(self, index):
        self._ensure_opened()

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = self.loads_data(byteflow)

        # Load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')


        if self.transform is not None:
            img = self.transform(img)

        # Convert PIL Image to numpy array
        #im2arr = np.array(img)

        # Load label
        target = unpacked[1]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    @staticmethod
    def loads_data(buf):
        return pickle.loads(buf)

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def folder2lmdb(dpath, name="train", write_frequency=5000):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    # generate lmdb
    folder2lmdb("/home/jiang/dataset/imagenet/", name="train")
    folder2lmdb("/home/jiang/dataset/imagenet/", name="val")
