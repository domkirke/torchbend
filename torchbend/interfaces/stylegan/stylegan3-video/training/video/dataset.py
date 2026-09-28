import numpy as np, torch, torchvision as tv, os, dill, re, random, pdb, tqdm, lardon
from torch.utils.data import Dataset
from math import ceil
from .transforms import VideoTransform
from torchvision import transforms as it

def check_file(file, types):
    return os.path.splitext(file)[1] in types and os.path.basename(file[0]) != "."

def checklist(item, n=1, copy=False):
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        else:
            item = [item]*n
    return item

def checktuple(item, n=1, copy=False):
    if not isinstance(item, tuple):
        if copy:
            item = tuple([copy.deepcopy(item) for _ in range(n)])
        else:
            item = tuple([item]*n)
    return item

def checkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

def check_file(file, types):
    return os.path.splitext(file)[1] in types and os.path.basename(file[0]) != "."

def checkdtype(dtype):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype), "dtype %s invalid"%dtype
        return dtype
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError("%s cannot be parsed as a dtype"%dtype)

class VideoDataset(Dataset):
    types = [".mp4", ".mov"]
    def __init__(self, 
                root_directory,
                transforms=[],
                flatten=False,
                refresh=False,
                dtype = torch.float32,
                **kwargs):
        self.root_directory = root_directory
        if isinstance(transforms, list):
            self.transforms = it.Compose(transforms)
        else:
            self.transforms = transforms 
        # self.augmentations = augmentations
        self.data = None
        self.files = []
        self.metadata = {}
        self.hash = {}
        self._shape = None
        self._flattened = False
        if os.path.isfile(f"{self.root_directory}/timestamps.ct") and not refresh:
            self.load_timestamps(f"{self.root_directory}/timestamps.ct")
        else:
            self.read_timestamps()
        if flatten:
            self.flatten_data()
            self._flattened = True
        self._sequence_mode = None
        self._sequence_length = None
        self._dtype = checkdtype(dtype)

    def __len__(self):
        return len(self.files)

    @property
    def shape(self):
        if len(self.files) > 0:
            return self[0][0].shape[-3:]

    def __getitem__(self, item, **kwargs):
        #if isinstance(item, slice):
        #    item = range(len(self))[item]
        data, seq = self._get_item(item, **kwargs)
        metadata = self._get_metadata(item, seq=seq, **kwargs)
        if self.transforms is not None:
            data = self.transforms(data)
        if "timestamps" in metadata:
            metadata['timestamps'] = float(metadata['timestamps'])
        data = data.to(self._dtype)
        return data, metadata

    def _get_item(self, item, **kwargs):
        sequence_mode = kwargs.get('sequence_mode', self._sequence_mode)
        sequence_length = kwargs.get('sequence_length', self._sequence_length)
        timestamps = self.metadata['timestamps'][item]
        if sequence_mode is not None:
            if sequence_mode == "random":
                start_idx = random.randrange(0, len(timestamps) - sequence_length)
                end_idx = start_idx + sequence_length
            elif sequence_mode == "start":
                start_idx = 0
                end_idx = sequence_length
            start = timestamps[start_idx]
            end = timestamps[end_idx]
        else:
            if hasattr(timestamps, "__iter__"):
                start = timestamps[0]
                end = timestamps[-1]
                start_idx = 0
                end_idx = len(timestamps)
            else:
                start = timestamps
                end = timestamps
                start_idx = 0
                end_idx = 0
        data, _, _ = tv.io.read_video(f"{self.root_directory}/data/{self.files[item]}", start, end, pts_unit="sec")
        if start == end:
            if data.shape[0] != 1:
                data = data[0][np.newaxis]
        if sequence_mode is not None:
            if data.shape[0] > sequence_length:
                data = data[:sequence_length]
        data = data.permute(0, 3, 1, 2)
        if data.shape[0] == 1:
            data = data[0]
        return data, (float(start_idx), float(end_idx))

    def _get_metadata(self, item, seq=None):
        metadata = {}
        for k, v in self.metadata.items():
            metadata[k] = v[item]
            if seq is not None:
                if hasattr(metadata[k], "__iter__"):
                    if seq[0] == seq[1]:
                        metadata[k] = metadata[k][seq[0]:seq[1]+1]
                    else:
                        metadata[k] = metadata[k][seq[0]:seq[1]]
        return metadata

    def drop_sequences(self, length, mode="random"):
        self._sequence_mode = mode
        self._sequence_length = length

    def read_timestamps(self):
        files = []
        for r, d, f in os.walk(f"{self.root_directory}/data"):
            valid_files = list(filter(lambda x: check_file(x, self.types), f))
            valid_files = [re.sub(f"{self.root_directory}/data/", "", f"{r}/{f}") for f in valid_files]
            files.extend(valid_files)
        if len(files) == 0:
            raise FileNotFoundError(f"no valid files found at {self.root_directory}")
        timestamps = {k: None for k in files}
        fps = {k: None for k in files}
        for video_path in tqdm.tqdm(files, desc="parsing video files...", total=len(files)):
            video_path_full = f"{self.root_directory}/data/{video_path}"
            tst, fps_tmp = tv.io.read_video_timestamps(video_path_full, pts_unit="sec")
            timestamps[video_path] = tst
            fps[video_path] = fps_tmp
        metadata_video = {'timestamps': timestamps, 'fps': fps}
        metadata_file = f"{self.root_directory}/timestamps.ct"
        with open(metadata_file, "wb") as f:
            dill.dump(metadata_video, f)
        self.files = files
        self.metadata = {'timestamps': [metadata_video['timestamps'][f] for f in files],
                         'fps': [metadata_video['fps'][f] for f in files]}
        self.hash = {i: f for f, i in enumerate(self.files)}

    def load_timestamps(self, filename):
        with open(filename, 'rb') as f:
            video_info = dill.load(f)
        self.files = list(video_info['timestamps'].keys())
        self.metadata = {'fps':[], 'timestamps':[]}
        for f in self.files:
            self.metadata['fps'].append(video_info['fps'][f])
            self.metadata['timestamps'].append(video_info['timestamps'][f])
        self.hash = {i: f for f, i in enumerate(self.files)}

    def write_transforms(self, path=None, force=False):
        """
        Write transforms in place, parsing the files to the target transform using lardon.
        Note : dataset should not be flattened, providing degenerated lardon pickling.
        :param name: transform name
        :param selector:
        :return:
        """
        checkdir(path)
        checkdir(f"{path}/data")
        new_timestamps = {}
        for i, d in enumerate(tqdm.tqdm(self.files, desc="exporting transforms...", total=len(self.files))):
            new_data = self.transforms(self._get_item(i)[0])
            new_data = new_data.permute(0, 2, 3, 1)
            current_path = f"{path}/data/{self.files[i]}"
            checkdir(os.path.dirname(current_path))
            tv.io.write_video(current_path, new_data, fps = self.metadata['fps'][i])
            new_timestamps[self.files[i]] = tv.io.read_video_timestamps(current_path, pts_unit="sec")[0]
        self.metadata['timestamps'] = [new_timestamps[self.files[i]] for i in range(len(self))]
        with open(f"{path}/transforms.ct", 'wb') as f:
            dill.dump(self.transforms, f)
        save_dict = self.get_attributes()
        with open(f"{path}/dataset.ct", "wb") as f:
            dill.dump(save_dict, f)
        with open(f"{path}/timestamps.ct", "wb") as f:
            fps = {self.files[i]: self.metadata['fps'][i] for i in range(len(self))}
            dill.dump({'timestamps':new_timestamps, 'fps': fps}, f)
        self.transforms = None
        self.root_directory = path

    def import_transform(self, transform):
        assert transform in self.available_transforms
        target_directory = f"{self.root_directory}/transforms/{transform}"
        if len(self.files) > 0:
            files = [os.path.splitext(f)[0] + lardon.lardon_ext for f in self.files]
        else:
            files = None
        self.data, self.metadata = lardon.parse_folder(target_directory, drop_metadata=True, files=files)
        with open(target_directory+'/transforms.ct', 'rb') as f:
            original_transform = dill.load(f)
        with open(target_directory + '/dataset.ct', 'rb') as f:
            save_dict = dill.load(f)
            self.load_dict(save_dict)
        self._pre_transform = original_transform
        return original_transform

    def get_attributes(self):
        return {'metadata': self.metadata}

    def load_dict(self, save_dict):
        self.metadata = save_dict['metadata']

    def flatten_data(self, axis=0):
        files = []
        metadata = {k: [] for k in self.metadata.keys()}
        hash = {f: [] for f in self.files}
        data = []
        for i, f in enumerate(self.files):
            if self.data is not None:
                data.extend(self.data.entries[i].scatter(axis))
            file_len = len(self.metadata['timestamps'][i])
            hash[f].extend(list(range(len(files), len(files)+file_len)))
            files.extend([f]*file_len)
            for k, v in self.metadata.items():
                current_value = v[i]
                if hasattr(current_value, "__iter__"):
                    metadata[k].extend(current_value)
                else:
                    metadata[k].extend([current_value]*file_len)
        self.files = files
        self.metadata = metadata
        self.hash = hash
        if self.data is not None:
            self.data = lardon.OfflineDataList(data)

    def make_partitions(self, names, balance, from_files=True):
        """
        Builds partitions from the data
        Args:
            names (list[str]) : list of partition names
            balance (list[float]) : list of partition balances (must sum to 1)
        """
        partition_files = {}
        if from_files:
            files = list(self.hash.keys())
            permutation = np.random.permutation(len(files))
            cum_ids = np.cumsum([0] + [ceil(n * (len(files)-1)) for n in balance])
            partitions = {}
            partition_files = {}
            for i, n in enumerate(names):
                partition_files[n] = [files[f] for f in permutation[cum_ids[i]:cum_ids[i + 1]]]
                partitions[n] = sum([checklist(self.hash[f]) for f in partition_files[n]], [])
        else:
            permutation = np.random.permutation(len(self.data))
            cum_ids = np.cumsum([0]+[int(n*len(self.data)) for n in balance])
            partitions = {}
            for i, n in enumerate(names):
                partitions[n] = permutation[cum_ids[i]:cum_ids[i+1]]
                partition_files[n] = [self.files[n] for n in permutation[cum_ids[i]:cum_ids[i+1]]]
        self.partitions = partitions
        self.partition_files = partition_files

    def retrieve(self, item):
        """
        Create a sub-dataset containing target items
        Args:
            item (iter[int] or str) : target data ids / partition

        Returns:
            subdataset (MidiDataset) : obtained sub-dataset
        """
        if isinstance(item, list):
            item = np.array(item)
        elif isinstance(item, torch.LongTensor):
            item = item.detach().numpy()
        elif isinstance(item, int):
            item = np.array([item])
        elif isinstance(item, str):
            item = self.partitions[item]
            if isinstance(item[0], str):
                item = sum([self.hash[i] for i in item], [])
        dataset = type(self)(self.root_directory, transforms=self.transforms)
        dataset.metadata = {k: (np.array(v)[item]).tolist() for k, v in self.metadata.items()}
        dataset.files = [self.files[f] for f in item]
        dataset.hash = {}
        for i, f in enumerate(dataset.files):
            dataset.hash[f] = dataset.hash.get(f, []) + [i]
        dataset._sequence_length = self._sequence_length
        dataset._sequence_mode = self._sequence_mode
        dataset._dtype = self._dtype
        return dataset
    

