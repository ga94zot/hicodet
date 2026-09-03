"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import numpy as np
import torch
from typing import Optional, List, Callable, Tuple
from pocket.data import ImageDataset, DataSubset

class HICODetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]
    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]
    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno
    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno
    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno

class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            llava_answer_path:Optional[str] = None,
            llava_token_path:Optional[str] = None,
            train_type:Optional[str] = 'default') -> None:
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80
        self.num_interation_cls = 600
        self.num_action_cls = 117
        self._anno_file = anno_file
        
        # Load annotations
        self._load_annotation_and_metadata(anno, train_type)
        if "train" in self._root:
            self.llava_answer_path = llava_answer_path.rstrip('/') + "/train/"
            self.llava_token_path = llava_token_path.rstrip('/') + "/train/"
        if "test" in self._root:
            self.llava_answer_path = llava_answer_path.rstrip('/') + "/test/"
            self.llava_token_path = llava_token_path.rstrip('/') + "/test/"
        self.train_type = train_type
        if self.train_type == 'default':
            self.train_idx = self._idx
        if self.train_type == 'RF_UC':
            self.train_idx = self._rf_uc_idx if "train" in self._root else self._idx
            self.seen = self._rf_seen
            self.unseen = self.rf_uc
        if self.train_type == 'NF_UC':
            self.train_idx = self._nf_uc_idx if "train" in self._root else self._idx
            self.seen = self._nf_seen
            self.unseen = self.nf_uc
        if self.train_type == 'UV':
            self.train_idx = self._uv_idx if "train" in self._root else self._idx
            self.seen = self._seen_v
            self.unseen = self.uv
        if self.train_type == "UO":
            self.train_idx = self._uo_idx if "train" in self._root else self._idx
            self.seen = self._seen_o
            self.unseen = self.uo
    def __len__(self) -> int:
        """Return the number of images"""
        return len(self.train_idx)
    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self.train_idx[i]
        with open(f"{self.llava_answer_path+self._filenames[intra_idx].replace('jpg', 'txt')}", "r") as f:
            llava_answer = f.readlines()
        try:
            # llava_answer_label = np.array(llava_answer[1].strip().split(" "), dtype=np.int64)
            # FIX (hoi_openworld, 2026-08-28): pocket.ops.relocate_to_cuda (called from the
            # DataLoader/collate path used by real training, not exercised by a bare
            # dataset[i] sanity check) only knows Tensor/list/dict and raises
            # `TypeError: Unsupported type of data <class 'numpy.ndarray'>` on a raw
            # numpy array. Wrap in torch.from_numpy() so the collated batch is all tensors.
            llava_answer_label = torch.from_numpy(
                np.array(llava_answer[1].strip().split(" "), dtype=np.int64)
            )
        except:
            # llava_answer_label = np.array([57], dtype=np.int64)
            llava_answer_label = torch.from_numpy(np.array([57], dtype=np.int64))
        llava_vision_feature = torch.load(f"{self.llava_token_path+self._filenames[intra_idx].replace('jpg', 'pt')}", "cpu").to(torch.float32)
        # 2026-09-03 (hoi_openworld, DINO-fusion experiment): if DINO_TOKEN_PATH is
        # set, load the DINOv3 dump for the same image and concatenate along the
        # feature dim. Both dumps are (577, D) with the SAME 24x24 patch grid at
        # 384x384 square resize (576 patches + 1 global token last), so the cat is
        # token-aligned: rows [:-1] are spatially corresponding patches. The model
        # side (pvic.py, gated by DINO_FUSE) splits [..., :1536] back out for the
        # decoder's VLM tokens and routes [..., 1536:] into FeatureHead. Default
        # (env unset) leaves this method byte-identical to before.
        _dino_root = os.environ.get("DINO_TOKEN_PATH")
        if _dino_root:
            # Reuse the train/test decision this class already made for
            # llava_token_path at __init__ (lines 84-87) instead of re-deriving it.
            _split = "train" if self.llava_token_path.rstrip("/").endswith("train") else "test"
            _dino_path = os.path.join(_dino_root, _split,
                                      self._filenames[intra_idx].replace("jpg", "pt"))
            _dino = torch.load(_dino_path, "cpu").to(torch.float32)
            llava_vision_feature = torch.cat([llava_vision_feature, _dino], dim=-1)

        return self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])), 
            self._anno[intra_idx]
            ), llava_answer_label, llava_vision_feature

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self) -> List[str]:
        """
        Object names 

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i] 
            for _, i, j in self._class_corr]

    @property
    def rare(self) -> List[int]:
        """
        List of rare class indices
        
        Returns:
            list[int]
        """
        return self._rare

    @property
    def non_rare(self) -> List [int]:
        """
        List of non-rare class indices

        Returns:
            list[int]
        """
        return self._non_rare
    
    @property
    def rf_uc(self) -> List[int]:
        """
        List of rare-first unseen combination class indices

        Returns:
        """
        return self._rf_uc
    
    @property
    def nf_uc(self) -> List[int]:
        """
        List of non-rare-first unseen combination class indices

        Returns:
        """
        return self._nf_uc
    
    @property
    def uv(self) -> List[int]:
        """
        List of unseen verb class indices

        Returns:
        """
        return self._uv
    
    @property
    def uo(self) -> List[int]:
        """
        List of unseen object class indices

        Returns:
        """
        return self._uo
    

    def split(self, ratio: float) -> Tuple[HICODetSubset, HICODetSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return HICODetSubset(self, perm[:n]), HICODetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict, train_type) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idx = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idx.remove(empty_idx)

        num_anno = [0 for _ in range(self.num_interation_cls)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idx
        self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._image_sizes = f['size']
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']
        self._rare = f['rare']
        self._non_rare = f['non_rare']
        hico_unseen_index = {
            "default": [],
            # start from 0
            "rare_first": [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                        70, 416,
                        389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                        345, 189,
                        205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                        158, 195,
                        238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                        216, 597,
                        77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                        55, 50,
                        198, 168, 391, 192, 595, 136, 581],  # 120
            "non_rare_first": [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                            212, 472, 61,
                            457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
                            230, 385, 73,
                            159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                            29, 594, 346,
                            456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
                            266, 304, 6, 572,
                            529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                            246, 173, 506,
                            383, 93, 516, 64],  # 120
            "unseen_object": [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                            126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                            294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                            338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                            429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                            463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                            537, 558, 559, 560, 561, 595, 596, 597, 598, 599],  # 100
            "unseen_verb": [4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97, 104, 113, 116, 118,
                            122, 129, 139, 147,
                            150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212, 219, 227, 228, 233, 235, 243, 298, 313,
                            315, 320, 326, 336,
                            342, 345, 354, 372, 401, 404, 409, 431, 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504,
                            519, 523, 535, 536,
                            541, 544, 562, 565, 569, 572, 591, 595]
            # 84, 20 unseen verbs: [41, 100, 99, 91, 34, 42, 97, 84, 26, 106, 38, 56, 92, 79, 19, 76, 80, 2, 114, 62]
        }
        self._rf_uc = hico_unseen_index['rare_first']
        self._rf_seen = list(range(600))
        for _ii_ in self._rf_uc:
            self._rf_seen.remove(_ii_)
        
        self._nf_uc = hico_unseen_index['non_rare_first']
        self._nf_seen = list(range(600))
        for _ii_ in self._nf_uc:
            self._nf_seen.remove(_ii_)
        
        self._uv = hico_unseen_index['unseen_verb']
        self._seen_v = list(range(600))
        for _ii_ in self._uv:
            self._seen_v.remove(_ii_)

        self._uo = hico_unseen_index['unseen_object']
        self._seen_o = list(range(600))
        for _ii_ in self._uo:
            self._seen_o.remove(_ii_)

        if train_type == "RF_UC":
            self._rf_uc_idx = self._idx.copy()
            if "train" in self._root:
                for i in idx:
                    for pair_idx, hoi in enumerate(self._anno[i]['hoi']):
                        if hoi in self._rf_uc:
                            for k in self._anno[i].keys():
                                self._anno[i][k].pop(pair_idx)
                        if len(self._anno[i]['hoi']) == 0:
                            self._rf_uc_idx.remove(i)
        
        if train_type == "NF_UC":
            self._nf_uc_idx = self._idx.copy()
            if "train" in self._root:
                for i in idx:
                    for pair_idx, hoi in enumerate(self._anno[i]['hoi']):
                        if hoi in self._nf_uc:
                            for k in self._anno[i].keys():
                                self._anno[i][k].pop(pair_idx)
                        if len(self._anno[i]['hoi']) == 0:
                            self._nf_uc_idx.remove(i)
        
        if train_type == "UV":
            self._uv_idx = self._idx.copy()
            if "train" in self._root:
                for i in idx:
                    for pair_idx, hoi in enumerate(self._anno[i]['hoi']):
                        if hoi in self._uv:
                            for k in self._anno[i].keys():
                                self._anno[i][k].pop(pair_idx)
                        if len(self._anno[i]['hoi']) == 0:
                            self._uv_idx.remove(i)
        
        if train_type == "UO":
            self._uo_idx = self._idx.copy()
            if "train" in self._root:
                for i in idx:
                    for pair_idx, hoi in enumerate(self._anno[i]['hoi']):
                        if hoi in self._uo:
                            for k in self._anno[i].keys():
                                self._anno[i][k].pop(pair_idx)
                        if len(self._anno[i]['hoi']) == 0:
                            self._uo_idx.remove(i)
