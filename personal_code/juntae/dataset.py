import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
    RandomHorizontalFlip,
    RandomErasing,
    GaussianBlur,
)
from collections import Counter
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BestAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((380, 380)),
                Resize(resize, Image.BILINEAR),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomErasing(p=0.5),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class OldAugmentation:
    def __init__(self, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((380, 380)),
                Resize(380, Image.BILINEAR),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(0.01, 0.01, 0.01, 0.01),
                GaussianBlur(kernel_size=(3, 3)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        old_agumentation=False,
        balanced_split=True,
        exec_age_drop=True,
        age_drop_mode=["train", "val"],
        drop_age=["57", "58", "59"],
        exec_remove_fake=True,
        remove_fake_mode=["train", "val"],
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None

        self.old_agumentation = old_agumentation
        self.old_transform = OldAugmentation(self.mean, self.std)

        self.balanced_split = balanced_split

        self.exec_age_drop = exec_age_drop
        self.age_drop_mode = age_drop_mode
        self.drop_age = drop_age

        self.exec_remove_fake = exec_remove_fake
        self.remove_fake_mode = remove_fake_mode

        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if self.old_agumentation == True and age_label == 2:
            image_transform = self.old_transform(image)
        else:
            image_transform = self.transform(image)

        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MultiLabelMaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        old_agumentation=False,
        balanced_split=True,
        exec_age_drop=True,
        age_drop_mode=["train", "val"],
        drop_age=["57", "58", "59"],
        exec_remove_fake=True,
        remove_fake_mode=["train", "val"],
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None

        self.old_agumentation = old_agumentation
        self.old_transform = OldAugmentation(self.mean, self.std)

        self.balanced_split = balanced_split

        self.exec_age_drop = exec_age_drop
        self.age_drop_mode = age_drop_mode
        self.drop_age = drop_age

        self.exec_remove_fake = exec_remove_fake
        self.remove_fake_mode = remove_fake_mode

        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if self.old_agumentation == True and age_label == 2:
            image_transform = self.old_transform(image)
        else:
            image_transform = self.transform(image)

        return image_transform, mask_label, gender_label, age_label, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌
    사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    total_labels = []
    train_labels = []
    val_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        old_agumentation=False,
        balanced_split=True,
        exec_age_drop=True,
        age_drop_mode=["train", "val"],
        drop_age=["57", "58", "59"],
        exec_remove_fake=True,
        remove_fake_mode=["train", "val"],
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio, old_agumentation, balanced_split, exec_age_drop, age_drop_mode, drop_age, exec_remove_fake, remove_fake_mode)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def balanced_split_profile(self, profiles, val_ratio):
        print("balanced_split_profile..on")
        df = pd.DataFrame()
        df["path"] = profiles

        gender = []
        age = []
        for fname in df["path"]:
            info = fname.split("_")
            age.append(int(info[3]))
            gender.append(info[1])

        df["gender"] = gender
        df["age"] = age

        new_label = []
        gender_list = []
        age_list = []

        for g, a in zip(df["gender"], df["age"]):
            if g == "male":
                gender_list.append(0)
            elif g == "female":
                gender_list.append(1)

            if int(a) < 30:
                age_list.append(0)
            elif 30 <= int(a) < 60:
                age_list.append(1)
            else:
                age_list.append(2)

        for i in range(len(df)):
            new_label.append(0 * 6 + gender_list[i] * 3 + age_list[i])

        df["gender_age"] = new_label
        train, val = train_test_split(
            df, test_size=val_ratio, random_state=42, stratify=df["gender_age"]
        )
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}

    def age_drop(
        self,
        split_profiles,
        profiles,
        mode=["train", "val"],
        age_list=["57", "58", "59"],
    ):  # mode에서 특정 나이 제거
        print("age_drop..")
        for phase, indices in split_profiles.items():
            removed_idx = []
            if phase in mode:
                for idx in indices:
                    if profiles[idx].split("_")[-1] in age_list:
                        removed_idx.append(idx)
            for ri in removed_idx:
                split_profiles[phase].remove(ri)

        return split_profiles

    def remove_fake(
        self,
        split_profiles,
        profiles,
        mode=["train", "val"],
    ):  # mode에서 mixup 데이터 제거
        print("remove fake..")
        for phase, indices in split_profiles.items():
            removed_idx = []
            if phase in mode:
                for idx in indices:
                    if profiles[idx].split("_")[2] == "Fake":
                        removed_idx.append(idx)
            for ri in removed_idx:
                split_profiles[phase].remove(ri)

        return split_profiles

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]

        if self.balanced_split == True:
            split_profiles = self.balanced_split_profile(profiles, self.val_ratio)
        else:
            split_profiles = self._split_profile(profiles, self.val_ratio)

        if self.exec_remove_fake:
            split_profiles = self.remove_fake(
                split_profiles, profiles, self.remove_fake_mode
            )

        if self.exec_age_drop:
            split_profiles = self.age_drop(
                split_profiles, profiles, self.age_drop_mode, self.drop_age
            )

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.total_labels.append(
                        self.encode_multi_class(mask_label, gender_label, age_label)
                    )

                    self.indices[phase].append(cnt)
                    cnt += 1

        for phase, indices in self.indices.items():
            if phase == "train":
                for t in indices:
                    self.train_labels.append(self.total_labels[t])
            else:
                for v in indices:
                    self.val_labels.append(self.total_labels[v])

        # ct = Counter(self.train_labels)
        # cv = Counter(self.val_labels)
        # print('train ratio',sorted(ct.items()))
        # print('val ratio',sorted(cv.items()))
        # print('train_labels len', len(self.train_labels))
        # print('val_labels len', len(self.val_labels))
        # print()

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class MultiLabelMaskSplitByProfileDataset(MultiLabelMaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌
    사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    total_labels = []
    train_labels = []
    val_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        old_agumentation=False,
        balanced_split=True,
        exec_age_drop=True,
        age_drop_mode=["train", "val"],
        drop_age=["57", "58", "59"],
        exec_remove_fake=True,
        remove_fake_mode=["train", "val"],
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio, old_agumentation, balanced_split, exec_age_drop, age_drop_mode, drop_age, exec_remove_fake, remove_fake_mode)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def balanced_split_profile(self, profiles, val_ratio):
        print("balanced_split_profile..on")
        df = pd.DataFrame()
        df["path"] = profiles

        gender = []
        age = []
        for fname in df["path"]:
            info = fname.split("_")
            age.append(int(info[3]))
            gender.append(info[1])

        df["gender"] = gender
        df["age"] = age

        new_label = []
        gender_list = []
        age_list = []

        for g, a in zip(df["gender"], df["age"]):
            if g == "male":
                gender_list.append(0)
            elif g == "female":
                gender_list.append(1)

            if int(a) < 30:
                age_list.append(0)
            elif 30 <= int(a) < 60:
                age_list.append(1)
            else:
                age_list.append(2)

        for i in range(len(df)):
            new_label.append(0 * 6 + gender_list[i] * 3 + age_list[i])

        df["gender_age"] = new_label
        train, val = train_test_split(
            df, test_size=val_ratio, random_state=42, stratify=df["gender_age"]
        )
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}

    def age_drop(
        self,
        split_profiles,
        profiles,
        mode=["train", "val"],
        age_list=["57", "58", "59"],
    ):  # mode에서 특정 나이 제거
        print("age_drop..")
        for phase, indices in split_profiles.items():
            removed_idx = []
            if phase in mode:
                for idx in indices:
                    if profiles[idx].split("_")[-1] in age_list:
                        removed_idx.append(idx)
            for ri in removed_idx:
                split_profiles[phase].remove(ri)

        return split_profiles

    def remove_fake(
        self,
        split_profiles,
        profiles,
        mode=["train", "val"],
    ):  # mode에서 mixup 데이터 제거
        print("remove fake..")
        for phase, indices in split_profiles.items():
            removed_idx = []
            if phase in mode:
                for idx in indices:
                    if profiles[idx].split("_")[2] == "Fake":
                        removed_idx.append(idx)
            for ri in removed_idx:
                split_profiles[phase].remove(ri)

        return split_profiles

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]

        if self.balanced_split == True:
            split_profiles = self.balanced_split_profile(profiles, self.val_ratio)
        else:
            split_profiles = self._split_profile(profiles, self.val_ratio)

        if self.exec_remove_fake:
            split_profiles = self.remove_fake(
                split_profiles, profiles, self.remove_fake_mode
            )

        if self.exec_age_drop:
            split_profiles = self.age_drop(
                split_profiles, profiles, self.age_drop_mode, self.drop_age
            )

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.total_labels.append(
                        self.encode_multi_class(mask_label, gender_label, age_label)
                    )

                    self.indices[phase].append(cnt)
                    cnt += 1

        for phase, indices in self.indices.items():
            if phase == "train":
                for t in indices:
                    self.train_labels.append(self.total_labels[t])
            else:
                for v in indices:
                    self.val_labels.append(self.total_labels[v])

        # ct = Counter(self.train_labels)
        # cv = Counter(self.val_labels)
        # print("train ratio", sorted(ct.items()))
        # print("val ratio", sorted(cv.items()))
        # print("train_labels len", len(self.train_labels))
        # print("val_labels len", len(self.val_labels))
        # print()

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


class TTADataset(Dataset):  # Apply TestTimeAugmentation
    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = Compose(
            [
                CenterCrop((380, 380)),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
        print("TTADataset is Used")

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
