import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import shared.utils as su


def normalize(X):
    """Normalize the input tensor X along the last dimension."""
    return torch.nn.functional.normalize(X, dim=-1)


def get_split_data(
    df_base,
    df_split,
    match_key,
    matrix,
    shuffle=False,
    match_col="triplet_id",
    id_col="id",
):
    """Get the split data for a given triplet id."""
    row = df_split[df_split[match_col] == match_key].iloc[0]
    ids = row.id_forward.split(";") + row.id_reverse.split(";")
    idx = np.array([np.where(df_base[id_col] == _id)[0][0] for _id in ids])
    X = matrix[idx]
    Y = np.concatenate(
        [
            np.ones(len(row.id_forward.split(";"))),
            np.zeros(len(row.id_reverse.split(";"))),
        ]
    )
    Y = torch.from_numpy(Y)

    if shuffle:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

    return X, Y


def chiral_action_recognition_avg_accuracy(
    X,
    df,
    df_train,
    df_valid,
    match_col="triplet_id",
    id_col="id",
    return_all=False,
):
    """Computes average accuracy of linear probes across chiral pairs."""
    assert len(X) == len(df)

    accs = []
    iterator = su.log.tqdm_iterator(
        df_train[match_col].unique(),
        desc="Evaluating linear probes across chiral actions",
    )
    for triplet_id in iterator:
        X_train, Y_train = get_split_data(
            df,
            df_train,
            triplet_id,
            X,
            shuffle=True,
            match_col=match_col,
            id_col=id_col,
        )
        X_valid, Y_valid = get_split_data(
            df,
            df_valid,
            triplet_id,
            X,
            shuffle=True,
            match_col=match_col,
            id_col=id_col,
        )
        acc = get_linear_probe_accuracy(
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            verbose=False,
        )
        # accs.append(acc)
        row = {
            "triplet_id": triplet_id,
            "accuracy": acc,
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "n_total": len(X_train) + len(X_valid),
            "verb_forward": df_train[df_train[match_col] == triplet_id]
            .iloc[0]
            .verb_forward,
            "verb_reverse": df_train[df_train[match_col] == triplet_id]
            .iloc[0]
            .verb_reverse,
        }
        if "noun_abstract" in df_train[df_train[match_col] == triplet_id].iloc[0]:
            row["noun"] = (
                df_train[df_train[match_col] == triplet_id].iloc[0].noun_abstract
            )
        elif "noun" in df_train[df_train[match_col] == triplet_id].iloc[0]:
            row["noun"] = df_train[df_train[match_col] == triplet_id].iloc[0].noun
        elif "noun_class" in df_train[df_train[match_col] == triplet_id].iloc[0]:
            row["noun"] = df_train[df_train[match_col] == triplet_id].iloc[0].noun_class
        else:
            row["noun"] = None
        accs.append(row)
    accs = pd.DataFrame(accs)
    if return_all:
        return accs
    else:
        acc = np.mean(accs["accuracy"])
        return acc


# https://arxiv.org/pdf/1907.08340v2
TEMPORAL_CLASSES_KINETICS = [
    "bouncing on trampoline",
    "breakdancing",
    "busking",
    "cartwheeling",
    "cleaning shoes",
    "country line dancing",
    "drop kicking",
    "gymnastics tumbling",
    "hammer throw",
    "high kick",
    "jumpstyle dancing",
    "kitesurfing",
    "parasailing",
    "playing cards",
    "playing cymbals",
    "playing drums",
    "playing ice hockey",
    "robot dancing",
    "shining shoes",
    "shuffling cards",
    "side kick",
    "ski jumping",
    "skiing (not slalom or crosscountry)",
    "skiing crosscountry",
    "skiing slalom",
    "snowboarding",
    "somersaulting",
    "tap dancing",
    "throwing ball",
    "throwing discus",
    "vault",
    "wrestling",
]
TEMPORAL_CLASSES_SSv2 = [
    "Turning [something] upside down",
    "Approaching [something] with your camera",
    "Moving away from [something] with your camera",
    "Moving [something] towards the camera",
    "Moving [something] away from the camera",
    "Moving [something] and [something] closer to each other",
    "Moving [something] and [something] away from each other",
    "Moving [something] away from [something]",
    "Moving [something] closer to [something]",
    "Lifting [something] with [something] on it",
    "Uncovering [something]",
    "Pretending to turn [something] upside down",
    "Covering [something] with [something]",
    "Lifting up one end of [something], then letting it drop down",
    "Lifting [something] up completely without letting it drop down",
    "Lifting [something] up completely, then letting it drop down",
    "Stuffing [something] into [something]",
    "Moving [something] and [something] so they collide with each other",
]


def load_dataframes(ds_name, add_video_path=True, split_version="v4.0-080225"):
    if ds_name == "ssv2":
        from video_language.datasets.ssv2 import (
            get_paths,
            get_video_path_from_id,
        )
        from adapt4change.datasets.ssv2 import (
            load_video_classification_metadata,
        )

        paths = get_paths(split_version=split_version)
        df_train = load_video_classification_metadata(
            paths,
            split="train",
            verbose=False,
        )
        df_valid = load_video_classification_metadata(
            paths,
            split="validation",
            verbose=False,
        )
        kwargs = dict(id_col="id", target_col="template")
    
    elif ds_name == "synthetic_motion":
        from video_language.datasets.misc import (
            get_paths_synthetic_motion,
            get_video_path_from_id_synthetic_motion as get_video_path_from_id,
            load_main_csv_synthetic_motion,
        )
        paths = get_paths_synthetic_motion()
        df_train = load_main_csv_synthetic_motion(paths, "all")
        df_valid = df_train.copy()
        kwargs = dict(id_col="video_id", target_col="motion_type")
    
    elif ds_name == "synthetic_trajectories":
        from video_language.datasets.misc import (
            get_paths_synthetic_trajectories,
            get_video_path_from_id_synthetic_trajectories as get_video_path_from_id,
            load_main_csv_synthetic_trajectories,
        )
        paths = get_paths_synthetic_trajectories()
        df_train = load_main_csv_synthetic_trajectories(paths, "train")
        df_valid = df_train.copy()
        kwargs = dict(id_col="video_id", target_col="traj_type")

    elif ds_name == "speedykinetics":
        from video_language.datasets.misc import (
            get_paths_speedykinetics,
            get_video_path_from_id_speedykinetics as get_video_path_from_id,
            load_main_csv_speedykinetics,
        )
        paths = get_paths_speedykinetics()
        df_train = load_main_csv_speedykinetics(paths, "train")
        df_valid = load_main_csv_speedykinetics(paths, "val")
        kwargs = dict(id_col="video_id", target_col="target")

    elif ds_name == "ego4d_subset":
        from video_language.datasets.ego4d_subset import (
            get_paths,
            get_video_path_from_id,
            load_main_csv,
        )
        paths = get_paths()
        df_train = load_main_csv(paths, "train")
        df_valid = load_main_csv(paths, "val")
        kwargs = dict(id_col="id", target_col="caption_forward")
    
    elif ds_name == "rtime":
        from video_language.datasets.rtime import (
            get_paths,
            get_video_path_from_id,
            load_main_csv as load_main_csv_rtime,
        )
        paths = get_paths()
        df = pd.read_csv(
            os.path.join(paths['split_dir'], "rtime_all-02072025.csv")
        )
        df_train = df[df['split'] == 'train']
        df_valid = df[df['split'] == 'valid']
        kwargs = dict(id_col="id", target_col="text")
        # df_train = load_main_csv_rtime(paths, "train")
        # df_train['split_original'] = 'train'
        # # HACK to get the valid set
        # df_valid = load_main_csv_rtime(paths, "valid")
        # df_valid['split_original'] = 'valid'
        # df_test = load_main_csv_rtime(paths, "test")
        # df_test['split_original'] = 'test'
        # df_valid = pd.concat([df_valid, df_test], ignore_index=True)
        # kwargs = dict(id_col="video_id", target_col="url")

    elif ds_name == "ssv2-temporal":
        from video_language.datasets.ssv2 import (
            get_paths,
            get_video_path_from_id,
        )
        from adapt4change.datasets.ssv2 import (
            load_video_classification_metadata,
        )

        paths = get_paths(split_version=split_version)
        df_train = load_video_classification_metadata(
            paths,
            split="train",
            verbose=False,
        )
        df_valid = load_video_classification_metadata(
            paths,
            split="validation",
            verbose=False,
        )

        # https://arxiv.org/pdf/1907.08340v2
        temporal_classes_ssv2 = TEMPORAL_CLASSES_SSv2
        print("Number of temporal classes: ", len(temporal_classes_ssv2))

        # Only keep the temporal classes
        df_train = df_train[df_train["template"].isin(temporal_classes_ssv2)]
        df_valid = df_valid[df_valid["template"].isin(temporal_classes_ssv2)]
        kwargs = dict(id_col="id", target_col="template")

    elif ds_name == "charades":
        from video_language.datasets.charades import (
            get_paths,
            get_video_path_from_id,
            load_main_csv,
        )

        paths = get_paths()
        _, df_train = load_main_csv(paths, "train")
        _, df_valid = load_main_csv(paths, "test")
        kwargs = dict(id_col="item_id", target_col="label")

    elif ds_name == "epic":
        from video_language.datasets.epic import (
            get_paths,
            get_video_path_from_id,
            load_main_csv,
        )

        paths = get_paths(split_version=split_version)
        df_train = load_main_csv(paths, "train")
        df_valid = load_main_csv(paths, "validation")
        kwargs = dict(id_col="path_id", target_col="verb_class")

        # Add a column for text descriptions
        df_train["text_desc"] = df_train[["verb", "noun"]].apply(
            lambda x: f"{x['verb']} {x['noun']}", axis=1
        )
        df_valid["text_desc"] = df_valid[["verb", "noun"]].apply(
            lambda x: f"{x['verb']} {x['noun']}", axis=1
        )
        df_train = df_train.drop_duplicates(subset=['path_id'])
        df_valid = df_valid.drop_duplicates(subset=['path_id'])

    elif ds_name == "kinetics400":
        from video_language.datasets.misc import (
            get_paths_kinetics400,
            get_video_path_from_id_kinetics400 as get_video_path_from_id,
            load_main_csv_kinetics400,
        )

        paths = get_paths_kinetics400()
        df_train = load_main_csv_kinetics400(paths, "train")
        # NOTE: Test labels are currently not available
        # df_valid = load_main_csv_kinetics400(paths, "test")
        df_valid = load_main_csv_kinetics400(paths, "val")
        kwargs = dict(id_col="id", target_col="class")

    elif ds_name == "kinetics400-temporal":
        from video_language.datasets.misc import (
            get_paths_kinetics400,
            get_video_path_from_id_kinetics400 as get_video_path_from_id,
            load_main_csv_kinetics400,
        )

        paths = get_paths_kinetics400()
        df_train = load_main_csv_kinetics400(paths, "train")
        df_valid = load_main_csv_kinetics400(paths, "val")

        # https://arxiv.org/pdf/1907.08340v2
        temporal_classes_kinetics = TEMPORAL_CLASSES_KINETICS
        print("Number of temporal classes: ", len(temporal_classes_kinetics))

        # Only keep the temporal classes
        df_train = df_train[df_train["class"].isin(temporal_classes_kinetics)]
        df_valid = df_valid[df_valid["class"].isin(temporal_classes_kinetics)]
        kwargs = dict(id_col="id", target_col="class")

    elif ds_name == "temporal":
        # Temporal dataset proposed in https://arxiv.org/pdf/1907.08340v2

        # 1. Load Kinetics400 dataset
        from video_language.datasets.misc import (
            get_paths_kinetics400,
            get_video_path_from_id_kinetics400,
            load_main_csv_kinetics400,
        )

        paths_kinetics = get_paths_kinetics400()
        df_train_kinetics = load_main_csv_kinetics400(paths_kinetics, "train")
        df_valid_kinetics = load_main_csv_kinetics400(paths_kinetics, "val")
        kwargs_kinetics = dict(id_col="id", target_col="class")
        tck = TEMPORAL_CLASSES_KINETICS
        print("Number of temporal classes in Kinetics400: ", len(tck))
        df_train_kinetics = df_train_kinetics[df_train_kinetics["class"].isin(tck)]
        df_valid_kinetics = df_valid_kinetics[df_valid_kinetics["class"].isin(tck)]

        # Rename Kinetics400 column "class" to "class_label"
        df_train_kinetics = df_train_kinetics.rename(
            columns={"class": "class_label"},
        )
        df_valid_kinetics = df_valid_kinetics.rename(
            columns={"class": "class_label"},
        )
        df_train_kinetics["dataset"] = "kinetics400"
        df_valid_kinetics["dataset"] = "kinetics400"
        print(
            "Number of samples in Kinetics400: ",
            len(df_train_kinetics) + len(df_valid_kinetics),
        )

        # 2. Load SSv2 dataset
        from video_language.datasets.ssv2 import (
            get_paths as get_paths_ssv2,
            get_video_path_from_id as get_video_path_from_id_ssv2,
        )
        from adapt4change.datasets.ssv2 import (
            load_video_classification_metadata,
        )

        paths_ssv2 = get_paths_ssv2(split_version=split_version)
        df_train_ssv2 = load_video_classification_metadata(
            paths_ssv2,
            split="train",
            verbose=False,
        )
        df_valid_ssv2 = load_video_classification_metadata(
            paths_ssv2,
            split="validation",
            verbose=False,
        )
        kwargs_ssv2 = dict(id_col="id", target_col="template")
        temporal_classes_ssv2 = TEMPORAL_CLASSES_SSv2
        print("Number of temporal classes in SSv2: ", len(temporal_classes_ssv2))
        df_train_ssv2 = df_train_ssv2[
            df_train_ssv2["template"].isin(temporal_classes_ssv2)
        ]
        df_valid_ssv2 = df_valid_ssv2[
            df_valid_ssv2["template"].isin(temporal_classes_ssv2)
        ]

        # Rename SSv2 column "template" to "class_label"
        df_train_ssv2 = df_train_ssv2.rename(columns={"template": "class_label"})
        df_valid_ssv2 = df_valid_ssv2.rename(columns={"template": "class_label"})
        df_train_ssv2["dataset"] = "ssv2"
        df_valid_ssv2["dataset"] = "ssv2"
        print("Number of samples in SSv2: ", len(df_train_ssv2) + len(df_valid_ssv2))

        # Combine the two datasets
        df_train = pd.concat([df_train_kinetics, df_train_ssv2], ignore_index=True)
        df_valid = pd.concat([df_valid_kinetics, df_valid_ssv2], ignore_index=True)
        kwargs = dict(id_col="id", target_col="class_label")

        # Define get_video_path_from_id function
        kinetics_ids = np.concatenate(
            [df_train_kinetics["id"].values, df_valid_kinetics["id"].values]
        )
        ssv2_ids = np.concatenate(
            [df_train_ssv2["id"].values, df_valid_ssv2["id"].values]
        )

        def get_video_path_from_id(id, video_dir, ext="mp4"):
            if id in kinetics_ids:
                return get_video_path_from_id_kinetics400(
                    id, paths_kinetics["video_dir"], ext
                )
            elif id in ssv2_ids:
                return get_video_path_from_id_ssv2(id, paths_ssv2["video_dir"], ext)
            else:
                raise ValueError(f"Unknown id: {id}")

        # Define paths (not used)
        paths = {
            "video_dir": paths_kinetics["video_dir"],
        }

    elif ds_name == "ucf101":
        from video_language.datasets.misc import (
            get_paths_ucf101,
            get_video_path_from_id_ucf101 as get_video_path_from_id,
            load_main_csv_ucf101,
        )

        paths = get_paths_ucf101(ds_name="UCF101")
        df_train = load_main_csv_ucf101(paths, "train01")
        df_valid = load_main_csv_ucf101(paths, "test01")
        kwargs = dict(id_col="id", target_col="class")

    elif ds_name == "ucf101_aot":
        from video_language.datasets.misc import (
            get_paths_ucf101,
            get_video_path_from_id_ucf101 as get_video_path_from_id,
            load_main_csv_ucf101,
        )

        paths = get_paths_ucf101(ds_name="UCF101ArrowOfTime")
        df_train = load_main_csv_ucf101(
            paths,
            "ucf101_irreversible_actions-train01",
            id_col="file_id",
        )
        df_valid = load_main_csv_ucf101(
            paths,
            "ucf101_irreversible_actions-test01",
            id_col="file_id",
        )
        kwargs = dict(id_col="file_id", target_col="label")

    elif ds_name == "hmdb51":
        from video_language.datasets.misc import (
            get_paths_hmdb51,
            get_video_path_from_id_hmdb51 as get_video_path_from_id,
            load_main_csv_hmdb51,
        )

        paths = get_paths_hmdb51()
        df_train = load_main_csv_hmdb51(paths, "train")
        df_valid = load_main_csv_hmdb51(paths, "val")
        kwargs = dict(id_col="id", target_col="class")

    elif ds_name == "synthetic":
        from video_language.datasets.synthetic import (
            get_paths as get_paths_synthetic,
            get_video_path_from_id,
            load_main_csv,
        )

        paths = get_paths_synthetic()
        # split_mode = "random"
        # split_mode = "color"
        split_mode = "shape"
        df_train = load_main_csv(paths, f"{split_mode}_train")
        df_valid = load_main_csv(paths, f"{split_mode}_test")
        kwargs = dict(id_col="video_id", target_col="chiral_label")
    
    elif ds_name == "aotbench-seqdirclf":
        from video_language.datasets.aotbench_seqdirclf import (
            get_paths,
            load_main_csv,
            get_video_path_from_id,
        )
        paths = get_paths()
        df_train = load_main_csv(paths, "train")
        df_valid = load_main_csv(paths, "validation")
        kwargs = dict(id_col="path_id", target_col="label")

    elif ds_name == "tsh":
        from video_language.datasets.tsh import (
            get_paths as get_paths_tsh,
            get_video_path_from_id,
            load_main_csv,
        )

        paths = get_paths_tsh(split_version=split_version)
        df_train = load_main_csv(paths, "train")
        df_valid = load_main_csv(paths, "validation")
        kwargs = dict(id_col="id", target_col="class_label")

    elif ds_name == "jester":
        from video_language.datasets.misc import (
            get_paths_jester,
            get_video_path_from_id_jester as get_video_path_from_id,
            load_main_csv_jester,
        )

        paths = get_paths_jester()
        df_train = load_main_csv_jester(paths, "train")
        df_valid = load_main_csv_jester(paths, "validation")
        kwargs = dict(id_col="id", target_col="label")

    elif ds_name == "webvid":
        from video_language.datasets.misc import (
            get_paths_webvid,
            get_video_path_from_id_webvid as get_video_path_from_id,
            load_main_csv_webvid,
        )

        paths = get_paths_webvid()
        # Only train set for this: hard coded for now: has 2.57M rows
        df_train = load_main_csv_webvid(paths=paths, split="webvid10m_motion")
        df_valid = df_train.copy()
        kwargs = dict(id_col="id", target_col="name")

    elif ds_name == "webvid-chiral-subset":
        from video_language.datasets.misc import (
            get_paths_webvid,
            get_video_path_from_id_webvid as get_video_path_from_id,
            load_main_csv_webvid,
        )

        paths = get_paths_webvid()
        # Only train set for this: hard coded for now: has 2.57M rows
        df_train = load_main_csv_webvid(paths=paths, split="webvid10M-chiral_subset_80k-sim=0.55-07072025")
        df_valid = df_train.copy()
        kwargs = dict(id_col="id", target_col="name")

    elif ds_name == "sequential_mnist":
        from video_language.datasets.misc import (
            get_paths_sequential_mnist,
            get_video_path_from_id_sequential_mnist as get_video_path_from_id,
            load_main_csv_sequential_mnist,
        )

        paths = get_paths_sequential_mnist()
        df_train = load_main_csv_sequential_mnist(paths, "all")
        df_train["video_id"] = df_train["video_id"].astype(str)
        df_valid = load_main_csv_sequential_mnist(paths, "all")
        df_valid["video_id"] = df_valid["video_id"].astype(str)
        kwargs = dict(id_col="video_id", target_col="digit_label")
    
    elif ds_name == "translation_mnist":
        from video_language.datasets.misc import (
            get_paths_translation_mnist,
            get_video_path_from_id_translation_mnist as get_video_path_from_id,
            load_main_csv_translation_mnist,
        )
        paths = get_paths_translation_mnist()
        df_train = load_main_csv_translation_mnist(paths, "all")
        df_valid = df_train.copy()
        kwargs = dict(id_col="video_id", target_col="desc")

    elif ds_name == "tempcompass":
        from video_language.datasets.misc import (
            get_paths_tempcompass,
            get_video_path_from_id_tempcompass as get_video_path_from_id,
            load_main_csv_tempcompass,
        )
        paths = get_paths_tempcompass()
        df_train = load_main_csv_tempcompass(paths, "tempcompass_yes_no")
        df_valid = df_train.copy()
        df_valid["video_id"] = df_valid["video_id"].astype(str)
        df_train["video_id"] = df_train["video_id"].astype(str)
        df_valid["video_id"] = df_valid["video_id"].astype(str)
        kwargs = dict(id_col="video_id", target_col="answer")
    
    elif ds_name == "driveandact":
        from video_language.datasets.misc import (
            get_paths_driveandact,
            get_video_path_from_id_driveandact as get_video_path_from_id,
            load_main_csv_driveandact,
        )
        paths = get_paths_driveandact()
        df_train = load_main_csv_driveandact(paths, "train")
        df_valid = load_main_csv_driveandact(paths, "val")
        kwargs = dict(id_col="clip_id", target_col="activity")

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    # Add video path to the dataframes
    if add_video_path:
        from tqdm import tqdm

        tqdm.pandas(
            desc="Adding video paths",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )
        df_train["video_path"] = df_train[kwargs["id_col"]].progress_apply(
            lambda x: get_video_path_from_id(x, paths["video_dir"])[0]
        )
        df_valid["video_path"] = df_valid[kwargs["id_col"]].progress_apply(
            lambda x: get_video_path_from_id(x, paths["video_dir"])[0]
        )

    return df_train, df_valid, paths, kwargs, get_video_path_from_id


def save_numpy(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, x)


class FeaturesDataset(Dataset):
    def __init__(self, df, id_col, target_col, feat_dir):
        super().__init__()
        self.df = df
        self.id_col = id_col
        self.target_col = target_col
        from sklearn.preprocessing import LabelEncoder

        self.le = LabelEncoder()
        self.le.fit(df[target_col].unique())
        self.feat_dir = feat_dir

    def get_feat_path_from_id(self, id):
        return os.path.join(self.feat_dir, f"{id}.npy")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i].to_dict()
        x = torch.from_numpy(np.load(self.get_feat_path_from_id(row[self.id_col])))
        # x = torch.mean(x, dim=1)
        y = row[self.target_col]
        y = self.le.transform([y])[0]
        return dict(x=x, y=y)


def gather_data(df, mode, feat_dir, num_workers=4, batch_size=256, **ds_args):
    ds_args["feat_dir"] = feat_dir
    ds = FeaturesDataset(df, **ds_args)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    iterator = su.log.tqdm_iterator(dl, desc=f"Loading features for {mode}")
    X = []
    Y = []
    for batch in iterator:
        X.append(batch["x"])
        Y.append(batch["y"])
    X = torch.cat(X)
    Y = torch.cat(Y)
    print("Number of samples: ", X.shape, Y.shape)
    return X, Y


def get_linear_probe_accuracy(
    X_train, Y_train, X_valid, Y_valid, verbose=True, clf="ridge"
):
    from sklearn.linear_model import RidgeClassifier

    if verbose:
        print(
            f"Fitting classifier on {X_train.shape} samples for {len(np.unique(Y_train))} classes ..."
        )
    if clf == "ridge":
        clf = RidgeClassifier()
    elif clf == "logistic":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
    elif clf == "kernel_svm":
        from sklearn.svm import SVC

        clf = SVC(kernel="rbf")
    else:
        raise ValueError(f"Unknown classifier: {clf}")
    clf.fit(X_train, Y_train)
    if verbose:
        print(f"Done fitting classifier. Evaluating ...")
    try:
        train_acc = np.round(
            100.0 * (clf.predict(X_train) == Y_train).float().mean().item(), 3
        )
        valid_acc = np.round(
            100.0 * (clf.predict(X_valid) == Y_valid).float().mean().item(), 3
        )
    except:
        train_acc = np.round(100.0 * (clf.predict(X_train) == Y_train).mean(), 3)
        valid_acc = np.round(100.0 * (clf.predict(X_valid) == Y_valid).mean(), 3)
    if verbose:
        print("." * 120)

    return valid_acc


def filter_chiral_subset(df, split_dir, id_col="id", prefix="chiral_actions"):
    df_chiral_train = pd.read_csv(os.path.join(split_dir, f"{prefix}_train.csv"))
    df_chiral_valid = pd.read_csv(os.path.join(split_dir, f"{prefix}_validation.csv"))
    chiral_ids = []
    for col in ["id_forward", "id_reverse"]:
        chiral_ids.append(
            np.concatenate(df_chiral_train[col].apply(lambda x: x.split(";")).values)
        )
        chiral_ids.append(
            np.concatenate(df_chiral_valid[col].apply(lambda x: x.split(";")).values)
        )
    chiral_ids = np.concatenate(chiral_ids)
    len(chiral_ids)
    df = df[df[id_col].isin(chiral_ids)]
    print("Only keeping videos relevant for chiral actions: ", len(df))
    return df


def add_chiral_labels(
        df,
        split_dir,
        split,
        id_col="id",
        prefix="chiral_actions",
        noun_col="noun_abstract",
        return_chiral_df=False,
        add_noun=True,
    ):
    df_chiral = pd.read_csv(os.path.join(split_dir, f"{prefix}_{split}.csv"))
    all_ids = []
    for i in range(len(df_chiral)):
        row_chiral = df_chiral.iloc[i].to_dict()
        ids_forward = row_chiral["id_forward"].split(";")
        df.loc[df[id_col].isin(ids_forward), "chiral_label"] = 1
        df.loc[df[id_col].isin(ids_forward), "chiral_triplet_id"] = row_chiral[
            "triplet_id"
        ]
        if add_noun:
            df.loc[df[id_col].isin(ids_forward), "noun"] = row_chiral.get(noun_col, "na")
        ids_reverse = row_chiral["id_reverse"].split(";")
        df.loc[df[id_col].isin(ids_reverse), "chiral_label"] = 0
        df.loc[df[id_col].isin(ids_reverse), "chiral_triplet_id"] = row_chiral[
            "triplet_id"
        ]
        if add_noun:
            df.loc[df[id_col].isin(ids_reverse), "noun"] = row_chiral.get(noun_col, "na")
        all_ids.extend(ids_forward)
        all_ids.extend(ids_reverse)
    print("Number of videos with chiral labels: ", len(all_ids))
    df = df[df[id_col].isin(all_ids)]
    if return_chiral_df:
        return df, df_chiral
    return df


def filter_chiral_samples(df_base, df_chiral):
    # Only keep the rows in df_base with chiral actions
    id_to_chiral_triplet_id = dict()
    id_to_chiral_label = dict()
    id_to_noun_abstract = dict()
    chiral_ids = []
    for i in range(len(df_chiral)):
        row = df_chiral.iloc[i].to_dict()
        _ids = row["id_forward"].split(";") + row["id_reverse"].split(";")
        chiral_ids.extend(_ids)
        id_to_chiral_triplet_id.update(
            {x: row['triplet_id'] for x in _ids}
        )
        id_to_chiral_label.update(
            **{x: 1 for x in row["id_forward"].split(";")}, # Forward is 1
            **{x: 0 for x in row["id_reverse"].split(";")}, # Reverse is 0
        )
        id_to_noun_abstract.update(
            **{x: row['noun_abstract'] for x in _ids},
        )
    df_base = df_base[df_base.id.isin(chiral_ids)]
    print("[:::] Number of chiral action videos:", len(df_base))
    df_base["chiral_triplet_id"] = df_base["id"].apply(
        lambda x: id_to_chiral_triplet_id[x])
    df_base["chiral_label"] = df_base["id"].apply(
        lambda x: id_to_chiral_label[x])
    df_base["noun_abstract"] = df_base["id"].apply(
        lambda x: id_to_noun_abstract[x])
    return df_base
