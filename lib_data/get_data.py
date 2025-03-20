import sys, os, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

from data_provider import RealDataOptimizablePoseProviderPose, RealDataOptimizablePoseProviderPoseStyle, DatabasePoseProvider
from personal_data_style import Dataset as PersonalDatasetStyle
import logging
import numpy as np
import torch
import pdb

def prepare_real_seq(
    seq_name,
    dataset_mode,
    split="train",
    image_zoom_ratio=0.5,
    balance=False,
    ins_avt_wild_start_end_skip=None,
):
    logging.info("Prepare real seq: {}".format(seq_name))
    # * Get dataset

    if dataset_mode == "personal_data":
        dataset = PersonalDataset(
            video_name=seq_name,
            split=split
        )
    else:
        raise NotImplementedError("Unknown mode: {}".format(dataset_mode))

    # prepare an optimizable data provider
    optimizable_data_provider = RealDataOptimizablePoseProviderPose(
        dataset,
        balance=balance,
    )
    return optimizable_data_provider, dataset


def prepare_real_seq_style(
    seq_name,
    dataset_mode,
    split="train",
    image_zoom_ratio=0.5,
    balance=False,
    ins_avt_wild_start_end_skip=None,
):
    logging.info("Prepare real seq: {}".format(seq_name))
    # * Get dataset

    if dataset_mode == "personal_data":
        dataset = PersonalDatasetStyle(
            video_name=seq_name,
            split=split
        )
    else:
        raise NotImplementedError("Unknown mode: {}".format(dataset_mode))

    # prepare an optimizable data provider
    optimizable_data_provider = RealDataOptimizablePoseProviderPoseStyle(
        dataset,
        balance=balance,
    )
    return optimizable_data_provider, dataset