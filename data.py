import mne
import warnings
import moabb
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery, SSVEP
from neuroxai.utils import topography
from neuroxai.explanation import BrainExplainer

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
            'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']
class_names = ['right_hand', 'left_hand']

dataset = Lee2019_MI()
sampling_rate = 1000
subject_idxs = [33]
paradigm = LeftRightImagery()
x, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_idxs[0]])
