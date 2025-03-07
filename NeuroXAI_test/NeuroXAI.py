import FBCSP

from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery, SSVEP
from neuroxai.utils import topography
from neuroxai.explanation import BrainExplainer
from data import x, meta, labels, np, class_names, ch_names, sampling_rate, mne
from preprocessing import rfreq, train_x, test_x, train_y, test_y
from FBCSP import model

class WrapperFBCSP(object):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predicate_prob(self, x):
        _, prob = self.model.predicate(x)
        return prob

model_wrapper = WrapperFBCSP(model=model)

explainer = BrainExplainer(class_names=class_names, kernel_width=25)
idx2chn = {i: ch_name for i, ch_name in enumerate(ch_names)}