import data
from data import x, meta, labels, np, class_names, ch_names, sampling_rate, mne

train_x, test_x = x[meta['session'] == '0'], x[meta['session'] == '1']
train_y, test_y = labels[meta['session'] == '0'], labels[meta['session'] == '1']
train_y, test_y = np.array([class_names.index(label) for label in train_y]), \
                  np.array([class_names.index(label) for label in test_y])

rfreq = 250

info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
train_raw, test_raw = mne.EpochsArray(train_x, info=info), mne.EpochsArray(test_x, info=info)

train_raw.resample(rfreq)
train_raw.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)

test_raw.resample(rfreq)
test_raw.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)

train_x, test_x = train_raw.get_data(), test_raw.get_data()
print(train_y.shape)