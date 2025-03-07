import preprocessing

from data import x, meta, labels, np, class_names, ch_names, sampling_rate, mne
from preprocessing import rfreq, train_x, test_x, train_y, test_y

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from scipy.signal import butter, lfilter


def butter_bandpass_filter(signal, low_cut, high_cut, fs, order=5):
    if low_cut == 0:
        low_cut = 0.5
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return y


class FBCSP(object):
    # Filter Bank Common Spectral Pattern
    def __init__(self, sampling_rate, n_components, n_select):
        self.sampling_rate = sampling_rate
        self.select_k = SelectKBest(mutual_info_classif, k=n_select)
        self.scaler = StandardScaler()
        self.low_cut, self.high_cut, self.interval = 4, 40, 4
        self.bands = np.arange(self.low_cut, self.high_cut, self.interval)
        self.csp_list = {
            '{}-{}'.format(band, band+self.interval):
                CSP(n_components=n_components) for band in self.bands
        }
        self.classifier = LinearDiscriminantAnalysis()

    def train(self, x, y):
        # [stage 1, 2] => acquire and combine features of different frequency bands
        x = self.temporal_spatial_filtering(x, y, train=True)

        # [stage 3] : band selection => get the best k features base on mutual information algorithm
        x = self.select_k.fit_transform(x, y)

        # [stage 4] : classification
        x = self.scaler.fit_transform(x)
        self.classifier.fit(x, y)

    def predicate(self, x):
        # [stage 1, 2] => acquire and combine features of different frequency bands
        x = self.temporal_spatial_filtering(x, train=False)

        # [stage 3] : band selection => get the best k features base on mutual information algorithm
        x = self.select_k.transform(x)

        # [stage 4] : classification
        x = self.scaler.transform(x)
        out, prob = self.classifier.predict(x), \
                    self.classifier.predict_proba(x)
        return out, prob

    def temporal_spatial_filtering(self, x, y=None, train=True):
        new_x = []
        for band in self.bands:
            start_band, end_band = band, band + self.interval

            # [stage 1] : temporal filtering
            x_filter = butter_bandpass_filter(signal=x,
                                              low_cut=start_band, high_cut=end_band,
                                              fs=self.sampling_rate)

            # [stage 2] : spectral filtering
            try:
                if train:
                    csp = self.csp_list['{}-{}'.format(start_band, end_band)]
                    x_filter = csp.fit_transform(x_filter, y)
                    self.csp_list['{}-{}'.format(start_band, end_band)] = csp
                else:
                    csp = self.csp_list['{}-{}'.format(start_band, end_band)]
                    x_filter = csp.transform(x_filter)
            except np.linalg.LinAlgError:
                del self.csp_list['{}-{}'.format(start_band, end_band)]
                continue
            except KeyError:
                continue

            new_x.append(x_filter)
        return np.concatenate(new_x, axis=1)

model = FBCSP(
    sampling_rate=rfreq,
    n_components=8,
    n_select=10
)

model.train(x=train_x, y=train_y)

out, prob = model.predicate(x=test_x)

from sklearn.metrics import classification_report

print(classification_report(y_true=test_y, y_pred=out))