from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import minmax_scale

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.segmentation import find_boundaries

from scipy import ndimage as ndi

import numpy as np


class Watershed:
    """Watershed segemention for two-dimensional data.

    Performs probability density estimation of data, then applies watershed
    to segement into discrete regions.

    Basically a wrapper on sklearn to manage preprocessing and store data.

    Args:
        prune_outliers (bool): To prune or not to prune.
        outlier_neighbors (int): Number of neighbors to use.
        outlier_thresholds (float): Outlier threshold.
        bandwidth (float): KDE bandwidth.
        ngrid (int): KDE grid resolution.
        ngrid_pad (float): Grid inset padding.
        peak_min_distance (int): Minimum number of pixels separating peaks.
        peak_threshold_rel (float): Minimum relative intensity of peaks.
        peak_dialation (float): Peak dialation factor.
        compactness (float): Compactness factor for compact watershed.

    Attributes:
        lof_: LocalOutlierFactor.
        kde_: KernelDensity estimator.
        Z_crop_: Z cropped for outliers.
        Z_norm_: Z normalized and inset.
        Z_labels_ (np array, shape (n_samples,)): Segment labels.
        P_ (np array, shape (ngrid, ngrid)): Probabilty density map.
        P_peaks_ (np array, shape (ngrid, ngrid)): KDE peaks.
        P_edt_ (np array, shape (ngrid, ngrid)): Euclidean distance transform.
        P_labels_ (np array, shape (ngrid, ngrid)): KDE Labels.
        P_bounds_ (np array, shape (ngrid, ngrid)): KDE Bounds .
        peaks_ (np array, shape (num_peaks, 2)): List of KDE peaks.
    """
    def __init__(self,
        prune_outliers=False,
        outlier_neighbors=1,
        outlier_threshold=0.7,
        bandwidth=1.0 / 40,
        ngrid=600,
        ngrid_pad=0.07,
        peak_min_distance=10,
        peak_threshold_rel=0.1,
        peak_dialation=2,
        compactness=0.01,
    ):
        self.prune_outliers = prune_outliers
        self.outlier_neighbors = outlier_neighbors
        self.outlier_threshold = outlier_threshold
        self.bandwidth = bandwidth
        self.ngrid = ngrid
        self.ngrid_pad = ngrid_pad
        self.peak_min_distance = peak_min_distance
        self.peak_threshold_rel = peak_threshold_rel
        self.peak_dialation = peak_dialation
        self.compactness = compactness


    def segment(self, Z, verbose=True):
        """Fit the model using Z as data to be segmented.

        Args:
            Z (np array, shape (n_samples, 2)): Data to be segemented.
            verbose (bool): Verbosity.

        Returns:
            labels (np array, (n_samples,): Segment label for each sample.
        """
        if verbose:
            print('Segmenting regions using watershed...')
            print('- num samples: {}'.format(len(Z)))

        # outliers
        if self.prune_outliers:
            if verbose:
                print('- pruning outliers')
            self.lof_ = LocalOutlierFactor(
                n_neighbors=self.outlier_neighbors, contamination=0.1
            )
            lof_pred = self.lof_.fit_predict(Z)
            lof_scores = self.lof_.negative_outlier_factor_
            lof_scores = minmax_scale(lof_scores)
            self.Z_crop_ = Z[lof_scores > self.outlier_threshold]
            self.Z_left_ = np.where(lof_scores > self.outlier_threshold)
            num_outliers = Z.shape[0] - self.Z_crop_.shape[0]
            print('-> outliers pruned: {}'.format(num_outliers))
        else:
            self.Z_crop_ = Z

        # normalize Z and inset
        self.Z_norm_ = minmax_scale(
            self.Z_crop_,
            feature_range=(0 + self.ngrid_pad, 1 - self.ngrid_pad),
            axis=0,
        )

        # estimate probability density using Gaussian kernal
        if verbose:
            print('- performing KDE')
        self.kde_ = KernelDensity(
            kernel='gaussian', bandwidth=self.bandwidth
        ).fit(self.Z_norm_)

        # convert density estimate to an image of probs and normalize
        if verbose:
            print('- scoring KDE')
        x, y = np.meshgrid(
            np.linspace(0, 1, self.ngrid), np.linspace(0, 1, self.ngrid)
        )
        log_dens = self.kde_.score_samples(
            np.array((x.flatten(), y.flatten())).T
        )
        self.P_ = np.reshape(log_dens, (self.ngrid, self.ngrid))
        self.P_ = np.exp(self.P_) / np.max(np.exp(self.P_))

        # find peaks
        if verbose:
            print('- finding peaks')
        self.peaks_ = peak_local_max(
            self.P_,
            min_distance=self.peak_min_distance,
            threshold_rel=self.peak_threshold_rel,
            exclude_border=False,
        )

        # convert peaks to image and dialate
        self.P_peaks_ = np.ones_like(self.P_)
        for peak in self.peaks_:
            for i in range(-self.peak_dialation, self.peak_dialation + 1):
                for j in range(-self.peak_dialation, self.peak_dialation + 1):
                    self.P_peaks_[(peak[0] + i, peak[1] + j)] = 0

        # euclidean distance transform
        if verbose:
            print('- computing edt')
        self.P_edt_ = ndi.distance_transform_edt(self.P_peaks_)

        # perform watershed on edt
        if verbose:
            print('- performing watershed on edt')
        markers = ndi.label(1 - self.P_peaks_)[0] # use peaks as seed markers
        self.P_labels_ = watershed(
            self.P_edt_, markers, compactness=self.compactness
        )

        # find boundaries
        if verbose:
            print('- finding boundaries')
        self.P_bounds_ = find_boundaries(self.P_labels_)

        # find labels for Zs
        indices = np.round(self.Z_norm_ * self.ngrid).astype(int)
        self.Z_labels_ = self.P_labels_[indices[:,1], indices[:,0]] # swap axes

        if verbose:
            print('-> num regions found: {}'.format(len(self.peaks_)))

        return self.Z_labels_
