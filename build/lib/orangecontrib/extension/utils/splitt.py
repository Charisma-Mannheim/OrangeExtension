



def train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets
    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState instance, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]
    >>> train_test_split(y, shuffle=False)
    [[0, 1, 2], [3, 4]]
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test,
                     train_size=n_train,
                     random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(chain.from_iterable((_safe_indexing(a, train),
                                     _safe_indexing(a, test)) for a in arrays))


# Tell nose that train_test_split is not a test.
# (Needed for external libraries that may use nose.)
# Use setattr to avoid mypy errors when monkeypatching.
setattr(train_test_split, '__test__', False)