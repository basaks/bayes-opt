import time
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=.25, random_state=0)


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms'.format((method.__name__, (te - ts) * 1000)))
        return result
    return timed


param_grid = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'gamma': (1e-6, 1e+1, 'log-uniform'),
    'degree': (1, 8),  # integer valued parameter
    'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
}

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    SVC(),
    param_grid,
    n_iter=32,
    cv=3
)


timeit(opt.fit(X_train, y_train))

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))


opt = GridSearchCV(
    SVC(),
    param_grid,
    cv=3,
    n_jobs=-1
)

timeit(opt.fit(X_train, y_train))


print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
