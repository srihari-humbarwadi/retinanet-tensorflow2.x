class AverageMeter:

    def __init__(self, name=None, momentum=0.997):
        if momentum >= 1 or momentum <= 0:
            raise AssertionError('`momentum` should be a non zero float less than 1')

        self.name = name
        self.momentum = momentum
        self._averaged_value = None
        self._count = 0

    def accumulate(self, x):
        if self._count < 10:
            self._averaged_value = x

        else:
            self._averaged_value = \
                self._averaged_value * self.momentum + (1 - self.momentum) * x

        self._count += 1

    @property
    def averaged_value(self):
        return self._averaged_value

    @property
    def data(self):
        return self._data


def format_eta(secs):
    eta = []
    for interval, unit in zip([3600, 60, 1], ['h', 'm', 's']):
        eta += ['{:02}{}'.format(int(secs // interval), unit)]
        secs %= interval
    return ' '.join(eta)
