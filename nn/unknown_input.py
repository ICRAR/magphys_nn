# Set of unknown input handlers
__author__ = 'ict310'


def replace_zeros(array, ignore=None):
    """
    Replaces all bad inputs with 0
    :param array:
    :param ignore:
    :return:
    """

    for x in range(0, len(array)):
        for y in range(0, len(array[0])):

            if ignore and ignore == y:
                continue

            if array[x][y] == 0 or array[x][y] == -999:
                array[x][y] = 0.0

    return array


def replace_mean(array, ignore=None):
    """
    Replaces all bad inputs with the mean value.
    :param array:
    :param ignore:
    :return:
    """

    means = [0] * len(array[0])
    totals = [0] * len(array[0])
    for x in range(0, len(array)):
        for y in range(0, len(array[0])):

            if ignore and ignore == y:
                continue

            if array[x][y] == -999 or array[x][y] == 0:
                continue

            means[y] += array[x][y]
            totals[y] += 1

    for i in range(0, len(array[0])):
        means[i] /= float(totals[i])

    print means

    for x in range(0, len(array)):
        for y in range(0, len(array[0])):

            if ignore and ignore == y:
                continue

            if array[x][y] == 0 or array[x][y] == -999:
                array[x][y] = means[y]

    return array