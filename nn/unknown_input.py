# Set of unknown input handlers
__author__ = 'ict310'


def replace_zeros(array, ignore=None):

    for x in range(0, len(array)):
        for y in range(0, len(array[0])):

            if ignore and ignore == y:
                continue

            if array[x][y] == 0 or array[x][y] == -999:
                array[x][y] = 0.0

    return array


def replace_zeros_validity(array, ignore=None):

    xlen = len(array)
    ylen = len(array[0])

    for x in range(0, xlen):
        for y in range(0, ylen):
            if array[x][y] == 0 or array[x][y] == -999:

                if ignore and ignore == y:
                    continue

                array[x][y] = 0.0
                 # put a 0 where there is invalid input
                array[x].append(0)
            else:
                array[x].append(1)

    return array


def replace_mean(array, ignore=None):

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


def replace_mean_validity(array, ignore=None):

    xlen = len(array)
    ylen = len(array[0])

    means = [0] * len(array[0])
    totals = [0] * len(array[0])
    for x in range(0, xlen):
        for y in range(0, ylen):

            if ignore and ignore == y:
                continue

            if array[x][y] == -999 or array[x][y] == 0:
                continue

            means[y] += array[x][y]
            totals[y] += 1

    for i in range(0, len(array[0])):
        means[i] /= float(totals[i])

    print means

    for x in range(0, xlen):
        for y in range(0, ylen):

            if ignore and ignore == y:
                continue

            if array[x][y] == 0 or array[x][y] == -999:
                array[x][y] = means[y]

                # put a 0 where there is invalid input
                array[x].append(0)
            else:
                # put a 1 where there is valid input
                array[x].append(1)

    return array