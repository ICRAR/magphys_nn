__author__ = 'ict310'
import math

omega0 = 0.0

def cosmol_c(h, omega, omega_lambda):

    cosmol = omega_lambda/(3.0 * pow(h, 2))

    if(omega_lambda == 0.0):
        q = omega/2.0
    else:
        q = (3.0 * omega/2.0) - 1.0

    return cosmol, q


def dl(h, q, z):
    """
    ---------------------------------------------------------------------------
    Computes luminosity distance corresponding to a redshift z.
    Uses Mattig formulae for qo both 0 and non 0
    Revised January 1991 to implement cosmolgical constant
    Ho in km/sec/Mpc, DL is in Mpc
    ===========================================================================
    :param h:
    :param q:
    :param z:
    :return:
    """

    global omega0
    s = 0

    if z <= 0:
        return 1.0e-5

    if q == 0:
        return ((3.0e5 * z) * (1+(z/2.0))) / h

    elif q > 0:
        d1 = (q * z) + ((q - 1.0) * (math.sqrt(1.0 + ((2.0 * q) * z)) - 1.0))
        d2 = ((h * q) * q) / 3.0e5
        return d1 / d2

    elif q < 0:
        omega0 = (2.0 * (q + 1.0)) / 3.0
        aa = 1.0
        bb = 1.0 + z
        success = False
        s0 = 1.e-10
        npts = 0

        while not success:
            npts += 1
            s = midpnt(funl, aa, bb, s, npts)
            espr = abs(s - s0) / s0
            if espr < 1.0e-4:
                success = True
            else:
                s0 = s

        dd1 = s

        dd2 = (3.0e5 * (1.0 + z)) / (h * math.sqrt(omega0))

        return dd1 * dd2


def midpnt(func, a, b, s, n):

    if n == 1:
        return (b-a) * func(0.5*(a+b))
    else:
        it = pow(3, (n-2))
        tnm = it
        dele = (b-a)/(3.0*tnm)
        ddel = dele + dele
        x = a + 0.5 * dele
        sum = 0

        for j in range(0, it):
            sum = sum + func(x)
            x = x + ddel
            sum = sum + func(x)
            x = x + ddel
        return (s + (b-a)*sum/tnm)/3.0


def funl(x):
    global omega0

    omegainv = 1. / omega0
    return 1. / math.sqrt((pow(x, 3) + omegainv) - 1.)


def read_file(filename):
    all_lines = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                current_line = line.split(' ')

                for a in range(0, len(current_line)):
                    current_line[a] = float(current_line[a])

                all_lines.append(current_line)

    return all_lines


def convert_to_luminosity(flux, sigma, redshift):

    num_filters = len(flux)
    w = [None] * num_filters

    #print flux_obs
    #print sigma
    #print w
    #print num_filters

    h = 70.
    omega = 0.3
    omega_lambda = 0.7

    clambda, q = cosmol_c(h, omega, omega_lambda)

    dist = dl(h, q, redshift)
    dist = dist * 3.086e24/math.sqrt(1.0 + redshift)

    for i in range(0, num_filters):
        if flux[i] > 0 and sigma[i] > 0:
            flux[i] = flux[i] * 1.e-23 * 3.283608731e-33 * dist ** 2
            sigma[i] = sigma[i] * 1.e-23 * 3.283608731e-33 * dist ** 2
        if flux[i] <= 0 and sigma[i] > 0:
            sigma[i] = sigma[i] * 1.e-23 * 3.283608731e-33 * dist ** 2
        if flux[i] > 0 and sigma[i] < 0.05 * flux[i]:
            sigma[i] = 0.05 * flux[i]

    for i in range(0, num_filters):
        if sigma[i] > 0.0:
            w[i] = 1.0 / (sigma[i]**2)

   # print flux
    #print sigma

    return flux, sigma

if __name__ == '__main__':

    all_gals = read_file('mygals.dat')

    flux_obs = []
    sigma = []

    redshift = all_gals[0][1]
    for i in range(2, len(all_gals[0])):
        if i % 2 == 0:
            flux_obs.append(all_gals[0][i])
        else:
            sigma.append(all_gals[0][i])

    del all_gals
    num_filters = len(flux_obs)

    w = [None] * num_filters

    print flux_obs
    print sigma
    print w
    print num_filters

    h = 70.
    omega = 0.3
    omega_lambda = 0.7

    clambda, q = cosmol_c(h, omega, omega_lambda)

    dist = dl(h, q, redshift)
    dist = dist * 3.086e24/math.sqrt(1.0 + redshift)

    for i in range(0, num_filters):
        if flux_obs[i] > 0 and sigma[i] > 0:
            flux_obs[i] = flux_obs[i] * 1.e-23 * 3.283608731e-33 * dist ** 2
            sigma[i] = sigma[i] * 1.e-23 * 3.283608731e-33 * dist ** 2
        if flux_obs[i] <= 0 and sigma[i] > 0:
            sigma[i] = sigma[i] * 1.e-23 * 3.283608731e-33 * dist ** 2
        if flux_obs[i] > 0 and sigma[i] < 0.05 * flux_obs[i]:
            sigma[i] = 0.05 * flux_obs[i]

    for i in range(0, num_filters):
        if sigma[i] > 0.0:
            w[i] = 1.0 / (sigma[i]**2)

    print flux_obs
    print sigma






