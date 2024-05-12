import numpy as np
from scipy.optimize import least_squares


def func_p(x0, a, b, c, A):
    x = x0
    F = A * ((a ** x) - 1) / ((b ** x) - 1) * (b ** x) - c  # least squares von F ist p
    return F


def calcp(x0, A, N1, N2, N3, fc1, fc2, fc3, D):
    a = (N2 / N3) ** (1 / D)
    b = (N1 / N2) ** (1 / D)
    c = (fc3 - fc2) / (fc2 - fc1)

    p = least_squares(func_p, x0, args=(a, b, c, A))

    return p.x[0]


def prel(p_opt, P):
    p_rel = abs((p_opt - P) / p_opt)
    return p_rel


def getGCI(N1, N2, N3, fc1, fc2, fc3, D, Fs=1.25):
    assert not (((fc2 > fc3) & (fc2 > fc1)) | ((fc2 < fc3) & (fc2 < fc1))), "values are not monotone"
    assert not ((N1 <= N2) or (N2 <= N3)), "Number of cells must be N1 > N2 > N3"

    r21 = (N1 / N2) ** (1 / D)
    r32 = (N2 / N3) ** (1 / D)

    epsilon32 = (fc3 - fc2)
    epsilon21 = (fc2 - fc1)

    x0 = np.log(epsilon32 / epsilon21) / np.log(r21)
    p = calcp(x0, 1, N1, N2, N3, fc1, fc2, fc3, D)

    GCI_1 = Fs * abs((epsilon21 / fc1)) * 1 / ((r21 ** p) - 1)
    GCI_2 = Fs * abs((epsilon32 / fc2)) * 1 / ((r32 ** p) - 1)
    GCI_3 = GCI_2 * r32 ** p

    f_extra = fc1 + (fc1 - fc2) / ((r21 ** p) - 1)

    EERE_1 = abs((f_extra - fc1) / f_extra)
    EERE_2 = abs((f_extra - fc2) / f_extra)
    EERE_3 = abs((f_extra - fc3) / f_extra)

    A_Flag = GCI_2 / (GCI_1 * (r21 ** p))

    if ((A_Flag > 1.15) | (A_Flag < 0.85)):
        print('Error: Your fc-values are not in the asymptotic range!')
        print('Refine your grid and repeat grid convergence study, or contact your CFD-expert')

    GCI_1_p1 = Fs * abs((epsilon21 / fc1)) * 1 / ((r21) - 1)
    GCI_2_p1 = Fs * abs((epsilon32 / fc2)) * 1 / ((r32) - 1)
    GCI_3_p1 = GCI_2_p1 * r32

    f_extra_p1 = fc1 + (fc1 - fc2) / ((r21) - 1)

    EERE_1_p1 = abs((f_extra_p1 - fc1) / f_extra_p1)
    EERE_2_p1 = abs((f_extra_p1 - fc2) / f_extra_p1)
    EERE_3_p1 = abs((f_extra_p1 - fc3) / f_extra_p1)

    text = ""
    text += f'{"GCI_1": <16} {GCI_1}\n'
    text += f'{"GCI_2": <16} {GCI_2}\n'
    text += f'{"GCI_3": <16} {GCI_3}\n'
    text += f'{"EERE_1": <16} {EERE_1}\n'
    text += f'{"EERE_2": <16} {EERE_2}\n'
    text += f'{"EERE_3": <16} {EERE_3}\n'
    text += f'{"GCI_1_p1": <16} {GCI_1_p1}\n'
    text += f'{"GCI_2_p1": <16} {GCI_2_p1}\n'
    text += f'{"GCI_3_p1": <16} {GCI_3_p1}\n'
    text += f'{"EERE_1_p1": <16} {EERE_1_p1}\n'
    text += f'{"EERE_2_p1": <16} {EERE_2_p1}\n'
    text += f'{"EERE_3_p1": <16} {EERE_3_p1}\n'
    print(text)
    return GCI_1, GCI_2, GCI_3
