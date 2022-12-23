def calc_inflow_cp(px, pt1, p1):
    """
    :param px: pressure at position
    :param pt1: total pressure inlet
    :param p1: pressure inlet
    :return: lift coefficient
    """
    cp = (px - pt1) / (p1 - pt1)
    return cp
