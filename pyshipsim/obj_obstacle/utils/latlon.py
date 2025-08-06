import sys
import numpy as np

__all__ = ["latlon2local"]


RD = 180 / np.pi
DR = np.pi / 180


def CalDistCo(ALong, ALat):
    # ****************************************************************************
    #   compute disntance traveled and course of ship by of 2 latitude and longitude of points using 漸長緯度航法
    #   input are DICIMAL DEGREE (ddd.dddd)
    #
    #    Input variables
    #    ALong: Longitude vector of two points [start end](deg.)
    #    ALat:  Latitude vector of two point [start end](deg.)
    #
    #    Output variables
    #    dist: distance traveled（min, to convert to meter, *1852）
    #    co: course (degree, north = 0)
    #
    # ****************************************************************************
    # Calculating the Earth as a sphere
    #  compute
    AM1 = 7915.7045 * np.log10(np.tan((45.0 + ALat[0] / 2.0) * DR))
    AM2 = 7915.7045 * np.log10(np.tan((45.0 + ALat[1] / 2.0) * DR))
    #  compute meri-dional parts (unit: min)
    AMD = AM2 - AM1
    #  compute latidude diff (unit: degree)
    ALatD = ALat[1] - ALat[0]
    #  compute longitude diff (unit:min)
    ALongD = (ALong[1] - ALong[0]) * 60.0
    # compute middle latitude
    ALatM = (ALat[1] + ALat[0]) / 2.0
    # compute touzaikyo(departure)
    ADep = ALongD * np.cos(ALatM * DR)
    # compute course by mean middle latitude sailing
    if (abs(ADep) <= sys.float_info.epsilon) and (abs(ALatD) <= sys.float_info.epsilon):
        co = 0.0
    else:
        co = np.arctan2(ADep / 60.0, ALatD)

    if co > 2 * np.pi:
        co = co - 2.0 * np.pi
    elif co < 0.0:
        co = co + 2.0 * np.pi

    co = co * RD

    ### COMPUTE traveled distance
    # around 90 deg and 270 deg, use middle latitude sailing
    if (co >= 89.0) and (co <= 91.0):
        dist = np.sqrt((ALatD * 60.0) * (ALatD * 60.0) + (ADep) * (ADep))
    elif (co >= 269.0) and (co <= 271.0):
        dist = np.sqrt((ALatD * 60.0) * (ALatD * 60.0) + (ADep) * (ADep))
    else:
        #  use mercator's method to compute course
        if (abs(ALongD) <= sys.float_info.epsilon) and (
            abs(AMD) <= sys.float_info.epsilon
        ):
            co = 0.0
        else:
            co = np.arctan2(ALongD, AMD)

        if co > 2.0 * np.pi:
            co = co - 2.0 * np.pi
        elif co < 0.0:
            co = co + 2.0 * np.pi
        co = co * RD
        # compute traveled distance
        dist = abs((ALatD) * 60.0 / np.cos(co * DR))
    return dist, co


def latlon2local(latlon, origin):
    # convert to local coord.
    diff_lon = [origin[0], latlon[0]]
    diff_lat = [origin[1], latlon[1]]
    dist, co = CalDistCo(diff_lon, diff_lat)
    dist_m = dist * 1852
    course = co * DR
    local = [dist_m * np.cos(course), dist_m * np.sin(course)]
    return local
