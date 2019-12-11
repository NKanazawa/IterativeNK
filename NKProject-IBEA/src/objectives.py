import numpy
import copy


class Objectives(object):
    def __init__(self):
        self.eps=1e-9

    @staticmethod
    def ikedaobj1(gnm):
        value = numpy.power(gnm[0], 2)
        for i in range(1, len(gnm)):
            value += 100 * numpy.power(gnm[0] - gnm[i], 2)
        return value

    @staticmethod
    def ikedaobj2(gnm):
        value = numpy.power(1 - gnm[0], 2)
        for i in range(1, len(gnm)):
            value += 100 * numpy.power(gnm[0] - gnm[i], 2)
        return value