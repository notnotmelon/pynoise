"""
.. module:: noisemodule
   :synopsis: Various interlinkable noise modules.

 .. moduleauthor:: Tim Butram
"""

import math

from pynoise.quality import Quality
from pynoise.noise import gradient_coherent_noise_3d, value_noise_3d
from pynoise.interpolators import linear_interp, cubic_interp, scurve3, scurve5
from pynoise.util import clamp

from sortedcontainers import SortedDict, SortedList

from functools import lru_cache

import numpy as np


from pynoise.gpu import GPU
gpu = GPU()
gpu.load_program()

class NoiseModule():
    frequency = 1

    def create_arrays(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        xa = np.linspace(min_x, max_x, width, dtype=np.float64)
        xa = np.tile(xa, height)
        xa *= self.frequency

        ya = np.linspace(min_y, max_y, height, dtype=np.float64)
        ya = np.repeat(ya, width)
        ya *= self.frequency

        if max_z is None:
            max_z = min_z

        za = np.linspace(min_z, max_z, width*height, dtype=np.float64)
        za *= self.frequency

        return xa, ya, za

    def get_value(self, x, y, z):
        raise NotImplementedError('Not Implemented.')

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        raise NotImplementedError('Not Implemented.')

class Abs(NoiseModule):
    """
    Returns the absolute value of the given source module.

    Args:
        source0 (NoiseModule): The module that Abs will apply to.
    """
    def __init__(self, source0):
        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return abs(self.source0.get_value(x, y, z))

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays=None):
        assert self.source0 is not None

        return np.absolute(self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays))

class Add(NoiseModule):
    """ Adds the two given source modules together.

        Args:
            source0 (NoiseModule): The NoiseModule to be added to source1
            source1 (NoiseModule): The NoiseModule to be added to soruce0

    """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)

        return self.source0.get_value(x, y, z) + self.source1.get_value(x, y, z)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None

        a = self.source0.get_values(width, height, min_x, max_x, min_y, max_y,min_z, max_z, use_arrays)
        b = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        return np.add(a, b)

class Billow(NoiseModule):
    """ Noise function that is more suitable for items like clouds or rocks.
    This module is nearly identical to Perlin noise, however each octave has
    the absolute value taken from the signal.

    Args:
        frequency (float): The frequency of the first octave. Default 1.
        lacunarity (float): Lacunarity is the frequency multiplier between
            successive octaves. For best results set to between 1.5 and 3.5.
            The default is 2.
        quality (Quality): How accuratly the noise is calculated. Higher
            Qualities is slower. Default is Quality.std
        octaves (int): The number of octaves controls the amount of detail in
            the noise. The default is 6.
        seed (int): The seed value for the noise. The default is 0.
    """
    def __init__(self, frequency=1, lacunarity=2, quality=Quality.std,
        octaves=6, persistence=0.5, seed=0):

        self.frequency = frequency
        self.lacunarity = lacunarity
        self.quality = quality
        self.octaves = octaves
        self.persistence = persistence
        self.seed = seed

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        value = 0
        signal = 0
        curPersistence = 1.0
        seed = 0

        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        for octave in range(self.octaves):
            seed = (self.seed + octave) & 0xffffffff
            signal = gradient_coherent_noise_3d(x, y, z, seed, self.quality)
            signal = 2 * abs(signal) - 1.0
            value += signal * curPersistence

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity

            curPersistence *= self.persistence

        value += 0.5

        return value

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        value = np.zeros_like(xa)
        curPersistence = 1

        for octave in range(self.octaves):
            seed = (self.seed + octave) & 0xffffffff
            signal = gpu.gradient_noise(xa, ya, za, np.int32(seed), np.int32(self.quality.value))
            signal = np.absolute(signal)
            signal *= 2
            signal -= 1

            value = value + (signal * curPersistence)

            xa *= self.lacunarity
            ya *= self.lacunarity
            za *= self.lacunarity

            curPersistence *= self.persistence

        value += 0.5

        return value

class Blend(NoiseModule):
    """ Blends source0 and source1 by performing a linear interpolation with
    source2 acting as the alpha

    Args:
        source0 (NoiseModule): The first of two sources to be blended.
        source1 (NoiseModule): The second of two sources to be blended.
        source2 (NoiseModule): The alpha value of the linear interpolation.
    """
    def __init__(self, source0, source1, source2):
        self.source0 = source0
        self.source1 = source1
        self.source2 = source2

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)
        assert (self.source2 is not None)

        v0 = self.source0.get_value(x, y, z)
        v1 = self.source1.get_value(x, y, z)
        alpha = (self.source2.get_value(x, y, z) + 1) / 2

        return linear_interp(v0, v1, alpha)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None
        assert self.source2 is not None

        v0 = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        v1 = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        alpha = (self.source2.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays) + 1) / 2

        return gpu.linear_interp(v0, v1, alpha)

class Checkerboard(NoiseModule):
    """
    Outputs a checkerboard patterned noise.
    """

    def _checker(self, ix, iy, iz):
        if ((int(ix) & 1 ^ int(iy) & 1 ^ int(iz) & 1) != 0):
            return -1
        else:
            return 1

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))

        return self._checker(ix, iy, iz)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]
        rv = np.empty_like(xa)
        it = np.nditer([xa, ya, za, rv], [], [['readonly'], ['readonly'], ['readonly'], ['writeonly']])

        while not it.finished:
            it[3] = self._checker(it[0], it[1], it[2])
            it.iternext()

        return rv

class Clamp():
    """
    Clamps a source between two values

    Args:
        source0 (NoiseModule): The source whose output is to be clamped.
        lower_bound (float): The lower bound of the clamp. Values lower than
            this are set to lower_bound. The default is -1.
        upper_bound (float): The upper bound of the clamp. Values higher than
            this are set to upper_bound. The default is 1.
    """
    def __init__(self, source0, lower_bound=-1, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        value = self.source0.get_value(x, y, z)

        if value < self.lower_bound:
            return self.lower_bound
        elif value > self.upper_bound:
            return self.upper_bound
        else:
            return value

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None

        v = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        return np.clip(v, self.lower_bound, self.upper_bound)

class Const(NoiseModule):
    """
    Outputs a constant value.

    Args:
        const (float): The output value of this module.
    """

    def __init__(self, const):
        self.const = const

    def get_value(self, x, y, z):
        return self.const

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        return (np.ones(width*height) * self.const)

class Curve(NoiseModule):
    """
    Maps the output of source0 to an arbitrary curve. This curve is defined
    by a list of given points, given as (input, output). The input module is
    then cubically interpolated against this curve.

    Args:
        source0 (NoiseModule): The source to be matched against the curve.
        points ([(float, float)]): A list of tuples that specify the inputs
            and outputs of the defined curve.
            Example: points = [(0, 0.1), (0.1,0.2),...]
            A minimum of 4 points must be added.
    """
    def __init__(self, source0, points=None):
        self.control_points = SortedDict()
        self.source0 = source0

        if points is not None:
            for point in points:
                self.add_control_point(point[0], point[1])

    def add_control_point(self, input, output):
        self.control_points[input] = output

    def clear_control_points(self):
        self.control_points = SortedDict()

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(len(self.control_points.keys()) >= 4)

        value = self.source0.get_value(x, y, z)

        for i, k in enumerate(self.control_points.keys()):
            if value < k:
                break

        index0 = clamp(i - 2, 0, len(self.control_points.keys())-1)
        index1 = clamp(i - 1, 0, len(self.control_points.keys())-1)
        index2 = clamp(i,     0, len(self.control_points.keys())-1)
        index3 = clamp(i + 1, 0, len(self.control_points.keys())-1)

        if index1 == index2:
            return self.control_points[k]

        l = list(self.control_points.keys())

        input0 = l[index1]
        input1 = l[index2]
        alpha = (value - input0) / (input1 - input0)

        return cubic_interp(
            self.control_points[l[index0]],
            self.control_points[l[index1]],
            self.control_points[l[index2]],
            self.control_points[l[index3]],
            alpha
        )

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert len(self.control_points.keys()) >= 4

        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        dest = np.empty_like(xa)

        for dex, l in enumerate(np.nditer([xa, ya, za])):
            value = self.source0.get_value(float(l[0]), float(l[1]), float(l[2]))

            for i, k in enumerate(self.control_points.keys()):
                if value < k:
                    break

            index0 = clamp(i - 2, 0, len(self.control_points.keys())-1)
            index1 = clamp(i - 1, 0, len(self.control_points.keys())-1)
            index2 = clamp(i,     0, len(self.control_points.keys())-1)
            index3 = clamp(i + 1, 0, len(self.control_points.keys())-1)

            if index1 == index2:
                return self.control_points[k]

            l = list(self.control_points.keys())

            input0 = l[index1]
            input1 = l[index2]
            alpha = (value - input0) / (input1 - input0)

            dest[dex] = cubic_interp(
                self.control_points[l[index0]],
                self.control_points[l[index1]],
                self.control_points[l[index2]],
                self.control_points[l[index3]],
                alpha)

        return dest

class Cylinders(NoiseModule):
    """ Outputs a series of concentric cyliners centered around 0,y,0.
    There is a new cylinder every `frequency` lengths from origin.

    Args:
        frequency (float): The frequency of a new concentric cylinder. Default
            is 1.
    """
    def __init__(self, frequency=1):
        self.frequency = frequency

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        x *= self.frequency
        z *= self.frequency

        center = (x**2 + z**2)**0.5
        small = center - math.floor(center)
        large = 1 - small
        nearest = min(small, large)

        return 1.0 - (nearest * 4.0)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        return gpu.cylinders(xa, za)

class Displace(NoiseModule):
    """ Displaces source3 output by the outputs of source0, source1, and source2.

    Args:
        source0 (NoiseModule): Controls the shift in x of source3.
        source1 (NoiseModule): Controls the shift in y of source3.
        source2 (NoiseModule): Controls the shift in z of source3.
        source3 (NoiseModule): The module whose output is shifted by source0,
            source1 and souce2.
    """
    def __init__(self, source0, source1, source2, source3):
        self.source0 = source0
        self.source1 = source1
        self.source2 = source2
        self.source3 = source3

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)
        assert(self.source2 is not None)
        assert(self.source3 is not None)

        xD = x + (self.source0.get_value(x, y, z))
        yD = y + (self.source1.get_value(x, y, z))
        zD = z + (self.source2.get_value(x, y, z))

        return self.source3.get_value(xD, yD, zD)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert(self.source0 is not None)
        assert(self.source1 is not None)
        assert(self.source2 is not None)
        assert(self.source3 is not None)

        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        xD = xa + self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        yD = ya + self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        zD = za + self.source2.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        if use_arrays is None:
            return self.source3.get_values(width, height, min_x + np.amin(xD),
                max_x + np.amax(xD), min_y + np.amin(yD), max_y + np.amax(yD),
                min_z + np.amin(zD), max_z + np.amax(zD))
        else:
            return self.source3.get_values(width, height, 0,0, 0,0, 0,0, [xD,yD,zD])

class Exponent(NoiseModule):
    """  Sets the exponent value to apply to the output value from the
    source module.

    Because most noise modules will output values that range from -1.0
    to +1.0, this noise module first normalizes this output value (the
    range becomes 0.0 to 1.0), maps that value onto an exponential
    curve, then rescales that value back to the original range.
    """
    def __init__(self, source0, exponent=1):
        self.exponent = exponent
        self.source0 = source0

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        value = self.source0.get_value(x, y, z)

        return math.pow(abs((value + 1) / 2), self.exponent) * 2 - 1

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None

        values = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        values += 1
        values /= 2
        values = np.absolute(values)
        values = np.power(values, self.exponent)
        values *= 2
        values -= 1

        return values

class Invert(NoiseModule):
    """ Inverts the sign of the given source. """
    def __init__(self, source0):
        self.source0 = source0

    def get_value(self, x, y, z):
        assert(self.source0 is not None)

        return -(self.source0.get_value(x, y, z))

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None

        v = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        return v * -1

class Max(NoiseModule):
    """ Chooses the larger value of source0 and source1. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)

        v0 = self.source0.get_value(x, y, z)
        v1 = self.source1.get_value(x, y, z)

        return max(v0, v1)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None

        a = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        b = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        return np.maximum(a, b)

class Min(NoiseModule):
    """ Chooses the lesser value of source0 and source1. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)

        v0 = self.source0.get_value(x, y, z)
        v1 = self.source1.get_value(x, y, z)

        return min(v0, v1)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None

        a = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        b = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        return np.minimum(a, b)

class Multiply(NoiseModule):
    """ Multiplies source0 by source1. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)

        return self.source0.get_value(x, y, z) * self.source1.get_value(x, y, z)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None

        a = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        b = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        return np.multiply(a, b)

class Perlin(NoiseModule):
    """ The classic noise. https://en.wikipedia.org/wiki/Perlin_noise """
    def __init__(self, frequency=1, lacunarity=2, octaves=6, persistence=0.5, seed=0, quality=Quality.std):
        self.frequency = frequency
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.seed = seed
        self.persistence = persistence
        self.quality = quality

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        value = 0.0
        signal = 0.0
        curPersistence = 1.0

        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        for i in range(self.octaves):
            seed = (self.seed + i) & 0xffffffff
            signal = gradient_coherent_noise_3d(x, y, z, seed, self.quality)
            value += signal * curPersistence

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity
            curPersistence *= self.persistence

        return value

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        value = np.zeros_like(xa, dtype=np.float64)
        curPersistence = 1.0

        for i in range(self.octaves):
            seed = (self.seed + i) & 0xffffffff
            signal = gpu.gradient_noise(xa, ya, za, np.int32(seed), np.int32(self.quality.value))

            value = value + (signal * curPersistence)

            xa = xa * self.lacunarity
            ya = ya * self.lacunarity
            za = za * self.lacunarity
            curPersistence *= self.persistence

        return value

class Power(NoiseModule):
    """ source0 ** source1. Does not apply scaling. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)

        return self.source0.get_value(x,y,z)**self.source1.get_value(x,y,z)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None

        a = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        b = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        return np.power(a, b)

class RidgedMulti(NoiseModule):
    """ This is much like perlin noise, however each octave is modified by
    abs(x*-exponent) where x is x *= frequency repeated over each octave. """
    def __init__(self, frequency=1, lacunarity=2, quality=Quality.std,
        octaves=6, seed=0, exponent=1, offset=1, gain=2):
        self.frequency = frequency
        self.lacunarity = lacunarity
        self.quality = quality
        self.octaves = octaves
        self.seed = seed
        self.exponent = exponent
        self.max_octaves = 30
        self.weights = [0] * self.max_octaves
        self.offset = offset
        self.gain = gain

        freq = 1
        for i in range(0, self.max_octaves):
            self.weights[i] = freq**-exponent
            freq *= self.lacunarity

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        signal = 0.0
        value = 0.0
        weight = 1.0

        for i in range(self.octaves):
            seed = (self.seed + i) & 0x7fffffff
            signal = gradient_coherent_noise_3d(x, y, z, seed, self.quality)

            signal = abs(signal)
            signal = self.offset - signal

            signal *= signal

            signal *= weight

            weight = signal * self.gain

            if weight > 1:
                weight = 1
            if weight < 0:
                weight = 0

            value += (signal * self.weights[i])

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity

        return (value * 1.25) - 1

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]
        value = np.zeros_like(xa)
        weight = np.ones_like(xa)

        for i in range(self.octaves):
            seed = (self.seed + i) & 0x7fffffff

            signal = gpu.gradient_noise(xa, ya, za, np.int32(seed), np.int32(self.quality.value))
            signal = np.absolute(signal)
            signal = self.offset - signal
            signal *= signal
            signal *= weight

            weight = signal * self.gain

            weight = np.clip(weight, 0, 1)

            value += (signal * self.weights[i])

            xa *= self.lacunarity
            ya *= self.lacunarity
            za *= self.lacunarity

        return (value * 1.25) - 1

class RotatePoint(NoiseModule):
    """ Rotates source0 around the origin before returning a value. This is a
    right hand system, xAngle increases to the right, yAngle increases upwards
    and zAngle increases inward.
    """
    def __init__(self, source0, xAngle=0, yAngle=0, zAngle=0):
        xCos = math.cos(math.radians(xAngle))
        yCos = math.cos(math.radians(yAngle))
        zCos = math.cos(math.radians(zAngle))

        xSin = math.sin(math.radians(xAngle))
        ySin = math.sin(math.radians(yAngle))
        zSin = math.sin(math.radians(zAngle))

        self.x1a = ySin * xSin * zSin + yCos * zCos
        self.y1a = xCos * zSin
        self.z1a = ySin * zCos - yCos * ySin * zSin

        self.x2a = ySin * xSin * zCos - yCos * zSin
        self.y2a = xCos * zCos
        self.z2a = -yCos * xSin * zCos - ySin * zSin

        self.x3a = -ySin * xCos
        self.y3a = xSin
        self.z3a = yCos * xCos

        self.source0 = source0

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        nx = (self.x1a * x) + (self.y1a * y) + (self.z1a * z)
        ny = (self.x2a * x) + (self.y2a * y) + (self.z2a * z)
        nz = (self.x3a * x) + (self.y3a * y) + (self.z3a * z)

        return self.source0.get_value(nx, ny, nz)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays=None):
        assert self.source0 is not None

        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        nx = (self.x1a * xa) + (self.y1a * ya) + (self.z1a * za)
        ny = (self.x2a * xa) + (self.y2a * ya) + (self.z2a * za)
        nz = (self.x3a * xa) + (self.y3a * ya) + (self.z3a * za)

        return self.source0.get_values(width, height, 0,0, 0,0, 0,0, use_arrays=[nx, ny, nz])

class ScaleBias(NoiseModule):
    """ Takes the value of source0 and multiplies by scale and adds by bias. """
    def __init__(self, source0, bias=0, scale=1):
        self.bias = bias
        self.scale = scale

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return self.source0.get_value(x, y, z) * self.scale + self.bias

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None

        return self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays) * self.scale + self.bias

class ScalePoint(NoiseModule):
    """ Scales the x,y,z before returning source0 value. """
    def __init__(self, source0, sx=1, sy=1, sz=1):
        self.sx = sx
        self.sy = sy
        self.sz = sz

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return self.source0.get_value(x * self.sx, y * self.sy, z * self.sz)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None

        if use_arrays is None:
            return self.source0.get_values(width, height, min_x * self.sx, max_x * self.sx,
                min_y * self.sy, max_y * self.sy, min_z * self.sz, max_z * self.sz)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]
            xa *= self.sx
            ya *= self.sy
            za *= self.sz

            return self.source0.get_values(width, height, 0,0, 0,0, 0,0, use_arrays=[xa,ya,za])

class Select(NoiseModule):
    """
    Noise module that outputs the value selected from one of two source
    modules chosen by the output value from a control module.

    Unlike most other noise modules, the index value assigned to a source
    module determines its role in the selection operation:
    - source0 outputs a value.
    - source1 outputs a value.
    - source2 is known as the **control module**.  The control module
      determines the value to select.  If the output value from the control
      module is within a range of values known as the **selection range**,
      this noise module outputs the value from source1.  Otherwise,
      this noise module outputs the value from source0.

    By default, there is an abrupt transition between the output values
    from the two source modules at the selection-range boundary.  To
    smooth the transition, pass a non-zero value to edge_falloff
    method.  Higher values result in a smoother transition.

    This noise module requires three source modules.
    """
    def __init__(self, source0, source1, source2, edge_falloff=0, lower_bound=-1, upper_bound=1):
        self.edge_falloff = edge_falloff
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.source0 = source0
        self.source1 = source1
        self.source2 = source2

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)
        assert (self.source2 is not None)

        control_value = self.source2.get_value(x, y, z)

        if self.edge_falloff > 0:
            if control_value < (self.lower_bound - self.edge_falloff):
                return self.source0.get_value(x,y,z)

            elif control_value < (self.lower_bound + self.edge_falloff):
                lower_curve = (self.lower_bound - self.edge_falloff)
                upper_curve = (self.lower_bound + self.edge_falloff)
                alpha = scurve3((control_value - lower_curve) / (upper_curve - lower_curve))

                return linear_interp(self.source0.get_value(x, y, z),
                    self.source1.get_value(x, y, z), alpha)

            elif control_value < (self.upper_bound - self.edge_falloff):
                return self.source1.get_value(x, y, z)

            elif control_value < (self.upper_bound + self.edge_falloff):
                lower_curve = (self.upper_bound - self.edge_falloff)
                upper_curve = (self.upper_bound + self.edge_falloff)
                alpha = scurve3((control_value - lower_curve) / (upper_curve - lower_curve))

                return linear_interp(self.source1.get_value(x, y, z),
                    self.source0.get_value(x, y, z), alpha)
            else:
                return self.source0.get_value(x, y, z)
        else:
            if control_value < self.lower_bound or control_value > self.upper_bound:
                return self.source0.get_value(x, y, z)
            else:
                return self.source1.get_value(x, y, z)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert self.source1 is not None
        assert self.source2 is not None

        s0a = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        s1a = self.source1.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        s2a = self.source2.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)
        rv = np.empty_like(s0a)

        if self.edge_falloff > 0:
            it = np.nditer([s0a, s1a, s2a, rv], [], [['readonly'], ['readonly'], ['readonly'], ['writeonly']])
            while not it.finished:
                control_value = it[2]
                if control_value < (self.lower_bound - self.edge_falloff):
                    it[3] = it[0]
                elif control_value < (self.lower_bound + self.edge_falloff):
                    lower_curve = (self.lower_bound - self.edge_falloff)
                    upper_curve = (self.lower_bound + self.edge_falloff)
                    alpha = scurve3((control_value - lower_curve) / (upper_curve - lower_curve))

                    it[3] = linear_interp(it[0], it[1], alpha)
                elif control_value < (self.upper_bound - self.edge_falloff):
                    it[3] = it[1]
                elif control_value < (self.upper_bound + self.edge_falloff):
                    lower_curve = (self.upper_bound - self.edge_falloff)
                    upper_curve = (self.upper_bound + self.edge_falloff)
                    alpha = scurve3((control_value - lower_curve) / (upper_curve - lower_curve))
                    it[3] = linear_interp(it[1], it[0], alpha)
                else:
                    it[3] = it[0]

                it.iternext()
        else:
            it = np.nditer([s0a, s1a, s2a, rv], [], [['readonly'], ['readonly'], ['readonly'], ['writeonly']])
            while not it.finished:
                if it[2] < self.lower_bound or it[2] > self.upper_bound:
                    it[3] = it[0]
                else:
                    it[3] = it[1]
                it.iternext()

        return rv

class Spheres(NoiseModule):
    """ Generates a series of concentric spheres centered around 0,0,0. """
    def __init__(self, frequency=1):
        self.frequency = frequency

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        center = (x*x + y*y + z*z)**(0.5)
        small = center - math.floor(center)
        large = 1 - small
        nearest = min(small, large)
        print(center, small, large, nearest)
        return 1 - (nearest*4)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        return gpu.spheres(xa, ya, za)

class Terrace(NoiseModule):
    """
    Noise module that maps the output value from a source module onto a
    terrace-forming curve.

    This noise module maps the output value from the source module onto a
    terrace-forming curve.  The start of this curve has a slope of zero;
    its slope then smoothly increases.  This curve also contains
    **control points** which resets the slope to zero at that point,
    producing a "terracing" effect.

    To add a control point to this noise module, call the
    add_control_point() method.

    An application must add a minimum of two control points to the curve.
    If this is not done, the get_value() method fails.  The control points
    can have any value, although no two control points can have the same
    value.  There is no limit to the number of control points that can be
    added to the curve.

    This noise module clamps the output value from the source module if
    that value is less than the value of the lowest control point or
    greater than the value of the highest control point.

    This noise module is often used to generate terrain features such as
    your stereotypical desert canyon.

    This noise module requires one source module.
    """
    def __init__(self, source0, control_points=None, invert_terraces=False):
        self.control_points = SortedList()
        self.invert_terraces = invert_terraces
        self.source0 = source0

        if control_points is not None:
            for c in control_points:
                self.add_control_point(c)

    def add_control_point(self, value):
        self.control_points.append(value)

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (len(self.control_points) > 1)

        value = self.source0.get_value(x, y, z)

        for i, cp in enumerate(self.control_points):
            if value < cp:
                break

        index0 = clamp(i - 1, 0, len(self.control_points)-1)
        index1 = clamp(i, 0, len(self.control_points)-1)

        if index0 == index1:
            return self.control_points[index1]

        value0 = self.control_points[index0]
        value1 = self.control_points[index1]
        alpha = (value - value0) / (value1 - value0)

        if self.invert_terraces:
            alpha = 1 - alpha
            x = value0
            value0 = value1
            value1 = x

        alpha *= alpha

        return linear_interp(value0, value1, alpha)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None
        assert len(self.control_points) > 1

        v = self.source0.get_values(width, height, min_x, max_x, min_y, max_y, min_z, max_z, use_arrays)

        rv = np.empty_like(v)

        for r, x in enumerate(v):
            for i, cp in enumerate(self.control_points):
                if x < cp:
                    break

            index0 = clamp(i - 1, 0, len(self.control_points)-1)
            index1 = clamp(i, 0, len(self.control_points)-1)

            if index0 == index1:
                rv[r] = self.control_points[index1]
                continue

            value0 = self.control_points[index0]
            value1 = self.control_points[index1]
            alpha = (v[r] - value0) / (value1 - value0)

            if self.invert_terraces:
                alpha = 1 - alpha
                x = value0
                value0 = value1
                value1 = x

            alpha *= alpha

            rv[r] = linear_interp(value0, value1, alpha)

        return rv

class TranslatePoint(NoiseModule):
    """ Translates the coordinates (x,y,z) of the input value
    (x+xtran, y+ytran, z+ztran) before returning any output.
    """
    def __init__(self, source0, xtran=0, ytran=0, ztran=0):
        self.xtran = xtran
        self.ytran = ytran
        self.ztran = ztran
        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return self.source0.get_value(x+self.xtran, y+self.ytran, z+self.ztran)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        assert self.source0 is not None

        if use_arrays is None:
            return self.source0.get_values(width, height, min_x+self.xtran, max_x+self.xtran,
                min_y+self.ytran, max_y+self.ytran, min_z+self.ztran, max_z+self.ztran)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]
            xa += self.xtran
            ya += self.ytran
            za += self.ztran

            return self.soruce0.get_values(width, height, 0,0, 0,0, 0,0, use_arrays=[xa,ya,za])

class Turbulence(NoiseModule):
    """
    Noise module that randomly displaces the input value before
    returning the output value from a source module.

    The get_value() method randomly displaces the (x, y, z)
    coordinates of the input value before retrieving the output value from
    the source0.  To control the turbulence, an application can
    modify its frequency, its power, and its roughness.

    The frequency of the turbulence determines how rapidly the
    displacement amount changes.  To specify the frequency, set the frequency
    parameter.

    The power of the turbulence determines the scaling factor that is
    applied to the displacement amount.  To specify the power, set the power
    parameter.

    The roughness of the turbulence determines the roughness of the
    changes to the displacement amount.  Low values smoothly change the
    displacement amount.  High values roughly change the displacement
    amount, which produces more "kinky" changes.  To specify the
    roughness, set the roughness parameter.

    Use of this noise module may require some trial and error.  Assuming
    that you are using a generator module as the source module, you
    should first:
     - Set the frequency to the same frequency as the source module.
     - Set the power to the reciprocal of the frequency.

    From these initial frequency and power values, modify these values
    until this noise module produce the desired changes in your terrain or
    texture.  For example:
    - Low frequency (1/8 initial frequency) and low power (1/8 initial
      power) produces very minor, almost unnoticeable changes.
    - Low frequency (1/8 initial frequency) and high power (8 times
      initial power) produces "ropey" lava-like terrain or marble-like
      textures.
    - High frequency (8 times initial frequency) and low power (1/8
      initial power) produces a noisy version of the initial terrain or
      texture.
    - High frequency (8 times initial frequency) and high power (8 times
      initial power) produces nearly pure noise, which isn't entirely
      useful.

    Displacing the input values result in more realistic terrain and
    textures.  If you are generating elevations for terrain height maps,
    you can use this noise module to produce more realistic mountain
    ranges or terrain features that look like flowing lava rock.  If you
    are generating values for textures, you can use this noise module to
    produce realistic marble-like or "oily" textures.

    Internally, there are three Perlin noise modules
    that displace the input value; one for the x, one for the y,
    and one for the z coordinate.

    This noise module requires one source module.
    """
    def __init__(self, source0, frequency=1, power=1, roughness=3, seed=0):
        self.frequency = frequency
        self.power = power
        self.roughness = roughness
        self.seed = seed
        self.xdm = Perlin(frequency=frequency, octaves=roughness, seed=seed)
        self.ydm = Perlin(frequency=frequency, octaves=roughness, seed=seed)
        self.zdm = Perlin(frequency=frequency, octaves=roughness, seed=seed)

        self.source0 = source0

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
          x0 = x + (12414.0 / 65536.0)
          y0 = y + (65124.0 / 65536.0)
          z0 = z + (31337.0 / 65536.0)
          x1 = x + (26519.0 / 65536.0)
          y1 = y + (18128.0 / 65536.0)
          z1 = z + (60493.0 / 65536.0)
          x2 = x + (53820.0 / 65536.0)
          y2 = y + (11213.0 / 65536.0)
          z2 = z + (44845.0 / 65536.0)

          xDistort = x + (self.xdm.get_value(x0, y0, z0) * self.power)
          yDistort = y + (self.ydm.get_value(x1, y1, z1) * self.power)
          zDistort = z + (self.zdm.get_value(x2, y2, z2) * self.power)

          return self.source0.get_value(xDistort, yDistort, zDistort)

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        x0 = min_x + (12414.0 / 65536.0)
        y0 = min_y + (65124.0 / 65536.0)
        z0 = min_z + (31337.0 / 65536.0)

        x1 = min_x + (26519.0 / 65536.0)
        y1 = min_y + (18128.0 / 65536.0)
        z1 = min_z + (60493.0 / 65536.0)

        x2 = min_x + (53820.0 / 65536.0)
        y2 = min_y + (11213.0 / 65536.0)
        z2 = min_z + (44845.0 / 65536.0)

        xa = self.xdm.get_values(width, height, min_x+x0, max_x+x0, min_y+y0, max_y+y0, min_z+z0, max_z+z0)
        ya = self.ydm.get_values(width, height, min_x+x1, max_x+x1, min_y+y1, max_y+y1, min_z+z1, max_z+z1)
        za = self.zdm.get_values(width, height, min_x+x2, max_x+x2, min_y+y2, max_y+y2, min_z+z2, max_z+z2)

        xa *= self.power
        ya *= self.power
        za *= self.power

        xa = np.amin(xa)
        ya = np.amin(ya)
        za = np.amin(za)

        if use_arrays is None:
            return self.source0.get_values(width, height, min_x+xa, max_x+xa, min_y+ya, max_y+ya, min_z+za, max_z+za)
        else:
            a,b,c = use_arrays[0], use_arrays[1], use_arrays[2]
            a += xa
            b += ya
            c += za

            return self.source0.get_values(width, height, 0,0, 0,0, 0,0, use_arrays=[a,b,c])

class Voronoi(NoiseModule):
    """
    Noise module that outputs Voronoi cells.

    In mathematics, a **Voronoi cell** is a region containing all the
    points that are closer to a specific **seed point** than to any
    other seed point.  These cells mesh with one another, producing
    polygon-like formations.

    By default, this noise module randomly places a seed point within
    each unit cube.  By modifying the **frequency** of the seed points,
    an application can change the distance between seed points.  The
    higher the frequency, the closer together this noise module places
    the seed points, which reduces the size of the cells.  To specify the
    frequency of the cells, set the frequency parameter.

    This noise module assigns each Voronoi cell with a random constant
    value from a coherent-noise function.  The **displacement value**
    controls the range of random values to assign to each cell.  The
    range of random values is +/- the displacement value.  To specify the
    displacement value, set the displacement parameter.

    To modify the random positions of the seed points, set the seed parameter
    to something different.

    This noise module can optionally add the distance from the nearest
    seed to the output value.  To enable this feature, set enable_distance
    to True. This causes the points in the Voronoi cells
    to increase in value the further away that point is from the nearest
    seed point.

    Voronoi cells are often used to generate cracked-mud terrain
    formations or crystal-like textures

    This noise module requires no source modules.
    """
    def __init__(self, displacement=1, enable_distance=False, frequency=1, seed=0):
        self.displacement = displacement
        self.enable_distance = enable_distance
        self.frequency = frequency
        self.seed = seed

    @lru_cache(maxsize=32)
    def get_value(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        xInt = int(x) if (x > 0) else int(x) -1
        yInt = int(y) if (y > 0) else int(y) -1
        zInt = int(z) if (z > 0) else int(z) -1

        minDist = 2147483647.0
        xCan = 0
        yCan = 0
        zCan = 0

        for zCur in range(zInt-2, zInt+2):
            for yCur in range(yInt-2, yInt+2):
                for xCur in range(xInt-2, xInt+2):
                    xPos = xCur + value_noise_3d(xCur, yCur, zCur, self.seed)
                    yPos = yCur + value_noise_3d(xCur, yCur, zCur, self.seed+1)
                    zPos = zCur + value_noise_3d(xCur, yCur, zCur, self.seed+2)

                    xDist = xPos - x
                    yDist = yPos - y
                    zDist = zPos - z
                    dist = xDist * xDist + yDist * yDist + zDist * zDist

                    if dist < minDist:
                        minDist = dist
                        xCan = xPos
                        yCan = yPos
                        zCan = zPos
        value = 0

        if self.enable_distance:
            xDist = xCan - x
            yDist = yCan - y
            zDist = zCan - z

            value = (math.sqrt(xDist * xDist + yDist * yDist + zDist * zDist)) *\
                math.sqrt(3) - 1

        return value + (self.displacement * value_noise_3d(math.floor(xCan),
                                                math.floor(yCan),
                                                math.floor(zCan), 0))

    def get_values(self, width, height, min_x, max_x, min_y, max_y, min_z, max_z=None, use_arrays=None):
        if use_arrays is None:
            xa, ya, za = self.create_arrays(width, height, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            xa, ya, za = use_arrays[0], use_arrays[1], use_arrays[2]

        rv = np.empty_like(xa)

        for i in range(width*height):
            rv[i] = self.get_value(xa[i], ya[i], za[i])

        return rv[i]
