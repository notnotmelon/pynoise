from pynoise.interpolators import scurve3, scurve5, linear_interp
from pynoise.vectortable import vector
from pynoise.quality import Quality

X_NOISE_GEN = 1619
Y_NOISE_GEN = 31337
Z_NOISE_GEN = 6971
SEED_NOISE_GEN = 1013
SHIFT_NOISE_GEN = 8

def gradient_coherent_noise_3d(x, y, z, seed, quality):
    if x > 0:
        x0 = int(x)
    else:
        x0 = int(x) - 1

    if y > 0:
        y0 = int(y)
    else:
        y0 = int(y) - 1

    if z > 0:
        z0 = int(z)
    else:
        z0 = int(z) - 1

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    xs = 0
    ys = 0
    zs = 0

    if quality == Quality.fast:
        xs = (x - x0)
        ys = (y - y0)
        zs = (z - z0)
    elif quality == Quality.std:
        xs = scurve3(x-x0)
        ys = scurve3(y-y0)
        zs = scurve3(z-z0)
    else:
        xs = scurve5(x-x0)
        ys = scurve5(y-y0)
        zs = scurve5(z-z0)

    n0 = gradient_noise_3d(x, y, z, x0, y0, z0, seed)
    n1 = gradient_noise_3d(x, y, z, x1, y0, z0, seed)
    ix0 = linear_interp(n0, n1, xs)

    n0 = gradient_noise_3d(x, y, z, x0, y1, z0, seed)
    n1 = gradient_noise_3d(x, y, z, x1, y1, z0, seed)
    ix1 = linear_interp(n0, n1, xs)
    iy0 = linear_interp(ix0, ix1, ys)

    n0 = gradient_noise_3d(x, y, z, x0, y0, z1, seed)
    n1 = gradient_noise_3d(x, y, z, x1, y0, z1, seed)
    ix0 = linear_interp(n0, n1, xs)

    n0 = gradient_noise_3d(x, y, z, x0, y1, z1, seed)
    n1 = gradient_noise_3d(x, y, z, x1, y1, z1, seed)
    ix1 = linear_interp(n0, n1, xs)
    iy1 = linear_interp(ix0, ix1, ys)

    return linear_interp(iy0, iy1, zs)

def gradient_noise_3d(fx, fy, fz, ix, iy, iz, seed):
    vectorIndex = (
    X_NOISE_GEN * ix +
    Y_NOISE_GEN * iy +
    Z_NOISE_GEN * iz +
    SEED_NOISE_GEN * seed
    )

    vectorIndex &= 0xffffffff
    vectorIndex = vectorIndex ^ (vectorIndex >> SHIFT_NOISE_GEN)
    vectorIndex &= 0xff

    xv = vector[vectorIndex << 2]
    yv = vector[(vectorIndex << 2) + 1]
    zv = vector[(vectorIndex << 2) + 2]

    xvp = (fx - ix)
    yvp = (fy - iy)
    zvp = (fz - iz)

    return ((xv * xvp) + (yv * yvp) + (zv * zvp)*2.12)

def int_value_noise_3d(x, y, z, seed):
    n = (
        X_NOISE_GEN * x +
        Y_NOISE_GEN * y +
        Z_NOISE_GEN * z +
        SEED_NOISE_GEN * seed
    ) & 0x7fffffff

    n = (n >> 13) ^ n

    return (n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff

def value_noise_3d(x, y, z, seed):
    return 1 - (int_value_noise_3d(x, y, z, seed) / 1073741824)
