from pynoise.noisemodule import gpu
import numpy as np

class Color(object):
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
   
    def get_value_tuple(self):
        return self.r, self.g, self.b

    def get_upscaled_value_tuple(self):
        return (int)(self.r*255), (int)(self.g*255), (int)(self.b*255)


def linear_interp_color(color0, color1, alpha):
    nc_r = linear_interp(color0.r, color1.r, alpha)
    nc_g = linear_interp(color0.g, color1.g, alpha)
    nc_b = linear_interp(color0.b, color1.b, alpha)

    return Color(nc_r, nc_g, rc_b)

def linear_interp_colors(colors0, colors1, alpha):
    length = len(colors0)

    nc0_r = np.empty(length)
    nc0_g = np.empty(length)
    nc0_b = np.empty(length)
    nc1_r = np.empty(length)
    nc1_g = np.empty(length)
    nc1_b = np.empty(length)

    i = 0

    while i < length:
        nc0_r[i] = colors0[i].r
        nc0_g[i] = colors0[i].g
        nc0_b[i] = colors0[i].b
        nc1_r[i] = colors1[i].r
        nc1_g[i] = colors1[i].g
        nc1_b[i] = colors1[i].b
        i = i + 1


    nci_r = gpu.linear_interp(nc0_r, nc1_r, alpha)
    nci_g = gpu.linear_interp(nc0_g, nc1_g, alpha)
    nci_b = gpu.linear_interp(nc0_b, nc1_b, alpha)

    rv = [None] * length
    i = 0

    while i < length:
        rv[i] = Color(nci_r[i], nci_g[i], nci_b[i])
        i = i+1

    return rv
