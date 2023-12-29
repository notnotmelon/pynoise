from sortedcontainers import SortedDict
import numpy as np
from PIL import Image, ImageColor
from pynoise.util import clamp
from pynoise.interpolators import linear_interp
from pynoise.noisemodule import gpu
from pynoise.colors import Color, linear_interp_colors
import math

class GradientColor():
    gradient_points = SortedDict()

    def add_gradient_point(self, position, color):
        self.gradient_points[position] = color

    def get_color(self, position):
        assert (len(self.gradient_points.keys()) > 1)

        for i, p in enumerate(self.gradient_points.keys()):
            if position < p:
                break

        index0 = clamp(i - 1, 0, len(self.gradient_points)-1)
        index1 = clamp(i, 0, len(self.gradient_points)-1)

        if index0 == index1:
            return self.gradient_points[self.gradient_points.keys()[index1]]

        input0 = self.gradient_points.keys()[index0]
        input1 = self.gradient_points.keys()[index1]

        alpha = (position - input0) / (input1 - input0)
        c0 = self.gradient_points[input0]
        c1 = self.gradient_points[input1]

        return linear_interp_color(c0, c1, alpha)

    def get_colors(self, noisemap):
        assert (len(self.gradient_points.keys()) > 1)
        length = noisemap.size

        c0 = [None] * length
        c1 = [None] * length
        alpha = np.empty(length)

        for i, nm in enumerate(noisemap):
            for gpi, p in enumerate(self.gradient_points.keys()):
                if nm < p:
                    break

            index0 = clamp(gpi - 1, 0, len(self.gradient_points)-1)
            index1 = index0 + 1

            if index0 == index1:
                alpha[i] = 0.5
                c0[i] = self.gradient_points[self.gradient_points.keys()[index1]]
                c1[i] = self.gradient_points[self.gradient_points.keys()[index1]]
                continue
            
            input0 = self.gradient_points.keys()[index0]
            input1 = self.gradient_points.keys()[index1]
            alpha[i] = (nm - input0) / (input1 - input0)

            c0[i] = self.gradient_points[input0]
            c1[i] = self.gradient_points[input1]

        return linear_interp_colors(c0, c1, alpha)

def noise_map_cylinder(width=0, height=0, lower_angle=0, upper_angle=0,
    lower_height=0, upper_height=0, source=None):
    assert lower_angle < upper_angle
    assert lower_height < upper_height
    assert width > 0
    assert height > 0
    assert source is not None
    assert img_name is not None

    angle_extent = upper_angle - lower_angle
    height_extent = upper_height - lower_height
    x_delta = angle_extent / width
    y_delta = angle_extent / height
    cur_angle = lower_angle
    cur_height = lower_height
    nm = np.zeros((height, width))

    for y in range(0, height):
        cur_angle = lower_angle
        for x in range(0, width):
            cx = math.cos(math.radians(cur_angle))
            cy = height
            cz = math.sin(math.radians(cur_angle))
            nm[y][x] = source.get_value(cx, cy, cz)
            cur_angle += x_delta
        cur_height += y_delta

    return np.flipud(nm)

def noise_map_plane(width=0, height=0, lower_x=0, upper_x=0, lower_z=0, upper_z=0,
    source=None, seamless=False, y=0, y_max=0):
    assert lower_x < upper_x
    assert lower_z < upper_z
    assert width > 0
    assert height > 0
    assert source is not None

    x_extent = upper_x - lower_x
    z_extent = upper_z - lower_z
    x_delta = x_extent / width
    z_delta = z_extent / height
    x_cur = lower_x
    z_cur = lower_z

    nm = np.zeros(height*width)
    i = 0

    for x in range(width):
        x_cur = lower_x
        for z in range(height):
            if not seamless:
                nm[i] = source.get_value(x_cur, y, z_cur)
            else:
                sw = source.get_value(x_cur, y, z_cur)
                se = source.get_value(x_cur+x_extent, y, z_cur)
                nw = source.get_value(x_cur, y, z_cur+z_extent)
                ne = source.get_value(x_cur+x_extent, y, z_cur+z_extent)
                swy = source.get_value(x_cur, y+y_max, z_cur)
                sey = source.get_value(x_cur+x_extent, y+y_max, z_cur)
                nwy = source.get_value(x_cur, y+y_max, z_cur+z_extent)
                ney = source.get_value(x_cur+x_extent, y+y_max, z_cur+z_extent)

                x_blend = 1 - ((x_cur - lower_x) / x_extent)
                y_blend = 1 - y / y_max
                z_blend = 1 - ((z_cur - lower_z) / z_extent)

                z0 = linear_interp(sw, se, x_blend)
                z1 = linear_interp(nw, ne, x_blend)
                z = linear_interp(z0, z1, z_blend)
                zy0 = linear_interp(swy, sey, x_blend)
                zy1 = linear_interp(nwy, ney, x_blend)
                zy = linear_interp(zy0, zy1, z_blend)
                nm[i] = linear_interp(z, zy, y_blend)

            x_cur += x_delta
            i += 1
        z_cur += z_delta

    return nm

def noise_map_plane_gpu(width=0, height=0, lower_x=0, upper_x=0, lower_z=0, upper_z=0,
    source=None, seamless=False, y=0, y_max=0):
    assert lower_x < upper_x
    assert lower_z < upper_z
    assert width > 0
    assert height > 0
    assert source is not None

    if not seamless:
        return source.get_values(width, height, lower_x, upper_x, y, y, lower_z, upper_z)
    else:
        x_extent = upper_x - lower_x
        z_extent = upper_z - lower_z
        x_delta = x_extent / width
        z_delta = z_extent / height

        se = source.get_values(width, height, upper_x+x_delta, upper_x+x_extent+x_delta, y, y, lower_z, upper_z)
        sw = source.get_values(width, height, lower_x, upper_x, y, y, lower_z, upper_z)
        nw = source.get_values(width, height, lower_x, upper_x, y, y, upper_z+z_delta, upper_z+z_extent+z_delta)
        ne = source.get_values(width, height, upper_x+x_delta, upper_x+x_extent+x_delta, y, y, upper_z+z_delta, upper_z+z_extent+z_delta)
        yy = y + y_max
        sey = source.get_values(width, height, upper_x+x_delta, upper_x+x_extent+x_delta, yy, yy, lower_z, upper_z)
        swy = source.get_values(width, height, lower_x, upper_x, yy, yy, lower_z, upper_z)
        nwy = source.get_values(width, height, lower_x, upper_x, yy, yy, upper_z+z_delta, upper_z+z_extent+z_delta)
        ney = source.get_values(width, height, upper_x+x_delta, upper_x+x_extent+x_delta, yy, yy, upper_z+z_delta, upper_z+z_extent+z_delta)

        x_blend = np.zeros(width*height)
        y_blend = np.zeros(width*height)
        z_blend = np.zeros(width*height)

        i = 0
        z_cur = lower_z
        for x in range(width):
            x_cur = lower_x
            for z in range(height):
                x_blend[i] = 1 - ((x_cur - lower_x) / x_extent)
                y_blend[i] = 1 - y / y_max
                z_blend[i] = 1 - ((z_cur - lower_z) / z_extent)
                x_cur += x_delta
                i += 1
            z_cur += z_delta

        z0 = gpu.linear_interp(sw, se, x_blend)
        z1 = gpu.linear_interp(nw, ne, x_blend)
        z2 = gpu.linear_interp(swy, sey, x_blend)
        z3 = gpu.linear_interp(nwy, ney, x_blend)

        z4 = gpu.linear_interp(z0, z1, z_blend)
        z5 = gpu.linear_interp(z2, z3, z_blend)

        return gpu.linear_interp(z4, z5, y_blend)

def noise_map_sphere(width=0, height=0, east_bound=0, west_bound=0,
    north_bound=0, south_bound=0, source=None):
    assert east_bound > west_bound
    assert north_bound > south_bound
    assert width > 0
    assert height > 0
    assert source is not None

    nm = np.zeros(height*width)

    lon_extent = east_bound - west_bound
    lat_extent = north_bound - south_bound
    x_delta = lon_extent / width
    y_delta = lat_extent / height
    cur_lon = west_bound
    cur_lat = south_bound

    i = 0

    for y in range(height):
        cur_lon = west_bound
        for x in range(width):
            r = math.cos(math.radians(cur_lat))
            xa = r * math.cos(math.radians(cur_lon))
            ya = math.sin(math.radians(cur_lat))
            za = r * math.sin(math.radians(cur_lon))

            nm[i] = source.get_value(xa, ya, za)

            cur_lon += x_delta
            i += 1
        cur_lat += y_delta

    return nm

def noise_map_sphere_gpu(width=0, height=0, east_bound=0, west_bound=0,
    north_bound=0, south_bound=0, source=None):

    lats = np.radians(np.linspace(south_bound, north_bound, height))
    longs = np.radians(np.linspace(west_bound, east_bound, width))

    longs = np.tile(longs, height)
    lats = np.repeat(lats, width)

    r = np.cos(lats)
    xa = r * np.cos(longs)
    ya = np.sin(lats)
    za = r * np.sin(longs)

    return source.get_values(width, height, 0,0, 0,0, 0,0, use_arrays=[xa,ya,za])

def grayscale_gradient():
    grad = GradientColor()
    grad.add_gradient_point(-1, Color(0, 0, 0))
    grad.add_gradient_point(1, Color(1, 1, 1))
    return grad

def terrain_gradient():
    grad = GradientColor()

    grad.add_gradient_point(-1.00, Color(  0,   0, 128/255))
    grad.add_gradient_point(-0.20, Color( 32/255,  64/255, 128/255))
    grad.add_gradient_point(-0.04, Color( 64/255,  96/255, 192/255))
    grad.add_gradient_point(-0.02, Color(192/255, 192/255, 128/255))
    grad.add_gradient_point( 0.00, Color(  0, 192/255,   0))
    grad.add_gradient_point( 0.25, Color(192/255, 192/255,   0))
    grad.add_gradient_point( 0.50, Color(160/255,  96/255,  64/255))
    grad.add_gradient_point( 0.75, Color(128/255, 1, 1))
    grad.add_gradient_point( 1.00, Color(1, 1, 1))

    return grad

class RenderImage():
    def __init__(self, light_enabled=False, wrap_enabled=False, light_azimuth=45,
        light_brightness=1, light_color=Color(1, 1, 1),
        light_contrast=1, light_elev=45, light_intensity=1):
        self.light_enabled = light_enabled
        self.wrap_enabled = wrap_enabled
        self.light_azimuth = light_azimuth
        self.light_brightness = light_brightness
        self.light_color = light_color
        self.light_contrast = light_contrast
        self.light_elev = light_elev
        self.light_intensity = light_intensity

        self.cos_azimuth = 0
        self.sin_azimuth = 0
        self.cos_elev = 0
        self.sin_elev = 0
        self.recalc_light = True

    def calc_dest_color(self, source, background, light_value):
        nc = source

        if self.light_enabled:
            ncr, ncg, ncb = nc.get_value_tuple()
            lcr, lcg, lcb = self.light_color.get_value_tuple()
            lcr *= light_value
            lcg *= light_value
            lcb *= light_value

            ncr *= lcr
            ncg *= lcg
            ncb *= lcb

            return Color(ncr, ncg, ncb)
        else:
            return nc

    def calc_light_intensity(self, center, left, right, up, down):
        if self.recalc_light:
            self.cos_azimuth = math.cos(math.radians(self.light_azimuth))
            self.sin_azimuth = math.sin(math.radians(self.light_azimuth))
            self.cos_elev = math.cos(math.radians(self.light_elev))
            self.sin_elev = math.sin(math.radians(self.light_elev))
            self.recalc_light = False

        I_MAX = 1
        io = I_MAX * math.sqrt(2) * self.sin_elev / 2
        ix = (I_MAX - io) * self.light_contrast * math.sqrt(2) * self.cos_elev * self.cos_azimuth
        iy = (I_MAX - io) * self.light_contrast * math.sqrt(2) * self.cos_elev * self.sin_azimuth
        intensity = (ix * (left - right) + iy * (down - up) + io)

        if intensity < 0:
            return 0
        else:
            return intensity

    def render(self, width, height, noisemap, gradient):
        img = Image.new('RGB', (width, height), '#ffffff')
        colors = gradient.get_colors(noisemap)
        i = -1

        for y in range(height):
            for x in range(width):
                i += 1
                dest_color = colors[i]

                light_intensity = 1

                if self.light_enabled:
                    x_left = 0
                    x_right = 0
                    y_up = 0
                    y_down = 0

                    if(self.wrap_enabled):
                        if x == 0:
                            x_left = width-1
                            x_right = 1
                        elif x == width-1:
                            x_left = -1
                            x_right = width-1
                        else:
                            x_left = -1
                            x_right = 1

                        if y == 0:
                            y_down = height-1
                            y_up = 1
                        elif y == height-1:
                            y_down = -1
                            y_up = -(height-1)
                        else:
                            y_down = -1
                            y_up = 1
                    else:
                        if x == 0:
                            x_left = 0
                            x_right = 1
                        elif x == width - 1:
                            x_left = -1
                            x_right = 0
                        else:
                            x_left = -1
                            x_right = 1

                        if y == 0:
                            y_down = 0
                            y_up = 1
                        elif y == height - 1:
                            y_down = -1
                            y_up = 0
                        else:
                            y_down = -1
                            y_up = 1

                    nc = noisemap[i]
                    nl = noisemap[i+x_left]
                    nr = noisemap[i+x_right]
                    nd = noisemap[i+(y_down*width)+y_down]
                    nu = noisemap[i+(width*y_up)]

                    light_intensity = self.calc_light_intensity(nc, nl, nr, nu, nd)
                    light_intensity *= self.light_brightness

                bg_color = Color(1, 1, 1)
                color = self.calc_dest_color(dest_color, bg_color, light_intensity)
                t = color.get_upscaled_value_tuple()
                img.putpixel((x,height-1-y), (t[0], t[1], t[2]))
        return img