if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append('/home/atrus/Copy/pynoise')

    from pynoise.noisemodule import Perlin

    from pynoise.noiseutil import noise_map_plane_gpu, terrain_gradient, RenderImage, noise_map_plane

    import time

    t =time.process_time()
    nm_perlin = noise_map_plane_gpu(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Perlin())
    print(time.process_time() - t)
    t = time.process_time()
    nm_perlin2 = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Perlin())
    print(time.process_time() - t)

    gradient = terrain_gradient()

    r = RenderImage()

    r.render_numpy(nm_perlin, 256, 256, 'perlin.png', gradient)
    r.render_numpy(nm_perlin2, 256, 256, 'perlin2.png', gradient)
