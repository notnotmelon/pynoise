import pyopencl as cl
import numpy as np
import os

class GPU:
    def __init__(self):
        platform = cl.get_platforms()[0]    # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        self.ctx = cl.Context([device])      # Create a context with your device
        self.queue = cl.CommandQueue(self.ctx)

    def load_program(self):
        data_path = os.path.dirname(__file__)
        f = open(os.path.join(data_path, 'kernels.cl'), 'r')

        fstr = "".join(f.readlines())

        self.program = cl.Program(self.ctx, fstr).build()

    def gradient_noise(self, x, y, z, seed, quality):
        mf = cl.mem_flags

        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
        z_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
        dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, x.nbytes)

        self.program.gradient_coherent_noise_3d(self.queue, x.shape, None,
        x_buf, y_buf, z_buf, dest_buf, seed, quality)


        rv = np.empty_like(x)

        cl._enqueue_read_buffer(self.queue, dest_buf, rv).wait()

        return rv

    def linear_interp(self, n0, n1, a):
        mf = cl.mem_flags

        n0_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = n0)
        n1_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = n1)
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = a)
        dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, n0.nbytes)

        self.program.linear_interp(self.queue, n0.shape, None,
            n0_buf, n1_buf, a_buf, dest_buf)

        rv = np.empty_like(n0)

        cl._enqueue_read_buffer(self.queue, dest_buf, rv).wait()

        return rv

    def cubic_interp(self, n0, n1, n2, n3, alpha):
        mf = cl.mem_flags

        n0_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = n0)
        n1_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = n1)
        n2_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = n2)
        n3_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = n3)
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = alpha)
        dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, n0.nbytes)

        self.program.cubic_interp(self.queue, n0.shape, None,
            n0_buf, n1_buf, n2_buf, n3_buf, a_buf, dest_buf)

        rv = np.empty_like(n0)

        cl._enqueue_read_buffer(self.queue, dest_buf, rv).wait()

        return rv

    def cylinders(self, xa, za):
        mf = cl.mem_flags

        xbuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = xa)
        zbuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = za)
        dest = cl.Buffer(self.ctx, mf.WRITE_ONLY, xa.nbytes)

        self.program.cylinders(self.queue, xa.shape, None,
            xbuf, zbuf, dest)

        rv = np.empty_like(xa)

        cl._enqueue_read_buffer(self.queue, dest, rv).wait()

        return rv

    def spheres(self, xa, ya, za):
        mf = cl.mem_flags

        xbuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = xa)
        ybuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = ya)
        zbuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = za)
        dest = cl.Buffer(self.ctx, mf.WRITE_ONLY, xa.nbytes)

        self.program.spheres(self.queue, xa.shape, None,
            xbuf, ybuf, zbuf, dest)

        rv = np.empty_like(xa)

        cl._enqueue_read_buffer(self.queue, dest, rv).wait()

        return rv
