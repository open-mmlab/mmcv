import numpy as np


def EnforceRange(x, MaxValue):
    return min(max(x, 0), MaxValue)


# pythran export flow_warp_c(float64[:,:,:], float64[:,:,2], int?, int?)
def flow_warp_c(img, flow, filling_value=0, interpolate_mode=1):
    out = np.zeros_like(img)
    height, width, channels = out.shape
    for h in range(height):
        for w in range(width):
            x = h + flow[h, w, 1]
            y = w + flow[h, w, 0]

            if x < 0 or x >= height - 1 or y < 0 or y >= width - 1:
                out[h, w] = filling_value
                continue

            if interpolate_mode == 0:
                BilinearInterpolate(img, width, height, x, y, out[h, w])
            elif interpolate_mode == 1:
                NNInterpolate(img, width, height, x, y, out[h, w])
            else:
                raise NotImplementedError("Interpolation Method")
    return out


def BilinearInterpolate(img, width, height, x, y, out):
    xx = int(x)
    yy = int(y)

    dx = max(min(x - xx, 1.), 0.)
    dy = max(min(y - yy, 1.), 0.)

    for m in range(2):
        for n in range(2):
            u = EnforceRange(yy + n, width)
            v = EnforceRange(xx + m, height)
            s = abs(1 - m - dx) * abs(1 - n - dy)
            out += img[v, u] * s


def NNInterpolate(img, width, height, x, y, out):
    xx = int(x)
    yy = int(y)

    dx = max(min(x - xx, 1.), 0.)
    dy = max(min(y - yy, 1.), 0.)

    m = 0 if dx < 0.5 else 1
    n = 0 if dy < 0.5 else 1

    u = EnforceRange(yy + n, width)
    v = EnforceRange(xx + m, height)
    out += img[v, u]
