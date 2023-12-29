def clamp(v, min, max):
    if v < min:
        return min
    elif v > max:
        return max
    else:
        return v
