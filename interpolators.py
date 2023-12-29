def cubic_interp(n0, n1, n2, n3, a):
    p = (n3 - n2) - (n0 - n1)
    q = (n0 - n1) - p
    r = n2 - n0
    s = n1

    return (p*a**3) + (q*a**2) + (r*a) + s

def linear_interp(n0, n1, a):
    return ((1 - a) * n0) + (a * n1)

def scurve3(a):
    return (a * a * (3-2*a))

def scurve5(a):
    return (6 * a**5) - (15 * a**4) + (10 * a**3)
