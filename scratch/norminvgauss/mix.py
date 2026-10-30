import time, mpmath as mp
mp.mp.dps = 100

def make(x, a, b):
    g = mp.sqrt((a + b) * (a - b))
    c0 = g - mp.log(2 * mp.pi) / 2 - mp.log(2)
    def L(s, sgn):
        v = mp.exp(s)
        t = sgn * (x - b * v) / mp.sqrt(v)
        return c0 - s / 2 - 1 / (2 * v) - g * g * v / 2 + mp.log(mp.erfc(-t / mp.sqrt(2)))
    return L, g

def peak(L, s0):
    with mp.workdps(30):
        best = max(range(-40, 41), key=lambda k: L(s0 + k))
        lo, hi = s0 + best - 1, s0 + best + 1
        phi = (mp.sqrt(5) - 1) / 2
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fc, fd = L(c), L(d)
        for _ in range(70):
            if fc > fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo); fc = L(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo); fd = L(d)
        sp = (lo + hi) / 2
        Lp = L(sp)
        h = mp.mpf('1e-4')
        while True:
            d2 = (L(sp + h) - 2 * Lp + L(sp - h)) / h ** 2
            if d2 < 0: break
            h *= 10
        w = min(1 / mp.sqrt(-d2), mp.mpf(1))
    return mp.mpf(sp), mp.mpf(w), mp.mpf(Lp)

def tailint(x, a, b, sgn):
    L, g = make(x, a, b)
    f = lambda s: L(s, sgn)
    sp, w, Lp = peak(f, -mp.log(g))
    if Lp < -1000:
        return mp.exp(Lp) * w * mp.sqrt(2 * mp.pi)
    Lp = f(sp)
    cut = (mp.mp.dps + 30) * mp.log(10)
    def trunc(sg):
        k = 1
        while f(sp + sg * w * k) - Lp > -cut:
            k *= 2
        return sp + sg * w * k
    lo, hi = trunc(-1), trunc(1)
    pts = [sp + k * w for k in (-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100)]
    nodes = [lo] + [t for t in pts if lo < t < hi] + [hi]
    return mp.quad(lambda s: mp.exp(f(s) - Lp), nodes) * mp.exp(Lp)

if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from scipy import stats
    for s in [(1.25, .5), (1e-3, 0), (1e4, 0), (1e6, 5e5), (10, 9.99999)]:
        for x in [-30., -1., 0., 0.3, 5., 100.]:
            t = time.time()
            c = tailint(mp.mpf(x), *map(mp.mpf, s), 1)
            sf_ = tailint(mp.mpf(x), *map(mp.mpf, s), -1)
            dt = time.time() - t
            print(s, x, mp.nstr(c, 17), mp.nstr(sf_, 17), mp.nstr(c + sf_ - 1, 3),
                  stats.norminvgauss.cdf(x, *s), round(dt, 2))
