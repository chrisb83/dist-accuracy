import mpmath as mp

def k1(z):
    z = mp.mpf(z)
    dps = mp.mp.dps
    if z > (dps + 10) * mp.log(10) / 2 + 5:
        # asymptotic expansion, truncated at the smallest term
        tol = mp.mpf(10) ** (-dps - 5)
        s = term = mp.mpf(1)
        k = 0
        while True:
            k += 1
            new = term * (4 - (2 * k - 1) ** 2) / (k * 8 * z)
            if abs(new) >= abs(term) or abs(new) < tol:
                s += new
                break
            term = new
            s += term
        return mp.sqrt(mp.pi / (2 * z)) * mp.exp(-z) * s
    extra = int(2 * z / mp.log(10)) + 15
    with mp.workdps(dps + extra):
        z = +z
        t = z * z / 4
        # I1 and the psi series together
        term = mp.mpf(1)  # (t^k)/(k!(k+1)!)
        psi1 = -mp.euler          # psi(k+1)
        psi2 = 1 - mp.euler       # psi(k+2)
        sI = mp.mpf(0)
        sP = mp.mpf(0)
        k = 0
        tol = mp.mpf(10) ** (-(dps + extra))
        while True:
            sI += term
            sP += (psi1 + psi2) * term
            k += 1
            term = term * t / (k * (k + 1))
            psi1 += mp.mpf(1) / k
            psi2 += mp.mpf(1) / (k + 1)
            if term < tol * abs(sI) and k > 5:
                break
        I1 = z / 2 * sI
        r = 1 / z + mp.log(z / 2) * I1 - z / 4 * sP
    return +r
