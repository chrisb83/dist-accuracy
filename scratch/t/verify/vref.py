# Independent verifier reference for scipy.stats.t (standard form).
# cdf via mp.quad of the docstring pdf (log-space substitution for tails),
# logpdf via loggamma, quantiles via findroot on a betainc cdf, then checked
# against the quad cdf.
import mpmath as mp

def logpdf(x, df):
    x = mp.mpf(x)
    if df == mp.inf:
        return -x*x/2 - mp.log(2*mp.pi)/2
    df = mp.mpf(df)
    return (mp.loggamma((df+1)/2) - mp.loggamma(df/2) - mp.log(mp.pi*df)/2
            - (df+1)/2*mp.log(1 + x*x/df))

def pdf(x, df):
    return mp.exp(logpdf(x, df))

def lower_tail_quad(a, df):
    # P(T < -a), a > 0, = int_{log a}^{inf} f(e^s) e^s ds
    s0 = mp.log(a)
    df = mp.mpf(df)
    g = lambda s: mp.exp(logpdf(mp.exp(s), df) + s)
    sc = max(1, 1/df)
    pts = [s0, s0+1, s0+5, s0+20, s0+20*sc, s0+200*sc, mp.inf]
    pts = sorted(set(pts))
    return mp.quad(g, pts)

def cdf_quad(x, df):
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(1)/2
    if abs(x) < 1:
        # 0.5 + int_0^x f
        return mp.mpf(1)/2 + mp.quad(lambda t: pdf(t, df), [0, x])
    if x < 0:
        return lower_tail_quad(-x, df)
    return 1 - lower_tail_quad(x, df)

def cdf_beta(x, df):
    x = mp.mpf(x); df = mp.mpf(df)
    z = df/(df + x*x)
    h = mp.betainc(df/2, mp.mpf(1)/2, 0, z, regularized=True)/2
    return h if x < 0 else 1 - h

def ppf(q, df):
    q = mp.mpf(q)
    sign = -1 if q < 0.5 else 1
    qq = q if q < 0.5 else 1 - q
    # solve in s = log|x|:  log P(T < -e^s) = log qq
    f = lambda s: mp.log(cdf_beta(-mp.exp(s), df)) - mp.log(qq)
    # starting guess from tail asymptote
    df_ = mp.mpf(df)
    C = mp.exp(mp.loggamma((df_+1)/2) - mp.loggamma(df_/2))/mp.sqrt(mp.pi*df_)
    s_guess = (mp.log(C*df_**((df_+1)/2)/df_) - mp.log(qq))/df_
    if s_guess < 0: s_guess = mp.mpf(0)
    try:
        s = mp.findroot(f, s_guess)
    except Exception:
        s = mp.findroot(f, (s_guess - 5, s_guess + 5), solver='anderson')
    return sign*mp.exp(s)
