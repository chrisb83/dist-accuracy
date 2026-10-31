import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
exec(open('f1_double.py').read().split('for x, k, df in')[0])
import mpmath as mp
from k2_ref import pdfB
mp.mp.dps=30
print('k=2 df=1e6 mine', pdf(1.0, 2.0, 1e6, rk_quad), 'exact', mp.nstr(pdfB(1,1e6),17))
print('k=2 df=10 mine', pdf(1.0, 2.0, 10, rk_quad), 'exact', mp.nstr(pdfB(1,10),17))
# wider outer range for k=1.5, df=1e6
df=1e6; sd=1/np.sqrt(2*df); x=1.0; k=1.5
g = lambda u: np.exp(logfS(np.exp(u), df) + 2*u)*rk_quad(x*np.exp(u), k)
e=np.linspace(-25,25,101)*sd
print('k=1.5 df=1e6 wide', sum(integrate.quad(g,a,b,epsabs=0,epsrel=1e-12)[0] for a,b in zip(e[:-1],e[1:])))
# infinite-df value (S=1) for scale of the 1/df effect
print('k=1.5 S=1', rk_quad(1.0,1.5))
