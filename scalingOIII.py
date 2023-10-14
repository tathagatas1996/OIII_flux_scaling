import numpy as np
import scipy.integrate as intg 
from scipy.interpolate import interp1d
import emcee

def load_ref_spec( ref_spec ):
    """LOAD THE REFERENCE SPECTRUM"""
    xR = np.loadtxt( ref_spec ,  usecols=(0,))
    R  = np.loadtxt( ref_spec ,  usecols=(1,))
    eR = np.loadtxt( ref_spec ,  usecols=(2,))
    return([xR,R,eR])

def load_target_spec( dir_path ):
    """LOAD THE TARGET SPECTRUM"""
    xT = np.loadtxt( dir_path,  usecols=(0,))
    T  = np.loadtxt( dir_path,  usecols=(1,))
    eT = np.loadtxt( dir_path,  usecols=(2,))
    return([xT,T,eT])
    

class scale:
    def __init__(self, ref_spec, spec_path):
        """INITIALIZE THE SPECTRA"""
        self.xr, self.fr, self.er = load_ref_spec( ref_spec)
        self.xt, self.ft, self.et = load_target_spec( spec_path )
        self.window = (5330,5450)
        self.line   = [(5340, 5380)]
    
    def cont(self, x, xcont, ycont):
        """INTERPOLATE THE UNDERLYING CONTINUUM"""
        f0 = interp1d(xcont, ycont, fill_value = "extrapolate")
        return f0(x)
    
    
    def spectral_segments(self):
        """ CREATE SEGMENTS OF THE REFERENCE SPECTRUM"""
        nomask = [(0,0)]
        self.xc, self.fc = self.trim_range( self.xr, self.fr, self.window, self.line )
        self.xc, self.ec = self.trim_range( self.xr, self.er, self.window, self.line )

        self.xT, self.fT = self.trim_range( self.xr, self.fr, self.window, nomask )
        self.xT, self.eT = self.trim_range( self.xr, self.er, self.window, nomask )
        
        self.fl = self.fT - self.cont( self.xT, self.xc, self.fc) #<<----- Subtract the continuum
        self.R0 = intg.simps(self.fl, self.xT)
        
        """ CREATE SEGMENTS OF THE TARGET SPECTRUM"""
        self.xOIII , self.fOIII  = self.trim_range( self.xt, self.ft, self.window, nomask)
        self.xOIII , self.eOIII  = self.trim_range( self.xt, self.et, self.window, nomask)

        self.xct , self.fct = self.trim_range( self.xOIII, self.fOIII, self.window, self.line)
        self.xct , self.ect = self.trim_range( self.xOIII, self.eOIII, self.window, self.line)

        self.fline = self.fOIII - self.cont( self.xOIII, self.xct, self.fct )

        idx = np.where(self.fline<0.0) #<---- This is removing the negative flux values near the line wings of the line.
        self.fline[idx] = 0
  
    
    def gauss(self, lam,lam0,sd):
        """GAUSSIAN PROFILE"""
        tt = ((lam - lam0)/sd)**2
        ff = np.exp(-tt/2)
        return ff
    
    
    def trim_range(self, lam, flux, 
                   window, mask):
        """SEGMENT THE SPECTRUM"""
        flux = np.delete(flux,np.where((lam<window[0]) | (lam>window[1])))
        lam = np.delete(lam,np.where((lam<window[0]) | (lam>window[1])))
        for wm in mask:
            flux = np.delete(flux,np.where((lam>wm[0]) & (lam<wm[1])))
            lam = np.delete(lam,np.where((lam>wm[0]) & (lam<wm[1])))
        return (lam,flux)

    def reference_spec_smooth(self, x, y, 
                              e, dlam):
        """SMOOTHING THE REFERENCE SPECTRUM"""
        lam_l = min(x)
        lam_h = max(x)
        xsm = np.arange(lam_l,lam_h,dlam)
        Sum = []
        err = []
    
        for i in range(xsm.size):
            Ns = np.sum( self.gauss(xsm[i],x,dlam))
            fs = np.sum( self.gauss(xsm[i],x,dlam) * y )
            es = np.sum( self.gauss(xsm[i],x,dlam) * e )
        
            Sum.append(fs/Ns)
            err.append(es/Ns)   
    
        Sum = np.array(Sum)
        err = np.array(err)
        ff = [xsm,Sum, err]
        return ff

    

    def fsmooth(self, xO, fO, eO, 
                      sd, shift):
        """ Gaussian smoothing of the observed spectrum """
        li = np.arange( min(xO), max(xO), 1 )
        ff = np.zeros( li.size, dtype=float)
        ee = np.zeros( li.size, dtype=float)
        for i in range(li.size):
            Ni = np.sum( self.gauss( li[i], xO, sd ) )
            ff[i] = (1/Ni) * np.sum( fO * self.gauss( li[i], xO, sd ) )
            ee[i] = (1/Ni) * np.sum( eO * self.gauss( li[i], xO, sd ) )
        li = li + shift
        return([li,ff, ee])


    def calculate_interp_error(self, xref,x,e):
        """ CALCULATE THE ERROR ON THE INTERPOLATED FLUX """
        eref = np.zeros( xref.size, dtype= float)
        for i in range (xref.size):
        
            idx1 = np.where( xref[i] >= x )[0]
            idx2 = np.where( xref[i] <  x )[0]
        
            if idx1.size != 0 and idx2.size != 0 :
                id1 = idx1[0]
                id2 = idx2[idx2.size-1]
                eref[i] = np.sqrt(1/e[id1]**2  + 1/e[id2]**2)**(-1)           
        return eref
    

    def f0(self, x, a, sd, shift, 
           xO, fO, eO):
        """ Smooth, Shift, Scale and Interpolate the spectrum at desired points"""
        a0 = self.fsmooth( xO, fO, eO, sd, shift )
        f01 = interp1d( a0[0], a0[1], fill_value='extrapolate' )
        ff = a * f01(x)
        ee = a * self.calculate_interp_error( x, a0[0], a0[2] )
        return [ff,ee]

    
    def log_likelihood(self, theta, x, y, yerr):
        """likelihood function"""
        a, sd, shift = theta
        # target being evaluated at the wavelength of the reference spectrum with scale, smoothing, and shift transformation:
        Olambda, eOlambda = self.f0( self.xT, a, sd, shift, x, y, yerr)
        #local variable being used for the reference spectra:
        Rlambda = self.fl
        # error of reference and error of target added in quadrature                 
        sigma2  =  eOlambda**2 + self.eT**2
        # likelihood estimation:
        l =  -0.5 * np.sum( (Olambda - Rlambda)**2/sigma2 )
        return l

    def log_prior(self, theta):
        """uniform priors"""
        a, sd, shift = theta
        if 0 < a < 10 and 5e-1 < sd < 10 and -8 < shift < 20:
            return 0.0
        return -np.inf

    def log_probability(self, theta, x, y, yerr):
        """total probalility"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)

    def run_emcee(self, x, f, e, nchains):
        """runs mcmc on a dataset"""
        init = np.array([1,4,2], dtype=float)
        pos = init + 1e-4 * np.random.randn(10, 3) 
        nwalkers, ndim = pos.shape
        # (x,f,e is the of the target spectrum)
        sampler = emcee.EnsembleSampler( nwalkers, ndim, self.log_probability, args=(x, f, e) )
        sampler.run_mcmc(pos, nchains, progress=True);
        flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
        return flat_samples
    
    def transform_spectrum(self, x, a, sd, s):
        flam, elam = self.f0(x, a, sd, s, self.xOIII, self.fline, self.eOIII)
        return (x, flam, elam)
    
