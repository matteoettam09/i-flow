""" Runs Foam, which is installed in foam_git/foam/, with the test functions
    of iflow_test.py. Test functions are transcribed in numpy here.
    Note that this Foam version, in the pipeline with scons, only runs on
    python 2.7.
    The output will be printed and can be saved in a file, to be read by
    iflow_test.py.

    Foam is taken from https://gitlab.com/shoeche/foam/, but Foam.C is modified as

--- a/Foam.C
+++ b/Foam.C
@@ -483,6 +483,7 @@ double Foam::Update(const int mode)
     THROW(fatal_error,"Summation does not agree.");
   double error(dabs(Sigma()/Mean()));
 #ifndef USING__PI_only
+  /*
   msg_Info()<<"  "<<om::bold<<m_vname<<om::reset<<" = "<<om::blue
            <<Mean()*m_scale<<" "<<m_uname<<om::reset<<" +- ( "
            <<error*Mean()*m_scale<<" "<<m_uname<<" = "<<om::red
@@ -490,15 +491,31 @@ double Foam::Update(const int mode)
            <<Mean()/m_max*100.0<<" %, n = "<<m_np
            <<" ( "<<m_nvp/m_np*100.0<<" % ), "
            <<m_channels.size()-m_point.size()
-           <<" cells      "<<mm(1,mm::up)<<bm::cr<<std::flush;
+           <<" cells      "<<mm(1,mm::up)<<bm::cr<<std::endl;
+  */
+  msg_Info()<<std::setprecision(15)<<Mean()*m_scale<<" +- "
+           <<error*Mean()*m_scale<<" ; "<<m_np<<std::setprecision(6)
+           <<" ; eff = "<<Mean()/m_max*100.0<<" %, n = "<<m_np
+           <<" ( "<<m_nvp/m_np*100.0<<" % ), "
+           <<m_channels.size()-m_point.size()
+           <<" cells      \n "<<mm(1,mm::up)<<bm::cr<<std::endl;
 #else
+  /*
   msg_Info()<<"  "<<m_vname<<" = "<<Mean()*m_scale<<" "<<m_uname
            <<" +- ( "<<error*Mean()*m_scale<<" "<<m_uname<<" = "
            <<error*100.0<<" % )\n  eff = "
            <<Mean()/m_max*100.0<<" %, n = "<<m_np
            <<" ( "<<m_nvp/m_np*100.0<<" % ), "
            <<m_channels.size()-m_point.size()
-           <<" cells      "<<mm_up(1)<<bm_cr<<std::flush;
+           <<" cells      "<<mm_up(1)<<bm_cr<<std::endl;
+  */
+  msg_Info()<<std::setprecision(15)<<Mean()*m_scale<<" +-  "
+           <<error*Mean()*m_scale
+           <<" ; "<<m_np<<std::setprecision(6)<<" ; eff = "
+           <<Mean()/m_max*100.0<<" %, n = "<<m_np
+           <<" ( "<<m_nvp/m_np*100.0<<" % ), "
+           <<m_channels.size()-m_point.size()
+           <<" cells      \n "<<mm_up(1)<<bm_cr<<std::endl;
 #endif
   return error;
 }

"""

import numpy as np
from scipy.special import erf

import foam_git.foam.Foam as fm
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('function', 'Gauss', 'The function to integrate',
                    short_name='f')
flags.DEFINE_integer('ndims', 4, 'The number of dimensions for the integral',
                     short_name='d')
flags.DEFINE_float('alpha', 0.2, 'The width of the Gaussians',
                   short_name='a')
flags.DEFINE_float('radius1', 0.45, 'The outer radius for the annulus integrand',
                   short_name='r1')
flags.DEFINE_float('radius2', 0.2, 'The inner radius for the annulus integrand',
                   short_name='r2')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('ptspepoch', 5000, 'Number of points to sample per epoch',
                     short_name='p')
flags.DEFINE_float('precision', 1e-5, 'Target precision in integrator comparison',
                   short_name='tp')
flags.DEFINE_bool('targetmode', False, 'Flag to trigger training until target precision is reached',
                  short_name='t')

class TestFunctions:
    """ Contains the functions discussed in the reference above.

    Attributes:
        ndims (int): dimensionality of the function to be integrated
        alpha (float): width of the Gaussians in the test functions
                       gauss and camel
        kwargs: additional parameters for the functions (not used at the moment)

    """
    def __init__(self, ndims, alpha, **kwargs):
        self.ndims = ndims
        self.alpha = alpha
        self.variables = kwargs
        self.calls = 0

    def gauss(self, x):
        """ Based on eq. 10 of [1], Gaussian function.

        Integral equals erf(1/(2*alpha)) ** ndims

        Args:
            x (np.array): Array with batch of points to evaluate

        Returns: np.array: functional values at the given points

        """
        x = np.array([x])
        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.0*np.sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        self.calls += 1
        return pre * np.exp(exponent)

    def camel(self, x):
        """ Based on eq. 12 of [1], Camel function.

        The Camel function consists of two Gaussians, centered at
        (1/3, 2/3) in each dimension.

        Integral equals
            (0.5*(erf(1/(3*alpha)) + erf(2/(3*alpha)) ))** ndims

        Args:
            x (np.array): Array with batch of points to evaluate

        Returns: np.array: functional values at the given points

        """
        x = np.array([x])
        pre = 1./(self.alpha*np.sqrt(np.pi))**self.ndims
        exponent1 = -1.*np.sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = -1.*np.sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        self.calls += 1
        return 0.5*pre*(np.exp(exponent1)+np.exp(exponent2))

    def circle(self, x):
        """ Based on eq. 14 of [1], two overlapping circles.

        Thickness and height change along the circles.

        Integral equals 0.0136848(1)

        Args:
            x (np.array): Array with batch of points to evaluate

        Returns:
            np.array: functional values at the given points.

        Raises:
            ValueError: If ndims is not equal to 2.

        """
        if self.ndims != 2:
            raise ValueError("ndims must be equal to 2 for circle function!")
        dx1, dy1, rr, w1, ee = 0.4, 0.6, 0.25, 1./0.004, 3.0
        res = (x[..., 1]**ee
               * np.exp(-w1*np.abs((x[..., 1]-dy1)**2
                                   + (x[..., 0]-dx1)**2-rr**2))
               + (1.0-x[..., 1]**ee)
               * np.exp(-w1*np.abs((x[..., 1]-1.0+dy1)**2
                                   + (x[..., 0]-1.0+dx1)**2-rr**2)))
        self.calls += 1
        return res

    class Ring:
        """ Class to store the annulus (ring) function.

        Attributes:
            radius1 (float): Outer radius of the annulus.
            radius2 (float): Inner radius of the annulus.

        """
        def __init__(self, radius1, radius2):
            """ Init annulus function. """

            # Ensure raidus1 is the large one
            if radius1 < radius2:
                radius1, radius2 = radius2, radius1

            self.radius12 = radius1**2
            self.radius22 = radius2**2

        def __call__(self, pts):
            """ Calculate annulus function.

            Args:
                x (np.array): Array with batch of points to evaluate

            Returns:
                np.array: 1. if on annulus, 0. otherwise

            """
            radius = np.sum((pts-0.5)**2, axis=-1)
            out_of_bounds = (radius < self.radius22) | (radius > self.radius12)
            ret = np.where(out_of_bounds, np.zeros_like(radius),
                           np.ones_like(radius))
            return ret

        @property
        def area(self):
            """ Get the area of annulus surface. """
            return np.pi*(self.radius12 - self.radius22)

    class TriangleIntegral:
        """ Class implementing the scalar one-loop triangle. """

        def __init__(self, mass_ext=None, mass_int=None):
            if len(mass_ext) != 3:
                raise ValueError('Triangle requires 3 external masses')
            if len(mass_int) != 3:
                raise ValueError('Triangle requires 3 external masses')
            self.mass_ext = np.array(mass_ext)**2
            self.mass_int = np.array(mass_int)**2

        def FTri(self, t1, t2, perm):
            """ Helper function to evaluate the triangle. """
            return (- self.mass_ext[perm[0]]*t1
                    - self.mass_ext[perm[1]]*t1*t2
                    - self.mass_ext[perm[2]]*t2
                    + (1 + t1 + t2)
                    * (t1*self.mass_int[perm[0]]
                       + t2*self.mass_int[perm[1]]
                       + self.mass_int[perm[2]]))

        def __call__(self, x):
            """ Calculate the one-loop triangle.

            Args:
                x (tf.Tensor): Numpy array with batch of points to evaluate

            Returns: np.ndarray: functional values the given points

            """
            numerator = (1 + x[..., 0] + x[..., 1])**-1.0
            denominator1 = self.FTri(x[..., 0], x[..., 1], [1, 2, 0])
            denominator2 = self.FTri(x[..., 0], x[..., 1], [2, 0, 1])
            denominator3 = self.FTri(x[..., 0], x[..., 1], [0, 1, 2])

            return (- numerator/denominator1
                    - numerator/denominator2
                    - numerator/denominator3)

    class BoxIntegral:
        """ Class implementing the scalar one-loop box. """

        def __init__(self, s12, s23, mass_ext=None, mass_int=None):
            if len(mass_ext) != 4:
                raise ValueError('Box requires 4 external masses')
            if len(mass_int) != 4:
                raise ValueError('Box requires 4 external masses')
            self.mass_ext = np.array(mass_ext)**2
            self.mass_int = np.array(mass_int)**2
            self.s12 = s12
            self.s23 = s23

        def FBox(self, s12, s23, t1, t2, t3, perm):
            """ Helper function to evaluate the box. """
            return (-s12*t2
                    - s23*t1*t3
                    - self.mass_ext[perm[0]]*t1
                    - self.mass_ext[perm[1]]*t1*t2
                    - self.mass_ext[perm[2]]*t2*t3
                    - self.mass_ext[perm[3]]*t3
                    + (1+t1+t2+t3)
                    * (t1*self.mass_int[perm[0]]+t2*self.mass_int[perm[1]]
                       + t3*self.mass_int[perm[2]]+self.mass_int[perm[3]]))

        def __call__(self, x):
            """ Calculate the one-loop box.

            Args:
                x (tf.Tensor): Numpy array with batch of points to evaluate

            Returns: np.ndarray: functional values the given points

            """
            denominator1 = self.FBox(self.s23, self.s12,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [1, 2, 3, 0])
            denominator2 = self.FBox(self.s12, self.s23,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [2, 3, 0, 1])
            denominator3 = self.FBox(self.s23, self.s12,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [3, 0, 1, 2])
            denominator4 = self.FBox(self.s12, self.s23,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [0, 1, 2, 3])

            return (1.0/denominator1**2
                    + 1.0/denominator2**2
                    + 1.0/denominator3**2
                    + 1.0/denominator4**2)


def main(argv):
    """ Main function to run Foam. """
    del argv

    # gauss: 2, 4, 8, or 16
    # camel: 2, 4, 8, or 16
    # circle: 2
    # annulus: 2
    # Box: ndims = 3, Triangle: ndims = 2
    ndims = FLAGS.ndims
    alpha = FLAGS.alpha

    func = TestFunctions(ndims, alpha)

    # select function:
    if FLAGS.function == 'Gauss':
        target = erf(1/(2.*alpha))**ndims
        integrand = func.gauss
    elif FLAGS.function == 'Camel':
        target = (0.5*(erf(1/(3.*alpha))+erf(2/(3.*alpha))))**ndims
        integrand = func.camel
    elif FLAGS.function == 'Circle':
        target = 0.0136848
        integrand = func.circle
    elif FLAGS.function == 'Ring':
        func_ring = func.Ring(FLAGS.radius1, FLAGS.radius2)
        target = func_ring.area
        integrand = func_ring
    elif FLAGS.function == 'Triangle':
        target = -1.70721682537767509e-5
        integrand = func.TriangleIntegral([0, 0, 125],
                                          [175, 175, 175])
    elif FLAGS.function == 'Box':
        target = 1.93696402386819321e-10
        integrand = func.BoxIntegral(130**2, -130**2/2.0,
                                     [0, 0, 0, 125],
                                     [175, 175, 175, 175])


    epochs = FLAGS.epochs
    ptspepoch = FLAGS.ptspepoch
    target_precision = FLAGS.precision

    integrator = fm.Foam()
    integrator.SetDimension(ndims)

    # tell Foam to store points for re-use
    integrator.SetStorePoints(True)

    integrator.SetNCells(epochs)
    integrator.SetNOpt(2*ptspepoch)
    integrator.SetNMax(2*ptspepoch*epochs)
    integrator.SetError(target_precision)
    integrator.Initialize()
    integrator.Integrate(integrand)

    print(func.calls)

if __name__ == '__main__':
    app.run(main)
