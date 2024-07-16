class Saltzmann_Maasch(object):
    '''
    The nondimensional parameters p, q, r, and s are each combinations of various physical parameters, 
    and they are all positive. Here, p and r represent the effective rate constants for how the
    CO2 concentration (y) changes as the NADW (z) and CO2 concentration change, respectively. 
    Next, q is the effective ratio of the characteristic time scales for the total global ice mass (x) and the
    volume of NADW; for physical reasons, q > 1. Then, the parameter s is a symmetry parameter. 
    With s = 0, the model possesses a reflection symmetry; if (x, y, z) is a solution, then so
    is (-x, -y, -z). In this special case, glaciation and deglaciation occur at the same rates. 
    Physically, however, it is observed that deglaciation occurs at a faster rate than glaciation, and s > 0
    guarantees this asymmetry. All of these nondimensional parameters incorporate several dimensional 
    rate constants as well as dimensional parameters and quantities related to the global mean
    sea surface temperature and the mean volume of permanent sea ice. Also, the three variables 
    are properly correlated: as the concentration of the atmospheric CO2 (a greenhouse gas) increases, 
    the climate gets warmer, and the total ice mass decreases (deglaciation); as the
    volume of NADW increases, the strength of the North Atlantic over-turning circulation increases, 
    more atmospheric CO2 is absorbed by the ocean and, consequently, the atmospheric CO2 concentration decreases.
    '''

    # class constructor
    def __init__(   self, r:float, 
                    v:float=0.2, 
                    p:float=1., 
                    s:float=0.6, 
                    q:float=2.5, 
                    eps:float=0.,
                    SDE_type:str='Ito')->None:
        self.v = v
        self.p = p
        self.r = r
        self.s = s
        self.q = q
        self.eps = eps
        self.SDE_type = SDE_type
        # setup random generator
        self.rng = Generator(PCG64())
        # check if the inserted parameters are valid:
        if(r<0 or v<0 or p<0 or s<0 or q<1):
            raise RuntimeError("Invalid parameters: it must be r>0, v>0, p>0, s>0, q>1")
        return


    def __checkInputs(self, T:float=10., N:int=10)->None:
        '''
        Given the inputs for a trajectory simulation, this method will
        check if they are correct
        '''
        # T check:
        if T<0:
            raise RuntimeError("Time interval must have a positive lenght")
        # N check:
        if N<=1:
            raise RuntimeError("The simulation must have at least two steps")
        return
    

    def __ODEs(self, r:tuple[float], t:float=None):
        X, Y, Z = r

        if (self.SDE_type == 'Ito'):
            return np.array([-X -Y -self.v*Z,                           # ... dx/dt
                            -self.p*Z +self.r*Y -self.s*(Y**2) -Y**3,   # ... dy/dt
                            -self.q*(X + Z)])                           # ... dz/dt
        
        elif (self.SDE_type == 'Stratonovich'):
            return np.array([-X -Y - self.v*Z,                          # ... dx/dt
                            -self.p*Z + self.r*Y -self.s*(Y**2) -Y**3,  # ... dy/dt
                            -self.q*(X + Z) + (Z*(self.eps**2))/2])     # ... dz/dt


    def __RK4(self, r_n:tuple[float], t_n:float, dt:float)->float: 
        '''
        Given a point in the trajectory, the time instant and the step lenght,
        this method will compute the variation for y using the RK4 for the 
        deterministic part of the PLS
        '''
        # Perform the check of the inputs. For N we hard code a good 
        self.__checkInputs(T = t_n)
        if dt<=0:
            raise RuntimeError("Given dt is negative or 0")

        k_1 = self.__ODEs(r_n)
        k_2 = self.__ODEs(r_n + dt*k_1/2)
        k_3 = self.__ODEs(r_n + dt*k_2/2)
        k_4 = self.__ODEs(r_n + dt*k_3)
        return dt*(k_1 + 2*k_2 + 2*k_3 + k_4)/6
        
    
    def simulateTraj(self, R0:tuple[float], T:float, N:int):
        '''
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        self.__checkInputs(T,N)
        if T==0:
            return np.array(R0)

        dt = T/N
        R = np.zeros((3,N+1), dtype=float)
        R[:,0] = R0
        time = np.zeros(N+1, dtype=float)
        time[0] = 0.0

        for i in range(1,N+1):
            time[i] = i*dt
            # deterministic part through RK method
            dR =  self.__RK4(R[:, i-1], time[i-1], dt)
            # white noise term
            dR[2] += self.eps*R[2, i-1]*self.rng.normal()*np.sqrt(dt)
            R[:, i] = R[:, i-1] + dR

        return R, time
        
        
        
 class Saltzmann_Maasch_advanced(object):
    '''
    The nondimensional parameters p, q, r, and s are each combinations of various physical parameters, 
    and they are all positive. Here, p and r represent the effective rate constants for how the
    CO2 concentration (y) changes as the NADW (z) and CO2 concentration change, respectively. 
    Next, q is the effective ratio of the characteristic time scales for the total global ice mass (x) and the
    volume of NADW; for physical reasons, q > 1. Then, the parameter s is a symmetry parameter. 
    With s = 0, the model possesses a reflection symmetry; if (x, y, z) is a solution, then so
    is (-x, -y, -z). In this special case, glaciation and deglaciation occur at the same rates. 
    Physically, however, it is observed that deglaciation occurs at a faster rate than glaciation, and s > 0
    guarantees this asymmetry. All of these nondimensional parameters incorporate several dimensional 
    rate constants as well as dimensional parameters and quantities related to the global mean
    sea surface temperature and the mean volume of permanent sea ice. Also, the three variables 
    are properly correlated: as the concentration of the atmospheric CO2 (a greenhouse gas) increases, 
    the climate gets warmer, and the total ice mass decreases (deglaciation); as the
    volume of NADW increases, the strength of the North Atlantic over-turning circulation increases, 
    more atmospheric CO2 is absorbed by the ocean and, consequently, the atmospheric CO2 concentration decreases.
    '''

    # class constructor
    def __init__(self,  r:float, 
                        a:float=1., 
                        v:float=0.2, 
                        p:float=1., 
                        s:float=0.6, 
                        b:float=0.1,
                        c:float=0.1,
                        q:float=2.5, 
                        eps:float=0.,
                        d:float = 0.,
                        noise_type = 'W',
                        omega_x:float=0.,
                        omega_y:float=0.,
                        SDE_type:str = 'Ito')->None:

        self.r = r
        self.a = a
        self.v = v
        self.p = p
        self.s = s
        self.b = b
        self.c = c
        self.q = q
        self.eps = eps
        self.d = d
        self.noise_type = noise_type
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.SDE_type = SDE_type
        
        # setup random generator
        self.rng = Generator(PCG64())
        # check if the inserted parameters are valid:
        if(r<0 or v<0 or p<0 or s<0 or q<1):
            raise RuntimeError("Invalid parameters: it must be r>0, v>0, p>0, s>0, q>1")
        return


    def __checkInputs(self, T:float=10., N:int=10)->None:
        '''
        Given the inputs for a trajectory simulation, this method will
        check if they are correct
        '''
        # T check:
        if T<0:
            raise RuntimeError("Time interval must have a positive lenght")
        # N check:
        if N<=1:
            raise RuntimeError("The simulation must have at least two steps")
        return
    

    def __ODEs(self, r:tuple[float], t:float=None):
        X, Y, Z = r

        if (self.SDE_type == 'Ito'):
            return np.array([   -self.a*(X + Y) -self.v*Z,                                                  # ... dx/dt
                                -self.p*Z + self.r*Y -self.s*(Y**2) -Y**3 +self.b*Z*Y +self.c*Z*(Y**2),     # ... dy/dt
                                -self.q*(X + Z)])                                                           # ... dz/dt
        
        #elif (self.SDE_type == 'Stratonovich'):
        #    return np.array([   -self.a*(X + Y) -self.v*Z,                                                  # ... dx/dt
        #                        -self.p*Z + self.r*Y -self.s*(Y**2) -Y**3 +self.b*Z*Y +self.c*Z*(Y**2),     # ... dy/dt
        #                        -self.q*(X + Z) + (Z*(self.eps**2))/2])                                     # ... dz/dt

    
    def deterministic_equilibria(self):
        # Calculate the common expressions to avoid redundant computations
        term1 = self.b * self.a / (self.a - self.v)
        term2 = self.c * self.a / (self.a - self.v) - 1
        term3 = self.r - self.p * self.a / (self.a - self.v)
        discriminant = (term1-self.s)**2 - 4 * term2 * term3
        denominator = 2 * term2

        if discriminant < 0:
            raise ValueError("The discriminant is negative, so there are no real solutions.")

        sqrt_discriminant = np.sqrt(discriminant)
        # Calculate the two possible values of Y
        Y1 = (self.s - term1 + sqrt_discriminant) / denominator
        Y2 = (self.s - term1 - sqrt_discriminant) / denominator
        E1 = (self.a*Y1/(self.v-self.a), Y1) 
        E2 = (self.a*Y2/(self.v-self.a), Y2)
        return E1, E2


    def __RK4(self, r_n:tuple[float], t_n:float, dt:float)->float: 
        '''
        Given a point in the trajectory, the time instant and the step lenght,
        this method will compute the variation for y using the RK4 for the 
        deterministic part of the PLS
        '''
        # Perform the check of the inputs. For N we hard code a good 
        self.__checkInputs(T = t_n)
        if dt<=0:
            raise RuntimeError("Given dt is negative or 0")

        k_1 = self.__ODEs(r_n)
        k_2 = self.__ODEs(r_n + dt*k_1/2)
        k_3 = self.__ODEs(r_n + dt*k_2/2)
        k_4 = self.__ODEs(r_n + dt*k_3)
        return dt*(k_1 + 2*k_2 + 2*k_3 + k_4)/6
        
    
    def simulateTraj(self, R0:tuple[float], T:float, N:int):
        '''
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        self.__checkInputs(T,N)
        if T==0:
            return np.array(R0)

        dt = T/N
        R = np.zeros((3,N+1), dtype=float)
        R[:,0] = R0
        time = np.zeros(N+1, dtype=float)
        time[0] = 0.0
        eta = np.zeros(N+1, dtype=float)

        for i in range(1,N+1):
            time[i] = i*dt
            # deterministic part through RK method
            dR =  self.__RK4(R[:, i-1], time[i-1], dt)
            # additive white noise terms in the first and second equations
            dR[0] += self.omega_x*self.rng.normal()*np.sqrt(dt)
            dR[1] += self.omega_y*self.rng.normal()*np.sqrt(dt)
            # multiplicative noise term in the first equation:
            # gaussian white noise
            #dR[2] += self.eps*R[2, i-1]*(1-self.d*R[0, i-1])*self.rng.normal()*np.sqrt(dt)
            # Ornstein-Uhlenbeck stochastic process
            eta[i] = eta[i-1] - 1.0*eta[i-1]*dt + 1.0*self.rng.normal()*np.sqrt(dt)
            dR[2] += self.eps*R[2, i-1]*(1-self.d*R[0, i-1])*eta[i-1]
            R[:, i] = R[:, i-1] + dR

        return R, time
