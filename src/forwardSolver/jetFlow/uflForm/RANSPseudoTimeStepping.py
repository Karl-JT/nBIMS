from ufl import *
set_level(DEBUG)

V = VectorElement('CG', triangle, 2)
Q = FiniteElement('CG', triangle, 1)

Vh_STATE = MixedElement([V, Q, Q, Q])
Momentum_STATE = MixedElement([V, Q])
Rans_STATE = MixedElement([Q, Q])

C_mu =  0.09
sigma_k =  1.00
sigma_e =  1.30
C_e1 =  1.44
C_e2 =  1.92
nu = 1e-4
u_ff  = VectorConstant(triangle)
k_ff  = 0
e_ff  = 0

reg_norm = 1e-1

beta_chi_inflow = 1.0/sqrt( nu )
shift_chi_inflow = 0.0
Cd = 1e5

h = 2.0*Circumradius(triangle)
n = FacetNormal(triangle)
tg = perp(n)

def _k_plus(k):
    return 0.5*( k + sqrt(k*k + 1.0e-8) )

def _e_plus(e):
    return 0.5*( e + sqrt(e*e + 1.0e-8) )

def _chi_inflow(u):
	return 0.5 - 0.5*tanh(beta_chi_inflow* dot(u, n) - shift_chi_inflow)

def _u_norm(u):
	return sqrt( dot(u, u) + reg_norm*nu*nu)
    
def _theta(k, e):
    kp = _k_plus(k)
    return e/kp

def _strain(u):
    return sym( grad(u))

def nu_t(k, e, m):
    ep = _e_plus(e)
    return  exp(m)*C_mu*k*k/ep

def production(u):
    return 2.* inner(_strain(u), _strain(u))

def sigma_n(nu, u):
    return dot( 2.*nu*_strain(u), n )

def tau(nu, u):
    Pe =   0.5*h*sqrt(dot(u, u))/nu
                
    num =  1.0 +  exp( -2.0*Pe )
    den =  1.0 -  exp( -2.0*Pe )

    a1 =  0.333554921691650
    a2 =  -0.004435991517475
        
    tau_1 = (num/den -  1.0/Pe)*h/sqrt(dot(u, u))
    tau_2 = (a1 + a2*Pe)* .5*h*h/nu       
            
    return  conditional( ge(Pe, .1), tau_1, tau_2)

def all_tau(xl, m):
    ul, pl, kl, el = split(xl)
    tau_all = [ tau(nu+nu_t(kl, el, m), ul),
            h*h*_u_norm(ul),
            tau( nu+nu_t(kl, el, m)/sigma_k, ul),
           	tau( nu+nu_t(kl, el, m)/sigma_e, ul)
           ]
    return tau_all


def stab(x, xt, m, xl):
    r_s = strong_residual(x, m, xl)
    
    r_s_prime =[None, None, None, None]
    
    for i in range(4):
        r_s_prime[i] =  derivative(r_s[i], x, xt)
    
    tau = all_tau(xl, m)
    
    res_stab = ( tau[0]* inner(r_s[0], r_s_prime[0]) + \
                 tau[1]* inner(r_s[1], r_s_prime[1]) + \
                 tau[2]* inner(r_s[2], r_s_prime[2]) + \
                 tau[3]* inner(r_s[3], r_s_prime[3]) )* dx            
    return res_stab
  

def momentum_stab(momentum, momentum_test, m, xl):
    r_s = strong_momentum_residual(momentum, m, xl)
    
    r_s_prime =[None, None]
    
    for i in range(2):
        r_s_prime[i] =  derivative(r_s[i], momentum, momentum_test)
    
    tau = all_tau(xl, m)
    
    res_stab = ( tau[0]* inner(r_s[0], r_s_prime[0]) + \
                 tau[1]* inner(r_s[1], r_s_prime[1]) )* dx            
    return res_stab

def rans_stab(rans, rans_test, m, xl):
    r_s = strong_rans_residual(rans, m, xl)
    
    r_s_prime =[None, None]
    
    for i in range(2):
        r_s_prime[i] =  derivative(r_s[i], rans, rans_test)
    
    tau = all_tau(xl, m)
    
    res_stab = ( tau[2]* inner(r_s[0], r_s_prime[0]) + \
                 tau[3]* inner(r_s[1], r_s_prime[1]) )* dx            
    return res_stab


def weak_residual(x, xt, sigma, m, xl):
    u, p, k, e     = split(x)
    ul, pl, kl, el = split(xl)
    u_test, p_test, k_test, e_test = split(xt)

    hbinv = 1.0/h

    res_u =  2.*(nu+nu_t(kl, el, m))* inner( _strain(u), _strain(u_test))* dx \
            + inner( grad(u)*u, u_test)* dx \
            - p* div(u_test)* dx \
            #+ Cd*(nu+nu_t(kl, el, m))*hbinv* dot(u-u_ff, tg)* dot(u_test, tg)*ds(3) \
            #- dot( sigma_n(nu+nu_t(kl, el, m), u), tg) *  dot(u_test, tg)*ds(3) \
            #- dot( sigma_n(nu+nu_t(kl, el, m), u_test), tg ) *  dot(u - u_ff, tg)*ds(3) \
            #- _chi_inflow(ul)*0.5*inner( dot(u,u)*n + dot(n,u)*u, u_test )*ds(3)


    res_p =   div(u)*p_test* dx

    res_k =  dot(ul,  grad(k))*k_test* dx \
            + (nu + nu_t(k,e,m)/sigma_k)* inner(  grad(k),  grad(k_test))* dx \
            + e*k_test* dx \
            - nu_t(k,e,m)*production(ul)*k_test* dx \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_k)* dot( grad(k), n)*k_test*ds(3) \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_k)* dot( grad(k_test), n)*(k-k_ff)*ds(3) \
            #+ _chi_inflow(ul)*Cd*(nu + nu_t(k,e,m)/sigma_k)*hbinv*(k-k_ff)*k_test*ds(3)
            
    res_e =  dot(ul,  grad(e))*e_test* dx \
            + (nu + nu_t(k,e,m)/sigma_e)* inner(  grad(e),  grad(e_test))* dx \
            + C_e2*e*_theta(k,e)*e_test* dx \
            - C_e1* exp(m)*C_mu*k*production(ul)*e_test* dx \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_e)* dot( grad(e), n)*k_test*ds(3) \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_e)* dot( grad(e_test), n)*(e-e_ff)*ds(3) \
            #+ _chi_inflow(ul)*Cd*(nu + nu_t(k,e,m)/sigma_e)*hbinv*(e-e_ff)*e_test*ds(3)

    time_derivative = sigma*(inner( u - ul, u_test) +  inner(k - kl, k_test) +  inner(e - el, e_test) )* dx

    return time_derivative + res_u + res_p + res_k + res_e

def momentum_residual(momentum, momentum_test, sigma, m, xl):
    u, p     = split(momentum)
    ul, pl, kl, el = split(xl)
    u_test, p_test = split(momentum_test)

    hbinv = 1.0/h

    res_u =  2.*(nu+nu_t(kl, el, m))* inner( _strain(u), _strain(u_test))* dx \
            + inner( grad(u)*u, u_test)* dx \
            - p* div(u_test)* dx \
            #+ Cd*(nu+nu_t(kl, el, m))*hbinv* dot(u-u_ff, tg)* dot(u_test, tg)*ds(3) \
            #- dot( sigma_n(nu+nu_t(kl, el, m), u), tg) *  dot(u_test, tg)*ds(3) \
            #- dot( sigma_n(nu+nu_t(kl, el, m), u_test), tg ) *  dot(u - u_ff, tg)*ds(3) \
            #- _chi_inflow(ul)*0.5*inner( dot(u,u)*n + dot(n,u)*u, u_test )*ds(3)


    res_p =   div(u)*p_test* dx

    time_derivative = sigma*(inner( u - ul, u_test))* dx

    return time_derivative + res_u + res_p


def rans_residual(rans, rans_test, sigma, m, xl):
    k, e     = split(rans)
    ul, pl, kl, el = split(xl)
    k_test, e_test = split(rans_test)

    hbinv = 1.0/h

    res_k =  dot(ul,  grad(k))*k_test* dx \
            + (nu + nu_t(k,e,m)/sigma_k)* inner(  grad(k),  grad(k_test))* dx \
            + e*k_test* dx \
            - nu_t(k,e,m)*production(ul)*k_test* dx \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_k)* dot( grad(k), n)*k_test*ds(3) \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_k)* dot( grad(k_test), n)*(k-k_ff)*ds(3) \
            #+ _chi_inflow(ul)*Cd*(nu + nu_t(k,e,m)/sigma_k)*hbinv*(k-k_ff)*k_test*ds(3)
            
    res_e =  dot(ul,  grad(e))*e_test* dx \
            + (nu + nu_t(k,e,m)/sigma_e)* inner(  grad(e),  grad(e_test))* dx \
            + C_e2*e*_theta(k,e)*e_test* dx \
            - C_e1* exp(m)*C_mu*k*production(ul)*e_test* dx \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_e)* dot( grad(e), n)*k_test*ds(3) \
            #- _chi_inflow(ul)*(nu + nu_t(k,e,m)/sigma_e)* dot( grad(e_test), n)*(e-e_ff)*ds(3) \
            #+ _chi_inflow(ul)*Cd*(nu + nu_t(k,e,m)/sigma_e)*hbinv*(e-e_ff)*e_test*ds(3)

    time_derivative = sigma*(inner(k - kl, k_test) +  inner(e - el, e_test) )* dx

    return time_derivative + res_k + res_e


def strong_residual(x, m, xl):
       
    u, p, k, e   = split(x)
    ul, pl, kl, el = split(xl)

    res_u = - div(  2.*(nu+nu_t(kl, el, m))*_strain(u) ) +  grad(u)*ul +  grad(p)
    res_p =  div( u )
            
    res_k = +  dot(ul,  grad(k)) \
            -  div( (nu + nu_t(kl,el,m)/sigma_k)*  grad(k) ) \
            + el \
            - nu_t(kl,el,m)*production(ul)
            
    res_e = +  dot(ul,  grad(e)) \
            -  div( (nu + nu_t(kl,el,m)/sigma_e)*  grad(e) ) \
            + C_e2*e*_theta(kl,e) \
            - C_e1* exp(m)*C_mu*kl*production(ul)

    return [res_u, res_p, res_k, res_e]



def strong_momentum_residual(momentum, m, xl):
       
    u, p  = split(momentum)
    ul, pl, kl, el = split(xl)

    res_u = - div(  2.*(nu+nu_t(kl, el, m))*_strain(u) ) +  grad(u)*ul +  grad(p)
    res_p =  div( u )

    return [res_u, res_p]


def strong_rans_residual(rans, m, xl):
       
    k, e  = split(rans)
    ul, pl, kl, el = split(xl)

    res_k = +  dot(ul,  grad(k)) \
            -  div( (nu + nu_t(kl,el,m)/sigma_k)*  grad(k) ) \
            + el \
            - nu_t(kl,el,m)*production(ul)
            
    res_e = +  dot(ul,  grad(e)) \
            -  div( (nu + nu_t(kl,el,m)/sigma_e)*  grad(e) ) \
            + C_e2*e*_theta(kl,e) \
            - C_e1* exp(m)*C_mu*kl*production(ul)

    return [res_k, res_e]

def totalResidual(x, xt, sigma, m, xl):
    return weak_residual(x, xt, sigma, m, xl) + stab(x, xt, m, xl)

def totalMomentumResidual(momentum, momentum_test, sigma, m, xl):
    return momentum_residual(momentum, momentum_test, sigma, m, xl) + momentum_stab(momentum, momentum_test, m, xl)

def totalRansResidual(rans, rans_test, sigma, m, xl):
    return rans_residual(rans, rans_test, sigma, m, xl) + rans_stab(rans, rans_test, m, xl)

sigma = Constant(triangle)

x  = Coefficient(Vh_STATE)
xl = Coefficient(Vh_STATE)
xt = TestFunction(Vh_STATE)
adjoints = Coefficient(Vh_STATE) 

momentum = Coefficient(Momentum_STATE)
momentum_test = Coefficient(Momentum_STATE)
rans = Coefficient(Rans_STATE)
rans_test = Coefficient(Rans_STATE)

m = Coefficient(Q)

F = totalResidual(x, xt, sigma, m, xl)
F_momentum = totalMomentumResidual(momentum, momentum_test, sigma, m, xl)
F_rans = totalRansResidual(rans, rans_test, sigma, m, xl)

dstate = TrialFunction(Vh_STATE)
dup = TrialFunction(Momentum_STATE)
dke = TrialFunction(Rans_STATE)
J = derivative(F, x, dstate)
J_momentum = derivative(F_momentum, momentum, dup)
J_rans = derivative(F_rans, rans, dke)
J_true = derivative(F, x, dstate) + derivative(F, xl, dstate);

Fm   = totalResidual(x, adjoints, sigma, m, xl)
dm   = TestFunction(Q)
Jm   = derivative(Fm, m, dm)
duu  = TestFunction(Vh_STATE)
Jx   = derivative(Fm, x, duu) + derivative(Fm, xl, duu)
Jadj = derivative(Fm, adjoints, duu) 

dmm = TrialFunction(Q)
Rform = derivative(Jm, m, dmm)
Aform = derivative(Jadj, x, dstate) + derivative(Jadj, xl, dstate)
Cform = derivative(Jadj, m, dmm)
WuuForm = derivative(Jx, x, dstate) + derivative(Jx, xl, dstate)

Coef = Constant(triangle)

elements = [Vh_STATE, V, Q]
forms = [F, F_momentum, F_rans, J, J_momentum, J_rans, J_true, Jm, Jx, Jadj, Rform, Aform, Cform, WuuForm]
