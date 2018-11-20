





# -*- coding:utf-8 -*-
#
# Convection 2D schema explicite
#      avec points fantomes
#
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg


###### affichage graphique
import matplotlib.pyplot as plt
plt.ion()
if 'qt' in plt.get_backend().lower():
    try:
        from PyQt4 import QtGui
    except ImportError:
        from PySide import QtGui

        
        
        
        
         
                  
def CFL_advection():
    """
    Condition CFL Advection pour dt
    return le nouveau 'dt' tel que

    abs(V) * dt < dx

    dt_new = 0.8*min(dx/umax,dy/vmax)

    """
    precautionADV = 0.8
    dt_cfa = ...

    return dt_cfa

def CFL_explicite():
    """
    Condition CFL pour la diffusion 
    en cas de schema explicite
  
    """
    precautionEXP = 0.3
    
    dt_cfl = ...
    return dt_cfl


####
def Advect():
    """
    Calcule la valeur interpolee qui correspond 
    a l'advection a la vitesse au temps n
    
    travaille bien sur le domaine reel [1:-1,1:-1]

    """ 
    global Resu, Resv, Tadv 

   

    # Calcul des matrices de resultat 
    # pour les vitesses u et v
    Resu[1:-1,1:-1] = ...
    
    Resv[1:-1,1:-1] = ...
    # advection temperature
    Tadv[1:-1,1:-1] = ...



def BuildLaPoisson():
    """
    pour l'etape de projection
    matrice de Laplacien phi
    avec CL Neumann pour phi

    BUT condition de Neumann pour phi 
    ==> non unicite de la solution

    besoin de fixer la pression en un point 
    pour lever la degenerescence: ici [0][1]
    
    ==> need to build a correction matrix

    """
    ### ne pas prendre en compte les points fantome (-2)
    NXi = nx
    NYi = ny

    ###### Definition of the 1D Lalace operator

    ###### AXE X
    ### Diagonal terms
    dataNXi = [np.ones(NXi), -2*np.ones(NXi), np.ones(NXi)]   
    
    ### Conditions aux limites : Neumann 
    ...  # SF left
    ...  # SF right

    ###### AXE Y
    ### Diagonal terms
    dataNYi = [np.ones(NYi), -2*np.ones(NYi), np.ones(NYi)] 
   
    ### Conditions aux limites : Neumann 
    ...  # SF low
    ...  # SF top

    ###### Their positions
    offsets = np.array([-1,0,1])                    
    DXX = sp.dia_matrix((dataNXi,offsets), shape=(NXi,NXi)) * dx_2
    DYY = sp.dia_matrix((dataNYi,offsets), shape=(NYi,NYi)) * dy_2
    #print DXX.todense()
    #print DYY.todense()
    
    ####### 2D Laplace operator
    LAP = sp.kron(sp.eye(NYi,NYi), DXX) + sp.kron(DYY, sp.eye(NXi,NXi))
    
    ####### BUILD CORRECTION MATRIX

    ### Upper Diagonal terms
    dataNYNXi = [np.zeros(NYi*NXi)]
    offset = np.array([1])

    ### Fix coef: 2+(-1) = 1 ==> Dirichlet en un point (redonne Laplacien)
    ### ATTENTION  COEF MULTIPLICATIF : dx_2 si M(j,i) j-NY i-NX
    dataNYNXi[0][1] = -1 * dx_2

    LAP0 = sp.dia_matrix((dataNYNXi,offset), shape=(NYi*NXi,NYi*NXi))
    
    # tmp = LAP + LAP0
    # print LAP.todense()
    # print LAP0.todense()
    # print tmp.todense()
  
    return LAP + LAP0

def ILUdecomposition(LAP):
    """
    return the Incomplete LU decomposition 
    of a sparse matrix LAP
    """
    return  lg.splu(LAP.tocsc(),)


def ResoLap(splu,RHS):
    """
    solve the system

    SPLU * x = RHS

    Args:
    --RHS: 2D array((NY,NX))
    --splu: (Incomplete) LU decomposed matrix 
            shape (NY*NX, NY*NX)

    Return: x = array[NY,NX]
    
    Rem1: taille matrice fonction des CL 

    """
    # array 2D -> array 1D
    f2 = RHS.ravel()

    # Solving the linear system
    x = splu.solve(f2)

    return x.reshape(RHS.shape)

####
def Laplacien(x):
    """
    calcule le laplacien scalaire 
    du champ scalaire x(i,j)
    
    pas de termes de bord car ghost points

    """
    rst = np.empty((NY,NX))
    
    rst[1:-1,1:-1] = ... 
    return rst

def divergence(u,v):
    """
    divergence avec points fantomes
    ne jamais utiliser les valeurs au bord

    """
    tmp = np.empty((NY,NX))
    
    tmp[1:-1,1:-1] = ...
        
    return tmp

def grad():
    """
    Calcule le gradient de phi (ordre 2)
    update gradphix and gradphiy
    
    """
    global gradphix, gradphiy

    gradphix[:, 1:-1] = ...
    gradphiy[1:-1, :] = ...

       
###
def VelocityGhostPoints(u,v):
    # NO SLIP BC
    ### left
    ...
    ### right      
    ... 
    ### bottom     
    ...
    ### top      
    ... 


def TemperatureGhostPoints(T):
    """
    global ==> pas de return 

    """
    ### left               
    ...
    ### right              
    ...
    ### bottom
    ...
    ### top                
    ...
        
def PhiGhostPoints(phi):
    """
    copie les points fantomes
    tjrs Neumann

    global ==> pas de return 

    """
    ### left               
    ...
    ### right             
    ...
    ### bottom   
    ...
    ### top               
    ...



#########################################
###### MAIN: Programme principal
#########################################


###### Taille adimensionnee du domaine
### aspect_ratio = LY/LX  

aspect_ratio = float(1.)
LY = float(1.)
LX = LY/aspect_ratio

###### GRID RESOLUTION

### Taille des tableaux (points fantomes inclus)

NX = int(100)
NY = int(100)

### Taille du domaine reel
nx = NX-2
ny = NY-2

###### Control parameters
Pr = float(1)
Ra = float(5e7)


###### LOOPING PARAMETERS
### Nombre d'iterations
nitermax = int(10000)

### Modulo
modulo = int(50)

###### CONDITIONS INITIALES

### Valeur initiale de la temperature T
T = np.zeros((NY,NX))

### Perturbation initiale
Tin  = 0.1

Xposi = NX/4
Yposi = NY/4

T[Yposi, Xposi-1] =  Tin
T[Yposi, Xposi+1] = -Tin

##### Valeurs initiales des vitesses
u = np.zeros((NY,NX)) 
v = np.zeros((NY,NX))

####################
###### COEF FOR ADIM

### Coef du Laplacien de la vitesse 
DeltaU = float(Pr)

### Coef du Laplacien T
DeltaT = float(1.)

### Coef Archimede
Buoyancy = float(Ra*Pr)

ForcingT = float(1.)

###### Elements differentiels 

dx = LX/(nx-1)
dy = LY/(ny-1)

dx_2 = 1./(dx*dx)
dy_2 = 1./(dy*dy)


### ATTENTION: dt_init calculer la CFL a chaque iteration... 
dt = float(1)

t = 0. # total time


### Tableaux avec points fantomes
### Matrices dans lesquelles se trouvent les extrapolations
Resu = np.zeros((NY,NX))
Resv = np.zeros((NY,NX))
Tadv = np.zeros((NY,NX))

### Definition des matrices ustar et vstar
ustar = np.zeros((NY,NX))
vstar = np.zeros((NY,NX))

### Definition de divstar
divstar = np.zeros((NY,NX))

### Definition de la pression phi
phi      = np.zeros((NY,NX))
gradphix = np.zeros((NY,NX))
gradphiy = np.zeros((NY,NX))


###### CONSTRUCTION des matrices et LU decomposition

### Matrix construction for projection step
LAPoisson = BuildLaPoisson() 
LUPoisson = ILUdecomposition(LAPoisson)


### Maillage pour affichage (inutile)
# ne pas compter les points fantomes
x = np.linspace(0,LX,nx) 
y = np.linspace(0,LY,ny)
[xx,yy] = np.meshgrid(x,y) 


###### CFL explicite
dt_exp = CFL_explicite()


###### Reference state
Tr = 1 - y

################
###### MAIN LOOP 
tStart = t

for niter in xrange(nitermax):
    ###### Check dt
    dt_adv = CFL_advection()
    
    dt_new = min(dt_adv,dt_exp)
    
    if (dt_new < dt):
        dt = dt_new
        
    ### Avancement du temps total
    t += dt

    ###### Etape d'advection semi-Lagrangienne
    Advect()

    ###### Etape de diffusion

    ustar = ...
    vstar = ...
    T     = ...

    ###### Conditions aux limites Vitesse
    ###### on impose sur ustar/vstar Att:ghost points
    ### left
    ... 
    ### right
    ...
    ### top
    ...
    ### bottom
    ... 
        
    ###### Temperature B.C    
     
    ### bottom
    ...
    ### top
    ...

    ###### END Conditions aux limites

    ###### Etape de projection
    
    ###### Mise a jour des points fantomes pour 
    ###### calculer la divergence(ustar,vstar) 
   
    VelocityGhostPoints(ustar,vstar)

    ### Update divstar 
    divstar = divergence(ustar,vstar)


    ### Solving the linear system
    phi[1:-1,1:-1] = ResoLap(LUPoisson, RHS=divstar[1:-1,1:-1])

    ### update Pressure ghost points 

    PhiGhostPoints(phi)

    ### Update gradphi

    grad()

    u = ...
    v = ...

    ###### Mise a jour des points fantomes
    ###### pour le champ de vitesse et T

    VelocityGhostPoints(u,v)

    TemperatureGhostPoints(T)

    if (niter%modulo==0):

        ###### logfile
        sys.stdout.write(
            '\niteration: %d -- %i %%\n'
            '\n'
            'total time     = %.2e\n'
            '\n'
            %(niter,                    
              float(niter)/nitermax*100,
              t))
        
        
        ###### FIGURE draw works only if plt.ion()
        plotlabel = "t = %1.5f" %(t)
        plt.title(plotlabel)
#        plt.pcolormesh(xx,yy,T[1:-1,1:-1]+Tr[:,np.newaxis])
        plt.imshow(T[1:-1,1:-1]+Tr[:,np.newaxis],origin='lower')
        plt.axis('image')
        plt.show()
        

        ###### Gael's tricks interactif
        if 'qt' in plt.get_backend().lower():
            QtGui.qApp.processEvents()
        
