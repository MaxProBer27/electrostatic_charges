import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import sympy as sp

def RK4(f,x0,h):
    """Calcula la aproximación a la ecuación
    diferencial por el método de Runge-Kutta
    de orden 4.

    Args:
        f (sympy function): Función de tipo sympy
        x0 (vector): Condición inicial del vector X
        h (float): Incremento en el tiempo

    Returns:
        vector: Vector solución al paso n+1
    """
    k1 = f(*x0)
    k2 = f(*(np.array(x0) + (h/2)*np.array(k1)))
    k3 = f(*(np.array(x0) + (h/2)*np.array(k2)))
    k4 = f(*(np.array(x0) + h*np.array(k3)))
    return list(np.array(x0) + (h/6)*(np.array(k1) +
                                 2*np.array(k2) +
                                 2*np.array(k3) +
                                 np.array(k4)))

class charges_animation():
    def __init__(self, magnetic_field = True, dt = 0.01, stream = False,
                 density = 1, full_screen = True):
        """Creacion del objeto charges_animation

        Args:
            dt (float, optional): Incremento de tiempo para la ODE.
            Defaults to 0.01.
        """        
        self.fig, self.ax = plt.subplots(figsize = (8,8)) # fig is the entire canvas and ax us just the plot
        self.dt = dt
        self.density = density
        self.draw_stream = stream
        self.mf = magnetic_field
        self.full_screen = full_screen

    def setup_fig_ax(self, lim = 300, color = 'black'):
        """Define las propiedades del fondo.

        Args:
            lim (int, optional): Límites del marco para 'x' y 'y'. Defaults to 300.
            color (str, optional): Color del fondo. Defaults to 'black'.
        """
        self.color = color
        self.lim = lim
        self.fig.set_facecolor(color) 
        self.ax.set_facecolor(color)
        if color == 'black':
            for spine in self.ax.spines.values():
                spine.set_edgecolor('white')
            self.ax.tick_params(color = 'white', labelcolor = 'white')
        else:
            for spine in self.ax.spines.values():
                spine.set_edgecolor('black')
            self.ax.tick_params(color = 'black', labelcolor = 'black')
        if self.full_screen:
            self.ax.set_xlim(-1920*lim/1080,1920*lim/1080)
        else:
            self.ax.set_xlim(-lim,lim)
        self.ax.set_ylim(-lim,lim)

    def set_static_charges(self, n_charges:int, F_charges = [],
                           alternating = False):
        """Coloca las cargas puntuales en el objeto axes

        Args:
            n_charges (int): numero de cargas
            F_charges (list): lista que contiene las cargas
        """        
        self.n = n_charges
        if alternating:
            self.F_static = [self.F_moving*(-1)**i for i in range(self.n)]
        else:    
            self.F_static = [-self.F_moving]*self.n if not F_charges else F_charges
        if self.n == 1:
            X_circ = [0]
            Y_circ = [0]    
        else:
            X_circ = [np.cos(i*2*np.pi/self.n)*120 for i in range(self.n)]
            Y_circ = [np.sin(i*2*np.pi/self.n)*120 for i in range(self.n)]
        self.coords = [[X_circ[i], Y_circ[i]] for i in range(self.n)]
        colors = cm.hsv(np.linspace(0,1,self.n + 1))
        for i in range(self.n):
            self.ax.plot(X_circ[i], Y_circ[i], color = colors[i],
                         markersize = 10, marker = 'o',
                         markeredgecolor='black')

    def set_moving_charge(self, x = 0, y = 0, vx = 0, vy = 0, F = 200):
        """Coloca la carga movil en el objeto axes

        Args:
            x (int): Coordenada x inicial de la carga
            y (int): Coordenada y inicial de la carga
            vx (int): Coordenada x inicial de la vel de carga
            vy (int): Coordenada y inicial de la vel de carga
            F (int): Carga de la particula
        """     
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.F_moving = F
        if self.color == 'black':
            self.mov_charge, = self.ax.plot([x],[y], 'wo', markersize = 10)
        else:
            self.mov_charge, = self.ax.plot([x],[y], 'ko', markersize = 10)
        self.new_x = [x] # Lists that will contain all values that self.X have taken, in order to draw a line for the trajectory
        self.new_y = [y]
        if self.color == 'black':
            self.traj_charge, = self.ax.plot([x],[y], color = 'white',
                                             linewidth = 1, alpha = 0.5)
        else:
            self.traj_charge, = self.ax.plot([x],[y], color = 'black',
                                             linewidth = 1, alpha = 0.5)
    
    def set_field(self, c = 0, alpha = 0.5):
        """Define el campo que afecta la carga movil.

        Args:
            c (int, optional): Fuerza del campo magnetico en caso de haber.
            Defaults to 0.
        """        
        x,y,vx,vy= sp.symbols('x,y,vx,vy')
        epsilon = 0.1
        q0 = self.F_moving
        F_cargas_ind = []
        for i in range(self.n):
            qi = self.F_static[i]
            ri = [self.coords[i][0], self.coords[i][1]]
            dist = sp.sqrt((x - ri[0])**2 + (y - ri[1])**2)
            Fi = (q0*qi/(epsilon*dist**3))*np.array([(x - ri[0]), (y - ri[1])])
            F_cargas_ind.append(Fi)
        E = sum((F_cargas_ind))
        B = c*np.array([vy,-vx]) if self.mf else np.array([0,0])
        F = sp.lambdify([x,y,vx,vy],[vx,vy] + list(E + B)) # Convert to list because sympy doesn't support numpy arrays
        self.field = F
        if self.draw_stream:
            X, Y = np.meshgrid(np.linspace(-1920*self.lim/1080,1920*self.lim/1080, 20),
                            np.linspace(-self.lim, self.lim, 20))
            V = np.zeros(X.shape)
            AX, AY = F(X,Y,V,V)[2], F(X,Y,V,V)[3]
            self.ax.streamplot(X,Y,AX,AY, color = str(alpha),
                            linewidth = 0.5, density = self.density,
                            arrowsize = 1)

    def update(self, i):
        for carga_x, carga_y in self.coords:
            dist = np.sqrt((self.x - carga_x)**2 + (self.y - carga_y)**2)
            if dist < 4:
                plt.pause(1)
                self.x = np.random.random()*(2*self.lim) - self.lim
                self.y = np.random.random()*(2*self.lim) - self.lim
                self.vx, self.vy = 0,0
                self.new_x = [self.x]
                self.new_y = [self.y]

        self.x, self.y, self.vx, self.vy = RK4(self.field,[self.x,self.y,self.vx,self.vy],self.dt)
        self.mov_charge.set_data([self.x], [self.y])
        self.new_x.append(self.x)
        self.new_y.append(self.y)
        self.traj_charge.set_data(self.new_x, self.new_y)
        return self.mov_charge, self.traj_charge

    def animate(self, inter = 5):
        a = animation.FuncAnimation(self.fig, self.update,
                              interval = inter, blit = True)
        if self.full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def Poincare_Mapping(self,n = 10, cycles = 100_000, lim = 250, figsize = 250):
        fig, ax = plt.subplots()
        fig.suptitle('Poincare Mapping')
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        for spine in ax.spines.values():
             spine.set_edgecolor('black')
             ax.tick_params(color = 'black', labelcolor = 'black')
        ax.grid(linestyle=':', linewidth=0.5)
        ax.set_xlim(-figsize,figsize)
        ax.set_ylim(-figsize,figsize)
        ax.scatter([x for x,y in self.coords],[y for x,y in self.coords], color = 'white')
        colors = cm.hsv(np.linspace(0,1,n + 1))
        for i in range(n):
            self.x = np.random.random()*(2*lim) - lim
            self.y = np.random.random()*(2*lim) - lim
            x0, y0 = self.x, self.y
            self.vx, self.vy = 0,0
            j = 0
            puntos_x, puntos_y = [],[]
            while j < cycles:
                for carga_x, carga_y in self.coords:
                    dist = np.sqrt((self.x - carga_x)**2 + (self.y - carga_y)**2)
                    if dist < 4:
                        self.x = np.random.random()*(2*lim) - lim
                        self.y = np.random.random()*(2*lim) - lim
                        x0, y0 = self.x, self.y
                        self.vx, self.vy = 0,0
                        puntos_x, puntos_y = [],[]
                        j = 0
                j += 1
                self.x, self.y, self.vx, self.vy = RK4(self.field,[self.x,self.y,self.vx,self.vy],self.dt)
                r = (self.x*self.vx + self.y*self.vy)/(np.sqrt(self.x**2 + self.y**2))
                if abs(r) < 0.1:
                    puntos_x.append(self.x)
                    puntos_y.append(self.y)
            ax.plot([x0],[y0], color = colors[i], marker = 'o', markersize = 10,
                    markeredgecolor = 'white')
            ax.scatter(puntos_x,puntos_y, color = colors[i], s = 8)
            print(f'[INFO] Finished {i+1} iterations')
            print(x0,y0,'\n')
        plt.show()


anim = charges_animation(magnetic_field = True, stream = True,
                         density = 4, full_screen=False)
anim.setup_fig_ax(450,color="white")
anim.set_moving_charge(x = 200, y = 0,
                       vx = 0, vy = -50,
                       F = 200)
anim.set_static_charges(n_charges = 3)
anim.set_field(c = 0.4,alpha = 0.8)
#anim.Poincare_Mapping(n = 10, cycles = 1_000_000, lim = 300, figsize=300)
anim.animate(1)