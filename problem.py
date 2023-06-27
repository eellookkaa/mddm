import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import ConvexHull

import pulp
from pulp import LpProblem, LpMinimize, LpVariable, value, LpStatus, lpSum

from utils import compute_coefficients, compute_no_fit_polygons

class Problem:

    def __init__(self, N, H, Kmax, xmax, ymax):
        self.N = N
        self.H = H
        self.Kmax = Kmax
        self.xmax = xmax
        self.ymax = ymax

        self.polygons = None
        self.use_no_fit = None
        self.a = None
        self.b = None
        self.c = None
        self.M = None

        self.no_fit_polygons = None

        self.problem = None
        self.w = None
        self.tx = None
        self.ty = None
        self.z = None


    def create_data(self, seed=None):

        # re-initialize self.polygons
        self.polygons = []
        # set the seed of pseudo-random number generator
        np.random.seed(seed)

        while len(self.polygons) < self.N:
            # generate Kmax points on the plane
            xs = np.random.uniform(0, self.xmax, self.Kmax, )
            ys = np.random.uniform(0, self.ymax, self.Kmax)
            points = np.stack((xs, ys), axis=1)
            # move points to match the origin with minimums of points' coordinates
            mins = np.min(points, axis=0)
            points = points - mins
            # compute height of the polygon
            height = np.max(points, axis=0)[1]
            # check if height is less than H. If yes, compute convex hull
            if height <= self.H:
                polygon = ConvexHull(points)
                self.polygons.append(polygon.points[polygon.vertices])


    def create_problem(self, filename=None, use_no_fit=False):

        if self.polygons is None:
            raise ValueError('You must create the data before the model!')

        self.use_no_fit = use_no_fit

        self.problem = LpProblem('polygons_packing', LpMinimize)

        self.w = LpVariable('w', 0)

        self.tx = LpVariable.dicts('tx', range(self.N), 0)
        self.ty = LpVariable.dicts('ty', range(self.N), 0)

        # objective
        self.problem.setObjective(self.w)

        # w constraints
        widths = [np.max(pol[:, 0]) - np.min(pol[:, 0]) for pol in self.polygons]
        for i in range(self.N):
            self.problem += self.w - self.tx[i] >= widths[i], f"w_constraint_{i}"

        # height constraints
        heights = [np.max(pol[:, 1]) - np.min(pol[:, 1]) for pol in self.polygons]
        for i in range(self.N):
            self.problem += self.ty[i] <= self.H - heights[i], f"height_constraint_{i}"
        
        self._add_non_overl_constr()
        
        if filename is not None:
            self.problem.writeLP(filename)


    def _add_non_overl_constr(self, M=None):
        
        if not self.use_no_fit:
            # we must distinguish two cases:
            # - we are creating the model for the first time
            # - we are creating new cuts using smaller M 
            # in the first case we must initialize variables and coefficients
            if M is None:
                z_idx = []
                for i in range(self.N):
                    for j in range(i+1, self.N):
                        K_i = self.polygons[i].shape[0]
                        K_j = self.polygons[j].shape[0]
                        for v in range(K_i + K_j):
                            z_idx.append((i, j, v))
                self.z = LpVariable.dicts('z', z_idx, 0, 1, 'Integer')
            
                # re-initialize coefficients
                self.a, self.b, self.c = {}, {}, {}

                for i in range(self.N):
                    a_list, b_list, c_list = compute_coefficients(self.polygons[i])
                    self.a[i] = a_list
                    self.b[i] = b_list
                    self.c[i] = c_list

                self.M = self._compute_M()
                constr_name = 'non_overlapping_constraint_'
    
            else:
                # in the second case we must only change self.M
                self.M = M
                constr_name = 'new_constr_'

            # non overlapping constraints
            cnt = 0
            for i in range(self.N):
                for j in range(i+1, self.N):
                    K_i = self.polygons[i].shape[0]
                    K_j = self.polygons[j].shape[0]
                    for v1 in range(K_i):
                        for v2 in range(K_j):
                            # 1
                            self.problem += self.a[i][v1] * self.tx[j] - self.a[i][v1] * self.tx[i] + \
                                self.b[i][v1] * self.ty[j] - self.b[i][v1] * self.ty[i] - self.M * self.z[(i, j, v1)] <= \
                                    - self.c[i][v1] - self.a[i][v1] * self.polygons[j][v2][0] - self.b[i][v1] * self.polygons[j][v2][1], constr_name + str(cnt)
                            cnt += 1
                            # 2
                            self.problem += self.a[j][v2] * self.tx[i] - self.a[j][v2] * self.tx[j] + \
                                self.b[j][v2] * self.ty[i] - self.b[j][v2] * self.ty[j] - self.M * self.z[(i, j, K_i + v2)]  <= \
                                    - self.c[j][v2] - self.a[j][v2] * self.polygons[i][v1][0] - self.b[j][v2] * self.polygons[i][v1][1], constr_name + str(cnt)
                            cnt += 1
            # 3, only if we are creating the model for the first time
            if M is None:
                for i in range(self.N):
                    for j in range(i+1, self.N):
                        K_i = self.polygons[i].shape[0]
                        K_j = self.polygons[j].shape[0]

                        self.problem += sum([self.z[(i, j, v)] for v in range(K_i + K_j)]) <= K_i + K_j - 1, constr_name + str(cnt)
                        cnt += 1
        
        if self.use_no_fit:
            # as in the other case
            if M is None:
                
                self.no_fit_polygons = compute_no_fit_polygons(self.polygons)

                z_idx = []
                for i in range(self.N):
                    for j in range(i+1, self.N):
                        for v in range(self.no_fit_polygons[(i, j)].shape[0]):
                            z_idx.append((i, j, v))
                self.z = LpVariable.dicts('z', z_idx, 0, 1, 'Integer')

                # re-initialize coefficients
                self.a, self.b, self.c = {}, {}, {}

                for i in range(self.N):
                    for j in range(i+1, self.N):
                        no_fit_pol = self.no_fit_polygons[(i, j)]
                        n_vert = no_fit_pol.shape[0]
                        
                        a_list, b_list, c_list = compute_coefficients(no_fit_pol)
                        self.a[(i, j)] = a_list
                        self.b[(i, j)] = b_list
                        self.c[(i, j)] = c_list

                self.M = self._compute_M()
                constr_name = 'non_overlapping_constraint_'

            else:
                self.M = M
                constr_name = 'new_constr_'
            
            # non overlapping constraints
            cnt = 0
            # 1
            for i in range(self.N):
                for j in range(i+1, self.N):
                    no_fit_pol = self.no_fit_polygons[(i, j)]
                    n_vert = no_fit_pol.shape[0]

                    for v in range(n_vert):
                        self.problem += self.a[(i, j)][v] * (self.tx[j] - self.tx[i]) + \
                                        self.b[(i, j)][v] * (self.ty[j] - self.ty[i]) + self.c[(i, j)][v]  \
                                        -self.M * (1 - self.z[(i, j, v)]) <= 0, constr_name + str(cnt)
                        cnt += 1
                            
            # 2, only if we are creating the model for the first time
            if M is None:
                for i in range(self.N):
                    for j in range(i+1, self.N):
                        no_fit_pol = self.no_fit_polygons[(i, j)]
                        n_vert = no_fit_pol.shape[0]

                        self.problem += sum([self.z[(i, j, v)] for v in range(n_vert)]) >= 1, constr_name + str(cnt)
                        cnt += 1


    def _compute_M(self, worst_val=None):

        if not self.use_no_fit:
            if worst_val is None:
                widths = [np.max(pol[:, 0]) - np.min(pol[:, 0]) for pol in self.polygons]
                worst_val = sum(widths)

            M = -float('inf')
        
            for n in range(self.N):
                n_edges = self.polygons[n].shape[0]
                
                for v in range(n_edges):
                    m1 = self.c[n][v]
                    m2 = self.b[n][v] * self.H + self.c[n][v]
                    m3 = self.a[n][v] * worst_val + self.c[n][v]
                    m4 = self.a[n][v] * worst_val + self.b[n][v] * self.H + self.c[n][v]

                    M = max([M, m1, m2, m3, m4])

            return M
        
        else:            
            heights = [np.max(pol[:, 1]) - np.min(pol[:, 1]) for pol in self.polygons]
            widths = [np.max(pol[:, 0]) - np.min(pol[:, 0]) for pol in self.polygons]
            
            if worst_val is None:
                worst_val = sum(widths)

            M = -float('inf')
            for i in range(self.N):
                for j in range(i+1, self.N):

                    n_edges = self.no_fit_polygons[(i, j)].shape[0]

                    min_diff_txs = -(worst_val - widths[i])
                    max_diff_txs = worst_val - widths[j]

                    min_diff_tys = -(self.H - heights[i])
                    max_diff_tys =  self.H - heights[j]

                    for v in range(n_edges):
                        m1 = self.a[(i, j)][v] * min_diff_txs + self.b[(i, j)][v] * min_diff_tys + self.c[(i, j)][v]
                        m2 = self.a[(i, j)][v] * min_diff_txs + self.b[(i, j)][v] * max_diff_tys + self.c[(i, j)][v]
                        m3 = self.a[(i, j)][v] * max_diff_txs + self.b[(i, j)][v] * min_diff_tys + self.c[(i, j)][v]
                        m4 = self.a[(i, j)][v] * max_diff_txs + self.b[(i, j)][v] * max_diff_tys + self.c[(i, j)][v]

                        M = max([M, m1, m2, m3, m4])

            return M

    
    def _solve(self, actual_solver, use_M_opt, first_solver):
        
        if self.problem is None:
            raise ValueError('You must create the model before solving it!')
         
        if not use_M_opt:        
            self.problem.solve(solver=actual_solver)
        
        else:
            assert first_solver is not None

            self.problem.solve(solver=first_solver)
            print('First solution obtained, computing new big-M value')

            new_M = self._compute_M(worst_val=value(self.problem.objective))

            self._add_non_overl_constr(new_M)

            self.problem.solve(solver=actual_solver)
        
        print(f'time = {self.problem.solutionTime}')
        print(f'status = {LpStatus[self.problem.status]}')
        print(f'objective = {value(self.problem.objective)}')


    def solve_cbc(self, 
            keepFiles = 0,
            mip = 1,
            msg = 1,
            cuts = 'on',
            presolve = 'on',
            strong = 0,
            heur = 'on',
            options = [],
            fracGap = None,
            maxSeconds = None,
            use_M_opt = False):
        
        #path = os.path.join(os.getcwd(), 'cbc.exe')
        path='cbc'
        #path='/Users/elnaragalimzhanova/Library/Logs/Homebrew/cbc'
        options.append(f'cuts {cuts}')
        options.append(f'presolve {presolve}')
        options.append(f'strongBranching {strong}')
        options.append(f'heur {heur}')

        solver = pulp.solvers.COIN_CMD(path, keepFiles, mip, msg, options=options, 
                                    fracGap=fracGap, maxSeconds=maxSeconds)

        if use_M_opt:
            help_solver = pulp.solvers.COIN_CMD(path, keepFiles, mip, msg=0, options=options, 
                                    fracGap=fracGap, maxSeconds=10)
        else:
            help_solver = None
        
        self._solve(solver, use_M_opt, help_solver)


    def solve_cplex(self,
            keepFiles=0, 
            mip=1, 
            msg=1, 
            options=[],
            timelimit = None, 
            mip_start=False,
            use_M_opt = False):
        
        path = os.path.join(os.getcwd(), 'cplex.exe')

        solver = pulp.solvers.CPLEX_CMD(path, keepFiles, mip, msg, options, timelimit, mip_start)

        if use_M_opt:
            help_solver = pulp.solvers.CPLEX_CMD(path, keepFiles, mip, msg=0, options=options, 
                                                    timelimit=10, mip_start=mip_start)
        else:
            help_solver = None
        
        self._solve(solver, use_M_opt, help_solver)


    def show_original_polygons(self, filename=None):
        
        plt.figure()
        max_width = 0   # need this variable to plot polygons next to each other
        colors = cm.get_cmap('rainbow')

        for i, polygon in enumerate(self.polygons):
            for j, v in enumerate(polygon):
                next_v = polygon[0] if j == polygon.shape[0] - 1 else polygon[j + 1]
                plt.plot((v[0] + max_width, next_v[0] + max_width), (v[1], next_v[1]), c=colors(float(i) / self.N))
            max_width += np.max(polygon, axis=0)[0]
        
        # roll plot
        plt.plot((0, 0), (0, self.H), c='k')
        plt.plot((0, max_width), (self.H, self.H), c='k')
        plt.plot((max_width, max_width), (self.H, 0), c='k')
        plt.plot((max_width, 0), (0, 0), c='k')

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        plt.close()
    

    def show_solution(self, filename=None):

        tx = [var.value() for var in self.problem.variables() if var.name[:2] == 'tx']
        ty = [var.value() for var in self.problem.variables() if var.name[:2] == 'ty']
        opt_val = value(self.problem.objective)

        plt.figure()
        colors = cm.get_cmap('rainbow')

        for i, polygon in enumerate(self.polygons):
            for j, v in enumerate(polygon):
                next_v = polygon[0] if j == polygon.shape[0] - 1 else polygon[j + 1]
                plt.plot((v[0] + tx[i], next_v[0] + tx[i]), (v[1] + ty[i], next_v[1] + ty[i]), c=colors(float(i) / self.N))

        # roll plot
        plt.plot((0, 0), (0, self.H), c='k')
        plt.plot((0, opt_val), (self.H, self.H), c='k')
        plt.plot((opt_val, opt_val), (self.H, 0), c='k')
        plt.plot((opt_val, 0), (0, 0), c='k')

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        plt.close()