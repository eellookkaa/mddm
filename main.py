from problem import Problem

# PuLP apis code at https://github.com/coin-or/pulp/tree/master/pulp/apis

N, H, Kmax, xmax, ymax = 8, 12, 6, 10, 10

P = Problem(N, H, Kmax, xmax, ymax)

# PuLP does not support the creation of a problem from file
# Set the seed if you want to reproduce the experiment
P.create_data(seed=None)
print(P.polygons)

# if filename is None, it doesn't save the model in an lp file
# use_no_fit if you want to use the model created using no-fit polygons
P.create_problem(filename='problem.lp', use_no_fit=True)

# if filename is None, it shows the figure without saving it
P.show_original_polygons(filename='problem_init.png')

# ------------------------------------------------------ #
# ----------------solve with cbc------------------------ #
# ------------------------------------------------------ #

# Possible cbc parameters at https://projects.coin-or.org/CoinBinary/export/1059/OptimizationSuite/trunk/Installer/files/doc/cbcCommandLine.pdf
# or run cbc.exe on your terminal and write ?
# use it in options, i.e. if you want to deactivate clique cuts:  options = ['clique off']  

keepFiles = 0
mip = 1
msg = 1
cuts = 'forceOn'        # on, off, root, ifmove, forceOn
presolve = 'on'         # on, off, more, file
strong = 5              # int 
heur = 'on'             # on, off
options = [] 
fracGap = None
maxSeconds = 300
# use_M_opt if you want to solve the model, then using a smaller big-M
# use it only when the instance is big
use_M_opt = False

P.solve_cbc(keepFiles, mip, msg, cuts, presolve, strong, heur, options, fracGap, maxSeconds, use_M_opt)

# filename as in show_original_polygons
P.show_solution(filename='problem_sol_cbc.png')

# ------------------------------------------------------ #
# ----------------solve with cplex---------------------- #
# ------------------------------------------------------ #

# Possible cplex parameters at https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.0/ilog.odms.ide.help/OPL_Studio/refoplrun/topics/oplrun_syntax_commandline.html

# keepFiles = 0
# mip = 1 
# msg = 1 
# options = []
# timelimit = 300 
# mip_start = False
# use_M_opt = False

# P.solve_cplex(keepFiles, mip, msg, options, timelimit, mip_start, use_M_opt)

# P.show_solution(filename='problem_sol_cplex.png')