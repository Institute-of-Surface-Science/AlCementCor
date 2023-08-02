import ufl
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from AlCementCor.config import *
from AlCementCor.info import *
from AlCementCor.input_file import *
from AlCementCor.material_properties import *

fe.parameters["form_compiler"]["representation"] = 'quadrature'
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


# Initialize a SimulationConfig object using the configuration file
simulation_config = SimulationConfig('simulation_config.json')

two_layers = simulation_config.use_two_material_layers
endTime = simulation_config.integration_time_limit
no_of_timesteps = simulation_config.total_timesteps
selected_hardening_model = simulation_config.hardening_model

# Check if the field_input_file is set in the configuration file
if simulation_config.field_input_file:
    # If it is, load the thickness and length from the file
    result = process_input_tensors(simulation_config.field_input_file, plot=True)

    # Access thickness and length directly from the result dictionary
    simulation_config.width = result[ExternalInput.WIDTH.value]
    simulation_config.length = result[ExternalInput.LENGTH.value]

# # Define old coordinates
# old_coordinates = np.array([result[input_file.ExternalInput.X.value], result[input_file.ExternalInput.Y.value],
#                             result[input_file.ExternalInput.Z.value]]).T

# Load material properties
properties_al = MaterialProperties('material_properties.json', 'Al6082-T6')
properties_ceramic = MaterialProperties('material_properties.json', 'Aluminium-Ceramic')

# Access properties for Al6082-T6
C_E = properties_al.youngs_modulus
C_nu = properties_al.poisson_ratio
C_sig0 = properties_al.yield_strength
C_mu = properties_al.shear_modulus
lmbda = properties_al.first_lame_parameter
C_Et = properties_al.tangent_modulus
C_linear_isotropic_hardening = properties_al.linear_isotropic_hardening
C_nlin_ludwik = properties_al.nonlinear_ludwik_parameter
C_exponent_ludwik = properties_al.exponent_ludwik
C_swift_eps0 = properties_al.swift_epsilon0
C_exponent_swift = properties_al.exponent_swift

# Access properties for Aluminium-Ceramic
C_E_outer = properties_ceramic.youngs_modulus
C_nu_outer = properties_ceramic.poisson_ratio
C_sig0_outer = properties_ceramic.yield_strength
C_mu_outer = properties_ceramic.shear_modulus
lmbda_outer = properties_ceramic.first_lame_parameter
C_Et_outer = properties_ceramic.tangent_modulus
C_linear_isotropic_hardening_outer = properties_ceramic.linear_isotropic_hardening

summarize_and_print_config(simulation_config, [properties_al, properties_ceramic])

# Length refers to the y-length
# Width refers to the x-length
# Geometry of the domain
##########################################
l_layer_x = 0.0  # mm
l_layer_y = 0.0  # mm

# todo: handle direction
if two_layers:
    l_layer_x = simulation_config.layer_1_thickness  # mm
    l_layer_y = 0.0  # mm

l_x = simulation_config.width + l_layer_x
l_y = simulation_config.length + l_layer_y

# Discretization of the domain
n_x = 200
n_y = 100
n_z = 2  # Number of elements

# C_strain_rate = fe.Constant(0.0001)  # 1/s
C_strain_rate = fe.Constant(0.000001)  # 0.01/s

# Initialize the mesh
mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(l_x, l_y), n_x, n_y)

# Declare Numerical Stuff
deg_u, deg_stress = 2, 2

V = fe.VectorFunctionSpace(mesh, "CG", deg_u)
u, du, Du = [fe.Function(V, name=n) for n in ["Total displacement", "Iteration correction", "Current increment"]]

DG = fe.FunctionSpace(mesh, "DG", 0)
We = fe.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W = fe.FunctionSpace(mesh, We)
sig, sig_old, n_elas = [fe.Function(W) for _ in range(3)]

func_names = ["Beta", "Cumulative plastic strain", "Hydrostatic stress", "local sig0",
              "local mu", "local lmbda", "local hardening factor"]
W0 = fe.FunctionSpace(mesh, fe.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default'))
beta, p, sig_hyd, sig_0_local, mu_local, lmbda_local, C_linear_h_local = [fe.Function(W0, name=n) for n in func_names]

P0 = fe.FunctionSpace(mesh, "DG", 0)
sig_hyd_avg, sig_0_test, lmbda_test = [fe.Function(P0, name=n) for n in ["Avg. Hydrostatic stress", "test", "test2"]]


# Boundary condition setup
def top(x, on_boundary):
    return on_boundary and fe.near(x[1], l_y)


def bottom(x, on_boundary):
    return on_boundary and fe.near(x[1], 0.0)


boundary_conditions = {
    'top': top,
    'bottom': bottom
}

displacement_conditions = {
    'top': fe.Expression("strain*t", t=0.0, degree=0, strain=C_strain_rate),
    'bottom': fe.Expression("-strain*t", t=0.0, degree=0, strain=C_strain_rate)
}

# top, bottom = boundary_condition(x, l_y), boundary_condition(x, 0.0)
# u_D = fe.Expression("strain*t", t=0.0, degree=0, strain=C_strain_rate)
# u_mD = fe.Expression("-strain*t", t=0.0, degree=0, strain=C_strain_rate)
bc1 = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), boundary_conditions['bottom'])
bc1_i = bc1

if two_layers:
    bc1 = fe.DirichletBC(V.sub(1), displacement_conditions['bottom'], boundary_conditions['bottom'])
    # reset displacement to 0
    bc1_i = fe.DirichletBC(V.sub(1), 0, boundary_conditions['bottom'])

# TOP
#########
# set the top to displace with a constant strain rate
bc2 = fe.DirichletBC(V.sub(1), displacement_conditions['top'], boundary_conditions['top'])

# reset displacement to 0
bc2_i = fe.DirichletBC(V.sub(1), 0, boundary_conditions['top'])

# set boundary condition
bc = [bc1, bc2]
# homogenized bc
bc_iter = [bc1_i, bc2_i]


# Util
########################################################################
def as_3D_tensor(X):
    return fe.as_tensor([[X[0], X[3], 0],
                         [X[3], X[1], 0],
                         [0, 0, X[2]]])


def local_project(v, V, u=None):
    dv = fe.TrialFunction(V)
    v_ = fe.TestFunction(V)
    a_proj = fe.inner(dv, v_) * dxm
    b_proj = fe.inner(v, v_) * dxm
    solver = fe.LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = fe.Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return


# Math!
########################################################################

def eps(displacement):
    """
    Calculate the strain tensor (epsilon)
    :param displacement: The displacement tensor
    :return: strain tensor
    """
    e = fe.sym(fe.nabla_grad(displacement))
    # e= 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)
    return fe.as_tensor([[e[0, 0], e[0, 1], 0],
                         [e[0, 1], e[1, 1], 0],
                         [0, 0, 0]])


def sigma(strain):
    """
    Calculate the Cauchy Stress Tensor
    :param strain: Strain (epsilon)
    :return: Cauchy Stress Tensor
    """
    # return lmbda * fe.tr(strain) * fe.Identity(3) + 2 * C_mu * strain
    # return lmbda * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_V * strain #todo: line is broken
    # return lmbda * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_DG * strain
    return lmbda_local_DG * fe.tr(strain) * fe.Identity(3) + 2 * mu_local_DG * strain


def sigma_tang(e):
    N_elas = as_3D_tensor(n_elas)
    # return sigma(e) - 3 * C_mu * (3 * C_mu / (3 * C_mu + C_linear_isotropic_hardening) - beta) * fe.inner(N_elas,
    #                                                                            e) * N_elas - 2 * C_mu * beta * fe.dev(e)

    return sigma(e) - 3 * mu_local_DG * (3 * mu_local_DG / (3 * mu_local_DG + C_linear_h_local_DG) - beta) * fe.inner(
        N_elas,
        e) * N_elas - 2 * mu_local_DG * beta * fe.dev(e)


# Von-Mises Stress
def sigma_v(strain):
    s = fe.dev(sigma(strain))
    return fe.sqrt(3 / 2. * fe.inner(s, s))


# Macaulays Bracket for <f_elastic>+
# only positive for x > 0
ppos = lambda x: (x + abs(x)) / 2.


# https://www.dynasupport.com/tutorial/computational-plasticity/radial-return
# https://www.dynasupport.com/tutorial/computational-plasticity/generalizing-the-yield-function
# https://www.dynasupport.com/tutorial/computational-plasticity/the-consistent-tangent-matrix
def proj_sig(deps, old_sig, old_p):
    # update stress from change in strain (deps)
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps)

    # trial stress
    s = fe.dev(sig_elas)
    # von-Mises stress or equivalent trial stress
    sig_eq = fe.sqrt(3 / 2. * fe.inner(s, s))

    # prevent division by zero
    # if np.mean(local_project(old_p, P0).vector()[:]) < 1E-12:
    #    old_p += 1E-12

    # https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_theory.06.29.html#3554826
    # Calculate flow stress based on hardening law
    # linear hardening
    # k_linear = C_sig0 + C_linear_isotropic_hardening * old_p
    k_linear = sig_0_local + C_linear_h_local * old_p

    #     # Ludwik hardening
    #     k_ludwik = C_sig0 + C_nlin_ludwik * pow(old_p + 1E-12, C_exponent_ludwik)

    #     # swift hardening
    #     k_swift = C_sig0 * pow(1 + old_p/C_swift_eps0, C_exponent_swift)

    # yield surface/ if trial stress <= yield_stress + H * old_p: elastic
    f_elas = sig_eq - k_linear
    # f_elas = sig_eq - k_ludwik
    # f_elas = sig_eq - k_swift

    # change of plastic strain =0 when f_elas < 0
    # in elastic case = 0
    dp = ppos(f_elas) / (3 * mu_local + C_linear_h_local)
    # dp_old = ppos(f_elas) / (3 * C_mu + C_linear_isotropic_hardening)
    # dp = ppos(f_elas) / (3 * C_mu)
    # print("dp", np.mean(local_project(dp, P0).vector()[:]), np.max(local_project(dp, P0).vector()[:]))
    # print("dp_old", np.mean(local_project(dp_old, P0).vector()[:]), np.max(local_project(dp_old, P0).vector()[:]))

    # normal vector on yield surface?
    # in elastic case = 0
    n_elas = s * ppos(f_elas) / (sig_eq * f_elas)

    # radial return mapping?
    # in elastic case = 0
    beta = 3 * C_mu * dp / sig_eq

    # updated cauchy stress tensor
    # in elastic case = sig_elas
    new_sig = sig_elas - beta * s

    # Hydrostatic stress
    sig_hyd = (1. / 3) * fe.tr(new_sig)

    return fe.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
        fe.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
        beta, dp, sig_hyd


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = ufl.dx(metadata=metadata)

v = fe.TrialFunction(V)
u_ = fe.TestFunction(V)


class set_layer_2(fe.UserExpression):
    # Return the diffusion coefficient at a point scaled by level set function
    def __init__(self, inner_value, outer_value, **kwargs):
        self.inner = inner_value
        self.outer = outer_value
        super().__init__(**kwargs)

    def eval(self, value, x):
        if x[0] > simulation_config.width:
            value[0] = self.outer
            value[1] = self.outer
        else:
            value[0] = self.inner
            value[1] = self.inner

    def value_shape(self):
        return (2,)


# calculate local mu
# test2 = set_layer_2(C_mu, C_mu)
# mu_local_V = fe.interpolate(test2, V)
mu_local_DG = fe.Function(DG)
dofmap = DG.tabulate_dof_coordinates()[:]
mu_vec = np.zeros(dofmap.shape[0])
mu_vec[:] = C_mu
mu_vec[dofmap[:, 0] > simulation_config.width] = C_mu_outer
mu_local_DG.vector()[:] = mu_vec
# print(dofmap)
# print(dofmap.shape)
# u_DG = fe.XDMFFile("u.xdmf")
# u_DG << fe.project(mu_local_DG, V)
# # mu_local_DG = fe.project(test2, DG)
# exit()

lmbda_local_DG = fe.Function(DG)
dofmap = DG.tabulate_dof_coordinates()[:]
lmbda_vec = np.zeros(dofmap.shape[0])
lmbda_vec[:] = lmbda
lmbda_vec[dofmap[:, 0] > simulation_config.width] = lmbda_outer
lmbda_local_DG.vector()[:] = lmbda_vec

# lmbda_test.assign(fe.project(lmbda_local_DG, P0))

# plt.figure()
# # plt.plot(results[:, 0], results[:, 1], "-o")
# ax = fe.plot(lmbda_test)
# cbar = plt.colorbar(ax)
# plt.xlabel("x")
# plt.ylabel("y$")
# # plt.show()
# plt.savefig("testlmbda.svg")
# plt.close()
# exit()

C_linear_h_local_DG = fe.Function(DG)
dofmap = DG.tabulate_dof_coordinates()[:]
C_linear_h_vec = np.zeros(dofmap.shape[0])
C_linear_h_vec[:] = C_linear_isotropic_hardening
C_linear_h_vec[dofmap[:, 0] > simulation_config.width] = C_linear_isotropic_hardening_outer
C_linear_h_local_DG.vector()[:] = C_linear_h_vec

a_Newton = fe.inner(eps(v), sigma_tang(eps(u_))) * dxm
# res = -inner(eps(u_), as_3D_tensor(sig)) * dxm + F_ext(u_)
res = -fe.inner(eps(u_), as_3D_tensor(sig)) * dxm

cellV = local_project(fe.CellVolume(mesh), P0)
# print("minV:", np.amin(cellV.vector()[:]), "maxV", np.amax(cellV.vector()[:]))
# print("H", H.values()[0])

file_results = fe.XDMFFile("plasticity_results.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
P0 = fe.FunctionSpace(mesh, "DG", 0)
p_avg = fe.Function(P0, name="Plastic strain")

Nitermax, tol = 100, 1e-8  # parameters of the Newton-Raphson procedure
time_step = endTime / (no_of_timesteps)


# assign local values to the layers
class set_yield(fe.UserExpression):
    # Return the diffusion coefficient at a point scaled by level set function
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, value, x):
        if x[0] > simulation_config.width:
            value[0] = C_sig0_outer
        else:
            value[0] = C_sig0

    def value_shape(self):
        return ()


class set_layer(fe.UserExpression):
    # Return the diffusion coefficient at a point scaled by level set function
    def __init__(self, inner_value, outer_value, **kwargs):
        self.inner = inner_value
        self.outer = outer_value
        super().__init__(**kwargs)

    def eval(self, value, x):
        if x[0] > simulation_config.width:
            value[0] = self.outer
        else:
            value[0] = self.inner

    def value_shape(self):
        return ()


# test = set_yield()
# sig_0_local = fe.interpolate(test, W0)

test = set_layer(C_sig0, C_sig0_outer)
sig_0_local = fe.interpolate(test, W0)

test = set_layer(C_mu, C_mu_outer)
mu_local = fe.interpolate(test, W0)

test = set_layer(C_linear_isotropic_hardening, C_linear_isotropic_hardening_outer)
C_linear_h_local = fe.interpolate(test, W0)

# #mu_local_V = fe.project(mu_local, V)
# test2 = set_layer_2(C_mu, C_mu)
# mu_local_V = fe.interpolate(test2, V)

# sig_0_test.assign(fe.project(sig_0_local, P0))

# plt.figure()
# # plt.plot(results[:, 0], results[:, 1], "-o")
# ax = fe.plot(sig_0_test)
# cbar = plt.colorbar(ax)
# plt.xlabel("x")
# plt.ylabel("y$")
# # plt.show()
# plt.savefig("test.svg")
# plt.close()

results = []
stress_max_t = []
stress_max_t += [0]
stress_mean_t = []
stress_mean_t += [0]
disp_t = []
disp_t += [0]
time_step_adjusted = False
time = 0
i = 0
not_adjusted_count = 0
# for (i, time) in enumerate(time_steps):
while time < endTime:
    time += time_step
    i += 1

    # update the displacement boundary
    displacement_conditions['top'].t = time_step
    if two_layers:
        displacement_conditions['bottom'].t = time_step

    # assemble system
    A, Res = fe.assemble_system(a_Newton, res, bc)

    # calculate residual
    nRes0 = Res.norm("l2")
    nRes = nRes0

    # reset Du to 0
    Du.interpolate(fe.Constant((0, 0)))

    print("Step:", str(i + 1), "time", time, "s")
    print("displacement", C_strain_rate.values()[0] * time, "mm")

    niter = 0
    # n_resold = 1000
    while niter == 0 or (nRes0 > 0 and nRes / nRes0 > tol and niter < Nitermax):
        # solve for du
        fe.solve(A, du.vector(), Res, "mumps")

        # update displacement gradient
        Du.assign(Du + du)

        # calculate strain gradient
        deps = eps(Du)

        # update variables
        sig_, n_elas_, beta_, dp_, sig_hyd_ = proj_sig(deps, sig_old, p)
        local_project(sig_, W, sig)
        local_project(n_elas_, W, n_elas)
        local_project(beta_, W0, beta)
        sig_hyd.assign(local_project(sig_hyd_, W0))

        # update system
        A, Res = fe.assemble_system(a_Newton, res, bc_iter)

        nRes = Res.norm("l2")
        print("    Residual:", nRes)
        # n_resold = nRes
        niter += 1

    # exit if diverged
    if nRes > 1 or np.isnan(nRes):
        print("ERROR: diverged!")
        exit(-1)

    # update displacement
    u.assign(u + Du)
    # update plastic strain
    p.assign(p + local_project(dp_, W0))

    # update stress fields
    sig_old.assign(sig)
    sig_hyd_avg.assign(fe.project(sig_hyd, P0))

    s11, s12, s21, s22 = sig.split(deepcopy=True)
    avg_stress_y = np.average(s22.vector()[:])
    avg_stress = np.average(sig.vector()[:])
    sig_n = as_3D_tensor(sig)
    s = fe.dev(sig_n)

    # calculate the von-mises stress
    sig_eq = fe.sqrt(3 / 2. * fe.inner(s, s))
    sig_eq_p = local_project(sig_eq, P0)

    if i % 10 == 0:
        plt.figure()
        # plt.plot(results[:, 0], results[:, 1], "-o")
        ax = fe.plot(sig_eq_p)
        cbar = plt.colorbar(ax)
        plt.xlabel("x")
        plt.ylabel("y$")
        # plt.show()
        plt.savefig("vm" + str(i) + ".svg")
        plt.close()

    # project the von-mises stress for plotting
    stress_max_t += [np.abs(np.amax(sig_eq_p.vector()[:]))]
    stress_mean_t += [np.abs(np.mean(sig_eq_p.vector()[:]))]

    # displacement at the middle of the bar in y-direction
    disp_t += [u(l_x / 2, l_y)[1]]

    # print("maximum stress ", np.amax(sig.vector()[:]))
    # print("maximum stress(y-y) ", np.amax(s22.vector()[:]))
    # print("plastic strain ", np.amax(p.vector()[:]))

    file_results.write(u, time)
    p_avg.assign(fe.project(p, P0))
    file_results.write(p_avg, time)
    results += [(np.abs(u(l_x / 2, l_y)[1]) / l_y, time)]

# ax = fe.plot(u, mode="displacement")
# cbar = plt.colorbar(ax)
# plt.show()

# plt.plot(results[:, 0], results[:, 1], "-o")
# plt.xlabel("Displacement of inner boundary")
# plt.ylabel(r"Applied pressure $q/q_{lim}$")
# # plt.show()
# plt.savefig("test_disp_pressure.svg")

# ax = fe.plot(sig_hyd_avg)
# cbar = plt.colorbar(ax)
# plt.show()

# data extracted from fig. 6 paper1
ref_strain = [4.5917e-7, 0.0022967, 0.0068471, 0.0077457, 0.008868360277136257,
              0.011511, 0.024988, 0.040012, 0.059011, 0.071602, 0.083751,
              0.10429, 0.12108, 0.13963, 0.16105, 0.17960]
ref_stress = [0.10395, 99.376, 250.10, 267.26, 275.09019989024057,
              279.21, 284.41, 290.64, 296.36, 299.48, 301.04, 302.60, 303.12, 302.60,
              301.04, 296.88]

# data extracted from fig. 10 paper 3
ref_strain2 = [0, 0.001072706, 0.001966627, 0.002324195, 0.003218117, 0.004469607, 0.007508939,
               0.009833135, 0.015554231, 0.027711561, 0.038796186, 0.056495828, 0.070262217,
               0.085637664, 0.10852205, 0.131585221, 0.155542312, 0.175029797, 0.178605483]

ref_stress2 = [0, 117.4641148, 172.0095694, 222.9665072, 236.2440191, 246.291866, 253.8277512,
               260.2870813, 264.9521531, 277.15311, 285.0478469, 294.7368421, 299.0430622,
               302.9904306, 307.2966507, 305.861244, 306.5789474, 300.8373206, 288.6363636]

# for linear deformation E = stress/strain -> strain = stress/E
linear_strain = np.multiply(np.array(stress_max_t), 1.0 / C_E.values()[0])
linear_deformation = linear_strain * l_y

print("max vm stress", np.max(stress_max_t))
print("mean vm stress", np.max(stress_mean_t))

# Plot the stress-deformation curve
plt.plot(disp_t, stress_max_t, "-o", label="sim", markevery=5)
plt.plot(linear_deformation, stress_max_t, label="linear")
plt.xlabel("displacement [mm]")
plt.ylabel(r"max. von-Mises stress [MPa]")
plt.legend()
plt.savefig("test_deformation_stress.svg")

# Plot the stress-strain curve
plt.figure()
# plt.plot(np.array(disp_t) / l_y, stress_max_t, "-o", label="sim-max", markevery=5)
plt.plot(np.array(disp_t) / l_y, stress_mean_t, "-o", label="sim", markevery=5)
plt.plot(linear_strain, stress_max_t, label="linear")
plt.plot(ref_strain, ref_stress, label="exp")
plt.plot(ref_strain2, ref_stress2, label="exp2")
# plt.plot(linear_strain, stress_mean_t, label="linear")
plt.xlabel("strain [-]")
plt.ylabel(r"max. von-Mises stress [MPa]")
plt.legend()
plt.savefig("test_strain_stress.svg")

# It can also be checked that the analytical limit load is also well reproduced
# when considering a zero hardening modulus.
#
# -----------
# References
# -----------
#
# .. [BON2014] Marc Bonnet, Attilio Frangi, Christian Rey.
#  *The finite element method in solid mechanics.* McGraw Hill Education, pp.365, 2014
# paper 1:Flow and fracture behavior of aluminum alloy 6082-T6 at different tensile strain rates and triaxialities
# paper 2:Fatigue Study of the Pre-Corroded 6082-T6 Aluminum Alloy in Saline Atmosphere
# Alejandro Fernández Muñoz 1 , José Luis Mier Buenhombre 2, *, Ana Isabel García-Diez 2 ,
# Carolina Camba Fabal 2 and Juan José Galán Díaz 2
# paper3: Analysis of manufacturing parameters on the shear strength of aluminium adhesive single-lap joints
# A.M. Pereira a,∗, J.M. Ferreira b, F.V. Antunes b, P.J. Bártolo
