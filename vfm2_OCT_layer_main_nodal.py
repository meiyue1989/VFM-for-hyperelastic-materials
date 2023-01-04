#%%  VFM-based inverse algorithm by FeniCS2019
from fenics import *
import ufl as uf 
import os
import numpy as np
from utils import proj_tensor2,proj_tensor4,post_plot_nodal
import time 
def forward(E,nu,Trac,bcs):   
    P= stress_nh(u,E,nu)    
    FV = inner(P, grad(v))*dx 
    # Traction at boundary
    T=Trac
    FT=dot(T,v)*ds1    
    # Whole system and its Jacobian
    FF = FV-FT
    JJ = derivative(FF, u)   
    # Initialize solver
    problem = NonlinearVariationalProblem(FF, u, bcs=bcs, J=JJ)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-10
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.solve()
    return u
def stress_nh(u,E,nu):
    """Returns 1st Piola-Kirchhoff stress and (local) mass balance for given u, p."""    
    mu = E/(2.0*(1.0 + nu)); lam= E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    F = I + grad(u)
    J = det(F)
    C = F.T * F
    S = mu*(I-inv(C))+lam*ln(J)*inv(C) # 2nd Piola-Kirchoff stress
    P = F*S # 1st Piola-Kirchhoff stress
    return P
def stress_grad_nh(u,E,nu):
    i,j,k,l,m=uf.indices(5)
    mu = E/(2.0*(1.0 + nu))
    lam= E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    dmu_dE = 1.0/(2.0*(1.0 + nu))
    dlam_dE= 1.0*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    dmu_dnu = -1.0*E/(2.0*(1.0 + nu)*(1.0 + nu))
    dlam_dnu= E*(2.0*nu*nu+1.0)/((1.0 + nu)*(1.0 - 2.0*nu)) /((1.0 + nu)*(1.0 - 2.0*nu))    
    F = I + grad(u)
    J = det(F)
    C=(F.T)*F
    C_inv=inv(C)
    S = mu*(I-inv(C))+lam*ln(J)*inv(C) # 2nd Piola-Kirchhoff stress
    dS_dE=dmu_dE*(I-inv(C))+dlam_dE*ln(J)*inv(C) # 2nd Piola-Kirchhoff stress
    dS_dnu=dmu_dnu*(I-inv(C))+dlam_dnu*ln(J)*inv(C) # 2nd Piola-Kirchhoff stress    
    K_tan=as_tensor(lam*C_inv[i,j]*C_inv[k,l]+(mu-lam*ln(J))*(C_inv[i,k]*C_inv[j,l]+C_inv[i,l]*C_inv[j,k]),(i,j,k,l))
    L = as_tensor(K_tan[i,j,k,l]+inv(F)[k,i]*inv(F)[l,m]*S[m,j],(i,j,k,l))
    return dS_dE,dS_dnu,L
def VFM(E_gue,nu_gue):
    u_o = Function(V)
    iter = 0
    while True:
        iter_start = time.time()
        E_o = Function(M)
        nu_o = Function(M)
        E_guess_vec = E_o.vector()
        nu_guess_vec = nu_o.vector()
        i = np.linspace(0,num_vertices-1,num_vertices)
        E_guess_vec[i] = E_gue
        nu_guess_vec[i] = nu_gue
        # E_o = interpolate(Expression('x[2]>-0.19635? E_top:E_bottom',degree=1,E_top=E_top_gue,E_bottom=E_bottom_gue), M)
        # nu_o = interpolate(Expression('x[2]>-0.19635? nu_top:nu_bottom',degree=1,nu_top=nu_top_gue,nu_bottom=nu_bottom_gue), M)
        # File("output/{}/{}/Parameters/Eo_iter{}.pvd".format(case_name,initial_name,iter)) << E_o
        # File("output/{}/{}/Parameters/nuo_iter{}.pvd".format(case_name,initial_name,iter)) << nu_o
        u_o.assign(forward(E_o,nu_o,traction1,bcs))    
        forward_time=time.time()
        print('{}th forward is done, consuming{}'.format(iter,forward_time-iter_start))
        error = assemble((inner(u_o-u_meas,u_o-u_meas)*dx))/assemble((inner(u_meas,u_meas)*dx))
        error_list.append(error)
        if error < tol:
            break
        if iter >MAX_ITER:
            break
        iter += 1 
        # kinematics in the intermediate configuration
        F=I+grad(u_o)
        C=(F.T)*F
        Strain_E = (C-I)/2
        # formulating Eq.(20) 
        dS_dE,dS_dnu,L = stress_grad_nh(u_o,E_o,nu_o)
        dS_dE_array,dS_dnu_array,Strain_E_array = proj_tensor2(dS_dE,dS_dnu,Strain_E,TT)
        L_array = proj_tensor4(L,TT_4)    
        proj_iter = time.time()
        print('{}th proj is done, consuming{}'.format(iter,proj_iter-forward_time))
        # solve VFM equation systems 
        Beta  = solve_VFM(dS_dE_array,dS_dnu_array,Strain_E_array,E_meas_array,L_array)
        os.makedirs('output/{}/{}/Beta_iter/'.format(case_name,initial_name),exist_ok=True)
        # np.savetxt('output/{}/{}/Beta_iter/E_iter{}.txt'.format(case_name,initial_name,iter),Beta)

        dE = list(Beta[:,0])
        dnu = list(Beta[:,1])
        for i in range(len(dE)):
            E_gue[i] += dE[i];nu_gue[i] += dnu[i] 
            if E_gue[i]<=0:
                E_gue[i] = 0.5
            if nu_gue[i] <=0:
                nu_gue[i] = 0.2
            if nu_gue[i] >=0.5:
                nu_gue[i] = 0.48
        iter_end = time.time()
        print('This is the {} iter, error is {},consuming {}'.format(iter,error,iter_end-iter_start))
def solve_VFM(dS_dE_array,dS_dnu_array,E_array,E_meas_array,L_array):
    Beta = np.zeros([num_vertices,2])    
    for ii in range(num_vertices):
        B = np.zeros([2,1])
        A = np.zeros([2,2])
        Vir_E1,Vir_E2,E_o,E_meas = calc_VF(dS_dE_array,dS_dnu_array,E_array,E_meas_array,L_array,ii)
        A[0,0] = np.dot(Vir_E1.T,Vir_E1)[0][0]
        A[0,1] = np.dot(Vir_E2.T,Vir_E1)[0][0]
        A[1,0] =np.dot(Vir_E1.T,Vir_E2)[0][0]
        A[1,1] = np.dot(Vir_E2.T,Vir_E2)[0][0]
        B[0] = -np.dot((E_meas-E_o).T,Vir_E1)[0][0]
        B[1] = -np.dot((E_meas-E_o).T,Vir_E2)[0][0]
        cond = np.linalg.cond(A)
        if cond>=1e6:
            temp_beta = np.linalg.lstsq(A,B)[0]
        else:
            temp_beta = np.linalg.solve(A,B)      
        Beta[ii,0] = temp_beta[0]
        Beta[ii,1] = temp_beta[1]    
    return Beta

def calc_VF(dS_dE_array,dS_dnu_array,E_array,E_meas_array,L_array,index):
    dS_dE_np = ((dS_dE_array[index].reshape([3,3])))
    dS_dnu_np = ((dS_dnu_array[index].reshape([3,3])))
    E = E_array[index].reshape([3,3])
    E_m = E_meas_array[index].reshape([3,3])
    L_np = (L_array[index].reshape([3,3,3,3]))
    index2D1=[0,1,2,1,0,0];index2D2=[0,1,2,2,2,1]
    S_E=np.zeros([6,1]);S_nu=np.zeros([6,1]);E_meas=np.zeros([6,1]);E_o=np.zeros([6,1]);LL=np.zeros([6,6])
    for i in range(6):
        for j in range(0,3):
            LL[i,j] = L_np[index2D1[i],index2D2[i],index2D1[j],index2D2[j]]
        for j in range(3,6):
            LL[i,j] = 2*L_np[index2D1[i],index2D2[i],index2D1[j],index2D2[j]]
        S_E[i] = dS_dE_np[index2D1[i],index2D2[i]]
        S_nu[i] = dS_dnu_np[index2D1[i],index2D2[i]]
        E_o[i] = E[index2D1[i],index2D2[i]]
        E_meas[i] = E_m[index2D1[i],index2D2[i]]
    inv_LL = np.linalg.inv(LL)    
    Vir_E1 = np.dot(inv_LL,S_E)
    Vir_E2 = np.dot(inv_LL,S_nu)
    return Vir_E1,Vir_E2,E_o,E_meas
#%% Model setup 
case_name = 'OCT_2layer_nodal' 
mesh_path = 'model/OCT' 
meshfile = 'OCT.xml'
# Geometry mesh
mesh = Mesh(os.path.join(os.curdir,mesh_path,meshfile)) 
num_vertices = mesh.num_vertices()
#%% FEniCS Functionspaces 
V = VectorFunctionSpace(mesh, 'P', 1)
u = Function(V)
v = TestFunction(V)
M = FunctionSpace(mesh, "CG", 1)  
TT = TensorFunctionSpace(mesh,'P',1)
shape = 4*(mesh.geometry().dim(),)
TT_4 = TensorFunctionSpace(mesh,'P',1,shape = shape)
I = Identity(3)
dof = vertex_to_dof_map(M)
dof2vtx = vertex_to_dof_map(M).argsort()
# Boundary definition
boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
bottom  = AutoSubDomain(lambda x: near(x[2], -0.3972))
top = AutoSubDomain(lambda x: near(x[2], 0))
bottom.mark(boundary_parts, 1)
top.mark(boundary_parts, 2)
dx = Measure("dx",mesh)
ds1 = Measure("ds", mesh,subdomain_data=boundary_parts, subdomain_id=2)
bc0 = DirichletBC(V.sub(0), Constant(0), boundary_parts, 1)
bc1 = DirichletBC(V.sub(1), Constant(0), boundary_parts, 1)
bc2 = DirichletBC(V.sub(2), Constant(0), boundary_parts, 1)
bcs = [bc0,bc1,bc2] 
normal_vector = FacetNormal(mesh)
traction1 = Constant((0.0, 0.0, -0.1))
#%% Synthetic Umeas  
E_target = Function(M)
nu_target = Function(M)
u_meas = Function(V)
# Set target parameter value by Expression
E_top = 10; E_bottom = 20
E_target = interpolate(Expression('x[2]>-0.19635? E_top:E_bottom',degree=1,E_top=E_top,E_bottom=E_bottom), M)
nu_top = 0.3; nu_bottom = 0.3
nu_target = interpolate(Expression('x[2]>-0.19635? nu_top:nu_bottom',degree=1,nu_top=nu_top,nu_bottom=nu_bottom), M)
u_meas.assign(forward(E_target,nu_target,traction1,bcs))
F_meas=I+grad(u_meas)
C_meas=(F_meas.T)*F_meas
E_meas=(C_meas-I)/2
E_meas_proj =  project(E_meas,TT)
E_meas_array = np.array(E_meas_proj.vector()).reshape([-1,9])
#%% Intermediate configurations 
nu_o = 0.2
E_o = 15
E_gue_old = E_o*np.ones([num_vertices,1]).flatten()
E_gue = []
nu_gue_old = nu_o*np.ones([num_vertices,1]).flatten()
nu_gue = []
initial_name = 'E_o{:.2f}_nu_o{:.2f}'.format(E_o,nu_o)
error_list = []
MAX_ITER = 100; tol = 1e-6
for index in dof2vtx:
    E_gue.append(E_gue_old[index])
    nu_gue.append(nu_gue_old[index])

start = time.time()
VFM(E_gue,nu_gue)
end = time.time()
print('TOTAL TIME CONSUMING:{}'.format(end-start))
#%% post_plot
post_plot_nodal(case_name,initial_name,error_list)
