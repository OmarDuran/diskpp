//
//  acoustics.cpp
//  acoustics
//
//  Created by Omar Durán on 4/8/20.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <fstream>
#include <list>
#include <getopt.h>


#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
using namespace Eigen;

#include "timecounter.h"
#include "methods/hho"
#include "geometry/geometry.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "output/silo.hpp"

// application common sources
#include "../common/display_settings.hpp"
#include "../common/fitted_geometry_builders.hpp"
#include "../common/linear_solver.hpp"
#include "../common/acoustic_material_data.hpp"
#include "../common/elastic_material_data.hpp"
#include "../common/scal_vec_analytic_functions.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

// implicit RK schemes
#include "../common/dirk_hho_scheme.hpp"
#include "../common/dirk_butcher_tableau.hpp"


//void IHHOSecondOrder(int argc, char **argv);

int main(int argc, char **argv)
{
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 1.0;
    size_t nx = 2;
    size_t ny = 2;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt     = (tf-ti)/nt;
    
    scal_vec_analytic_functions functions;
    functions.set_function_type(scal_vec_analytic_functions::EFunctionType::EFunctionNonPolynomial);
    RealType t = ti;
    
    auto u_fun     = functions.Evaluate_u(t);
    auto v_fun     = functions.Evaluate_v(t);
    auto a_fun     = functions.Evaluate_a(t);
    auto f_fun      = functions.Evaluate_f(t);
    auto flux_fun     = functions.Evaluate_sigma(t);
    
    auto s_u_fun     = functions.Evaluate_s_u(t);
    auto s_v_fun     = functions.Evaluate_s_v(t);
    auto s_a_fun     = functions.Evaluate_s_a(t);
    auto s_f_fun     = functions.Evaluate_s_f(t);
    auto s_flux_fun     = functions.Evaluate_s_q(t);
    
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
//    bnd.addDirichletEverywhere(exact_scal_fun); // easy because boundary assumes zero every where any time.
    
//    tc.tic();
//    auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
//
//    // simple material
//    RealType rho = 1.0;
//    RealType vp = 1.0;
//    acoustic_material_data<RealType> material(rho,vp);
//    assembler.load_material_data(msh,material);
//    if(sim_data.m_hdg_stabilization_Q){
//        assembler.set_hdg_stabilization();
//    }
//    tc.toc();
//    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
//
//    tc.tic();
//    assembler.assemble_mass(msh);
//    tc.toc();
//    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
//
//    // Projecting initial scalar, velocity and acceleration
//    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
//    assembler.project_over_cells(msh, p_dof_n, exact_scal_fun);
//    assembler.project_over_cells(msh, v_dof_n, exact_vel_fun);
//    assembler.project_over_cells(msh, a_dof_n, exact_accel_fun);
//
//    if (sim_data.m_render_silo_files_Q) {
//        size_t it = 0;
//        std::string silo_file_name = "scalar_";
//        postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
//    }
//
//    std::ofstream simulation_log("acoustic_one_field.txt");
//
//    if (sim_data.m_report_energy_Q) {
//        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
//    }
//
//    bool standar_Q = true;
//    // Newmark process
//    {
//        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;
//
//        RealType beta = 0.25;
//        RealType gamma = 0.5;
//        if (!standar_Q) {
//            RealType kappa = 0.25;
//            gamma = 1.5;
//            beta = kappa*(gamma+0.5)*(gamma+0.5);
//        }
//
//        tc.tic();
//        assembler.assemble(msh, rhs_fun);
//        SparseMatrix<double> Kg = assembler.LHS;
//        assembler.LHS *= beta*(dt*dt);
//        assembler.LHS += assembler.MASS;
//        linear_solver<RealType> analysis;
//        if (sim_data.m_sc_Q) {
//            analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
//            analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
//        }else{
//            analysis.set_Kg(assembler.LHS);
//        }
//        analysis.set_iterative_solver(true);
//        analysis.factorize();
//        tc.toc();
//        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
//
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            // Manufactured solution
//            RealType t = dt*it+ti;
//            auto exact_scal_fun     = functions.Evaluate_u(t);
//            auto exact_flux_fun     = functions.Evaluate_q(t);
//            auto rhs_fun            = functions.Evaluate_f(t);
//
//            tc.tic();
//            assembler.get_bc_conditions().updateDirichletFunction(exact_scal_fun, 0);
//            assembler.assemble_rhs(msh, rhs_fun);
//
//            // Compute intermediate state for scalar and rate
//            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
//            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
//            Matrix<RealType, Dynamic, 1> res = Kg*p_dof_n;
//
//            assembler.RHS -= res;
//            tc.toc();
//            std::cout << bold << cyan << "Rhs assembly completed: " << tc << " seconds" << reset << std::endl;
//
//            tc.tic();
//            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//            // update scalar and rate
//            p_dof_n += beta*dt*dt*a_dof_np;
//            v_dof_n += gamma*dt*a_dof_np;
//            a_dof_n  = a_dof_np;
//
//            if (sim_data.m_render_silo_files_Q) {
//                std::string silo_file_name = "scalar_";
//                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
//            }
//
//            if (sim_data.m_report_energy_Q) {
//                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
//            }
//
//            if(it == nt){
//                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, p_dof_n, exact_scal_fun, exact_flux_fun, simulation_log);
//            }
//
//        }
//        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
//        simulation_log << "Number of time steps =  " << nt << std::endl;
//        simulation_log << "Step size =  " << dt << std::endl;
//        simulation_log.flush();
//    }
    
    return 0;
}
