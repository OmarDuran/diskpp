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
#include "../common/acoustic_material_data.hpp"
#include "../common/scal_analytic_functions.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

// implicit RK schemes
#include "../common/dirk_hho_scheme.hpp"
#include "../common/dirk_butcher_tableau.hpp"

// explicit RK schemes
#include "../common/ssprk_hho_scheme.hpp"
#include "../common/ssprk_shu_osher_tableau.hpp"

void HeterogeneousPulseIHHOFirstOrder(int argc, char **argv);

void HeterogeneousPulseIHHOSecondOrder(int argc, char **argv);

void HeterogeneousEHHOFirstOrder(int argc, char **argv);

void EHHOFirstOrder(int argc, char **argv);

void HeterogeneousIHHOFirstOrder(int argc, char **argv);

void IHHOFirstOrder(int argc, char **argv);

void HeterogeneousIHHOSecondOrder(int argc, char **argv);

void IHHOSecondOrder(int argc, char **argv);

void HHOOneFieldConvergenceExample(int argc, char **argv);

void HHOTwoFieldsConvergenceExample(int argc, char **argv);

int main(int argc, char **argv)
{
    
//    HeterogeneousPulseIHHOFirstOrder(argc, argv);
//
//    HeterogeneousPulseIHHOSecondOrder(argc, argv);
//
//    HeterogeneousEHHOFirstOrder(argc, argv);
//
    HeterogeneousIHHOFirstOrder(argc, argv);
//
//    HeterogeneousIHHOSecondOrder(argc, argv);

//    EHHOFirstOrder(argc, argv);
//
//    IHHOFirstOrder(argc, argv);
    
//    IHHOSecondOrder(argc, argv);
    
//    // Examples using main app objects for solving the laplacian with optimal convergence rates
//    // Primal HHO
//    HHOOneFieldConvergenceExample(argc, argv);
//
//    // Dual HHO
//    HHOTwoFieldsConvergenceExample(argc, argv);
    
    return 0;
}

void HHOOneFieldConvergenceExample(int argc, char **argv){

    using RealType = double;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    
    simulation_data sim_data = preprocessor::process_convergence_test_args(argc, argv);
    sim_data.print_simulation_data();

    // Manufactured exact solution
    bool quadratic_function_Q = sim_data.m_quadratic_function_Q;
    auto exact_scal_fun = [quadratic_function_Q](const mesh_type::point_type& pt) -> RealType {
        if(quadratic_function_Q){
            return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
        }else{
            return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        }
        
    };

    auto exact_flux_fun = [quadratic_function_Q](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        if(quadratic_function_Q){
            flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
            flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
            return flux;
        }else{
            flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
            flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
            return flux;
        }

    };

    auto rhs_fun = [quadratic_function_Q](const typename mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        if(quadratic_function_Q){
            return -2.0*((x - 1)*x + (y - 1)*y);
        }else{
            return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        }
    };

    // simple material
    RealType rho = 1.0;
    RealType vp = 1.0;
    acoustic_material_data<RealType> material(rho,vp);
    
    std::ofstream error_file("steady_scalar_error.txt");
    
    for(size_t k = 0; k <= sim_data.m_k_degree; k++){
        std::cout << bold << cyan << "Running an approximation with k : " << k << reset << std::endl;
        error_file << "Approximation with k : " << k << std::endl;
        for(size_t l = 0; l <= sim_data.m_n_divs; l++){
            
            // Building a cartesian mesh
            timecounter tc;
            tc.tic();
            RealType lx = 1.0;
            RealType ly = 1.0;
            size_t nx = 2;
            size_t ny = 2;
            mesh_type msh;
            
            cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
            mesh_builder.refine_mesh(l);
            mesh_builder.build_mesh();
            mesh_builder.move_to_mesh_storage(msh);
            std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Creating HHO approximation spaces and corresponding linear operator
            size_t cell_k_degree = k;
            if(sim_data.m_hdg_stabilization_Q){
                cell_k_degree++;
            }
            disk::hho_degree_info hho_di(cell_k_degree,k);

            // Solving a scalar primal HHO problem
            boundary_type bnd(msh);
            bnd.addDirichletEverywhere(exact_scal_fun);
            tc.tic();
            auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
            if(sim_data.m_hdg_stabilization_Q){
                assembler.set_hdg_stabilization();
            }
            assembler.load_material_data(msh,material);
            assembler.assemble(msh, rhs_fun);
            assembler.apply_bc(msh);
            tc.toc();
            std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Solving LS
            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            Matrix<RealType, Dynamic, 1> x_dof = analysis.solve(assembler.RHS);
            tc.toc();
            std::cout << bold << cyan << "Number of equations : " << assembler.RHS.rows() << reset << std::endl;
            std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Computing errors
            postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, x_dof, exact_scal_fun, exact_flux_fun,error_file);
            
            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "steady_scalar_k" + std::to_string(k) + "_";
                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, l, msh, hho_di, x_dof, exact_scal_fun, false);
            }
        }
        error_file << std::endl << std::endl;
    }
    error_file.close();
}

void HHOTwoFieldsConvergenceExample(int argc, char **argv){

    using RealType = double;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    
    simulation_data sim_data = preprocessor::process_convergence_test_args(argc, argv);
    sim_data.print_simulation_data();

    // Manufactured exact solution
    bool quadratic_function_Q = sim_data.m_quadratic_function_Q;
    auto exact_scal_fun = [quadratic_function_Q](const mesh_type::point_type& pt) -> RealType {
        if(quadratic_function_Q){
            return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
        }else{
            return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        }
        
    };

    auto exact_flux_fun = [quadratic_function_Q](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        if(quadratic_function_Q){
            flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
            flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
            return flux;
        }else{
            flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
            flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
            return flux;
        }

    };

    auto rhs_fun = [quadratic_function_Q](const typename mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        if(quadratic_function_Q){
            return -2.0*((x - 1)*x + (y - 1)*y);
        }else{
            return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        }
    };

    // simple material
    RealType rho = 1.0;
    RealType vp = 1.0;
    acoustic_material_data<RealType> material(rho,vp);
    
    std::ofstream error_file("steady_scalar_mixed_error.txt");
    
    for(size_t k = 0; k <= sim_data.m_k_degree; k++){
        std::cout << bold << cyan << "Running an approximation with k : " << k << reset << std::endl;
        error_file << "Approximation with k : " << k << std::endl;
        for(size_t l = 0; l <= sim_data.m_n_divs; l++){
            
            // Building a cartesian mesh
            timecounter tc;
            tc.tic();
            RealType lx = 1.0;
            RealType ly = 1.0;
            size_t nx = 2;
            size_t ny = 2;
            mesh_type msh;
            
            cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
            mesh_builder.refine_mesh(l);
            mesh_builder.build_mesh();
            mesh_builder.move_to_mesh_storage(msh);
            std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Creating HHO approximation spaces and corresponding linear operator
            size_t cell_k_degree = k;
            if(sim_data.m_hdg_stabilization_Q){
                cell_k_degree++;
            }
            disk::hho_degree_info hho_di(cell_k_degree,k);

            // Solving a scalar primal HHO problem
            boundary_type bnd(msh);
            bnd.addDirichletEverywhere(exact_scal_fun);
            tc.tic();
            auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
            if(sim_data.m_hdg_stabilization_Q){
                assembler.set_hdg_stabilization();
            }
            assembler.load_material_data(msh,material);
            assembler.assemble(msh, rhs_fun);
            assembler.assemble_mass(msh, false);
            assembler.apply_bc(msh);
            tc.toc();
            std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Solving LS
            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS+assembler.MASS);
            analysis.factorize(assembler.LHS+assembler.MASS);
            Matrix<RealType, Dynamic, 1> x_dof = analysis.solve(assembler.RHS); // new state
            tc.toc();
            std::cout << bold << cyan << "Number of equations : " << assembler.RHS.rows() << reset << std::endl;
            std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_scal_fun, exact_flux_fun, error_file);
            
            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "steady_scalar_mixed_k" + std::to_string(k) + "_";
                postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, l, msh, hho_di, x_dof, exact_scal_fun, exact_flux_fun, false);
            }
        }
        error_file << std::endl << std::endl;
    }
    error_file.close();
}

void IHHOSecondOrder(int argc, char **argv){

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
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
    RealType t = ti;
    auto exact_scal_fun     = functions.Evaluate_u(t);
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_accel_fun    = functions.Evaluate_a(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_scal_fun); // easy because boundary assumes zero every where any time.
    tc.tic();
    auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
    
    // simple material
    RealType rho = 1.0;
    RealType vp = 1.0;
    acoustic_material_data<RealType> material(rho,vp);
    assembler.load_material_data(msh,material);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, p_dof_n, exact_scal_fun);
    assembler.project_over_cells(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_cells(msh, a_dof_n, exact_accel_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "scalar_";
        postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_one_field.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
    }
    
    bool standar_Q = true;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 1.5;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        tc.tic();
        assembler.assemble(msh, rhs_fun);
        SparseMatrix<double> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        SparseLU<SparseMatrix<RealType>> analysis;
        analysis.analyzePattern(assembler.LHS);
        analysis.factorize(assembler.LHS);
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
        
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_fun     = functions.Evaluate_u(t);
            auto exact_flux_fun     = functions.Evaluate_q(t);
            auto rhs_fun            = functions.Evaluate_f(t);
            
            tc.tic();
            assembler.get_bc_conditions().updateDirichletFunction(exact_scal_fun, 0);
            assembler.assemble_rhs(msh, rhs_fun);

            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = Kg*p_dof_n;

            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Rhs assembly completed: " << tc << " seconds" << reset << std::endl;

            tc.tic();
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
            
            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "scalar_";
                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
            }
            
            if (sim_data.m_report_energy_Q) {
                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
            }
            
            if(it == nt){
                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, p_dof_n, exact_scal_fun, exact_flux_fun, simulation_log);
            }
            
        }
        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
}

void HeterogeneousIHHOSecondOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.2;
    size_t nx = 10;
    size_t ny = 1;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh_x_direction(sim_data.m_n_divs);
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);

    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.5
    size_t nt = 5;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 0.5;
    RealType dt     = tf/nt;

    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_scal_fun     = functions.Evaluate_u(t);
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_accel_fun    = functions.Evaluate_a(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_scal_fun); // easy because boundary assumes zero every where any time.
    tc.tic();
    auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        rho = 1.0;
        if (x < 0.5) {
            vp = 10.0;
        }else{
            vp = 1.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, p_dof_n, exact_scal_fun);
    assembler.project_over_faces(msh, p_dof_n, exact_scal_fun);
    assembler.project_over_cells(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_faces(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_cells(msh, a_dof_n, exact_accel_fun);
    assembler.project_over_faces(msh, a_dof_n, exact_accel_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "scalar_";
        postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_one_field.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
    }
    
    bool standar_Q = true;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 1.5;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        tc.tic();
        assembler.assemble(msh, rhs_fun);
        SparseMatrix<double> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        SparseLU<SparseMatrix<RealType>> analysis;
        analysis.analyzePattern(assembler.LHS);
        analysis.factorize(assembler.LHS);
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
        
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_fun     = functions.Evaluate_u(t);
            auto exact_flux_fun     = functions.Evaluate_q(t);
            auto rhs_fun            = functions.Evaluate_f(t);

            assembler.get_bc_conditions().updateDirichletFunction(exact_scal_fun, 0);
            assembler.assemble_rhs(msh, rhs_fun);

            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = Kg*p_dof_n;

            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Rhs assembly completed: " << tc << " seconds" << reset << std::endl;

            tc.tic();
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
            
            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "scalar_";
                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
            }
            
            if (sim_data.m_report_energy_Q) {
                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
            }
            
            if(it == nt){
                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, p_dof_n, exact_scal_fun, exact_flux_fun, simulation_log);
            }
            
        }
        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
}

void IHHOFirstOrder(int argc, char **argv){
    
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
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
    RealType t = ti;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vel_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    
    // simple material
    RealType rho = 1.0;
    RealType vp = 1.0;
    acoustic_material_data<RealType> material(rho,vp);
    assembler.load_material_data(msh,material);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, exact_vel_fun, exact_flux_fun);
    assembler.project_over_faces(msh, x_dof, exact_vel_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_two_fields.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    int s = 3;
    bool is_sdirk_Q = false;
    
    if (is_sdirk_Q) {
        dirk_butcher_tableau::sdirk_tables(s, a, b, c);
    }else{
        dirk_butcher_tableau::dirk_tables(s, a, b, c);
    }
    
    tc.tic();
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
    dirk_hho_scheme dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (is_sdirk_Q) {
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }
    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        RealType tn = dt*(it-1)+ti;
        
        // DIRK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<double, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {
                
                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }
                
                {
                    RealType t = tn + c(i,0) * dt;
                    auto exact_vel_fun      = functions.Evaluate_v(t);
                    auto rhs_fun            = functions.Evaluate_f(t);
                    assembler.get_bc_conditions().updateDirichletFunction(exact_vel_fun, 0);
                    assembler.assemble_rhs(msh, rhs_fun);
                    dirk_an.SetFg(assembler.RHS);
                    dirk_an.irk_weight(yn, ki, dt, a(i,i),is_sdirk_Q);
                }

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "DIRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;
        
        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }
        
        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }

    }
    
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}

void HeterogeneousIHHOFirstOrder(int argc, char **argv){
    
    // An explicit pseudo-energy conserving time-integration scheme for Hamiltonian dynamics
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.2;
    size_t nx = 10;
    size_t ny = 1;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh_x_direction(sim_data.m_n_divs);
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.5
    size_t nt = 5;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 0.5;
    RealType dt     = (tf-ti)/nt;
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vel_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        if (x < 0.5) {
            vp = 10.0;
        }else{
            vp = 1.0;
        }
        rho = 1.0/(vp*vp); // this is required to make both formulation compatible keeping kappa = 1
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    std::string silo_file_props_name = "properties_map";
    postprocessor<mesh_type>::write_silo_acoustic_property_map(silo_file_props_name, msh, assembler.get_material_data());
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, exact_vel_fun, exact_flux_fun);
    assembler.project_over_faces(msh, x_dof, exact_vel_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_two_fields.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    int s = 3;
    bool is_sdirk_Q = true;
    
    if (is_sdirk_Q) {
        dirk_butcher_tableau::sdirk_tables(s, a, b, c);
    }else{
        dirk_butcher_tableau::dirk_tables(s, a, b, c);
    }
    
    tc.tic();
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
    dirk_hho_scheme dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (is_sdirk_Q) {
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }
    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        RealType tn = dt*(it-1)+ti;
        
        // DIRK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<double, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {
                
                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }
                
                {
                    RealType t = tn + c(i,0) * dt;
                    auto exact_vel_fun      = functions.Evaluate_v(t);
                    auto rhs_fun            = functions.Evaluate_f(t);
                    assembler.get_bc_conditions().updateDirichletFunction(exact_vel_fun, 0);
                    assembler.assemble_rhs(msh, rhs_fun);
                    dirk_an.SetFg(assembler.RHS);
                    dirk_an.irk_weight(yn, ki, dt, a(i,i),is_sdirk_Q);
                }

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "DIRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;
        
        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }
        
        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }

    }
    
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}


void EHHOFirstOrder(int argc, char **argv){
    
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
    RealType dt     = tf/nt;
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
    RealType t = ti;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vel_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    assembler.load_material_data(msh);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, exact_vel_fun, exact_flux_fun);
    assembler.project_over_faces(msh, x_dof, exact_vel_fun);
    
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_two_fields_explicit.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    int s = 3;
    Matrix<double, Dynamic, Dynamic> alpha;
    Matrix<double, Dynamic, Dynamic> beta;
    ssprk_shu_osher_tableau::OSSPRKSS(s, alpha, beta);

    tc.tic();
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness and rhs assembly completed: " << tc << " seconds" << reset << std::endl;
    size_t n_face_dof = assembler.get_n_face_dof();
    ssprk_hho_scheme ssprk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
    tc.toc();

    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        
        {   // Updating rhs
            RealType t = dt*(it)+ti;
            auto rhs_fun            = functions.Evaluate_f(t);
            assembler.assemble_rhs(msh, rhs_fun);
            ssprk_an.SetFg(assembler.RHS);
        }
        RealType tn = dt*(it-1)+ti;
        tc.tic();
        {
            
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);
        
            Matrix<double, Dynamic, 1> yn, ysi, yj;
            ys.block(0, 0, n_dof, 1) = x_dof;
            for (int i = 0; i < s; i++) {
        
                ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
                for (int j = 0; j <= i; j++) {
                    yn = ys.block(0, j, n_dof, 1);
                    ssprk_an.explicit_rk_weight(yn, yj, dt, alpha(i,j), beta(i,j));
                    ysi += yj;
                }
                ys.block(0, i+1, n_dof, 1) = ysi;
            }
        
            x_dof_n = ys.block(0, s, n_dof, 1);
        }
        tc.toc();
        std::cout << bold << cyan << "SSPRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;

        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }

        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }
    }
    
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of SSPRKSS steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
}

void HeterogeneousEHHOFirstOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.25;
    size_t nx = 10;
    size_t ny = 1;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh_x_direction(sim_data.m_n_divs);
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.5
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 0.5;
    RealType dt     = (tf-ti)/nt;
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vel_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);

    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        rho = 1.0;
        if (x < 0.5) {
            vp = 2.0;
        }else{
            vp = 1.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, exact_vel_fun, exact_flux_fun);
    assembler.project_over_faces(msh, x_dof, exact_vel_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_two_fields_explicit.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    int s = 3;
    Matrix<double, Dynamic, Dynamic> alpha;
    Matrix<double, Dynamic, Dynamic> beta;
    ssprk_shu_osher_tableau::OSSPRKSS(s, alpha, beta);

    tc.tic();
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness and rhs assembly completed: " << tc << " seconds" << reset << std::endl;
    size_t n_face_dof = assembler.get_n_face_dof();
    ssprk_hho_scheme ssprk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
    tc.toc();

    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        
        {   // Updating rhs
            RealType t = dt*(it)+ti;
            auto rhs_fun            = functions.Evaluate_f(t);
            assembler.assemble_rhs(msh, rhs_fun);
            ssprk_an.SetFg(assembler.RHS);
        }
        RealType tn = dt*(it-1)+ti;
        tc.tic();
        {
            
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);
        
            Matrix<double, Dynamic, 1> yn, ysi, yj;
            ys.block(0, 0, n_dof, 1) = x_dof;
            for (int i = 0; i < s; i++) {
        
                ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
                for (int j = 0; j <= i; j++) {
                    yn = ys.block(0, j, n_dof, 1);
                    ssprk_an.explicit_rk_weight(yn, yj, dt, alpha(i,j), beta(i,j));
                    ysi += yj;
                }
                ys.block(0, i+1, n_dof, 1) = ysi;
            }
        
            x_dof_n = ys.block(0, s, n_dof, 1);
        }
        tc.toc();
        std::cout << bold << cyan << "SSPRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;

        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }

        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }
    }
    
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of SSPRKSS steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}


void HeterogeneousPulseIHHOFirstOrder(int argc, char **argv){
    
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
    RealType tf = 0.25;
    RealType dt     = tf/nt;
    
    auto null_fun = [](const mesh_type::point_type& pt) -> RealType {
            RealType x,y;
            x = pt.x();
            y = pt.y();
            return 0.0;
    };
    
    auto null_flux_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        return {0,0};
    };
    
    auto vel_fun = [](const mesh_type::point_type& pt) -> RealType {
            RealType x,y,xc,yc,r,wave;
            x = pt.x();
            y = pt.y();
            xc = 0.5;
            yc = 0.25;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = 0.1*(-4*std::sqrt(10.0/3.0)*(-1 + 1600.0*r*r))/(std::exp(800*r*r)*std::pow(M_PI,0.25));
            return wave;
    };
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(null_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        rho = 1.0;
        if (y < 0.5) {
            vp = 10.0;
        }else{
            vp = 1.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, vel_fun, null_flux_fun);
    assembler.project_over_faces(msh, x_dof, vel_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "inhomogeneous_scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, vel_fun, null_flux_fun, false);
    }
    
    std::ofstream simulation_log("inhomogeneous_acoustic_two_fields.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, ti, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    int s = 2;
    bool is_sdirk_Q = true;
    
    if (is_sdirk_Q) {
        dirk_butcher_tableau::sdirk_tables(s, a, b, c);
    }else{
        dirk_butcher_tableau::dirk_tables(s, a, b, c);
    }
    
    tc.tic();
    assembler.assemble(msh, null_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
    dirk_hho_scheme dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (is_sdirk_Q) {
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }
    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        RealType tn = dt*(it-1)+ti;
        
        // DIRK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<double, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {
                
                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }
                
                dirk_an.irk_weight(yn, ki, dt, a(i,i),is_sdirk_Q);

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "DIRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;
        
        RealType t = tn + dt;
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "inhomogeneous_scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, vel_fun, null_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }

    }
    
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
}


void HeterogeneousPulseIHHOSecondOrder(int argc, char **argv){
    
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
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);

    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.5
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 0.25;
    RealType dt     = tf/nt;

    auto null_fun = [](const mesh_type::point_type& pt) -> RealType {
            RealType x,y;
            x = pt.x();
            y = pt.y();
            return 0.0;
    };
    
    auto vel_fun = [](const mesh_type::point_type& pt) -> RealType {
            RealType x,y,xc,yc,r,wave;
            x = pt.x();
            y = pt.y();
            xc = 0.5;
            yc = 0.25;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = 0.1*(-4*std::sqrt(10.0/3.0)*(-1 + 1600.0*r*r))/(std::exp(800*r*r)*std::pow(M_PI,0.25));
            return wave;
    };
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(null_fun); // easy because boundary assumes zero every where any time.
    tc.tic();
    auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        rho = 1.0;
        if (y < 0.5) {
            vp = 10.0;
        }else{
            vp = 1.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, p_dof_n, null_fun);
    assembler.project_over_faces(msh, p_dof_n, null_fun);
    assembler.project_over_cells(msh, v_dof_n, vel_fun);
    assembler.project_over_faces(msh, v_dof_n, vel_fun);
    assembler.project_over_cells(msh, a_dof_n, null_fun);
    assembler.project_over_faces(msh, a_dof_n, null_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "inhomogeneous_scalar_";
        postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, v_dof_n, vel_fun, false);
    }
    
    std::ofstream simulation_log("inhomogeneous_acoustic_one_field.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, ti, p_dof_n, v_dof_n, simulation_log);
    }
    
    bool standar_Q = false;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 0.5125;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        tc.tic();
        assembler.assemble(msh, null_fun);
        SparseMatrix<double> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        SparseLU<SparseMatrix<RealType>> analysis;
        analysis.analyzePattern(assembler.LHS);
        analysis.factorize(assembler.LHS);
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
        
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;

            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = Kg*p_dof_n;
            
            assembler.RHS.setZero();
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Rhs assembly completed: " << tc << " seconds" << reset << std::endl;

            tc.tic();
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
            
            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "inhomogeneous_scalar_";
                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, v_dof_n, vel_fun, false);
            }
            
            if (sim_data.m_report_energy_Q) {
                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
            }
            
        }
        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
}

