//
//  elastodynamics.cpp
//  elastodynamics
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

#include "loaders/loader.hpp" // candidate to delete

// application common sources
#include "../common/display_settings.hpp"
#include "../common/fitted_geometry_builders.hpp"
#include "../common/vec_analytic_functions.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

void IHHOSecondOrder(int argc, char **argv);

void HHOSecondOrderExample(int argc, char **argv);

void HHOFirstOrderExample(int argc, char **argv);


int main(int argc, char **argv)
{

    IHHOSecondOrder(argc, argv);
    
    // Examples solving the vector laplacian with optimal HHO convergence properties
//    HHOFirstOrderExample(argc, argv);
//    HHOSecondOrderExample(argc, argv);
    
    return 0;
}

void IHHOSecondOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();

    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    std::string filename = "mesh.txt";
    typedef disk::cartesian_mesh<RealType, 2>   mesh_type;
    typedef disk::BoundaryConditions<mesh_type, false> boundary_type;
    mesh_type msh;
    disk::cartesian_mesh_loader<RealType, 2> loader;
    if (!loader.read_mesh(filename))
    {
        std::cout << "Problem loading mesh." << std::endl;
        return;
    }
    loader.populate_mesh(msh);
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;

    // Final time value 1.0
    std::vector<size_t> nt_v = {10,20,40,80,160,320,640};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625};
    int tref = 3;
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti     = 0.0;

    vec_analytic_functions functions;
    functions.set_function_type(vec_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
    RealType t = ti;
    auto exact_vec_fun      = functions.Evaluate_u(t);
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_accel_fun    = functions.Evaluate_a(t);
    auto exact_flux_fun     = functions.Evaluate_sigma(t);

    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vec_fun); // easy because boundary assumes zero every where any time.

    tc.tic();
    auto assembler = elastodynamic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
    assembler.load_material_data(msh);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler created: " << tc.to_double() << " seconds" << reset << std::endl;

    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;

    // Projecting initial displacement, velocity and acceleration
    tc.tic();
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, u_dof_n, exact_vec_fun);
    assembler.project_over_faces(msh, u_dof_n, exact_vec_fun);
    
    assembler.project_over_cells(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_faces(msh, v_dof_n, exact_vel_fun);
    
    assembler.project_over_cells(msh, a_dof_n, exact_accel_fun);
    assembler.project_over_faces(msh, a_dof_n, exact_accel_fun);
    tc.toc();
    std::cout << bold << cyan << "Initialization completed: " << tc << " seconds" << reset << std::endl;
    
    size_t it = 0;
    std::string silo_file_name = "vec_";
    postprocessor<mesh_type>::write_silo_one_field_vectorial(silo_file_name, it, msh, hho_di, u_dof_n, exact_vec_fun, false);
    
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_vec_fun      = functions.Evaluate_u(t);
            auto exact_flux_fun     = functions.Evaluate_sigma(t);
            auto rhs_fun            = functions.Evaluate_f(t);

            assembler.assemble(msh, rhs_fun);

            // Compute intermediate state for scalar and rate
            u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*u_dof_n;
            
            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += assembler.MASS;
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
            
            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update displacement, velocity and acceleration
            u_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;

            std::string silo_file_name = "vec_";
            postprocessor<mesh_type>::write_silo_one_field_vectorial(silo_file_name, it, msh, hho_di, u_dof_n, exact_vec_fun, false);

            if(it == nt){
                auto assembler_c = one_field_vectorial_assembler<mesh_type>(msh, hho_di, bnd);
                postprocessor<mesh_type>::compute_errors_one_field_vectorial(msh, hho_di, assembler_c, u_dof_n, exact_vec_fun, exact_flux_fun);
            }

        }
        std::cout << green << "Number of time steps =  " << nt << reset << std::endl;
        std::cout << green << "Step size =  " << dt << reset << std::endl;
    }
    
}
#define quadratic_space_solution_Q
void HHOFirstOrderExample(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    std::string filename = "mesh.txt";
    typedef disk::cartesian_mesh<RealType, 2>   mesh_type;
    typedef disk::BoundaryConditions<mesh_type, false> boundary_type;
    mesh_type msh;
    disk::cartesian_mesh_loader<RealType, 2> loader;
    if (!loader.read_mesh(filename))
    {
        std::cout << "Problem loading mesh." << std::endl;
        return;
    }
    loader.populate_mesh(msh);
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Manufactured solution
#ifdef quadratic_space_solution_Q

    auto exact_vec_fun = [](const point<RealType, 2>& p) -> static_vector<RealType, 2> {
        RealType x,y;
        x = p.x();
        y = p.y();
        RealType ux = (1 - x)*x*(1 - y)*y;
        RealType uy = (1 - x)*x*(1 - y)*y;
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto exact_flux_fun = [](const mesh_type::point_type& p) -> static_matrix<RealType,2,2> {
        RealType x,y;
        x = p.x();
        y = p.y();
        static_matrix<RealType, 2, 2> sigma = static_matrix<RealType,2,2>::Zero(2,2);
        RealType sxx = 2*(1 - x)*(1 - y)*y - 2*x*(1 - y)*y + (2*(1 - x)*x*(1 - y) - 2*(1 - x)*x*y)/2. + (2*(1 - x)*(1 - y)*y - 2*x*(1 - y)*y)/2.;
        RealType sxy = (1 - x)*x*(1 - y) - (1 - x)*x*y + (1 - x)*(1 - y)*y - x*(1 - y)*y;
        RealType syy = 2*(1 - x)*x*(1 - y) - 2*(1 - x)*x*y + (2*(1 - x)*x*(1 - y) - 2*(1 - x)*x*y)/2. + (2*(1 - x)*(1 - y)*y - 2*x*(1 - y)*y)/2.;
        sigma(0,0) = sxx;
        sigma(0,1) = sxy;
        sigma(1,0) = sxy;
        sigma(1,1) = syy;
        return sigma;
    };
    
    auto rhs_fun = [](const mesh_type::point_type& p) -> static_vector<RealType, 2> {
        RealType x,y;
        x = p.x();
        y = p.y();
        RealType fx = 2*(1 + x*x + y*(-5 + 3*y) + x*(-3 + 4*y));
        RealType fy = 2*(1 + 3*x*x + (-3 + y)*y + x*(-5 + 4*y));
        return static_vector<RealType, 2>{-fx, -fy};
    };

#else

    auto exact_vec_fun = [](const point<RealType, 2>& p) -> static_vector<RealType, 2> {
        RealType ux = std::sin(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
        RealType uy = std::sin(3.0 * M_PI * p.x()) * std::sin(3.0 * M_PI * p.y());
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto exact_flux_fun = [](const mesh_type::point_type& p) -> static_matrix<RealType,2,2> {
        static_matrix<RealType, 2, 2> sigma = static_matrix<RealType,2,2>::Zero(2,2);
        RealType sxx = 3.0 * M_PI * std::sin(3.0 * M_PI * p.x()) * std::cos(3.0 * M_PI * p.y())
                     + 6.0 * M_PI * std::cos(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
        RealType sxy = 2.0 * M_PI * std::sin(2.0 * M_PI * p.x()) * std::cos(2.0 * M_PI * p.y())
                     + 3.0 * M_PI * std::cos(3.0 * M_PI * p.x()) * std::sin(3.0 * M_PI * p.y());
        RealType syy = 9.0 * M_PI * std::sin(3.0 * M_PI * p.x()) * std::cos(3.0 * M_PI * p.y())
                     + 2.0 * M_PI * std::cos(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
        sigma(0,0) = sxx;
        sigma(0,1) = sxy;
        sigma(1,0) = sxy;
        sigma(1,1) = syy;
        return sigma;
    };
    
    auto rhs_fun = [](const mesh_type::point_type& p) -> static_vector<RealType, 2> {
        RealType fx = 2.0 * M_PI * M_PI * (
                      9.0 * std::cos(3.0 * M_PI * p.x()) * std::cos(3.0 * M_PI * p.y())
                    - 8.0 * std::sin(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y()));
        RealType fy = 4.0 * M_PI * M_PI * (
                      2.0 * std::cos(2.0 * M_PI * p.x()) * std::cos(2.0 * M_PI * p.y())
                    - 9.0 * std::sin(3.0 * M_PI * p.x()) * std::sin(3.0 * M_PI * p.y()));
        return static_vector<RealType, 2>{-fx, -fy};
    };

#endif

    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);


    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vec_fun);
    tc.tic();
    auto assembler = three_fields_vectorial_assembler<mesh_type>(msh, hho_di, bnd);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(assembler.LHS);
    analysis_t.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> x_dof = analysis_t.solve(assembler.RHS); // new state
    tc.toc();
    std::cout << bold << cyan << "Number of equations : " << assembler.RHS.rows() << reset << std::endl;
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Computing errors
    postprocessor<mesh_type>::compute_errors_three_fields_vectorial(msh, hho_di, x_dof, exact_vec_fun, exact_flux_fun);
    
    size_t it = 0;
    std::string silo_file_name = "vec_mixed_";
    postprocessor<mesh_type>::write_silo_three_fields_vectorial(silo_file_name, it, msh, hho_di, x_dof, exact_vec_fun, exact_flux_fun, false);
    
}

void HHOSecondOrderExample(int argc, char **argv){

    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    std::string filename = "mesh.txt";
    typedef disk::cartesian_mesh<RealType, 2>   mesh_type;
    typedef disk::BoundaryConditions<mesh_type, false> boundary_type;
    mesh_type msh;
    disk::cartesian_mesh_loader<RealType, 2> loader;
    if (!loader.read_mesh(filename))
    {
        std::cout << "Problem loading mesh." << std::endl;
        return;
    }
    loader.populate_mesh(msh);
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Manufactured solution
#ifdef quadratic_space_solution_Q

    auto exact_vec_fun = [](const point<RealType, 2>& p) -> static_vector<RealType, 2> {
        RealType x,y;
        x = p.x();
        y = p.y();
        RealType ux = (1 - x)*x*(1 - y)*y;
        RealType uy = (1 - x)*x*(1 - y)*y;
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto exact_flux_fun = [](const mesh_type::point_type& p) -> static_matrix<RealType,2,2> {
        RealType x,y;
        x = p.x();
        y = p.y();
        static_matrix<RealType, 2, 2> sigma = static_matrix<RealType,2,2>::Zero(2,2);
        RealType sxx = 2*(1 - x)*(1 - y)*y - 2*x*(1 - y)*y + (2*(1 - x)*x*(1 - y) - 2*(1 - x)*x*y)/2. + (2*(1 - x)*(1 - y)*y - 2*x*(1 - y)*y)/2.;
        RealType sxy = (1 - x)*x*(1 - y) - (1 - x)*x*y + (1 - x)*(1 - y)*y - x*(1 - y)*y;
        RealType syy = 2*(1 - x)*x*(1 - y) - 2*(1 - x)*x*y + (2*(1 - x)*x*(1 - y) - 2*(1 - x)*x*y)/2. + (2*(1 - x)*(1 - y)*y - 2*x*(1 - y)*y)/2.;
        sigma(0,0) = sxx;
        sigma(0,1) = sxy;
        sigma(1,0) = sxy;
        sigma(1,1) = syy;
        return sigma;
    };
    
    auto rhs_fun = [](const mesh_type::point_type& p) -> static_vector<RealType, 2> {
        RealType x,y;
        x = p.x();
        y = p.y();
        RealType fx = 2*(1 + x*x + y*(-5 + 3*y) + x*(-3 + 4*y));
        RealType fy = 2*(1 + 3*x*x + (-3 + y)*y + x*(-5 + 4*y));
        return static_vector<RealType, 2>{-fx, -fy};
    };

#else

    auto exact_vec_fun = [](const point<RealType, 2>& p) -> static_vector<RealType, 2> {
        RealType ux = std::sin(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
        RealType uy = std::sin(3.0 * M_PI * p.x()) * std::sin(3.0 * M_PI * p.y());
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto exact_flux_fun = [](const mesh_type::point_type& p) -> static_matrix<RealType,2,2> {
        static_matrix<RealType, 2, 2> sigma = static_matrix<RealType,2,2>::Zero(2,2);
        RealType sxx = 3.0 * M_PI * std::sin(3.0 * M_PI * p.x()) * std::cos(3.0 * M_PI * p.y())
                     + 6.0 * M_PI * std::cos(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
        RealType sxy = 2.0 * M_PI * std::sin(2.0 * M_PI * p.x()) * std::cos(2.0 * M_PI * p.y())
                     + 3.0 * M_PI * std::cos(3.0 * M_PI * p.x()) * std::sin(3.0 * M_PI * p.y());
        RealType syy = 9.0 * M_PI * std::sin(3.0 * M_PI * p.x()) * std::cos(3.0 * M_PI * p.y())
                     + 2.0 * M_PI * std::cos(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
        sigma(0,0) = sxx;
        sigma(0,1) = sxy;
        sigma(1,0) = sxy;
        sigma(1,1) = syy;
        return sigma;
    };
    
    auto rhs_fun = [](const mesh_type::point_type& p) -> static_vector<RealType, 2> {
        RealType fx = 2.0 * M_PI * M_PI * (
                      9.0 * std::cos(3.0 * M_PI * p.x()) * std::cos(3.0 * M_PI * p.y())
                    - 8.0 * std::sin(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y()));
        RealType fy = 4.0 * M_PI * M_PI * (
                      2.0 * std::cos(2.0 * M_PI * p.x()) * std::cos(2.0 * M_PI * p.y())
                    - 9.0 * std::sin(3.0 * M_PI * p.x()) * std::sin(3.0 * M_PI * p.y()));
        return static_vector<RealType, 2>{-fx, -fy};
    };

#endif

    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);


    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vec_fun);
    tc.tic();
    auto assembler = one_field_vectorial_assembler<mesh_type>(msh, hho_di, bnd);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(assembler.LHS);
    analysis_t.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> x_dof = analysis_t.solve(assembler.RHS); // new state
    tc.toc();
    std::cout << bold << cyan << "Number of equations : " << assembler.RHS.rows() << reset << std::endl;
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Computing errors
    postprocessor<mesh_type>::compute_errors_one_field_vectorial(msh, hho_di, assembler, x_dof, exact_vec_fun, exact_flux_fun);
    
    size_t it = 0;
    std::string silo_file_name = "vec_";
    postprocessor<mesh_type>::write_silo_one_field_vectorial(silo_file_name, it, msh, hho_di, x_dof, exact_vec_fun, false);
}
