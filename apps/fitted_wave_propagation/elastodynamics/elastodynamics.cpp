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
#include "../common/scal_analytic_functions.hpp"
#include "../common/one_field_assembler.hpp" // to deletion
#include "../common/one_field_vectorial_assembler.hpp"
#include "../common/two_fields_assembler.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"


void HHOSecondOrderExample(int argc, char **argv);

void HHOFirstOrderExample(int argc, char **argv);

#define quadratic_space_solution_Q
int main(int argc, char **argv)
{

//    HHOFirstOrderExample(argc, argv);
//    HHOSecondOrderExample(argc, argv);
    
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
    
    return 0;
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
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
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

    auto exact_scal_fun = [](const mesh_type::point_type& pt) -> RealType {
        return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
    };

    auto exact_flux_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
        flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };

    auto rhs_fun = [](const typename mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return -2.0*((x - 1)*x + (y - 1)*y);
    };

#else

    auto exact_scal_fun = [](const mesh_type::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };

    auto exact_flux_fun = [](const mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
        flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };

    auto rhs_fun = [](const mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
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
    bnd.addDirichletEverywhere(exact_scal_fun);
    tc.tic();
    auto assembler = one_field_assembler<mesh_type>(msh, hho_di, bnd);
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
    postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, x_dof, exact_scal_fun, exact_flux_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_";
    postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, x_dof, exact_scal_fun, false);
}
