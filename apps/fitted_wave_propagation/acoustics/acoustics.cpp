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
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Eigenvalues>
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
#include "../common/scal_analytic_functions.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

// implicit RK schemes
#include "../common/dirk_hho_scheme.hpp"
#include "../common/dirk_butcher_tableau.hpp"

// explicit RK schemes
#include "../common/erk_hho_scheme.hpp"
#include "../common/erk_butcher_tableau.hpp"
#include "../common/ssprk_hho_scheme.hpp"
#include "../common/ssprk_shu_osher_tableau.hpp"


/// Enumerate defining available function prototypes
enum EAcousticPrototype {
    OneFieldConvTest = 0,
    TwoFieldsConvTest = 1,
    IOneFieldAcoustic = 2,
    ITwoFieldsAcoustic = 3,
    ETwoFieldsAcoustic = 4,
    Hete1DIOneFieldAcoustic = 5,
    Hete1DITwoFieldsAcoustic = 6,
    Hete1DETwoFieldsAcoustic = 7,
    Hete2DIOneFieldAcoustic = 8,
    Hete2DITwoFieldsAcoustic = 9,
    Hete2DETwoFieldsAcoustic = 10
};

// ----- common data types ------------------------------
using RealType = double;
typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
typedef disk::BoundaryConditions<mesh_type, true> boundary_type;

// ----- function prototypes ------------------------------

// Convergece tests for elliptic problem
void EllipticOneFieldConvergenceTest(char **argv);
void EllipticTwoFieldsConvergenceTest(char **argv);

// Convergece tests for the acoustic wave equation
void IHHOSecondOrder(char **argv);
void IHHOFirstOrder(char **argv);
void EHHOFirstOrder(char **argv);

// Simulation for a heterogeneous acoustics problem
void HeterogeneousIHHOSecondOrder(char **argv);
void HeterogeneousIHHOFirstOrder(char **argv);
void HeterogeneousEHHOFirstOrder(char **argv);

// Simulation for a heterogeneous acoustics problem on polygonal meshes
void HeterogeneousPulseIHHOSecondOrder(char **argv);
void HeterogeneousPulseIHHOFirstOrder(char **argv);
void HeterogeneousPulseEHHOFirstOrder(char **argv);

// Comparison with Gar6more2D
void HeterogeneousGar6more2DIHHOSecondOrder(int argc, char **argv);
void HeterogeneousGar6more2DIHHOFirstOrder(int argc, char **argv);

void EHHOFirstOrderCFL(int argc, char **argv);
void EHHOFirstOrderEigenCFL(int argc, char **argv);

int prototype_selector(char **argv, EAcousticPrototype prototype);

void print_prototype_description();

int main(int argc, char **argv)
{
//    EHHOFirstOrderCFL(argc,argv);
//    EHHOFirstOrderEigenCFL(argc,argv);
    HeterogeneousGar6more2DIHHOFirstOrder(argc,argv);
//    HeterogeneousGar6more2DIHHOSecondOrder(argc,argv);
    return 0;
    EAcousticPrototype prototype;
    const char* const short_opts = "p:h:";
    const option long_opts[] = {
            {"prototype", required_argument, nullptr, 'p'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, no_argument, nullptr, 0}
    };
    
    while (true)
    {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
        if (-1 == opt)
            break;

        switch (opt)
        {
        case 'p':
            prototype = EAcousticPrototype(std::stoi(optarg));
            break;
        case 'h':
             print_prototype_description();
            break;
        case '?':
        default:
            throw std::invalid_argument("Invalid option.");
            break;
        }
    }
    
    if (argc != 4)
    {
        throw std::invalid_argument("Please specify and option and a lua configuration file.");
        return 0;
    }

    return prototype_selector(argv, prototype);
}

void print_prototype_description(){
    
}

int prototype_selector(char **argv, EAcousticPrototype prototype){
    
    switch (prototype) {
        case EAcousticPrototype::OneFieldConvTest:
            {
                EllipticOneFieldConvergenceTest(argv);
            }
            break;
        case EAcousticPrototype::TwoFieldsConvTest:
            {
                EllipticTwoFieldsConvergenceTest(argv);
            }
            break;
        case EAcousticPrototype::IOneFieldAcoustic:
            {
                IHHOSecondOrder(argv);
            }
            break;
        case EAcousticPrototype::ITwoFieldsAcoustic:
            {
                IHHOFirstOrder(argv);
            }
            break;
        case EAcousticPrototype::ETwoFieldsAcoustic:
            {
                EHHOFirstOrder(argv);
            }
            break;
        case EAcousticPrototype::Hete1DIOneFieldAcoustic:
            {
                HeterogeneousIHHOSecondOrder(argv);
            }
            break;
        case EAcousticPrototype::Hete1DITwoFieldsAcoustic:
            {
                HeterogeneousIHHOFirstOrder(argv);
            }
            break;
        case EAcousticPrototype::Hete1DETwoFieldsAcoustic:
            {
                HeterogeneousEHHOFirstOrder(argv);
            }
            break;
        case EAcousticPrototype::Hete2DIOneFieldAcoustic:
            {
                HeterogeneousPulseIHHOSecondOrder(argv);
            }
            break;
        case EAcousticPrototype::Hete2DITwoFieldsAcoustic:
            {
                HeterogeneousPulseIHHOFirstOrder(argv);
            }
            break;
        case EAcousticPrototype::Hete2DETwoFieldsAcoustic:
            {
                HeterogeneousPulseEHHOFirstOrder(argv);
            }
            break;
        default:
            {
                throw std::invalid_argument("Prototype not available.");
            }
            break;
    }
    
    //    HeterogeneousGar6more2DIHHOSecondOrder(argc, argv);

}

void EllipticOneFieldConvergenceTest(char **argv){

    simulation_data sim_data = preprocessor::process_convergence_test_lua_file(argv);
    sim_data.print_simulation_data();

    // Manufactured exact solution
    bool quadratic_function_Q = sim_data.m_exact_functions == 1;
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
    
    if (sim_data.m_polygonal_mesh_Q) {
        if (sim_data.m_n_divs > 8) {
            sim_data.m_n_divs = 8;
        }
    }

    // simple material
    RealType rho = 1.0;
    RealType vp = 1.0;
    acoustic_material_data<RealType> material(rho,vp);
    
    std::ofstream error_file("steady_scalar_polygon_error.txt");
    
    for(size_t k = 0; k <= sim_data.m_k_degree; k++){
        std::cout << bold << cyan << "Running an approximation with k : " << k << reset << std::endl;
        error_file << "Approximation with k : " << k << std::endl;
        for(size_t l = 0; l < sim_data.m_n_divs; l++){
            
            // Reading the polygonal mesh
            timecounter tc;
            
            tc.tic();
            mesh_type msh;
            if (sim_data.m_polygonal_mesh_Q) {
                polygon_2d_mesh_reader<RealType> mesh_builder;
                std::vector<std::string> mesh_files;
                mesh_files.push_back("meshes/unit_square_polymesh_nel_20.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_40.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_80.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_160.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_320.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_640.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_1280.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_2560.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_5120.txt");
                
                // Reading the polygonal mesh
                mesh_builder.set_poly_mesh_file(mesh_files[l]);
                mesh_builder.build_mesh();
                mesh_builder.move_to_mesh_storage(msh);
            }else{
                RealType lx = 1.0;
                RealType ly = 1.0;
                size_t nx = 2;
                size_t ny = 2;
                
                cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
                mesh_builder.refine_mesh(l);
                mesh_builder.build_mesh();
                mesh_builder.move_to_mesh_storage(msh);
            }
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
            Matrix<RealType, Dynamic, 1> x_dof;
            if (sim_data.m_sc_Q) {
                tc.tic();
                linear_solver<RealType> analysis(assembler.LHS,assembler.get_n_face_dof());
                analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
                tc.toc();
                std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                if(sim_data.m_iterative_solver_Q){
                  analysis.set_iterative_solver(true);
                }
                
                tc.tic();
                analysis.factorize();
                tc.toc();
                std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                tc.tic();
                x_dof = analysis.solve(assembler.RHS);
                tc.toc();
                std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
                error_file << "Number of equations (SC) : " << analysis.n_equations() << std::endl;
            }else{
                tc.tic();
                linear_solver<RealType> analysis(assembler.LHS);
                tc.toc();
                std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                if(sim_data.m_iterative_solver_Q){
                  analysis.set_iterative_solver(true);
                }
                
                tc.tic();
                analysis.factorize();
                tc.toc();
                std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                tc.tic();
                x_dof = analysis.solve(assembler.RHS);
                tc.toc();
                std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
                error_file << "Number of equations : " << analysis.n_equations() << std::endl;
            }
            
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

void EllipticTwoFieldsConvergenceTest(char **argv){
    
    simulation_data sim_data = preprocessor::process_convergence_test_lua_file(argv);
    sim_data.print_simulation_data();

    // Manufactured exact solution
    bool quadratic_function_Q = sim_data.m_exact_functions == 1;
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
    
    if (sim_data.m_polygonal_mesh_Q) {
        if (sim_data.m_n_divs > 8) {
            sim_data.m_n_divs = 8;
        }
    }

    // simple material
    RealType rho = 1.0;
    RealType vp = 1.0;
    acoustic_material_data<RealType> material(rho,vp);
    
    std::ofstream error_file("steady_scalar_mixed_polygon_error.txt");
    
    for(size_t k = 0; k <= sim_data.m_k_degree; k++){
        std::cout << bold << cyan << "Running an approximation with k : " << k << reset << std::endl;
        error_file << "Approximation with k : " << k << std::endl;
        for(size_t l = 0; l < sim_data.m_n_divs; l++){
            
            // Reading the polygonal mesh
            timecounter tc;
            tc.tic();
            mesh_type msh;
            if (sim_data.m_polygonal_mesh_Q) {
                polygon_2d_mesh_reader<RealType> mesh_builder;
                std::vector<std::string> mesh_files;
                mesh_files.push_back("meshes/unit_square_polymesh_nel_20.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_40.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_80.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_160.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_320.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_640.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_1280.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_2560.txt");
                mesh_files.push_back("meshes/unit_square_polymesh_nel_5120.txt");
                
                // Reading the polygonal mesh
                mesh_builder.set_poly_mesh_file(mesh_files[l]);
                mesh_builder.build_mesh();
                mesh_builder.move_to_mesh_storage(msh);
            }else{
                RealType lx = 1.0;
                RealType ly = 1.0;
                size_t nx = 2;
                size_t ny = 2;
                
                cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
                mesh_builder.refine_mesh(l);
                mesh_builder.build_mesh();
                mesh_builder.move_to_mesh_storage(msh);
            }
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
            if(sim_data.m_scaled_stabilization_Q){
                assembler.set_scaled_stabilization();
            }
            assembler.load_material_data(msh,material);
            assembler.assemble(msh, rhs_fun);
            assembler.assemble_mass(msh, false);
            assembler.apply_bc(msh);
            tc.toc();
            std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
            
            // Solving LS
            Matrix<RealType, Dynamic, 1> x_dof;
            if (sim_data.m_sc_Q) {
                tc.tic();
                SparseMatrix<RealType> Kg = assembler.LHS+assembler.MASS;
                linear_solver<RealType> analysis(Kg,assembler.get_n_face_dof());
                analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
                tc.toc();
                std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                if(sim_data.m_iterative_solver_Q){
                  analysis.set_iterative_solver();
                }
                
                tc.tic();
                analysis.factorize();
                tc.toc();
                std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                tc.tic();
                x_dof = analysis.solve(assembler.RHS);
                tc.toc();
                std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
                error_file << "Number of equations (SC) : " << analysis.n_equations() << std::endl;
            }else{
                tc.tic();
                SparseMatrix<RealType> Kg = assembler.LHS+assembler.MASS;
                linear_solver<RealType> analysis(Kg);
                tc.toc();
                std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                if(sim_data.m_iterative_solver_Q){
                  analysis.set_iterative_solver();
                }
                
                tc.tic();
                analysis.factorize();
                tc.toc();
                std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
                
                tc.tic();
                x_dof = analysis.solve(assembler.RHS);
                tc.toc();
                std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
                error_file << "Number of equations : " << analysis.n_equations() << std::endl;
            }
            
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, assembler, x_dof, exact_scal_fun, exact_flux_fun, error_file);
            
            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "steady_scalar_mixed_k" + std::to_string(k) + "_";
                postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, l, msh, hho_di, x_dof, exact_scal_fun, exact_flux_fun, false);
            }
        }
        error_file << std::endl << std::endl;
    }
    error_file.close();
}

void IHHOSecondOrder(char **argv){

    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    
    if (sim_data.m_polygonal_mesh_Q) {
        if (sim_data.m_n_divs > 8) {
            sim_data.m_n_divs = 8;
        }
    }
    
    mesh_type msh;
    if (sim_data.m_polygonal_mesh_Q) {
        size_t l = sim_data.m_n_divs;
        polygon_2d_mesh_reader<RealType> mesh_builder;
        std::vector<std::string> mesh_files;
        mesh_files.push_back("meshes/unit_square_polymesh_nel_20.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_40.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_80.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_160.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_320.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_640.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_1280.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_2560.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_5120.txt");
        
        // Reading the polygonal mesh
        mesh_builder.set_poly_mesh_file(mesh_files[l]);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }else{
        RealType lx = 1.0;
        RealType ly = 1.0;
        size_t nx = 2;
        size_t ny = 2;
        
        cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
        mesh_builder.refine_mesh(sim_data.m_n_divs);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }
    
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
    switch (sim_data.m_exact_functions) {
        case 1:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
            break;
        case 2:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInTime);
            break;
        default:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
            break;
    }
    
    
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
    bnd.addDirichletEverywhere(exact_scal_fun);
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
        SparseMatrix<RealType> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
            analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
            analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
        }else{
            analysis.set_Kg(assembler.LHS);
        }
        
        if (sim_data.m_iterative_solver_Q) {
            analysis.set_iterative_solver(true);
        }else{
            analysis.set_direct_solver(true); // symmetric matrix case
        }
        
        analysis.factorize();
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
        simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
}

void IHHOFirstOrder(char **argv){
    
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    
    if (sim_data.m_polygonal_mesh_Q) {
        if (sim_data.m_n_divs > 8) {
            sim_data.m_n_divs = 8;
        }
    }
    
    mesh_type msh;
    if (sim_data.m_polygonal_mesh_Q) {
        size_t l = sim_data.m_n_divs;
        polygon_2d_mesh_reader<RealType> mesh_builder;
        std::vector<std::string> mesh_files;
        mesh_files.push_back("meshes/unit_square_polymesh_nel_20.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_40.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_80.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_160.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_320.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_640.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_1280.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_2560.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_5120.txt");
        
        // Reading the polygonal mesh
        mesh_builder.set_poly_mesh_file(mesh_files[l]);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }else{
        RealType lx = 1.0;
        RealType ly = 1.0;
        size_t nx = 2;
        size_t ny = 2;
        
        cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
        mesh_builder.refine_mesh(sim_data.m_n_divs);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }
    
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
    switch (sim_data.m_exact_functions) {
        case 1:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
            break;
        case 2:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInTime);
            break;
        default:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
            break;
    }

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
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
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
    dirk_hho_scheme<RealType> dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (sim_data.m_sc_Q) {
        dirk_an.set_static_condensation_data(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()), assembler.get_n_face_dof());
    }
    
    if (is_sdirk_Q) {
        // SDIRK case
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.ComposeMatrix();
        
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
        
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }else{
        // DIRK case
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
    }
    
    Matrix<RealType, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        RealType tn = dt*(it-1)+ti;
        
        // DIRK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<RealType, Dynamic, Dynamic> k = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<RealType, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<RealType, Dynamic, 1> yn, ki;

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
                    assembler.apply_bc(msh);
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
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, assembler, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }

    }
    
    simulation_log << "Number of equations : " << dirk_an.DirkAnalysis().n_equations() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}

void EHHOFirstOrder(char **argv){
    
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    if (sim_data.m_polygonal_mesh_Q) {
        if (sim_data.m_n_divs > 8) {
            sim_data.m_n_divs = 8;
        }
    }
    
    mesh_type msh;
    if (sim_data.m_polygonal_mesh_Q) {
        size_t l = sim_data.m_n_divs;
        polygon_2d_mesh_reader<RealType> mesh_builder;
        std::vector<std::string> mesh_files;
        mesh_files.push_back("meshes/unit_square_polymesh_nel_20.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_40.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_80.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_160.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_320.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_640.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_1280.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_2560.txt");
        mesh_files.push_back("meshes/unit_square_polymesh_nel_5120.txt");
        
        // Reading the polygonal mesh
        mesh_builder.set_poly_mesh_file(mesh_files[l]);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }else{
        RealType lx = 1.0;
        RealType ly = 1.0;
        size_t nx = 2;
        size_t ny = 2;
        
        cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
        mesh_builder.refine_mesh(sim_data.m_n_divs);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }
    
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
    switch (sim_data.m_exact_functions) {
        case 1:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
            break;
        case 2:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInTime);
            break;
        default:
            functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
            break;
    }
    
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
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
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
        std::string silo_file_name = "e_scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_two_fields_explicit.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    int s = 4;
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    erk_butcher_tableau::erk_tables(s, a, b, c);

    tc.tic();
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness and rhs assembly completed: " << tc << " seconds" << reset << std::endl;
    size_t n_face_dof = assembler.get_n_face_dof();
    tc.tic();
    erk_hho_scheme<RealType> erk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
    erk_an.Kcc_inverse(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
    if(sim_data.m_hdg_stabilization_Q){
        erk_an.Sff_inverse(std::make_pair(assembler.get_n_faces(), assembler.get_face_basis_data()));
    }else{
        if (sim_data.m_iterative_solver_Q) {
            erk_an.setIterativeSolver();
        }
        erk_an.DecomposeFaceTerm();
    }
    tc.toc();
    std::cout << bold << cyan << "ERK analysis created: " << tc << " seconds" << reset << std::endl;
    
    erk_an.refresh_faces_unknowns(x_dof);
    Matrix<RealType, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        
        RealType tn = dt*(it-1)+ti;
        // ERK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<RealType, Dynamic, Dynamic> k = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<RealType, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<RealType, Dynamic, 1> yn, ki;

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
                    assembler.apply_bc(msh);
                    erk_an.SetFg(assembler.RHS);
                    erk_an.erk_weight(yn, ki);
                }

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "ERK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;

        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "e_scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }

        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, assembler, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }
    }
    
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of ERK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
}

void HeterogeneousIHHOSecondOrder(char **argv){
    
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.2;
    size_t nx = 10;
    size_t ny = 1;
    
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
        if (x < 0.5) {
            vp = 3.0;
        }else{
            vp = 1.0;
        }
        rho = 1.0/(vp*vp); // this is required to make both formulations compatible by keeping kappa = 1
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
        postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, v_dof_n, exact_vel_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_one_field.txt");
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
    }
    
    linear_solver<RealType> analysis;
    bool standar_Q = true;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 0.6;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        tc.tic();
        assembler.assemble(msh, rhs_fun);
        SparseMatrix<RealType> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
            analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
            analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
        }else{
            analysis.set_Kg(assembler.LHS);
        }
        
        if (sim_data.m_iterative_solver_Q) {
            analysis.set_iterative_solver(true);
        }else{
            analysis.set_direct_solver(true); // symmetric matrix case
        }
        
        analysis.factorize();
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
        
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_fun     = functions.Evaluate_u(t);
            auto exact_vel_fun      = functions.Evaluate_v(t);
            auto exact_flux_fun     = functions.Evaluate_q(t);

            assembler.get_bc_conditions().updateDirichletFunction(exact_scal_fun, 0);
            assembler.RHS.setZero(); // Optimization: this is a problem with (f = 0)
            assembler.apply_bc(msh);

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
                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, v_dof_n, exact_vel_fun, false);
            }
            
            if (sim_data.m_report_energy_Q) {
                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
            }
            
            if(it == nt){
                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, p_dof_n, exact_scal_fun, exact_flux_fun, simulation_log);
            }
            
        }
        simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
}



void HeterogeneousIHHOFirstOrder(char **argv){
    
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.2;
    size_t nx = 10;
    size_t ny = 1;
    
    
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
        if (x < 0.5) {
            vp = 3.0;
        }else{
            vp = 1.0;
        }
        rho = 1.0/(vp*vp); // this is required to make both formulations compatible by keeping kappa = 1
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
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
    dirk_hho_scheme<RealType> dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (sim_data.m_sc_Q) {
        dirk_an.set_static_condensation_data(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()), assembler.get_n_face_dof());
    }
    
    if (is_sdirk_Q) {
        // SDIRK case
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.ComposeMatrix();
        
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
        
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }else{
        // DIRK case
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
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
                    assembler.RHS.setZero(); // Optimization: this is a problem with (f = 0)
                    assembler.apply_bc(msh);
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
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, assembler, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
        }

    }
    
    simulation_log << "Number of equations : " << dirk_an.DirkAnalysis().n_equations() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}

void HeterogeneousEHHOFirstOrder(char **argv){
    
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.2;
    size_t nx = 10;
    size_t ny = 1;
    
    
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
        if (x < 0.5) {
            vp = 3.0;
        }else{
            vp = 1.0;
        }
        rho = 1.0/(vp*vp); // this is required to make both formulations compatible by keeping kappa = 1
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
        std::string silo_file_name = "e_scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    }
    
    std::ofstream simulation_log("acoustic_two_fields_explicit.txt");
    
    if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, ti, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    int s = 4;
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    erk_butcher_tableau::erk_tables(s, a, b, c);

    tc.tic();
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness and rhs assembly completed: " << tc << " seconds" << reset << std::endl;
    size_t n_face_dof = assembler.get_n_face_dof();
    tc.tic();
    erk_hho_scheme<RealType> erk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
    erk_an.Kcc_inverse(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
    if(sim_data.m_hdg_stabilization_Q){
        erk_an.Sff_inverse(std::make_pair(assembler.get_n_faces(), assembler.get_face_basis_data()));
    }else{
        if (sim_data.m_iterative_solver_Q) {
            erk_an.setIterativeSolver();
        }
        erk_an.DecomposeFaceTerm();
    }
    tc.toc();
    std::cout << bold << cyan << "ERK analysis created: " << tc << " seconds" << reset << std::endl;
    
    erk_an.refresh_faces_unknowns(x_dof);
    Matrix<RealType, Dynamic, 1> x_dof_n;
    timecounter simulation_tc;
    simulation_tc.tic();
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        
        RealType tn = dt*(it-1)+ti;
        // ERK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<RealType, Dynamic, Dynamic> k = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<RealType, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<RealType, Dynamic, 1> yn, ki;

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
                    assembler.RHS.setZero();
                    assembler.apply_bc(msh);
                    erk_an.SetFg(assembler.RHS);
                    erk_an.erk_weight(yn, ki);
                }

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "ERK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;

        RealType t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "e_inhomogeneous_scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        }
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }
    }
    simulation_tc.toc();
    simulation_log << "Simulation time : " << simulation_tc << " seconds" << std::endl;
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of ERK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}


void HeterogeneousPulseIHHOSecondOrder(char **argv){
    
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    
    mesh_type msh;
    if (sim_data.m_polygonal_mesh_Q) {
        
        auto validate_l = [](size_t l) -> size_t {
            if ((0 < l) && (l < 5) ) {
                return l;
            }else{
                std::cout << "Warning:: Only five meshes available. Running level 4." << std::endl;
                return 4;
            }
        };
        
        size_t l = validate_l(sim_data.m_n_divs);
        polygon_2d_mesh_reader<RealType> mesh_builder;
        std::vector<std::string> mesh_files;
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_4096.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_5120.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_10240.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_16384.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_65536.txt");
        
        // Reading the polygonal mesh
        mesh_builder.set_poly_mesh_file(mesh_files[l]);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
        
    }else{
        RealType lx = 1.0;
        RealType ly = 1.0;
        size_t nx = 2;
        size_t ny = 2;
        
        cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
        mesh_builder.refine_mesh(sim_data.m_n_divs);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }
    
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
            vp = 5.0;
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
    
    std::ofstream sensor_top_log("top_sensor_acoustic_one_field.csv");
    std::ofstream sensor_bot_log("bot_sensor_acoustic_one_field.csv");
    typename mesh_type::point_type top_pt(0.5, 2.0/3.0);
    typename mesh_type::point_type bot_pt(0.5, 1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> top_pt_cell = std::make_pair(top_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> bot_pt_cell = std::make_pair(bot_pt, -1);
    
    postprocessor<mesh_type>::record_data_acoustic_one_field(0, top_pt_cell, msh, hho_di, v_dof_n, sensor_top_log);
    postprocessor<mesh_type>::record_data_acoustic_one_field(0, bot_pt_cell, msh, hho_di, v_dof_n, sensor_bot_log);
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, ti, p_dof_n, v_dof_n, simulation_log);
    }
    
    timecounter simulation_tc;
    bool standar_Q = true;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 1.0;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        tc.tic();
        assembler.assemble(msh, null_fun);
        SparseMatrix<RealType> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
           analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
           analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
        }else{
           analysis.set_Kg(assembler.LHS);
        }

        if (sim_data.m_iterative_solver_Q) {
           analysis.set_iterative_solver(true);
        }else{
           analysis.set_direct_solver(true); // symmetric matrix case
        }

        analysis.factorize();
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
        
        simulation_tc.tic();
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;

            tc.tic();
            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = Kg*p_dof_n;
            
            assembler.RHS.setZero(); // Optimization: this is a problem with (f = 0) and (p_D = 0)
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
            
            postprocessor<mesh_type>::record_data_acoustic_one_field(it, top_pt_cell, msh, hho_di, v_dof_n, sensor_top_log);
            postprocessor<mesh_type>::record_data_acoustic_one_field(it, bot_pt_cell, msh, hho_di, v_dof_n, sensor_bot_log);
            
            if (sim_data.m_report_energy_Q) {
                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
            }
            
        }
        simulation_tc.toc();
        simulation_log << "Simulation time : " << simulation_tc << " seconds" << std::endl;
        simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
}

void HeterogeneousPulseIHHOFirstOrder(char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    mesh_type msh;
    if (sim_data.m_polygonal_mesh_Q) {
        
        auto validate_l = [](size_t l) -> size_t {
            if ((0 < l) && (l < 5) ) {
                return l;
            }else{
                std::cout << "Warning:: Only five meshes available. Running level 4." << std::endl;
                return 4;
            }
        };
        
        size_t l = validate_l(sim_data.m_n_divs);
        polygon_2d_mesh_reader<RealType> mesh_builder;
        std::vector<std::string> mesh_files;
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_4096.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_5120.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_10240.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_16384.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_65536.txt");
        
        // Reading the polygonal mesh
        mesh_builder.set_poly_mesh_file(mesh_files[l]);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
        
    }else{
        RealType lx = 1.0;
        RealType ly = 1.0;
        size_t nx = 2;
        size_t ny = 2;
        
        cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
        mesh_builder.refine_mesh(sim_data.m_n_divs);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }
    
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.25
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
            vp = 5.0;
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
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
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
    
    std::ofstream sensor_top_log("top_sensor_acoustic_two_fields.csv");
    std::ofstream sensor_bot_log("bot_sensor_acoustic_two_fields.csv");
    typename mesh_type::point_type top_pt(0.5, 2.0/3.0);
    typename mesh_type::point_type bot_pt(0.5, 1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> top_pt_cell = std::make_pair(top_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> bot_pt_cell = std::make_pair(bot_pt, -1);
    
    postprocessor<mesh_type>::record_data_acoustic_two_fields(0, top_pt_cell, msh, hho_di, x_dof, sensor_top_log);
    postprocessor<mesh_type>::record_data_acoustic_two_fields(0, bot_pt_cell, msh, hho_di, x_dof, sensor_bot_log);
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, ti, x_dof, simulation_log);
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
    assembler.assemble(msh, null_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
    dirk_hho_scheme<RealType> dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (sim_data.m_sc_Q) {
        dirk_an.set_static_condensation_data(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()), assembler.get_n_face_dof());
    }
    
    if (is_sdirk_Q) {
        // SDIRK case
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.ComposeMatrix();
        
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
        
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }else{
        // DIRK case
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
    }

    Matrix<double, Dynamic, 1> x_dof_n;
    timecounter simulation_tc;
    simulation_tc.tic();
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
                    assembler.RHS.setZero();
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
        
        RealType t = tn + dt;
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "inhomogeneous_scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, vel_fun, null_flux_fun, false);
        }
        
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, top_pt_cell, msh, hho_di, x_dof, sensor_top_log);
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, bot_pt_cell, msh, hho_di, x_dof, sensor_bot_log);
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }

    }
    simulation_tc.toc();
    simulation_log << "Simulation time : " << simulation_tc << " seconds" << std::endl;
    simulation_log << "Number of equations : " << dirk_an.DirkAnalysis().n_equations() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
}

void HeterogeneousPulseEHHOFirstOrder(char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_acoustics_lua_file(argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 1.0;
    size_t nx = 2;
    size_t ny = 2;
    
    
    mesh_type msh;
    if (sim_data.m_polygonal_mesh_Q) {
        
        auto validate_l = [](size_t l) -> size_t {
            if ((0 < l) && (l < 5) ) {
                return l;
            }else{
                std::cout << "Warning:: Only five meshes available. Running level 4." << std::endl;
                return 4;
            }
        };
        
        size_t l = validate_l(sim_data.m_n_divs);
        polygon_2d_mesh_reader<RealType> mesh_builder;
        std::vector<std::string> mesh_files;
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_4096.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_5120.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_10240.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_16384.txt");
        mesh_files.push_back("meshes/mexican_hat_polymesh_nel_65536.txt");
        
        // Reading the polygonal mesh
        mesh_builder.set_poly_mesh_file(mesh_files[l]);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
        
    }else{
        RealType lx = 1.0;
        RealType ly = 1.0;
        size_t nx = 2;
        size_t ny = 2;
        
        cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
        mesh_builder.refine_mesh(sim_data.m_n_divs);
        mesh_builder.build_mesh();
        mesh_builder.move_to_mesh_storage(msh);
    }
    
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.25
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
            vp = 5.0;
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
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
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
        std::string silo_file_name = "e_inhomogeneous_scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, vel_fun, null_flux_fun, false);
    }
    
    std::ofstream simulation_log("inhomogeneous_acoustic_two_fields_explicit.txt");
        
    std::ofstream sensor_top_log("top_sensor_e_acoustic_two_fields.csv");
    std::ofstream sensor_bot_log("bot_sensor_e_acoustic_two_fields.csv");
    typename mesh_type::point_type top_pt(0.5, 2.0/3.0);
    typename mesh_type::point_type bot_pt(0.5, 1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> top_pt_cell = std::make_pair(top_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> bot_pt_cell = std::make_pair(bot_pt, -1);
    
    postprocessor<mesh_type>::record_data_acoustic_two_fields(0, top_pt_cell, msh, hho_di, x_dof, sensor_top_log);
    postprocessor<mesh_type>::record_data_acoustic_two_fields(0, bot_pt_cell, msh, hho_di, x_dof, sensor_bot_log);
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, ti, x_dof, simulation_log);
    }
    
    // Solving a first order equation HDG/HHO propagation problem
    int s = 4;
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    erk_butcher_tableau::erk_tables(s, a, b, c);

    tc.tic();
    assembler.assemble(msh, null_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness and rhs assembly completed: " << tc << " seconds" << reset << std::endl;
    size_t n_face_dof = assembler.get_n_face_dof();
    tc.tic();
    erk_hho_scheme<RealType> erk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
    erk_an.Kcc_inverse(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
    if(sim_data.m_hdg_stabilization_Q){
        erk_an.Sff_inverse(std::make_pair(assembler.get_n_faces(), assembler.get_face_basis_data()));
    }else{
        if (sim_data.m_iterative_solver_Q) {
            erk_an.setIterativeSolver();
        }
        erk_an.DecomposeFaceTerm();
    }
    tc.toc();
    std::cout << bold << cyan << "ERK analysis created: " << tc << " seconds" << reset << std::endl;
    
    erk_an.refresh_faces_unknowns(x_dof);
    Matrix<RealType, Dynamic, 1> x_dof_n;
    timecounter simulation_tc;
    simulation_tc.tic();
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        
        RealType tn = dt*(it-1)+ti;
        // ERK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<RealType, Dynamic, Dynamic> k = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<RealType, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
            
            Matrix<RealType, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {
                
                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }
                
                {
//                    RealType t = tn + c(i,0) * dt;
//                    auto exact_vel_fun      = functions.Evaluate_v(t);
//                    auto rhs_fun            = functions.Evaluate_f(t);
//                    assembler.get_bc_conditions().updateDirichletFunction(exact_vel_fun, 0);
//                    assembler.assemble_rhs(msh, rhs_fun);
//                    assembler.apply_bc(msh);
//                    erk_an.SetFg(assembler.RHS);
                    erk_an.erk_weight(yn, ki);
                }

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "ERK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;

        RealType t = tn + dt;
        
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "e_inhomogeneous_scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, vel_fun, null_flux_fun, false);
        }
        
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, top_pt_cell, msh, hho_di, x_dof, sensor_top_log);
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, bot_pt_cell, msh, hho_di, x_dof, sensor_bot_log);
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }
    }
    simulation_tc.toc();
    simulation_log << "Simulation time : " << simulation_tc << " seconds" << std::endl;
    simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
    simulation_log << "Number of ERK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
}

void HeterogeneousGar6more2DIHHOSecondOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 3.0;
    RealType ly = 3.0;
    size_t nx = 3;
    size_t ny = 3;
    
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.set_translation_data(-1.5, -1.5);
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

    auto null_fun = [](const mesh_type::point_type& pt) -> RealType {
            RealType x,y;
            x = pt.x();
            y = pt.y();
            return 0.0;
    };
    
    auto p_fun = [](const mesh_type::point_type& pt) -> RealType {
        RealType x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
        x = pt.x();
        y = pt.y();
        xc = 0.0;
        yc = 2.0/3.0;
        c = 10.0;
        lp = 1.0*std::sqrt(64.0)/c;
        r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
        wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
        factor = (lp*lp/(2.0*M_PI*M_PI));
        return factor*wave;
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
        if (y < 0.0) {
            vp = 1.0*std::sqrt(3.0);
            rho = 1.0/3.0;
        }else{
            vp = 1.0*std::sqrt(64.0);
            rho = 1.0/64.0;
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
    assembler.project_over_cells(msh, p_dof_n, p_fun);
    assembler.project_over_faces(msh, p_dof_n, p_fun);
    assembler.project_over_cells(msh, v_dof_n, null_fun);
    assembler.project_over_faces(msh, v_dof_n, null_fun);
    assembler.project_over_cells(msh, a_dof_n, null_fun);
    assembler.project_over_faces(msh, a_dof_n, null_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "inhomogeneous_scalar_";
        postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, p_fun, false);
    }
    
    std::ofstream simulation_log("inhomogeneous_acoustic_one_field.txt");
    
    std::ofstream sensor_1_log("s1_acoustic_one_field_h.csv");
    std::ofstream sensor_2_log("s2_acoustic_one_field_h.csv");
    std::ofstream sensor_3_log("s3_acoustic_one_field_h.csv");
    typename mesh_type::point_type s1_pt(+3.0/4.0, -1.0/3.0);
    typename mesh_type::point_type s2_pt( 0.0, +1.0/3.0);
    typename mesh_type::point_type s3_pt(+3.0/4.0, +1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> s1_pt_cell = std::make_pair(s1_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s2_pt_cell = std::make_pair(s2_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s3_pt_cell = std::make_pair(s3_pt, -1);
    
    postprocessor<mesh_type>::record_velocity_data_acoustic_one_field(0, s1_pt_cell, msh, hho_di, assembler, p_dof_n, sensor_1_log);
    postprocessor<mesh_type>::record_velocity_data_acoustic_one_field(0, s2_pt_cell, msh, hho_di, assembler, p_dof_n, sensor_2_log);
    postprocessor<mesh_type>::record_velocity_data_acoustic_one_field(0, s3_pt_cell, msh, hho_di, assembler, p_dof_n, sensor_3_log);
    
    if (sim_data.m_report_energy_Q) {
        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, ti, p_dof_n, v_dof_n, simulation_log);
    }
        
    timecounter simulation_tc;
    bool standar_Q = true;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 1.0;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        tc.tic();
        assembler.assemble(msh, null_fun);
        SparseMatrix<double> Kg = assembler.LHS;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += assembler.MASS;
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
        
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
            tc.tic();
            analysis.set_Kg(assembler.LHS,assembler.get_n_face_dof());
            analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
            tc.toc();
            std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
            
//            analysis.set_iterative_solver(true, 1.0e-10);
            analysis.set_direct_solver(true);
            
            tc.tic();
            analysis.factorize();
            tc.toc();
            std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
        }else{
            tc.tic();
            analysis.set_Kg(assembler.LHS);
            tc.toc();
            std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
            
            tc.tic();
            analysis.factorize();
            tc.toc();
            std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
        }
        
        simulation_tc.tic();
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;

            tc.tic();
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
                postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, p_fun, false);
            }
            
            postprocessor<mesh_type>::record_velocity_data_acoustic_one_field(it, s1_pt_cell, msh, hho_di, assembler, p_dof_n, sensor_1_log);
            postprocessor<mesh_type>::record_velocity_data_acoustic_one_field(it, s2_pt_cell, msh, hho_di, assembler, p_dof_n, sensor_2_log);
            postprocessor<mesh_type>::record_velocity_data_acoustic_one_field(it, s3_pt_cell, msh, hho_di, assembler, p_dof_n, sensor_3_log);
            
            if (sim_data.m_report_energy_Q) {
                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
            }
            
        }
        simulation_tc.toc();
        simulation_log << "Simulation time : " << simulation_tc << " seconds" << std::endl;
        simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
    
}

void HeterogeneousGar6more2DIHHOFirstOrder(int argc, char **argv){
    
    bool reference_Q = false;
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType scal = 1.0;
    if (reference_Q) {
        scal = 3.0;
    }
    
    RealType lx = 3.0*scal;
    RealType ly = 3.0*scal;
    size_t nx = 3;
    size_t ny = 3;
    
    
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.set_translation_data(-1.5*scal, -1.5*scal);
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
    
    auto null_fun = [](const mesh_type::point_type& pt) -> RealType {
            RealType x,y;
            x = pt.x();
            y = pt.y();
            return 0.0;
    };
    
    auto flux_fun = [](const mesh_type::point_type& pt) -> std::vector<RealType> {
        RealType x,y,xc,yc,r,wave,vx,vy,c,lp;
        x = pt.x();
        y = pt.y();
        xc = 0.0;
        yc = 2.0/3.0;
        c = 10.0;
        lp = std::sqrt(9.0)/c;
        r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
        wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
        vx = -wave*(x-xc);
        vy = -wave*(y-yc);
        std::vector<RealType> v = {vx,vy};
        return v;
    };
    
    RealType t = ti;
    
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
        if (y < 0.0) {
            vp = 1.0*std::sqrt(3.0);
            rho = 1.0/3.0;
        }else{
            vp = 1.0*std::sqrt(9.0);
            rho = 1.0/9.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
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
    assembler.project_over_cells(msh, x_dof, null_fun, flux_fun);
    assembler.project_over_faces(msh, x_dof, null_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "inhomogeneous_scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, null_fun, flux_fun, false);
    }
    
    std::ofstream simulation_log("inhomogeneous_acoustic_two_fields.txt");
    
    std::ofstream sensor_1_log("s1_acoustic_two_fields_h.csv");
    std::ofstream sensor_2_log("s2_acoustic_two_fields_h.csv");
    std::ofstream sensor_3_log("s3_acoustic_two_fields_h.csv");
    typename mesh_type::point_type s1_pt(+3.0/4.0, -1.0/3.0);
    typename mesh_type::point_type s2_pt( 0.0, +1.0/3.0);
    typename mesh_type::point_type s3_pt(+3.0/4.0, +1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> s1_pt_cell = std::make_pair(s1_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s2_pt_cell = std::make_pair(s2_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s3_pt_cell = std::make_pair(s3_pt, -1);
    
    postprocessor<mesh_type>::record_velocity_data_acoustic_two_fields(0, s1_pt_cell, msh, hho_di, x_dof, sensor_1_log);
    postprocessor<mesh_type>::record_velocity_data_acoustic_two_fields(0, s2_pt_cell, msh, hho_di, x_dof, sensor_2_log);
    postprocessor<mesh_type>::record_velocity_data_acoustic_two_fields(0, s3_pt_cell, msh, hho_di, x_dof, sensor_3_log);
    
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
    assembler.assemble(msh, null_fun);
    tc.toc();
    std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
    dirk_hho_scheme<RealType> dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (sim_data.m_sc_Q) {
        dirk_an.set_static_condensation_data(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()), assembler.get_n_face_dof());
    }
    
    if (is_sdirk_Q) {
        // SDIRK case
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.ComposeMatrix();
        
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
        
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }else{
        // DIRK case
        if (sim_data.m_iterative_solver_Q) {
            dirk_an.setIterativeSolver();
        }
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
                    assembler.RHS.setZero(); // Optimization: this is a problem with (f = 0)
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
                
        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "inhomogeneous_scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, null_fun, flux_fun, false);
        }
        
        postprocessor<mesh_type>::record_velocity_data_acoustic_two_fields(it, s1_pt_cell, msh, hho_di, x_dof, sensor_1_log);
        postprocessor<mesh_type>::record_velocity_data_acoustic_two_fields(it, s2_pt_cell, msh, hho_di, x_dof, sensor_2_log);
        postprocessor<mesh_type>::record_velocity_data_acoustic_two_fields(it, s3_pt_cell, msh, hho_di, x_dof, sensor_3_log);
        
        if (sim_data.m_report_energy_Q) {
            postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
        }
        
    }
    
    simulation_log << "Number of equations : " << dirk_an.DirkAnalysis().n_equations() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
    
}


void EHHOFirstOrderCFL(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    int k_ind = sim_data.m_k_degree;
    std::vector<RealType> tf_vec;
    
    if(sim_data.m_hdg_stabilization_Q){
       tf_vec = {0.5,0.4,0.3,0.2};
    }else{
       tf_vec = {0.5/4,0.4/4,0.3/4,0.2/4};
    }
    
    RealType ti = 0.0;
    RealType tf = tf_vec[k_ind];
    int nt_base = sim_data.m_nt_divs;
    
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
    std::ofstream simulation_log("acoustic_two_fields_explicit_cfl.txt");
    
    for (int s = 1; s < 5; s++) {
        simulation_log << " ******************************* " << std::endl;
        simulation_log << " number of stages s =  " << s << std::endl;
        simulation_log << std::endl;
        for(size_t l = 0; l <= sim_data.m_n_divs; l++){
                
                // Building a cartesian mesh
                timecounter tc;
                tc.tic();

                RealType lx = 1.0;
                RealType ly = 1.0;
                size_t nx = 10+l;
                size_t ny = 10+l;
                typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
                typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
                mesh_type msh;

                cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
                mesh_builder.build_mesh();
                mesh_builder.move_to_mesh_storage(msh);
                std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
                
                // Time controls : Final time value 1.0
                size_t nt = nt_base;
                for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
                
                    RealType dt     = (tf-ti)/nt;
                    
                    // Solving a primal HHO mixed problem
                    boundary_type bnd(msh);
                    bnd.addDirichletEverywhere(exact_vel_fun);
                    tc.tic();
                    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
                    assembler.load_material_data(msh);
                    if(sim_data.m_hdg_stabilization_Q){
                        assembler.set_hdg_stabilization();
                    }
                    if(sim_data.m_scaled_stabilization_Q){
                        assembler.set_scaled_stabilization();
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
                        std::string silo_file_name = "e_scalar_mixed_";
                        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
                    }
                    
                    RealType energy_0 = postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, ti, x_dof, simulation_log);
                    
                    // Solving a first order equation HDG/HHO propagation problem
                    Matrix<RealType, Dynamic, Dynamic> a;
                    Matrix<RealType, Dynamic, 1> b;
                    Matrix<RealType, Dynamic, 1> c;
                    erk_butcher_tableau::erk_tables(s, a, b, c);
        //            erk_butcher_tableau::ssprk_tables(s, a, b, c);

                    tc.tic();
                    assembler.assemble(msh, rhs_fun);
                    tc.toc();
                    std::cout << bold << cyan << "Stiffness and rhs assembly completed: " << tc << " seconds" << reset << std::endl;
                    size_t n_face_dof = assembler.get_n_face_dof();
                    tc.tic();
                    erk_hho_scheme<RealType> erk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
                    erk_an.Kcc_inverse(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
                    if(sim_data.m_hdg_stabilization_Q){
                        erk_an.Sff_inverse(std::make_pair(assembler.get_n_faces(), assembler.get_face_basis_data()));
                    }else{
        //                erk_an.setIterativeSolver();
                        erk_an.DecomposeFaceTerm();
                    }
                    tc.toc();
                    std::cout << bold << cyan << "ERK analysis created: " << tc << " seconds" << reset << std::endl;
                    
                    erk_an.refresh_faces_unknowns(x_dof);
                    Matrix<RealType, Dynamic, 1> x_dof_n;
                    bool approx_fail_check_Q = false;
                    RealType energy = energy_0;;
                    for(size_t it = 1; it <= nt; it++){

                        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
                        
                        RealType tn = dt*(it-1)+ti;
                        // ERK step
                        tc.tic();
                        {
                            size_t n_dof = x_dof.rows();
                            Matrix<RealType, Dynamic, Dynamic> k = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, s);
                            Matrix<RealType, Dynamic, 1> Fg, Fg_c,xd;
                            xd = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
                            
                            Matrix<RealType, Dynamic, 1> yn, ki;

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
                                    assembler.apply_bc(msh);
                                    erk_an.SetFg(assembler.RHS);
                                    erk_an.erk_weight(yn, ki);
                                }

                                // Accumulated solution
                                x_dof_n += dt*b(i,0)*ki;
                                k.block(0, i, n_dof, 1) = ki;
                            }
                        }
                        tc.toc();
                        std::cout << bold << cyan << "ERK step completed: " << tc << " seconds" << reset << std::endl;
                        x_dof = x_dof_n;

                        t = tn + dt;
                        auto exact_vel_fun = functions.Evaluate_v(t);
                        auto exact_flux_fun = functions.Evaluate_q(t);
                        
                        if (sim_data.m_render_silo_files_Q) {
                            std::string silo_file_name = "e_scalar_mixed_";
                            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
                        }
                        RealType energy_n = postprocessor<mesh_type>::compute_acoustic_energy_two_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
                        
                        RealType relative_energy = (energy_n - energy) / energy;
                        RealType relative_energy_0 = (energy_n - energy_0) / energy_0;
                        bool unstable_check_Q = (relative_energy > 1.0e-2) || (relative_energy_0 >= 1.0e-2);
                        if (unstable_check_Q) { // energy is increasing
                            approx_fail_check_Q = true;
                            // Computing errors
                              postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, assembler, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
                            break;
                        }
                        energy = energy_n;
                        if(it == nt){
                            // Computing errors
                            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, assembler, x_dof, exact_vel_fun, exact_flux_fun,simulation_log);
                        }
                    }

                    if(approx_fail_check_Q){
                        simulation_log << std::endl;
                        simulation_log << "Simulation is unstable for :"<< std::endl;
                        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
                        simulation_log << "Number of ERK steps =  " << s << std::endl;
                        simulation_log << "Number of time steps =  " << nt << std::endl;
                        simulation_log << "dt size =  " << dt << std::endl;
                        simulation_log << "h size =  " << lx/mesh_builder.get_nx() << std::endl;
                        simulation_log << "CFL (dt/h) =  " << dt/(lx/mesh_builder.get_nx()) << std::endl;
                        simulation_log << std::endl;
                        simulation_log.flush();
                        break;
                    }else{
                        simulation_log << "Simulation is stable for :"<< std::endl;
                        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
                        simulation_log << "Number of ERK steps =  " << s << std::endl;
                        simulation_log << "Number of time steps =  " << nt << std::endl;
                        simulation_log << "dt size =  " << dt << std::endl;
                        simulation_log << "h size =  " << lx/mesh_builder.get_nx() << std::endl;
                        simulation_log << "CFL (dt/h) =  " << dt/(lx/mesh_builder.get_nx()) << std::endl;
                        simulation_log << std::endl;
                        simulation_log.flush();
                        nt -= 5;
                        continue;
                    }
                }
            }
        simulation_log << " ******************************* " << std::endl;
        simulation_log << std::endl << std::endl;
    }
    
}

void EHHOFirstOrderEigenCFL(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
        
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionNonPolynomial);
    RealType t = 0.0;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    std::ofstream simulation_log("acoustic_two_fields_explicit_cfl.txt");
    
    for (int k = 0; k <= sim_data.m_k_degree; k++) {
        
        // Creating HHO approximation spaces and corresponding linear operator
        size_t cell_k_degree = k;
        if(sim_data.m_hdg_stabilization_Q){
            cell_k_degree++;
        }
        disk::hho_degree_info hho_di(cell_k_degree,k);
        
        simulation_log << " ******************************* " << std::endl;
        simulation_log << " Polynomial degree =  " << k << std::endl;
        simulation_log << std::endl;
        for(size_t l = 0; l <= sim_data.m_n_divs; l++){
                
                // Building a cartesian mesh
                timecounter tc;
                tc.tic();

                RealType lx = 1.0;
                RealType ly = 1.0;
                size_t nx = 10;
                size_t ny = 10;
                for (unsigned int i = 0; i < l; i++) {
                  nx *= 2;
                  ny *= 2;
                }
                typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
                typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
                mesh_type msh;

                cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
                mesh_builder.build_mesh();
                mesh_builder.move_to_mesh_storage(msh);
                std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
            
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
                    vp = 2.0;
                    rho = 1.0/(vp*vp); // this is required to make both formulations compatible by keeping kappa = 1
                    mat_data[0] = rho; // rho
                    mat_data[1] = vp; // seismic compressional velocity vp
                    return mat_data;
                };
                assembler.load_material_data(msh,acoustic_mat_fun);
                if(sim_data.m_hdg_stabilization_Q){
                    assembler.set_hdg_stabilization();
                }
                if(sim_data.m_scaled_stabilization_Q){
                    assembler.set_scaled_stabilization();
                }
                tc.toc();
                std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
            
                tc.tic();
                assembler.assemble_mass(msh);
                assembler.assemble(msh,rhs_fun);
                tc.toc();
                std::cout << bold << cyan << "Matrix assembly completed: " << tc << " seconds" << reset << std::endl;
            
                size_t n_face_dof = assembler.get_n_face_dof();
                tc.tic();
                erk_hho_scheme<RealType> erk_an(assembler.LHS,assembler.RHS,assembler.MASS,n_face_dof);
                erk_an.Kcc_inverse(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
                tc.toc();
                std::cout << bold << cyan << "ERK analysis created: " << tc << " seconds" << reset << std::endl;
            
                RealType lambda_max = 0;
                {
                SparseMatrix<RealType> Kg = assembler.LHS;
                SparseMatrix<RealType> Mg = assembler.MASS;
                tc.tic();
            //                for (int i = 0; i < n_cell_dof; i++) {
            //                    for (int j = 0; j < n_cell_dof; j++) {
            //                        Mg.coeffRef(i, j) = analysis.Mc_inv().coeffRef(i,j);
            //                    }
            //                }
                Kg = erk_an.Mc_inv()*Kg;
                
                Spectra::SparseGenMatProd<RealType> op(Kg);
                Spectra::GenEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                        Spectra::SparseGenMatProd<RealType> > max_eigs(&op, 1, 8);
                tc.toc();
                simulation_log << "Generalized Eigen Solver creation time: " << tc << " seconds" << std::endl;
                
                tc.tic();
                max_eigs.init();
                max_eigs.compute();
                tc.toc();
                if(max_eigs.info() == Spectra::SUCCESSFUL){
                    lambda_max = max_eigs.eigenvalues()(0).real();
                }
                simulation_log << "Generalized Eigen Solver compute time: " << tc << " seconds" << std::endl;
                    

                RealType h_T = std::numeric_limits<RealType>::max();
                for (auto& cell : msh) {
                 
                     RealType h = diameter(msh, cell);
                     if (h < h_T) {
                         h_T = h;
                     }
                }
                    
                auto beta_cfl = 1.0/(lambda_max);
                simulation_log << "Number of equations : " << Kg.rows() << std::endl;
                simulation_log << "Largest eigenvalue : " << lambda_max << std::endl;
                simulation_log << "l :  " << l << std::endl;
                simulation_log << "h :  " << h_T << std::endl;
                simulation_log << "Beta-CFL :  " << beta_cfl << std::endl;
                if(sim_data.m_scaled_stabilization_Q){
                    simulation_log << "CFL :  " << beta_cfl/(h_T*h_T) << std::endl;
                }else{
                    simulation_log << "CFL :  " << beta_cfl/h_T << std::endl;
                }
                simulation_log << std::endl;
                simulation_log.flush();
            }
        }
    }
    
    simulation_log << " ******************************* " << std::endl;
    simulation_log << std::endl << std::endl;
    
}
