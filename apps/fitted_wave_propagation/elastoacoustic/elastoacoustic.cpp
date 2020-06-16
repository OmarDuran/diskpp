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

    RealType lx = 2.0;
    RealType ly = 1.0;
    size_t nx = 4;
    size_t ny = 2;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, false> e_boundary_type;
    typedef disk::BoundaryConditions<mesh_type, true> a_boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.set_translation_data(-1.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.5;
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

    
    // Classify cells per material data and bc faces
    
    auto elastic_mat_fun = [](const typename mesh_type::point_type& pt) -> elastic_material_data<RealType> {
            double x,y;
            x = pt.x();
            y = pt.y();
            RealType rho, vp, vs;
            rho = 1.0; // fluid mass density
            vp = std::sqrt(3.0); // seismic compressional velocity vp
            vs = 1.0; // seismic shear velocity vs
            elastic_material_data<RealType> material(rho,vp,vs);
            return material;
        };
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> acoustic_material_data<RealType> {
            double x,y;
            x = pt.x();
            y = pt.y();
            RealType rho, vp;
            rho = 1.0; // fluid mass density
            vp = 1.0; // seismic compressional velocity vp
            acoustic_material_data<RealType> material(rho,vp);
            return material;
        };
    
    std::map<size_t,elastic_material_data<RealType>> e_material;
    std::map<size_t,acoustic_material_data<RealType>> a_material;
    std::set<size_t> elastic_bc_face_indexes, acoustic_bc_face_indexes, interface_face_indexes;
    std::map<size_t,std::pair<size_t,size_t>> interface_cell_pair_indexes;
    
    RealType eps = 1.0e-10;
    for (auto face_it = msh.faces_begin(); face_it != msh.faces_end(); face_it++)
    {
        const auto face = *face_it;
        mesh_type::point_type bar = barycenter(msh, face);
        auto fc_id = msh.lookup(face);
        if (std::fabs(bar.x()) < eps) {
            interface_face_indexes.insert(fc_id);
            continue;
        }
        
    }
    
    for (auto & cell : msh ) {
        auto cell_ind = msh.lookup(cell);
        mesh_type::point_type bar = barycenter(msh, cell);
        if (bar.x() > 0) {
            acoustic_material_data<RealType> material = acoustic_mat_fun(bar);
            a_material.insert(std::make_pair(cell_ind,material));
        }else{
            elastic_material_data<RealType> material = elastic_mat_fun(bar);
            e_material.insert(std::make_pair(cell_ind,material));
        }
        
        auto cell_faces = faces(msh,cell);
        for (auto face :cell_faces) {
            auto fc_id = msh.lookup(face);
            bool is_member_Q = interface_face_indexes.find(fc_id) != interface_face_indexes.end();
            if (is_member_Q) {
                if (bar.x() > 0) {
                    interface_cell_pair_indexes[fc_id].second = cell_ind;
                }else{
                    interface_cell_pair_indexes[fc_id].first = cell_ind;
                }
            }
        }
    }
    
    size_t bc_elastic_id = 0;
    size_t bc_acoustic_id = 1;
    for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
    {
        auto face = *face_it;
        mesh_type::point_type bar = barycenter(msh, face);
        auto fc_id = msh.lookup(face);
        if (bar.x() > 0) {
            disk::bnd_info bi{bc_acoustic_id, true};
            msh.backend_storage()->boundary_info.at(fc_id) = bi;
            acoustic_bc_face_indexes.insert(fc_id);
        }else{
            disk::bnd_info bi{bc_elastic_id, true};
            msh.backend_storage()->boundary_info.at(fc_id) = bi;
            elastic_bc_face_indexes.insert(fc_id);
        }
        
    }
    
    // detect interface elastic - acoustic
    e_boundary_type e_bnd(msh);
    a_boundary_type a_bnd(msh);
    e_bnd.addDirichletBC(disk::DirichletType::DIRICHLET, bc_elastic_id, u_fun);
    a_bnd.addDirichletBC(disk::DirichletType::DIRICHLET, bc_acoustic_id, s_u_fun);
    
    // Solving a primal HHO mixed problem

    
//    tc.tic();

    auto assembler = elastoacoustic_two_fields_assembler<mesh_type>(msh, hho_di, e_bnd, a_bnd, e_material, a_material);
    assembler.set_interface_cell_indexes(interface_cell_pair_indexes);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;

    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;

    //    std::cout << "M = " << assembler.MASS.toDense() << std::endl;
    
    tc.tic();
    assembler.assemble_coupling_terms(msh);
    tc.toc();
    std::cout << bold << cyan << "Coupling Assembly completed: " << tc << " seconds" << reset << std::endl;
    
//    std::cout << "C = " << assembler.COUPLING.toDense() << std::endl;

    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, u_dof_n, u_fun, s_u_fun);
    assembler.project_over_cells(msh, v_dof_n, v_fun, s_v_fun);
    assembler.project_over_cells(msh, a_dof_n, a_fun, s_a_fun);
        
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "elasto_acoustic_two_fields_";
        postprocessor<mesh_type>::write_silo_two_fields_elastoacoustic(silo_file_name, it, msh, hho_di, u_dof_n, u_fun, s_u_fun, e_material, a_material, true);
    }

    std::ofstream simulation_log("acoustic_one_field.txt");

//    if (sim_data.m_report_energy_Q) {
//        postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
//    }

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
        assembler.assemble(msh, f_fun, s_f_fun);
//        std::cout << "K = " << assembler.LHS.toDense() << std::endl;
    
        SparseMatrix<RealType> Kg = assembler.LHS;
        SparseMatrix<RealType> C = assembler.COUPLING;
        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += gamma*dt*C;
        assembler.LHS += assembler.MASS;
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
            analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
            analysis.condense_equations(std::make_pair(msh.cells_size(), assembler.get_cell_basis_data()));
        }else{
            analysis.set_Kg(assembler.LHS);
        }
        analysis.set_iterative_solver();
        analysis.factorize();
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;

        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto u_fun      = functions.Evaluate_u(t);
            auto f_fun      = functions.Evaluate_f(t);
            auto s_u_fun    = functions.Evaluate_s_u(t);
            auto s_f_fun    = functions.Evaluate_s_f(t);
            

            tc.tic();
            assembler.get_e_bc_conditions().updateDirichletFunction(u_fun, 0);
            assembler.get_a_bc_conditions().updateDirichletFunction(s_u_fun, 0);
            assembler.assemble(msh, f_fun, s_f_fun);
            assembler.apply_bc(msh);
//            assembler.assemble_rhs(msh, rhs_fun);

//            std::cout << "r = " << assembler.RHS << std::endl;
            
            // Compute intermediate state for scalar and rate
            u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = Kg*u_dof_n;
            Matrix<RealType, Dynamic, 1> res_v = C*v_dof_n;
            assembler.RHS -= res;
            assembler.RHS -= res_v;
            tc.toc();
            std::cout << bold << cyan << "Rhs assembly completed: " << tc << " seconds" << reset << std::endl;

            tc.tic();
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update scalar and rate
            u_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;

            if (sim_data.m_render_silo_files_Q) {
                std::string silo_file_name = "elasto_acoustic_two_fields_";
                postprocessor<mesh_type>::write_silo_two_fields_elastoacoustic(silo_file_name, it, msh, hho_di, u_dof_n, u_fun, s_u_fun, e_material, a_material, true);
            }

//            if (sim_data.m_report_energy_Q) {
//                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
//            }

//            if(it == nt){
//                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, p_dof_n, exact_scal_fun, exact_flux_fun, simulation_log);
//            }

        }
        simulation_log << "Number of equations : " << assembler.RHS.rows() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
    
    return 0;
}
