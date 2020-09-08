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

void HeterogeneousGar6more2DIHHOSecondOrder(int argc, char **argv);

void IHHOFirstOrder(int argc, char **argv);

void IHHOSecondOrder(int argc, char **argv);

int main(int argc, char **argv)
{
        
//    HeterogeneousGar6more2DIHHOSecondOrder(argc, argv);
    IHHOFirstOrder(argc, argv);
//    IHHOSecondOrder(argc, argv);
}

void IHHOFirstOrder(int argc, char **argv){
    
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
    tc.toc();
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
    functions.set_function_type(scal_vec_analytic_functions::EFunctionType::EFunctionQuadraticInTime);
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
    
    tc.tic();
    auto assembler = elastoacoustic_four_fields_assembler<mesh_type>(msh, hho_di, e_bnd, a_bnd, e_material, a_material);
    assembler.set_interface_cell_indexes(interface_cell_pair_indexes);
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
    
    tc.tic();
    assembler.assemble_coupling_terms(msh);
    tc.toc();
    std::cout << bold << cyan << "Coupling Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, v_fun, flux_fun, s_v_fun, s_flux_fun);
    assembler.project_over_faces(msh, x_dof, v_fun, s_v_fun);
    
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "elasto_acoustic_four_fields_";
        postprocessor<mesh_type>::write_silo_four_fields_elastoacoustic(silo_file_name, it, msh, hho_di, x_dof, e_material, a_material, false);
    }
    
    std::ofstream simulation_log("elasto_acoustic_four_fields.txt");
    
//    if (sim_data.m_report_energy_Q) {
//        postprocessor<mesh_type>::compute_elastic_energy_three_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
//    }
    
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
    assembler.assemble(msh, f_fun, s_f_fun);
    tc.toc();
    std::cout << bold << cyan << "First stiffness assembly completed: " << tc << " seconds" << reset << std::endl;
    assembler.LHS += assembler.COUPLING; // Line to delete.
    dirk_hho_scheme<RealType> dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (sim_data.m_sc_Q) {
        std::vector<std::pair<size_t,size_t>> vec_cell_basis_data(2);
        vec_cell_basis_data[0] = std::make_pair(assembler.get_e_material_data().size(), assembler.get_e_cell_basis_data());
        vec_cell_basis_data[1] = std::make_pair(assembler.get_a_material_data().size(), assembler.get_a_cell_basis_data());
        dirk_an.set_static_condensation_data(vec_cell_basis_data, assembler.get_n_face_dof());
    }
    
    if (is_sdirk_Q) {
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        tc.tic();
        dirk_an.ComposeMatrix();
//        dirk_an.setIterativeSolver();
        dirk_an.DecomposeMatrix();
        tc.toc();
        std::cout << bold << cyan << "Matrix decomposed: " << tc << " seconds" << reset << std::endl;
    }
    Matrix<RealType, Dynamic, 1> x_dof_n;
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

            RealType t;
            Matrix<double, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {

                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }

                t = tn + c(i,0) * dt;
                // Manufactured solution
                auto v_fun      = functions.Evaluate_v(t);
                auto f_fun      = functions.Evaluate_f(t);
                auto s_v_fun    = functions.Evaluate_s_v(t);
                auto s_f_fun    = functions.Evaluate_s_f(t);
                
                assembler.get_e_bc_conditions().updateDirichletFunction(v_fun, 0);
                assembler.get_a_bc_conditions().updateDirichletFunction(s_v_fun, 0);
                assembler.assemble_rhs(msh, f_fun, s_f_fun);
                dirk_an.SetFg(assembler.RHS);
                dirk_an.irk_weight(yn, ki, dt, a(i,i),is_sdirk_Q);
                
                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "DIRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;
        t = tn + dt;
        auto v_fun     = functions.Evaluate_v(t);
        auto flux_fun     = functions.Evaluate_sigma(t);
        auto s_v_fun     = functions.Evaluate_s_v(t);
        auto s_flux_fun     = functions.Evaluate_s_q(t);

        if (sim_data.m_render_silo_files_Q) {
            std::string silo_file_name = "elasto_acoustic_four_fields_";
            postprocessor<mesh_type>::write_silo_four_fields_elastoacoustic(silo_file_name, it, msh, hho_di, x_dof, e_material, a_material, false);
        }
        
//        if (sim_data.m_report_energy_Q) {
//            postprocessor<mesh_type>::compute_elastic_energy_three_fields(msh, hho_di, assembler, t, x_dof, simulation_log);
//        }

        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_four_fields_elastoacoustic(msh, hho_di, assembler, x_dof, v_fun, flux_fun, s_v_fun, s_flux_fun, simulation_log);
        }

    }
    
    simulation_log << "Number of equations : " << dirk_an.DirkAnalysis().n_equations() << std::endl;
    simulation_log << "Number of DIRK steps =  " << s << std::endl;
    simulation_log << "Number of time steps =  " << nt << std::endl;
    simulation_log << "Step size =  " << dt << std::endl;
    simulation_log.flush();
}

void IHHOSecondOrder(int argc, char **argv){
    
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
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt = (tf-ti)/nt;
    
    scal_vec_analytic_functions functions;
    functions.set_function_type(scal_vec_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
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

    
    tc.tic();
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
    
    tc.tic();
    assembler.assemble_coupling_terms(msh);
    tc.toc();
    std::cout << bold << cyan << "Coupling Assembly completed: " << tc << " seconds" << reset << std::endl;

    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, u_dof_n, u_fun, s_u_fun);
    assembler.project_over_faces(msh, u_dof_n, u_fun, s_u_fun);
    assembler.project_over_cells(msh, v_dof_n, v_fun, s_v_fun);
    assembler.project_over_faces(msh, v_dof_n, v_fun, s_v_fun);
    assembler.project_over_cells(msh, a_dof_n, a_fun, s_a_fun);
    assembler.project_over_faces(msh, a_dof_n, a_fun, s_a_fun);
        
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "elasto_acoustic_two_fields_";
        postprocessor<mesh_type>::write_silo_two_fields_elastoacoustic(silo_file_name, it, msh, hho_di, v_dof_n, e_material, a_material, false);
    }

    std::ofstream simulation_log("elasto_acoustic_two_fields.txt");

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
        SparseMatrix<RealType> Kg = assembler.LHS;
        SparseMatrix<RealType> C = assembler.COUPLING;

        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += gamma*dt*C;
        assembler.LHS += assembler.MASS;
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
            analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
            std::vector<std::pair<size_t,size_t>> vec_cell_basis_data(2);
            vec_cell_basis_data[0] = std::make_pair(assembler.get_e_material_data().size(), assembler.get_e_cell_basis_data());
            vec_cell_basis_data[1] = std::make_pair(assembler.get_a_material_data().size(), assembler.get_a_cell_basis_data());
            analysis.condense_equations(vec_cell_basis_data);
        }else{
            analysis.set_Kg(assembler.LHS);
        }
//        analysis.set_iterative_solver();
        analysis.factorize();
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;

        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto u_fun      = functions.Evaluate_u(t);
            auto v_fun      = functions.Evaluate_v(t);
            auto f_fun      = functions.Evaluate_f(t);
            auto flux_fun   = functions.Evaluate_sigma(t);
            auto s_u_fun    = functions.Evaluate_s_u(t);
            auto s_v_fun    = functions.Evaluate_s_v(t);
            auto s_f_fun    = functions.Evaluate_s_f(t);
            auto s_flux_fun  = functions.Evaluate_s_q(t);
            

            tc.tic();
            assembler.get_e_bc_conditions().updateDirichletFunction(u_fun, 0);
            assembler.get_a_bc_conditions().updateDirichletFunction(s_u_fun, 0);
            assembler.assemble_rhs(msh, f_fun, s_f_fun);
            
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
                postprocessor<mesh_type>::write_silo_two_fields_elastoacoustic(silo_file_name, it, msh, hho_di, v_dof_n, e_material, a_material, false);
            }

//            if (sim_data.m_report_energy_Q) {
//                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
//            }

            if(it == nt){
                postprocessor<mesh_type>::compute_errors_two_fields_elastoacoustic(msh, hho_di, assembler, u_dof_n, u_fun, flux_fun, s_u_fun, s_flux_fun, simulation_log);
            }

        }
        simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
    
    return 0;
    
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
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, false> e_boundary_type;
    typedef disk::BoundaryConditions<mesh_type, true> a_boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.set_translation_data(-1.5, -1.5);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt = (tf-ti)/nt;
    
    scal_vec_analytic_functions functions;
    functions.set_function_type(scal_vec_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
    RealType t = ti;
        
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
            vp = std::sqrt(9.0); // seismic compressional velocity vp
            acoustic_material_data<RealType> material(rho,vp);
            return material;
        };
    
    auto null_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
            static_vector<RealType, 2> f{0,0};
            return f;
    };
    auto null_s_fun = [](const mesh_type::point_type& pt) -> RealType {
            return 0;
    };
    
    auto u_s_fun = [](const mesh_type::point_type& pt) -> RealType {
        RealType x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
        x = pt.x();
        y = pt.y();
        xc = 0.0;
        yc = 2.0/3.0;
        c = 10.0;
        lp = std::sqrt(9.0)/10.0;
        r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
        wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
        factor = (lp*lp/(2.0*M_PI*M_PI));
        return factor*wave;
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
        if (std::fabs(bar.y()) < eps) {
            interface_face_indexes.insert(fc_id);
            continue;
        }
        
    }
    
    for (auto & cell : msh ) {
        auto cell_ind = msh.lookup(cell);
        mesh_type::point_type bar = barycenter(msh, cell);
        if (bar.y() > 0) {
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
                if (bar.y() > 0) {
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
        if (bar.y() > 0) {
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
    e_bnd.addDirichletBC(disk::DirichletType::DIRICHLET, bc_elastic_id, null_fun);
    a_bnd.addDirichletBC(disk::DirichletType::DIRICHLET, bc_acoustic_id, null_s_fun);
    
    // Solving a primal HHO mixed problem

    
    tc.tic();
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
    
    tc.tic();
    assembler.assemble_coupling_terms(msh);
    tc.toc();
    std::cout << bold << cyan << "Coupling Assembly completed: " << tc << " seconds" << reset << std::endl;

    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, u_dof_n, null_fun, u_s_fun);
    assembler.project_over_faces(msh, u_dof_n, null_fun, u_s_fun);
    assembler.project_over_cells(msh, v_dof_n, null_fun, null_s_fun);
    assembler.project_over_faces(msh, v_dof_n, null_fun, null_s_fun);
    assembler.project_over_cells(msh, a_dof_n, null_fun, null_s_fun);
    assembler.project_over_faces(msh, a_dof_n, null_fun, null_s_fun);
        
    if (sim_data.m_render_silo_files_Q) {
        size_t it = 0;
        std::string silo_file_name = "elasto_acoustic_inhomogeneous_two_fields_";
        postprocessor<mesh_type>::write_silo_two_fields_elastoacoustic(silo_file_name, it, msh, hho_di, v_dof_n, e_material, a_material, false);
    }

    std::ofstream simulation_log("elasto_acoustic_inhomogeneous_two_fields.txt");
    
    std::ofstream sensor_1_log("s1_elasto_acoustic_one_field_h.csv");
    std::ofstream sensor_2_log("s2_elasto_acoustic_one_field_h.csv");
    std::ofstream sensor_3_log("s3_elasto_acoustic_one_field_h.csv");
    bool e_side_Q = true;
    typename mesh_type::point_type s1_pt(-1.0/3.0, -1.0/3.0);
    typename mesh_type::point_type s2_pt( 0.0, -1.0/3.0);
    typename mesh_type::point_type s3_pt(+1.0/3.0, -1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> s1_pt_cell = std::make_pair(s1_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s2_pt_cell = std::make_pair(s2_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s3_pt_cell = std::make_pair(s3_pt, -1);

    postprocessor<mesh_type>::record_velocity_data_elasto_acoustic_two_fields(0, s1_pt_cell, msh, hho_di, assembler, u_dof_n, v_dof_n, e_side_Q, sensor_1_log);
    postprocessor<mesh_type>::record_velocity_data_elasto_acoustic_two_fields(0, s2_pt_cell, msh, hho_di, assembler, u_dof_n, v_dof_n, e_side_Q, sensor_2_log);
    postprocessor<mesh_type>::record_velocity_data_elasto_acoustic_two_fields(0, s3_pt_cell, msh, hho_di, assembler, u_dof_n, v_dof_n, e_side_Q, sensor_3_log);
    
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
        assembler.assemble(msh, null_fun, null_s_fun);
        SparseMatrix<RealType> Kg = assembler.LHS;
        SparseMatrix<RealType> C = assembler.COUPLING;

        assembler.LHS *= beta*(dt*dt);
        assembler.LHS += gamma*dt*C;
        assembler.LHS += assembler.MASS;
        linear_solver<RealType> analysis;
        if (sim_data.m_sc_Q) {
            analysis.set_Kg(assembler.LHS, assembler.get_n_face_dof());
            
            std::vector<std::pair<size_t,size_t>> vec_cell_basis_data(2);
            vec_cell_basis_data[0] = std::make_pair(assembler.get_e_material_data().size(), assembler.get_e_cell_basis_data());
            vec_cell_basis_data[1] = std::make_pair(assembler.get_a_material_data().size(), assembler.get_a_cell_basis_data());
            analysis.condense_equations(vec_cell_basis_data);
        }else{
            analysis.set_Kg(assembler.LHS);
        }
//        analysis.set_iterative_solver();
        analysis.factorize();
        tc.toc();
        std::cout << bold << cyan << "Stiffness assembly completed: " << tc << " seconds" << reset << std::endl;

        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;

            tc.tic();
            assembler.RHS.setZero();
            assembler.apply_bc(msh);
            
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
                std::string silo_file_name = "elasto_acoustic_inhomogeneous_two_fields_";
                postprocessor<mesh_type>::write_silo_two_fields_elastoacoustic(silo_file_name, it, msh, hho_di, v_dof_n, e_material, a_material, false);
            }

            postprocessor<mesh_type>::record_velocity_data_elasto_acoustic_two_fields(it, s1_pt_cell, msh, hho_di, assembler, u_dof_n, v_dof_n, e_side_Q, sensor_1_log);
            postprocessor<mesh_type>::record_velocity_data_elasto_acoustic_two_fields(it, s2_pt_cell, msh, hho_di, assembler, u_dof_n, v_dof_n, e_side_Q, sensor_2_log);
            postprocessor<mesh_type>::record_velocity_data_elasto_acoustic_two_fields(it, s3_pt_cell, msh, hho_di, assembler, u_dof_n, v_dof_n, e_side_Q, sensor_3_log);
            
//            if (sim_data.m_report_energy_Q) {
//                postprocessor<mesh_type>::compute_acoustic_energy_one_field(msh, hho_di, assembler, t, p_dof_n, v_dof_n, simulation_log);
//            }

        }
        
        simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
        simulation_log << "Number of time steps =  " << nt << std::endl;
        simulation_log << "Step size =  " << dt << std::endl;
        simulation_log.flush();
    }
    
    return 0;
    
}
