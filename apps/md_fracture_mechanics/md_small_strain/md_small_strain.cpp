//
//  md_small_strain.cpp
//  md_small_strain
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

// application common sources
#include "../common/display_settings.hpp"
#include "../common/fitted_geometry_builders.hpp"
#include "../common/linear_solver.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

// ----- common data types ------------------------------
using RealType = double;
typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
typedef disk::mesh<RealType, 3, disk::generic_mesh_storage<RealType, 3>>  mesh_type_3d;
typedef disk::mesh<RealType, 1, disk::generic_mesh_storage<RealType, 1>>  mesh1d_type;
typedef disk::BoundaryConditions<mesh_type, false> boundary_type;
typedef disk::BoundaryConditions<mesh_type_3d, false> boundary_type_3d;
typedef disk::BoundaryConditions<mesh1d_type, true> boundary_1d_type;

void SurfaceStrain(simulation_data & sim_data);

void Fratures2D(simulation_data & sim_data);

void Fratures3D(simulation_data & sim_data);

int main(int argc, char **argv)
{
    
    simulation_data sim_data = preprocessor::process_convergence_test_args(argc, argv);
    sim_data.print_simulation_data();
    
    Fratures2D(sim_data);
//    Fratures3D(sim_data);
    return 0;
}

void Fratures3D(simulation_data & sim_data){
    
    timecounter tc;
    
    // Reading the polygonal mesh
    tc.tic();
    mesh_type_3d msh;
        
    gmsh_3d_reader<RealType> mesh_builder;
    std::string mesh_file = "meshes/fractured_reservoir_3d.msh";
    mesh_builder.set_gmsh_file(mesh_file);

    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    std::string silo_mesh_file = "mesh";
    postprocessor<mesh_type_3d>::write_silo_mesh(silo_mesh_file, msh);
    
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    std::vector<std::pair<size_t,size_t>> fracture_pairs;
    std::vector<std::pair<size_t,size_t>> f0_pairs,f1_pairs,f2_pairs;
    std::vector<mesh_type_3d::point_type> fracture_bars;
    fracture_bars.reserve(msh.faces_size());
    auto storage = msh.backend_storage();
    for (size_t face_id = 0; face_id < msh.faces_size(); face_id++)
    {
        auto& egde = storage->surfaces[face_id];
        mesh_type_3d::point_type bar = barycenter(msh, egde);
        fracture_bars.push_back(bar);
    }

    auto are_equal_Q = [](const mesh_type_3d::point_type& a, const mesh_type_3d::point_type& b)-> bool {
        bool check_Q = fabs(a.x() - b.x()) <= 1.0e-3 && fabs(a.y() - b.y()) <= 1.0e-3 && fabs(a.z() - b.z()) <= 1.0e-3;
        return check_Q;
    };

    for (size_t i = 0; i < fracture_bars.size(); i++) {
        mesh_type_3d::point_type bar_i = fracture_bars.at(i);
        for (size_t j = i+1; j < fracture_bars.size(); j++) {
            mesh_type_3d::point_type bar_j = fracture_bars.at(j);
             if (are_equal_Q(bar_i,bar_j)) {
                 fracture_pairs.push_back(std::make_pair(i, j));
             }
        }
    }


    
//    // detect end point mortars
    std::vector<std::pair<size_t,size_t>> end_point_mortars;
    size_t fracture_cell_ind = 0;
    for (auto chunk: fracture_pairs) {

        auto& edge_l = storage->surfaces[chunk.first];
        auto& edge_r = storage->surfaces[chunk.second];
        auto points_l = edge_l.point_ids();
        auto points_r = edge_r.point_ids();

        std::vector<size_t> indexes;
        for (auto index : points_l) {
            indexes.push_back(index);
        }
        for (auto index : points_r) {
            indexes.push_back(index);
        }

        std::sort(indexes.begin(), indexes.end());
        const auto duplicate = std::adjacent_find(indexes.begin(), indexes.end());
        if (duplicate != indexes.end()){
            size_t index = *duplicate;
            std::cout << "Duplicate element = " << *duplicate << "\n";
            end_point_mortars.push_back(std::make_pair(fracture_cell_ind, index));
        }
        fracture_cell_ind++;
    }

//    end_point_mortars.clear();

//    if(0){// find points
//        mesh_type_3d::point_type p;
//        p.x() = 10.0;
//        p.y() = 1.0;
//        auto storage = msh.backend_storage();
//        for (size_t i = 0; i < msh.points_size(); i++)
//        {
//            auto point = storage->points.at(i);
//            if (are_equal_Q(point,p)){
//                int aka = 0;
//            }
//        }
//
//        // print points
//        for(auto chunk : f1_pairs){
//            auto &face_l = storage->edges.at(chunk.first);
//            auto &face_r = storage->edges.at(chunk.second);
//
//            auto ptids_l = face_l.point_ids();
//            auto ptids_r = face_r.point_ids();
//
//            std::cout << "left " << std::endl;
//            std::cout << ptids_l[0] << ", " << ptids_l[1] << std::endl;
//            std::cout << "right " << std::endl;
//            std::cout << ptids_r[0] << ", " << ptids_r[1] << std::endl;
//            int aka = 0;
//        }
//
//    }

    tc.toc();
    std::cout << bold << cyan << "Fracture mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;

    // filling up fractures
    std::vector<fracture_3d<mesh_type_3d> > fractures;
    std::vector<restriction_3d > restrictions;
    if(1){
        fracture_3d<mesh_type_3d> f;
        f.m_pairs = fracture_pairs;
//        f.m_bl_index = end_point_mortars[0].second;
//        f.m_el_index = end_point_mortars[1].second;
//        f.m_br_index = end_point_mortars[0].second;
//        f.m_er_index = end_point_mortars[1].second;
        f.build(msh);

        fractures.push_back(f);

        restriction_3d r0;
        r0.m_f_index = {0,0};
        r0.m_p_index = {0,0};
        r0.m_s_index = {0,1};
        restrictions.push_back(r0);

        restriction_3d r1;
        r1.m_f_index = {0,0};
        r1.m_p_index = {1,1};
        r1.m_s_index = {0,1};
        restrictions.push_back(r1);
    }


    // Constant elastic properties
    RealType rho,l,mu;
    rho = 1.0;
    l = 1.0;//2000.0;
    mu = 1.0;//2000.0;
    elastic_material_data<RealType> material(rho,l,mu);

    // Creating HHO approximation spaces and corresponding linear operator
    size_t face_k_degree = sim_data.m_k_degree;
    size_t cell_k_degree = face_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,face_k_degree);

    // Solving a scalar primal HHO problem
    auto null_v_fun = [](const mesh_type_3d::point_type& pt) -> static_vector<RealType, 3> {
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        RealType ux = 0.0;
        RealType uy = 0.0;
        RealType uz = 0.0;
        return static_vector<RealType, 3>{ux, uy, uz};
    };
    
    auto u_east_fun = [](const mesh_type_3d::point_type& pt) -> static_vector<RealType, 3> {
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        RealType ux = -0.1;
        RealType uy = -0.0;
        RealType uz = -0.0;
        return static_vector<RealType, 3>{ux, uy, uz};
    };
    
    auto u_north_fun = [](const mesh_type_3d::point_type& pt) -> static_vector<RealType, 3> {
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        RealType ux = -0.0;
        RealType uy = -0.1;
        RealType uz = -0.0;
        return static_vector<RealType, 3>{ux, uy, uz};
    };
    
    auto u_top_fun = [](const mesh_type_3d::point_type& pt) -> static_vector<RealType, 3> {
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        RealType ux = -0.0;
        RealType uy = -0.0;
        RealType uz = -0.1;
        return static_vector<RealType, 3>{ux, uy, uz};
    };
    

    auto rhs_fun = [](const mesh_type_3d::point_type& pt) -> static_vector<RealType, 3> {
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        RealType rx = 0.0;
        RealType ry = 0.0;
        RealType rz = 0.0;
        return static_vector<RealType, 3>{rx, ry, rz};
    };

    boundary_type_3d bnd(msh);

    RealType minlx = -10.0;
    RealType minly = -10.0;
    RealType minlz = -10.0;
    RealType maxlx = 10.0;
    RealType maxly = 10.0;
    RealType maxlz = 10.0;
    // defining boundary conditions
    {
        size_t bc_west_id = 0;
        size_t bc_north_id = 1;
        size_t bc_east_id = 2;
        size_t bc_south_id = 3;
        size_t bc_bottom_id = 4;
        size_t bc_top_id = 5;
        RealType eps = 1.0e-4;

        for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
        {
            auto face = *face_it;
            mesh_type_3d::point_type bar = barycenter(msh, face);
            auto fc_id = msh.lookup(face);
            
            if(std::fabs(bar.x()-minlx) < eps){
                disk::bnd_info bi{bc_west_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(std::fabs(bar.x()-maxlx) < eps){
                disk::bnd_info bi{bc_east_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(std::fabs(bar.y()-minly) < eps){
                disk::bnd_info bi{bc_south_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(std::fabs(bar.y()-maxly) < eps){
                disk::bnd_info bi{bc_north_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.z()-minlz) < eps) {
                disk::bnd_info bi{bc_bottom_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.z()-maxlz) < eps) {
                disk::bnd_info bi{bc_top_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }

        }

        bnd.addDirichletBC(disk::DY, bc_south_id, null_v_fun);
        bnd.addDirichletBC(disk::DX, bc_east_id, u_east_fun);
        bnd.addDirichletBC(disk::DY, bc_north_id, u_north_fun);
        bnd.addDirichletBC(disk::DX, bc_west_id, null_v_fun);
        bnd.addDirichletBC(disk::DZ, bc_bottom_id, null_v_fun);
        bnd.addDirichletBC(disk::DZ, bc_top_id, u_top_fun);
    }

    tc.tic();
//    auto assembler = elastic_two_fields_assembler_3d<mesh_type_3d>(msh, hho_di, bnd, fracture_pairs, end_point_mortars, fractures,restrictions);
    auto assembler = elastic_two_fields_assembler_3d<mesh_type_3d>(msh, hho_di, bnd, fractures);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
    }
    assembler.load_material_data(msh,material);
    assembler.assemble(msh, rhs_fun);
    assembler.apply_bc(msh);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    std::cout << bold << cyan << "ndof : " << assembler.LHS.rows() << reset << std::endl;

    bool write_kg_Q = false;
    if(write_kg_Q){
        std::ofstream mat_file;
        mat_file.open ("matrix.txt");
        size_t n_cells_dof = assembler.get_n_cells_dofs();
        size_t n_dof = assembler.LHS.rows();
        mat_file << assembler.LHS.block(n_cells_dof, n_cells_dof, n_dof-n_cells_dof, n_dof-n_cells_dof).toDense() <<  std::endl;
        mat_file.close();
    }

    // Solving LS
    Matrix<RealType, Dynamic, 1> x_dof;
    tc.tic();
    linear_solver<RealType> analysis(assembler.LHS);
    analysis.set_direct_solver(false);
    tc.toc();
    std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;

    tc.tic();
    analysis.factorize();
    tc.toc();
    std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;

    tc.tic();
    x_dof = analysis.solve(assembler.RHS);
    tc.toc();
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    std::cout << bold << cyan << "Number of equations : " << analysis.n_equations() << reset << std::endl;

    
    auto ue_fun = [](const mesh_type_3d::point_type& pt) -> static_vector<RealType, 3> {
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        RealType ux = -(x+10.0)/200.0;
        RealType uy = -(y+10.0)/200.0;
        RealType uz = -(z+10.0)/200.0;
        return static_vector<RealType, 3>{ux, uy, uz};
    };
    
    auto flux_fun = [](const mesh_type_3d::point_type& pt) -> static_matrix<RealType, 3,3> {
        
        static_matrix<RealType, 3,3> sigma = static_matrix<RealType, 3,3>::Zero(3,3);
        RealType x,y,z;
        x = pt.x();
        y = pt.y();
        z = pt.z();
        sigma(0,0) = -0.025;
        sigma(1,1) = -0.025;
        sigma(2,2) = -0.025;
        return sigma;
    };
    
//    assembler.project_over_cells(msh,x_dof,ue_fun,flux_fun);
//    assembler.cells_residuals(msh,ue_fun,flux_fun);

    // render silo
    size_t it = 0;
    std::string silo_file_name = "single_fracture_3d";
    postprocessor<mesh_type_3d>::write_silo_u_field_3d(silo_file_name, it, msh, hho_di, x_dof);

//    // sigma n and t
//    size_t f_ind = 0;
//    {
//        fracture<mesh_type> f = fractures[f_ind];
//        auto storage = msh.backend_storage();
//        size_t n_cells_dof = assembler.get_n_cells_dofs();
//        size_t n_faces_dofs = assembler.get_n_faces_dofs();
//        size_t n_hybrid_dofs = assembler.get_n_hybrid_dofs();
//        size_t n_skins_dofs = assembler.get_n_skin_dof();
//        size_t cell_ind = 0;
//        size_t n_cells = f.m_pairs.size();
//        size_t sigma_degree = hho_di.face_degree()-1;
//        size_t n_f_sigma_bs = disk::scalar_basis_size(sigma_degree, mesh_type::dimension - 1);
//        size_t n_data = 2*n_cells;
//
//        Matrix<RealType, Dynamic, 2> data_n = Matrix<RealType, Dynamic, Dynamic>::Zero(n_cells, 2);
//        Matrix<RealType, Dynamic, 2> data_t = Matrix<RealType, Dynamic, Dynamic>::Zero(n_cells, 2);
//
//        Matrix<RealType, Dynamic, 3> data_u_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//        Matrix<RealType, Dynamic, 3> data_u_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//
//        Matrix<RealType, Dynamic, 3> data_div_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//        Matrix<RealType, Dynamic, 3> data_div_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//
//        Matrix<RealType, Dynamic, 3> data_sig_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//        Matrix<RealType, Dynamic, 3> data_sig_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//
//        Matrix<RealType, Dynamic, 3> data_s_n = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//        Matrix<RealType, Dynamic, 3> data_s_t = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
//
//        mesh_type::point_type p0;
//        for (auto chunk : f.m_pairs) {
//
//            size_t cell_ind_l = f.m_elements[cell_ind].first;
//            size_t cell_ind_r = f.m_elements[cell_ind].second;
//            auto& face_l = storage->edges[chunk.first];
//            auto& face_r = storage->edges[chunk.second];
//            auto& cell_l = storage->surfaces[cell_ind_l];
//            auto& cell_r = storage->surfaces[cell_ind_r];
//
//            auto points = face_l.point_ids();
//
//            for (size_t ip = 0; ip < points.size(); ip++) {
//
//                auto pt_id = points[ip];
//                auto bar = *std::next(msh.points_begin(), pt_id);
//
//                if ((ip == 0) && (cell_ind == 0)) {
//                    p0 = bar;
//                }
//
//                // hybrid sigma evaluation
//                {
//
//                    auto face_basis = make_scalar_monomial_basis(msh, face_l, sigma_degree);
//                    size_t offset_n = cell_ind*2*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
//                    Matrix<RealType, Dynamic, 1> sigma_n_x_dof = x_dof.block(offset_n, 0, n_f_sigma_bs, 1);
//
//                    size_t offset_t = cell_ind*2*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_f_sigma_bs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
//                    Matrix<RealType, Dynamic, 1> sigma_t_x_dof = x_dof.block(offset_t, 0, n_f_sigma_bs, 1);
//
//                    auto t_phi = face_basis.eval_functions( bar );
//                    assert(t_phi.rows() == face_basis.size());
//
//                    auto snh = disk::eval(sigma_n_x_dof, t_phi);
//                    auto sth = disk::eval(sigma_t_x_dof, t_phi);
//
//                    RealType dv = (bar-p0).to_vector().norm();
//                    data_n(cell_ind,0) += 0.5*dv;
//                    data_n(cell_ind,1) += 0.5*snh;
//
//                    data_t(cell_ind,0) += 0.5*dv;
//                    data_t(cell_ind,1) += 0.5*sth;
//                }
//
//                // u evaluation
//                {
//                    size_t face_l_offset = assembler.get_n_cells_dofs() + assembler.compress_indexes().at(chunk.first);
//
//                    size_t face_r_offset = assembler.get_n_cells_dofs() + assembler.compress_indexes().at(chunk.second);
//
//                    auto face_basis_l = make_vector_monomial_basis(msh, face_l, hho_di.face_degree());
//                    auto face_basis_r = make_vector_monomial_basis(msh, face_r, hho_di.face_degree());
//                    size_t n_u_bs = disk::vector_basis_size(hho_di.face_degree(), mesh_type::dimension - 1, mesh_type::dimension);
//
//                    Matrix<RealType, Dynamic, 1> u_l_x_dof = x_dof.block(face_l_offset, 0, n_u_bs, 1);
//                    Matrix<RealType, Dynamic, 1> u_r_x_dof = x_dof.block(face_r_offset, 0, n_u_bs, 1);
//
//                    auto t_phi_l = face_basis_l.eval_functions( bar );
//                    auto t_phi_r = face_basis_r.eval_functions( bar );
//                    assert(t_phi_l.rows() == face_basis_l.size());
//                    assert(t_phi_r.rows() == face_basis_r.size());
//
//                    auto ul = disk::eval(u_l_x_dof, t_phi_l);
//                    auto ur = disk::eval(u_r_x_dof, t_phi_r);
//
//                    RealType dv = (bar-p0).to_vector().norm();
//                    {
//                        const auto n = disk::normal(msh, cell_l, face_l);
//                        const auto t = disk::tanget(msh, cell_l, face_l);
//                        auto unl = ul.dot(n);
//                        auto utl = ul.dot(t);
//                        data_u_l(2*cell_ind+ip,0) = dv;
//                        data_u_l(2*cell_ind+ip,1) = unl;
//                        data_u_l(2*cell_ind+ip,2) = utl;
//                    }
//                    {
//                        const auto n = disk::normal(msh, cell_r, face_r);
//                        const auto t = disk::tanget(msh, cell_r, face_r);
//                        auto unr = ur.dot(n);
//                        auto utr = ur.dot(t);
//                        data_u_r(2*cell_ind+ip,0) = dv;
//                        data_u_r(2*cell_ind+ip,1) = unr;
//                        data_u_r(2*cell_ind+ip,2) = utr;
//                    }
//                }
//
//                // skins div
//                {
//
//                    auto face_basis_l = make_scalar_monomial_basis(msh, face_l, hho_di.face_degree());
//                    auto face_basis_r = make_scalar_monomial_basis(msh, face_r, hho_di.face_degree());
//                    if (f.m_flips_l.at(cell_ind)) {
//                        face_basis_l.swap_nodes();
//                    }
//                    if (f.m_flips_r.at(cell_ind)) {
//                        face_basis_r.swap_nodes();
//                    }
//
//
//                    size_t sig_bs = 3;
//                    size_t n_skin_bs = f.m_skin_bs;
//                    size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
//                    size_t base = n_cells_dof + n_faces_dofs + f_offset;
//                    size_t p_sn_l = base+cell_ind*sig_bs+0*n_skin_bs;
//                    size_t p_st_l = base+cell_ind*sig_bs+1*n_skin_bs;
//                    size_t p_sn_r = base+cell_ind*sig_bs+2*n_skin_bs;
//                    size_t p_st_r = base+cell_ind*sig_bs+3*n_skin_bs;
//                    Matrix<RealType, Dynamic, 1> sn_l_dof = x_dof.block(p_sn_l, 0, sig_bs, 1);
//                    Matrix<RealType, Dynamic, 1> st_l_dof = x_dof.block(p_st_l, 0, sig_bs, 1);
//                    Matrix<RealType, Dynamic, 1> sn_r_dof = x_dof.block(p_sn_r, 0, sig_bs, 1);
//                    Matrix<RealType, Dynamic, 1> st_r_dof = x_dof.block(p_st_r, 0, sig_bs, 1);
//
//                    auto t_div_phi_l = face_basis_l.eval_div_flux_functions( bar );
//                    auto t_div_phi_r = face_basis_r.eval_div_flux_functions( bar );
//
////                    std::cout << st_l_dof << std::endl;
////                    std::cout << st_r_dof << std::endl;
//
//                    auto div_sn_l = disk::eval(sn_l_dof, t_div_phi_l);
//                    auto div_st_l = disk::eval(st_l_dof, t_div_phi_l);
//
//                    auto div_sn_r = disk::eval(sn_r_dof, t_div_phi_r);
//                    auto div_st_r = disk::eval(st_r_dof, t_div_phi_r);
//
//                    RealType dv = (bar-p0).to_vector().norm();
//                    data_div_l(2*cell_ind+ip,0) = dv;
//                    data_div_l(2*cell_ind+ip,1) = div_sn_l;
//                    data_div_l(2*cell_ind+ip,2) = div_st_l;
//
//                    data_div_r(2*cell_ind+ip,0) = dv;
//                    data_div_r(2*cell_ind+ip,1) = div_sn_r;
//                    data_div_r(2*cell_ind+ip,2) = div_st_r;
//                }
//
//                // skins sigma
//                {
//                    auto face_basis_l = make_scalar_monomial_basis(msh, face_l, hho_di.face_degree());
//                    auto face_basis_r = make_scalar_monomial_basis(msh, face_r, hho_di.face_degree());
//                    if (f.m_flips_l.at(cell_ind)) {
//                        face_basis_l.swap_nodes();
//                    }
//                    if (f.m_flips_r.at(cell_ind)) {
//                        face_basis_r.swap_nodes();
//                    }
//
//
//                    size_t sig_bs = 3;
//                    size_t n_skin_bs = f.m_skin_bs;
//                    size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
//                    size_t base = n_cells_dof + n_faces_dofs + f_offset;
//                    size_t p_sn_l = base+cell_ind*sig_bs+0*n_skin_bs;
//                    size_t p_st_l = base+cell_ind*sig_bs+1*n_skin_bs;
//                    size_t p_sn_r = base+cell_ind*sig_bs+2*n_skin_bs;
//                    size_t p_st_r = base+cell_ind*sig_bs+3*n_skin_bs;
//                    Matrix<RealType, Dynamic, 1> sn_l_dof = x_dof.block(p_sn_l, 0, sig_bs, 1);
//                    Matrix<RealType, Dynamic, 1> st_l_dof = x_dof.block(p_st_l, 0, sig_bs, 1);
//                    Matrix<RealType, Dynamic, 1> sn_r_dof = x_dof.block(p_sn_r, 0, sig_bs, 1);
//                    Matrix<RealType, Dynamic, 1> st_r_dof = x_dof.block(p_st_r, 0, sig_bs, 1);
//
//                    auto t_phi_l = face_basis_l.eval_flux_functions( bar );
//                    auto t_phi_r = face_basis_r.eval_flux_functions( bar );
//
//                    auto sn_l = disk::eval(sn_l_dof, t_phi_l);
//                    auto st_l = disk::eval(st_l_dof, t_phi_l);
//
//                    auto sn_r = disk::eval(sn_r_dof, t_phi_r);
//                    auto st_r = disk::eval(st_r_dof, t_phi_r);
//
//                    RealType dv = (bar-p0).to_vector().norm();
//                    data_sig_l(2*cell_ind+ip,0) = dv;
//                    data_sig_l(2*cell_ind+ip,1) = sn_l;
//                    data_sig_l(2*cell_ind+ip,2) = st_l;
//
//                    data_sig_r(2*cell_ind+ip,0) = dv;
//                    data_sig_r(2*cell_ind+ip,1) = sn_r;
//                    data_sig_r(2*cell_ind+ip,2) = st_r;
//                }
//
//
//             }
//
//            cell_ind++;
//        }
//
//        size_t pre = 15;
//        // skins Lagrange multiplier
//        if(1){
//
//            size_t sig_bs = 3;
//            size_t n_skin_bs = f.m_skin_bs;
//            size_t uL_bs = cell_ind+1;
//            size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
//            size_t base = n_cells_dof + n_faces_dofs + f_offset;
//            size_t p_sn_l = base+n_cells*sig_bs+0*n_skin_bs;
//            size_t p_st_l = base+n_cells*sig_bs+1*n_skin_bs;
//            size_t p_sn_r = base+n_cells*sig_bs+2*n_skin_bs;
//            size_t p_st_r = base+n_cells*sig_bs+3*n_skin_bs;
//
//            Matrix<RealType, Dynamic, 1> un_l_dof = x_dof.block(p_sn_l, 0, uL_bs, 1);
//            Matrix<RealType, Dynamic, 1> ut_l_dof = x_dof.block(p_st_l, 0, uL_bs, 1);
//            Matrix<RealType, Dynamic, 1> un_r_dof = x_dof.block(p_sn_r, 0, uL_bs, 1);
//            Matrix<RealType, Dynamic, 1> ut_r_dof = x_dof.block(p_st_r, 0, uL_bs, 1);
//
//            std::cout << std::setprecision(pre) << "unL_l =  " << un_l_dof << std::endl;
//            std::cout << std::setprecision(pre) << "unL_r =  " << un_r_dof << std::endl;
//            std::cout << std::setprecision(pre) << "utL_l =  " << ut_l_dof << std::endl;
//            std::cout << std::setprecision(pre) << "utL_r =  " << ut_r_dof << std::endl;
//        }
//
//        // sigma normal and mortar u0 evaluation
//        if(1){
//            size_t n_mortar_displacements = 2*2;
//            size_t n_skin_bs = assembler.get_n_skin_dof();
//            size_t n_f_hybrid_dofs = assembler.get_n_f_hybrid_dofs();
//            size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
//            size_t points_offset = n_cells_dof + n_faces_dofs + n_skin_bs + n_f_hybrid_dofs + f_ind * n_mortar_displacements;
//            Matrix<RealType, Dynamic, 1> sigma_dof = x_dof.block(points_offset,0,n_mortar_displacements,1);
//            std::cout << std::setprecision(pre) << "sigma =  " << sigma_dof << std::endl;
//        }
//
//        {
//            std::ofstream sn_file;
//            sn_file.open ("sigma_n.txt");
//            sn_file << std::setprecision(pre) << data_n <<  std::endl;
//            sn_file.close();
//
//            std::ofstream st_file;
//            st_file.open ("sigma_t.txt");
//            st_file << std::setprecision(pre) << data_t <<  std::endl;
//            st_file.close();
//
//            std::ofstream ul_file;
//            ul_file.open ("u_l.txt");
//            ul_file << std::setprecision(pre) << data_u_l <<  std::endl;
//            ul_file.close();
//
//            std::ofstream ur_file;
//            ur_file.open ("u_r.txt");
//            ur_file << std::setprecision(pre) << data_u_r <<  std::endl;
//            ur_file.close();
//
//        }
//
//        {
//            std::ofstream divs_l_file;
//            divs_l_file.open("divs_l.txt");
//            divs_l_file << std::setprecision(pre) << data_div_l <<  std::endl;
//            divs_l_file.close();
//
//            std::ofstream divs_r_file;
//            divs_r_file.open("divs_r.txt");
//            divs_r_file << std::setprecision(pre) << data_div_r <<  std::endl;
//            divs_r_file.close();
//
//            std::ofstream s_l_file;
//            s_l_file.open ("s_l.txt");
//            s_l_file << std::setprecision(pre) << data_sig_l <<  std::endl;
//            s_l_file.close();
//
//            std::ofstream s_r_file;
//            s_r_file.open ("s_r.txt");
//            s_r_file << std::setprecision(pre) << data_sig_r <<  std::endl;
//            s_r_file.close();
//
//        }
//
//    }
    
    return;
    
}

void Fratures2D(simulation_data & sim_data){
    
    timecounter tc;
    
    // Reading the polygonal mesh
    tc.tic();
    mesh_type msh;
    
    // Reading the polygonal mesh
    polygon_2d_mesh_reader<RealType> mesh_builder;
//    std::string mesh_file = "meshes/simple_mesh_single_crack_nel_2.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_nel_4.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_4.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_8.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_42.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_20.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_32.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_40.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_735.txt";
//
//    std::string mesh_file = "meshes/base_polymesh_cross_fracture_nel_22.txt";
//    std::string mesh_file = "meshes/base_polymesh_cross_nel_22.txt";
//    std::string mesh_file = "meshes/base_polymesh_cross_fracture_nel_88.txt";
//    std::string mesh_file = "meshes/base_polymesh_cross_nel_88.txt";
//    std::string mesh_file = "meshes/base_polymesh_cross_fracture_nel_352.txt";
//    std::string mesh_file = "meshes/base_polymesh_cross_nel_352.txt";
//
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_111.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_nel_111.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_444.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_nel_444.txt";
//      std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_581.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_1533.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_1965.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_fracture_nel_11588.txt";
//    std::string mesh_file = "meshes/base_polymesh_internal_nel_1965.txt";
//
    std::string mesh_file = "meshes/base_polymesh_yshape_fracture_nel_414.txt";
//    std::string mesh_file = "meshes/base_polymesh_yshape_fracture_nel_801.txt";
    mesh_builder.set_poly_mesh_file(mesh_file);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    
//    gmsh_2d_reader<RealType> mesh_builder;
//    std::string mesh_file = "meshes/fractured_reservoir.msh";
//    mesh_builder.set_gmsh_file(mesh_file);
//    mesh_builder.build_mesh();
//    mesh_builder.move_to_mesh_storage(msh);
//    std::string silo_mesh_file = "mesh";
//    postprocessor<mesh_type>::write_silo_mesh(silo_mesh_file, msh);
    
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    std::vector<std::pair<size_t,size_t>> fracture_pairs;
    std::vector<std::pair<size_t,size_t>> f0_pairs,f1_pairs,f2_pairs;
    std::vector<mesh_type::point_type> fracture_bars;
    fracture_bars.reserve(msh.faces_size());
    auto storage = msh.backend_storage();
    for (size_t face_id = 0; face_id < msh.faces_size(); face_id++)
    {
        auto& egde = storage->edges[face_id];
        mesh_type::point_type bar = barycenter(msh, egde);
        fracture_bars.push_back(bar);
    }
    
    auto are_equal_Q = [](const mesh_type::point_type& a, const mesh_type::point_type& b)-> bool {
        bool check_Q = fabs(a.x() - b.x()) <= 1.0e-4 && fabs(a.y() - b.y()) <= 1.0e-4;
        return check_Q;
    };
    
    auto is_diag_Q = [](const mesh_type::point_type& a)-> bool {
        bool check_Q = fabs(a.x() - a.y()) <= 1.0e-3;
        return check_Q;
    };
    
    auto is_horizontal_Q = [](const mesh_type::point_type& a)-> bool {
        bool check_Q = fabs(1.0 - a.y()) <= 1.0e-3;
        return check_Q;
    };

    for (size_t i = 0; i < fracture_bars.size(); i++) {
        mesh_type::point_type bar_i = fracture_bars.at(i);
        for (size_t j = i+1; j < fracture_bars.size(); j++) {
            mesh_type::point_type bar_j = fracture_bars.at(j);
             if (are_equal_Q(bar_i,bar_j)) {
                 fracture_pairs.push_back(std::make_pair(i, j));
                 if (is_diag_Q(bar_i)) {
                     f0_pairs.push_back(std::make_pair(i, j));
                 } else if (is_horizontal_Q(bar_i)) {
                     f1_pairs.push_back(std::make_pair(i, j));
                 } else{
                     f2_pairs.push_back(std::make_pair(i, j));
                 }
             }
        }
    }
    
    // detect end point mortars
    std::vector<std::pair<size_t,size_t>> end_point_mortars;
    size_t fracture_cell_ind = 0;
    for (auto chunk: fracture_pairs) {
        
        auto& edge_l = storage->edges[chunk.first];
        auto& edge_r = storage->edges[chunk.second];
        auto points_l = edge_l.point_ids();
        auto points_r = edge_r.point_ids();
        
        std::vector<size_t> indexes;
        for (auto index : points_l) {
            indexes.push_back(index);
        }
        for (auto index : points_r) {
            indexes.push_back(index);
        }
        
        std::sort(indexes.begin(), indexes.end());
        const auto duplicate = std::adjacent_find(indexes.begin(), indexes.end());
        if (duplicate != indexes.end()){
            size_t index = *duplicate;
            std::cout << "Duplicate element = " << *duplicate << "\n";
            end_point_mortars.push_back(std::make_pair(fracture_cell_ind, index));
        }
        fracture_cell_ind++;
    }
    
//    end_point_mortars.clear();
    
    if(0){// find points
        mesh_type::point_type p;
        p.x() = 10.0;
        p.y() = 1.0;
        auto storage = msh.backend_storage();
        for (size_t i = 0; i < msh.points_size(); i++)
        {
            auto point = storage->points.at(i);
            if (are_equal_Q(point,p)){
                int aka = 0;
            }
        }
        
        // print points
        for(auto chunk : f1_pairs){
            auto &face_l = storage->edges.at(chunk.first);
            auto &face_r = storage->edges.at(chunk.second);

            auto ptids_l = face_l.point_ids();
            auto ptids_r = face_r.point_ids();

            std::cout << "left " << std::endl;
            std::cout << ptids_l[0] << ", " << ptids_l[1] << std::endl;
            std::cout << "right " << std::endl;
            std::cout << ptids_r[0] << ", " << ptids_r[1] << std::endl;
            int aka = 0;
        }
        
    }
    
    tc.toc();
    std::cout << bold << cyan << "Fracture mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // filling up fractures
    std::vector<fracture<mesh_type> > fractures;
    std::vector<restriction > restrictions;
    if(0){
        fracture<mesh_type> f;
        f.m_pairs = fracture_pairs;
        f.m_bl_index = 0;//mesh_builder.fracture_nodes()[0].second;
        f.m_el_index = 1;//mesh_builder.fracture_nodes()[1].second;
        f.m_br_index = 2;//mesh_builder.fracture_nodes()[0].second;
        f.m_er_index = 3;//mesh_builder.fracture_nodes()[1].second;
        f.build(msh);

        fractures.push_back(f);
        
        restriction r0;
        r0.m_f_index = {0,0};
        r0.m_p_index = {0,0};
        r0.m_s_index = {0,1};
        restrictions.push_back(r0);
        
        restriction r1;
        r1.m_f_index = {0,0};
        r1.m_p_index = {1,1};
        r1.m_s_index = {0,1};
        restrictions.push_back(r1);
    }
    
    if(1){
        fracture<mesh_type> f0;
        f0.m_pairs = f0_pairs;
        f0.m_bl_index = 6;
        f0.m_el_index = 7;
        f0.m_br_index = 6;
        f0.m_er_index = 449;
        f0.build(msh);
        fractures.push_back(f0);
        
        fracture<mesh_type> f1;
        f1.m_pairs = f1_pairs;
        f1.m_bl_index = 4;
        f1.m_el_index = 7;
        f1.m_br_index = 472;
        f1.m_er_index = 475;
        f1.build(msh);
        f1.m_bc_type = {1,1};
        f1.m_bc_data = {{-0.05,0},{-0.05,0}};
        fractures.push_back(f1);
        
        fracture<mesh_type> f2;
        f2.m_pairs = f2_pairs;
        f2.m_bl_index = 5;
        f2.m_el_index = 475;
        f2.m_br_index = 455;
        f2.m_er_index = 449;
        f2.build(msh);
        f2.m_bc_type = {1,1};
        f2.m_bc_data = {{0,-0.2},{0,-0.2}};
        fractures.push_back(f2);
        

        
        
        restriction r0;
        r0.m_f_index = {0,0};
        r0.m_p_index = {0,0};
        r0.m_s_index = {0,1};
        restrictions.push_back(r0);
        
//        restriction r1;
//        r1.m_f_index = {0,2};
//        r1.m_p_index = {1,1};
//        r1.m_s_index = {1,1};
//        restrictions.push_back(r1);
//
//        restriction r2;
//        r2.m_f_index = {2,1};
//        r2.m_p_index = {1,1};
//        r2.m_s_index = {0,1};
//        restrictions.push_back(r2);
//
//        restriction r3;
//        r3.m_f_index = {0,1};
//        r3.m_p_index = {1,1};
//        r3.m_s_index = {0,0};
//        restrictions.push_back(r3);
        
//        restriction r4;
//        r4.m_f_index = {1,1};
//        r4.m_p_index = {0,0};
//        r4.m_s_index = {0,1};
//        restrictions.push_back(r4);
        
//        restriction r5;
//        r5.m_f_index = {2,2};
//        r5.m_p_index = {0,0};
//        r5.m_s_index = {0,1};
//        restrictions.push_back(r5);
    }
    
    // Constant elastic properties
    RealType rho,l,mu;
    rho = 1.0;
    l = 1.0;//2000.0;
    mu = 1.0;//2000.0;
    elastic_material_data<RealType> material(rho,l,mu);

    // Creating HHO approximation spaces and corresponding linear operator
    size_t face_k_degree = sim_data.m_k_degree;
    size_t cell_k_degree = face_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,face_k_degree);

    // Solving a scalar primal HHO problem
    auto null_v_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType ux = 0.0;
        RealType uy = 0.0;
        return static_vector<RealType, 2>{ux, uy};
    };
    auto u_top_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType ux = -0.0;
        RealType uy = -0.2;
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto u_right_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType ux = -0.05;
        RealType uy = -0.0;
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto rhs_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType rx = 0.0;
        RealType ry = 0.0;
        return static_vector<RealType, 2>{rx, ry};
    };
    
    boundary_type bnd(msh);
//    RealType minlx = 0.0;
//    RealType minly = 0.0;
//    RealType maxlx = 7.0;
//    RealType maxly = 6.0;
    RealType minlx = -10.0;
    RealType minly = -10.0;
    RealType maxlx = 10.0;
    RealType maxly = 10.0;
    // defining boundary conditions
    {
        size_t bc_D_bot_id = 0;
        size_t bc_N_right_id = 1;
        size_t bc_D_top_id = 2;
        size_t bc_N_left_id = 3;
        RealType eps = 1.0e-4;
        
        for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
        {
            auto face = *face_it;
            mesh_type::point_type bar = barycenter(msh, face);
            auto fc_id = msh.lookup(face);
            if (std::fabs(bar.y()-minly) < eps) {
                disk::bnd_info bi{bc_D_bot_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(std::fabs(bar.x()-maxlx) < eps){
                disk::bnd_info bi{bc_N_right_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.y()-maxly) < eps) {
                disk::bnd_info bi{bc_D_top_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.x()-minlx) < eps) {
                disk::bnd_info bi{bc_N_left_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
        }
        
//        bnd.addDirichletBC(disk::DY, bc_D_bot_id, null_v_fun);
//        bnd.addDirichletBC(disk::DX, bc_N_right_id, null_v_fun);
//        bnd.addNeumannBC(disk::NEUMANN, bc_D_top_id, u_top_fun);
//        bnd.addDirichletBC(disk::DX, bc_N_left_id, null_v_fun);
        
//        bnd.addDirichletBC(disk::DY, bc_D_bot_id, null_v_fun);
//        bnd.addDirichletBC(disk::DX, bc_N_right_id, null_v_fun);
//        bnd.addDirichletBC(disk::DY, bc_D_top_id, u_top_fun);
//        bnd.addDirichletBC(disk::DX, bc_N_left_id, null_v_fun);
        
//        bnd.addDirichletBC(disk::DY, bc_D_bot_id, null_v_fun);
//        bnd.addNeumannBC(disk::NEUMANN, bc_N_right_id, null_v_fun);
//        bnd.addDirichletBC(disk::DY, bc_D_top_id, u_top_fun);
//        bnd.addNeumannBC(disk::NEUMANN, bc_N_left_id, null_v_fun);
        
//        bnd.addDirichletBC(disk::DIRICHLET, bc_D_bot_id, null_v_fun);
//        bnd.addNeumannBC(disk::NEUMANN, bc_N_right_id, null_v_fun);
//        bnd.addDirichletBC(disk::DY, bc_D_top_id, u_top_fun);
//        bnd.addNeumannBC(disk::NEUMANN, bc_N_left_id, null_v_fun);
        
        bnd.addDirichletBC(disk::DY, bc_D_bot_id, null_v_fun);
        bnd.addDirichletBC(disk::DX, bc_N_right_id, u_right_fun);
        bnd.addDirichletBC(disk::DY, bc_D_top_id, u_top_fun);
        bnd.addDirichletBC(disk::DX, bc_N_left_id, null_v_fun);
        
    }

    tc.tic();
    auto assembler = elastic_two_fields_assembler<mesh_type>(msh, hho_di, bnd, fracture_pairs, end_point_mortars, fractures,restrictions);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
    }
    assembler.load_material_data(msh,material);
    assembler.assemble(msh, rhs_fun);
    assembler.apply_bc(msh);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    bool write_kg_Q = false;
    if(write_kg_Q){
        std::ofstream mat_file;
        mat_file.open ("matrix.txt");
        size_t n_cells_dof = assembler.get_n_cells_dofs();
        size_t n_dof = assembler.LHS.rows();
        mat_file << assembler.LHS.block(n_cells_dof, n_cells_dof, n_dof-n_cells_dof, n_dof-n_cells_dof).toDense() <<  std::endl;
        mat_file.close();
    }
    
    // Solving LS
    Matrix<RealType, Dynamic, 1> x_dof;
    tc.tic();
    linear_solver<RealType> analysis(assembler.LHS);
    analysis.set_direct_solver(false);
    tc.toc();
    std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    analysis.factorize();
    tc.toc();
    std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    x_dof = analysis.solve(assembler.RHS);
    tc.toc();
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    std::cout << bold << cyan << "Number of equations : " << analysis.n_equations() << reset << std::endl;
    
    assembler.project_over_skin_cells(msh,x_dof);
//    std::cout << "x = " << x_dof.tail(4) << std::endl;
//    std::cout << "r = " << assembler.LHS * x_dof - assembler.RHS << std::endl;
    
    // render silo
    size_t it = 0;
    std::string silo_file_name = "single_fracture";
    postprocessor<mesh_type>::write_silo_u_field(silo_file_name, it, msh, hho_di, x_dof);
    
    // sigma n and t
    size_t f_ind = 0;
    {
        fracture<mesh_type> f = fractures[f_ind];
        auto storage = msh.backend_storage();
        size_t n_cells_dof = assembler.get_n_cells_dofs();
        size_t n_faces_dofs = assembler.get_n_faces_dofs();
        size_t n_hybrid_dofs = assembler.get_n_hybrid_dofs();
        size_t n_skins_dofs = assembler.get_n_skin_dof();
        size_t cell_ind = 0;
        size_t n_cells = f.m_pairs.size();
        size_t sigma_degree = hho_di.face_degree()-1;
        size_t n_f_sigma_bs = disk::scalar_basis_size(sigma_degree, mesh_type::dimension - 1);
        size_t n_data = 2*n_cells;
        
        Matrix<RealType, Dynamic, 2> data_n = Matrix<RealType, Dynamic, Dynamic>::Zero(n_cells, 2);
        Matrix<RealType, Dynamic, 2> data_t = Matrix<RealType, Dynamic, Dynamic>::Zero(n_cells, 2);
        
        Matrix<RealType, Dynamic, 3> data_u_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        Matrix<RealType, Dynamic, 3> data_u_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        
        Matrix<RealType, Dynamic, 3> data_div_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        Matrix<RealType, Dynamic, 3> data_div_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);

        Matrix<RealType, Dynamic, 3> data_sig_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        Matrix<RealType, Dynamic, 3> data_sig_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        
        Matrix<RealType, Dynamic, 3> data_s_n = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        Matrix<RealType, Dynamic, 3> data_s_t = Matrix<RealType, Dynamic, Dynamic>::Zero(n_data, 3);
        
        mesh_type::point_type p0;
        for (auto chunk : f.m_pairs) {
            
            size_t cell_ind_l = f.m_elements[cell_ind].first;
            size_t cell_ind_r = f.m_elements[cell_ind].second;
            auto& face_l = storage->edges[chunk.first];
            auto& face_r = storage->edges[chunk.second];
            auto& cell_l = storage->surfaces[cell_ind_l];
            auto& cell_r = storage->surfaces[cell_ind_r];
            
            auto points = face_l.point_ids();
            
            for (size_t ip = 0; ip < points.size(); ip++) {
                
                auto pt_id = points[ip];
                auto bar = *std::next(msh.points_begin(), pt_id);
                
                if ((ip == 0) && (cell_ind == 0)) {
                    p0 = bar;
                }

                // hybrid sigma evaluation
                {

                    auto face_basis = make_scalar_monomial_basis(msh, face_l, sigma_degree);
                    size_t offset_n = cell_ind*2*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_n_x_dof = x_dof.block(offset_n, 0, n_f_sigma_bs, 1);
                    
                    size_t offset_t = cell_ind*2*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_f_sigma_bs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_t_x_dof = x_dof.block(offset_t, 0, n_f_sigma_bs, 1);
                    
                    auto t_phi = face_basis.eval_functions( bar );
                    assert(t_phi.rows() == face_basis.size());
                    
                    auto snh = disk::eval(sigma_n_x_dof, t_phi);
                    auto sth = disk::eval(sigma_t_x_dof, t_phi);

                    RealType dv = (bar-p0).to_vector().norm();
                    data_n(cell_ind,0) += 0.5*dv;
                    data_n(cell_ind,1) += 0.5*snh;
                    
                    data_t(cell_ind,0) += 0.5*dv;
                    data_t(cell_ind,1) += 0.5*sth;
                }
                
                // u evaluation
                {
                    size_t face_l_offset = assembler.get_n_cells_dofs() + assembler.compress_indexes().at(chunk.first);
                    
                    size_t face_r_offset = assembler.get_n_cells_dofs() + assembler.compress_indexes().at(chunk.second);
                    
                    auto face_basis_l = make_vector_monomial_basis(msh, face_l, hho_di.face_degree());
                    auto face_basis_r = make_vector_monomial_basis(msh, face_r, hho_di.face_degree());
                    size_t n_u_bs = disk::vector_basis_size(hho_di.face_degree(), mesh_type::dimension - 1, mesh_type::dimension);
                    
                    Matrix<RealType, Dynamic, 1> u_l_x_dof = x_dof.block(face_l_offset, 0, n_u_bs, 1);
                    Matrix<RealType, Dynamic, 1> u_r_x_dof = x_dof.block(face_r_offset, 0, n_u_bs, 1);
                    
                    auto t_phi_l = face_basis_l.eval_functions( bar );
                    auto t_phi_r = face_basis_r.eval_functions( bar );
                    assert(t_phi_l.rows() == face_basis_l.size());
                    assert(t_phi_r.rows() == face_basis_r.size());
                    
                    auto ul = disk::eval(u_l_x_dof, t_phi_l);
                    auto ur = disk::eval(u_r_x_dof, t_phi_r);

                    RealType dv = (bar-p0).to_vector().norm();
                    {
                        const auto n = disk::normal(msh, cell_l, face_l);
                        const auto t = disk::tanget(msh, cell_l, face_l);
                        auto unl = ul.dot(n);
                        auto utl = ul.dot(t);
                        data_u_l(2*cell_ind+ip,0) = dv;
                        data_u_l(2*cell_ind+ip,1) = unl;
                        data_u_l(2*cell_ind+ip,2) = utl;
                    }
                    {
                        const auto n = disk::normal(msh, cell_r, face_r);
                        const auto t = disk::tanget(msh, cell_r, face_r);
                        auto unr = ur.dot(n);
                        auto utr = ur.dot(t);
                        data_u_r(2*cell_ind+ip,0) = dv;
                        data_u_r(2*cell_ind+ip,1) = unr;
                        data_u_r(2*cell_ind+ip,2) = utr;
                    }
                }
                
                // skins div
                {

                    auto face_basis_l = make_scalar_monomial_basis(msh, face_l, hho_di.face_degree());
                    auto face_basis_r = make_scalar_monomial_basis(msh, face_r, hho_di.face_degree());
                    if (f.m_flips_l.at(cell_ind)) {
                        face_basis_l.swap_nodes();
                    }
                    if (f.m_flips_r.at(cell_ind)) {
                        face_basis_r.swap_nodes();
                    }
                    
                    
                    size_t sig_bs = 3;
                    size_t n_skin_bs = f.m_skin_bs;
                    size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
                    size_t base = n_cells_dof + n_faces_dofs + f_offset;
                    size_t p_sn_l = base+cell_ind*sig_bs+0*n_skin_bs;
                    size_t p_st_l = base+cell_ind*sig_bs+1*n_skin_bs;
                    size_t p_sn_r = base+cell_ind*sig_bs+2*n_skin_bs;
                    size_t p_st_r = base+cell_ind*sig_bs+3*n_skin_bs;
                    Matrix<RealType, Dynamic, 1> sn_l_dof = x_dof.block(p_sn_l, 0, sig_bs, 1);
                    Matrix<RealType, Dynamic, 1> st_l_dof = x_dof.block(p_st_l, 0, sig_bs, 1);
                    Matrix<RealType, Dynamic, 1> sn_r_dof = x_dof.block(p_sn_r, 0, sig_bs, 1);
                    Matrix<RealType, Dynamic, 1> st_r_dof = x_dof.block(p_st_r, 0, sig_bs, 1);
                    
                    auto t_div_phi_l = face_basis_l.eval_div_flux_functions( bar );
                    auto t_div_phi_r = face_basis_r.eval_div_flux_functions( bar );
                    
//                    std::cout << st_l_dof << std::endl;
//                    std::cout << st_r_dof << std::endl;
                    
                    auto div_sn_l = disk::eval(sn_l_dof, t_div_phi_l);
                    auto div_st_l = disk::eval(st_l_dof, t_div_phi_l);
                    
                    auto div_sn_r = disk::eval(sn_r_dof, t_div_phi_r);
                    auto div_st_r = disk::eval(st_r_dof, t_div_phi_r);
                    
                    RealType dv = (bar-p0).to_vector().norm();
                    data_div_l(2*cell_ind+ip,0) = dv;
                    data_div_l(2*cell_ind+ip,1) = div_sn_l;
                    data_div_l(2*cell_ind+ip,2) = div_st_l;
                    
                    data_div_r(2*cell_ind+ip,0) = dv;
                    data_div_r(2*cell_ind+ip,1) = div_sn_r;
                    data_div_r(2*cell_ind+ip,2) = div_st_r;
                }
                
                // skins sigma
                {
                    auto face_basis_l = make_scalar_monomial_basis(msh, face_l, hho_di.face_degree());
                    auto face_basis_r = make_scalar_monomial_basis(msh, face_r, hho_di.face_degree());
                    if (f.m_flips_l.at(cell_ind)) {
                        face_basis_l.swap_nodes();
                    }
                    if (f.m_flips_r.at(cell_ind)) {
                        face_basis_r.swap_nodes();
                    }
                    
                    
                    size_t sig_bs = 3;
                    size_t n_skin_bs = f.m_skin_bs;
                    size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
                    size_t base = n_cells_dof + n_faces_dofs + f_offset;
                    size_t p_sn_l = base+cell_ind*sig_bs+0*n_skin_bs;
                    size_t p_st_l = base+cell_ind*sig_bs+1*n_skin_bs;
                    size_t p_sn_r = base+cell_ind*sig_bs+2*n_skin_bs;
                    size_t p_st_r = base+cell_ind*sig_bs+3*n_skin_bs;
                    Matrix<RealType, Dynamic, 1> sn_l_dof = x_dof.block(p_sn_l, 0, sig_bs, 1);
                    Matrix<RealType, Dynamic, 1> st_l_dof = x_dof.block(p_st_l, 0, sig_bs, 1);
                    Matrix<RealType, Dynamic, 1> sn_r_dof = x_dof.block(p_sn_r, 0, sig_bs, 1);
                    Matrix<RealType, Dynamic, 1> st_r_dof = x_dof.block(p_st_r, 0, sig_bs, 1);
                    
                    auto t_phi_l = face_basis_l.eval_flux_functions( bar );
                    auto t_phi_r = face_basis_r.eval_flux_functions( bar );
                    
                    auto sn_l = disk::eval(sn_l_dof, t_phi_l);
                    auto st_l = disk::eval(st_l_dof, t_phi_l);
                    
                    auto sn_r = disk::eval(sn_r_dof, t_phi_r);
                    auto st_r = disk::eval(st_r_dof, t_phi_r);
                    
                    RealType dv = (bar-p0).to_vector().norm();
                    data_sig_l(2*cell_ind+ip,0) = dv;
                    data_sig_l(2*cell_ind+ip,1) = sn_l;
                    data_sig_l(2*cell_ind+ip,2) = st_l;
                    
                    data_sig_r(2*cell_ind+ip,0) = dv;
                    data_sig_r(2*cell_ind+ip,1) = sn_r;
                    data_sig_r(2*cell_ind+ip,2) = st_r;
                }
                
                                                
             }
            
            cell_ind++;
        }
        
        size_t pre = 15;
        // skins Lagrange multiplier
        if(1){
            
            size_t sig_bs = 3;
            size_t n_skin_bs = f.m_skin_bs;
            size_t uL_bs = cell_ind+1;
            size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
            size_t base = n_cells_dof + n_faces_dofs + f_offset;
            size_t p_sn_l = base+n_cells*sig_bs+0*n_skin_bs;
            size_t p_st_l = base+n_cells*sig_bs+1*n_skin_bs;
            size_t p_sn_r = base+n_cells*sig_bs+2*n_skin_bs;
            size_t p_st_r = base+n_cells*sig_bs+3*n_skin_bs;
            
            Matrix<RealType, Dynamic, 1> un_l_dof = x_dof.block(p_sn_l, 0, uL_bs, 1);
            Matrix<RealType, Dynamic, 1> ut_l_dof = x_dof.block(p_st_l, 0, uL_bs, 1);
            Matrix<RealType, Dynamic, 1> un_r_dof = x_dof.block(p_sn_r, 0, uL_bs, 1);
            Matrix<RealType, Dynamic, 1> ut_r_dof = x_dof.block(p_st_r, 0, uL_bs, 1);
            
            std::cout << std::setprecision(pre) << "unL_l =  " << un_l_dof << std::endl;
            std::cout << std::setprecision(pre) << "unL_r =  " << un_r_dof << std::endl;
            std::cout << std::setprecision(pre) << "utL_l =  " << ut_l_dof << std::endl;
            std::cout << std::setprecision(pre) << "utL_r =  " << ut_r_dof << std::endl;
        }
        
        // sigma normal and mortar u0 evaluation
        if(1){
            size_t n_mortar_displacements = 2*2;
            size_t n_skin_bs = assembler.get_n_skin_dof();
            size_t n_f_hybrid_dofs = assembler.get_n_f_hybrid_dofs();
            size_t f_offset = assembler.compress_fracture_indexes().at(f_ind);
            size_t points_offset = n_cells_dof + n_faces_dofs + n_skin_bs + n_f_hybrid_dofs + f_ind * n_mortar_displacements;
            Matrix<RealType, Dynamic, 1> sigma_dof = x_dof.block(points_offset,0,n_mortar_displacements,1);
            std::cout << std::setprecision(pre) << "sigma =  " << sigma_dof << std::endl;
        }
        
        {
            std::ofstream sn_file;
            sn_file.open ("sigma_n.txt");
            sn_file << std::setprecision(pre) << data_n <<  std::endl;
            sn_file.close();
            
            std::ofstream st_file;
            st_file.open ("sigma_t.txt");
            st_file << std::setprecision(pre) << data_t <<  std::endl;
            st_file.close();
            
            std::ofstream ul_file;
            ul_file.open ("u_l.txt");
            ul_file << std::setprecision(pre) << data_u_l <<  std::endl;
            ul_file.close();
            
            std::ofstream ur_file;
            ur_file.open ("u_r.txt");
            ur_file << std::setprecision(pre) << data_u_r <<  std::endl;
            ur_file.close();
            
        }
        
        {
            std::ofstream divs_l_file;
            divs_l_file.open("divs_l.txt");
            divs_l_file << std::setprecision(pre) << data_div_l <<  std::endl;
            divs_l_file.close();
            
            std::ofstream divs_r_file;
            divs_r_file.open("divs_r.txt");
            divs_r_file << std::setprecision(pre) << data_div_r <<  std::endl;
            divs_r_file.close();
            
            std::ofstream s_l_file;
            s_l_file.open ("s_l.txt");
            s_l_file << std::setprecision(pre) << data_sig_l <<  std::endl;
            s_l_file.close();
            
            std::ofstream s_r_file;
            s_r_file.open ("s_r.txt");
            s_r_file << std::setprecision(pre) << data_sig_r <<  std::endl;
            s_r_file.close();
            
        }
        
    }
    
    return;
    
}

void SurfaceStrain(simulation_data & sim_data){
    
    mesh1d_type msh;
    
    line_1d_mesh_reader<RealType> mesh_builder;
    std::string mesh_file = "meshes/base_line_nel_4.txt";
    mesh_builder.set_line_mesh_file(mesh_file);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    
    // Constant elastic properties
    RealType rho,l,mu;
    rho = 1.0;
    l = 1.0;//2000.0;
    mu = 1.0;//2000.0;
    elastic_material_data<RealType> material(rho,l,mu);

    // Creating HHO approximation spaces and corresponding linear operator
    size_t face_k_degree = sim_data.m_k_degree;
    size_t cell_k_degree = face_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,face_k_degree);
    
    auto rhs_fun = [](const mesh1d_type::point_type& pt) -> RealType {
        RealType x;
        x = pt.x();
        RealType r = 6.0;
        return r;
    };
    
    auto null_u_fun = [](const mesh1d_type::point_type& pt) -> RealType {
        RealType x;
        x = pt.x();
        RealType u = 0.0;
        return u;
    };
    
    auto u_fun = [](const mesh1d_type::point_type& pt) -> RealType {
        RealType x;
        x = pt.x();
        RealType u = x * (1-x);
        return u;
    };
    
    auto s_fun = [](const mesh1d_type::point_type& pt) -> RealType {
        RealType x;
        x = pt.x();
        RealType s = 3 - 6.0*x;
        return s;
    };
    
    boundary_1d_type bnd(msh);
    // defining boundary conditions
    if(1){
        size_t bc_D_left_id = 0;
        size_t bc_D_right_id = 1;
        RealType eps = 1.0e-4;
        RealType minlx = 0.0;
        RealType maxlx = 1.0;
        for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
        {
            auto face = *face_it;
            mesh1d_type::point_type bar = barycenter(msh, face);
            auto fc_id = msh.lookup(face);
            if (std::fabs(bar.x()-minlx) < eps) {
                disk::bnd_info bi{bc_D_left_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(std::fabs(bar.x()-maxlx) < eps){
                disk::bnd_info bi{bc_D_right_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
        }
                
        bnd.addDirichletBC(disk::DIRICHLET, bc_D_left_id, null_u_fun);
        bnd.addDirichletBC(disk::DIRICHLET, bc_D_right_id, null_u_fun);

    }
    
    auto assembler = elastic_1d_two_fields_assembler<mesh1d_type>(msh, hho_di, bnd);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
    }
    assembler.load_material_data(msh,material);
    assembler.assemble(msh, rhs_fun);

    
    std::ofstream mat_file;
    mat_file.open ("matrix.txt");
    mat_file << assembler.LHS.toDense() <<  std::endl;
    mat_file.close();

    
    // Solving LS
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, u_fun, s_fun);
    assembler.project_over_faces(msh, x_dof, u_fun);
    
    linear_solver<RealType> analysis(assembler.LHS);
    analysis.set_direct_solver(false);
    analysis.factorize();
    x_dof = analysis.solve(assembler.RHS);
    
    // render silo
    size_t it = 0;
    std::string silo_file_name = "line_strain";
    postprocessor<mesh1d_type>::write_silo_um_field(silo_file_name, it, msh, hho_di, x_dof);
    
}
