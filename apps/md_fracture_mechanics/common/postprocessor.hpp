//
//  postprocessor.hpp
//  acoustics
//
//  Created by Omar Durán on 4/10/20.
//

#pragma once
#ifndef postprocessor_hpp
#define postprocessor_hpp

#include <iomanip>
#include "../common/elastic_two_fields_assembler_3d.hpp"
#include "../common/elastic_two_fields_assembler.hpp"
#include "../common/elastic_1d_two_fields_assembler.hpp"

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
class postprocessor {
    
public:
    
    // Write a silo file for cell displacements
    static void write_silo_mesh(std::string silo_file_name, Mesh & msh){
        
        timecounter tc;
        tc.tic();

        disk::silo_database silo;
        silo_file_name += ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
        
    }
    
    // Write a silo file for cell displacements
    static void write_silo_u_field(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof){
        
        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> approx_ux, approx_uy;
        std::vector<RealType> approx_sxx, approx_sxy, approx_syy;
        size_t n_ten_cbs = disk::sym_matrix_basis_size(hho_di.grad_degree(), dim, dim);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(),dim, dim);
        size_t cell_dof = n_ten_cbs + n_vec_cbs;

        approx_ux.reserve( num_points );
        approx_uy.reserve( num_points );
        approx_sxx.reserve( num_points );
        approx_sxy.reserve( num_points );
        approx_syy.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh)
        {
            auto points = cell.point_ids();
            size_t n_p = points.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = points[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }

        for (auto& pt_id : point_to_cell)
        {
            auto bar = *std::next(msh.points_begin(), pt_id.first);
            cell_i = pt_id.second;
            auto cell = *std::next(msh.cells_begin(), cell_i);
            
            // vector evaluation
            {
                auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof + n_ten_cbs, 0, n_vec_cbs, 1);
                auto t_phi = cell_basis.eval_functions( bar );
                assert(t_phi.rows() == cell_basis.size());
                auto uh = disk::eval(vec_x_cell_dof, t_phi);
                approx_ux.push_back(uh(0,0));
                approx_uy.push_back(uh(1,0));
            }
            
            // tensor evaluation
            {
                auto ten_basis = make_sym_matrix_monomial_basis(msh, cell, hho_di.grad_degree());
                Matrix<RealType, Dynamic, 1> ten_x_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_ten_cbs, 1);

                auto t_ten_phi = ten_basis.eval_functions( bar );
                assert(t_ten_phi.size() == ten_basis.size());
                auto sigma_h = disk::eval(ten_x_cell_dof, t_ten_phi);
                
                approx_sxx.push_back(sigma_h(0,0));
                approx_sxy.push_back(sigma_h(0,1));
                approx_syy.push_back(sigma_h(1,1));
            }
        }

        disk::silo_database silo;
        silo_file_name += std::to_string(0) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_nodal_variable<double> vx_silo("ux", approx_ux);
        disk::silo_nodal_variable<double> vy_silo("uy", approx_uy);
        disk::silo_nodal_variable<double> sxx_silo("sxx", approx_sxx);
        disk::silo_nodal_variable<double> sxy_silo("sxy", approx_sxy);
        disk::silo_nodal_variable<double> syy_silo("syy", approx_syy);
        silo.add_variable("mesh", vx_silo);
        silo.add_variable("mesh", vy_silo);
        silo.add_variable("mesh", sxx_silo);
        silo.add_variable("mesh", sxy_silo);
        silo.add_variable("mesh", syy_silo);

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
        
    }
    
    // Write a silo file for cell displacements
    static void write_silo_u_field_3d(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof){
        
        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> approx_ux, approx_uy, approx_uz;
        std::vector<RealType> approx_sxx, approx_syy, approx_szz, approx_sxy, approx_sxz, approx_syz;
        size_t n_ten_cbs = disk::sym_matrix_basis_size(hho_di.grad_degree(), dim, dim);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(),dim, dim);
        size_t cell_dof = n_ten_cbs + n_vec_cbs;

        approx_ux.reserve( num_points );
        approx_uy.reserve( num_points );
        approx_sxx.reserve( num_points );
        approx_sxy.reserve( num_points );
        approx_syy.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh)
        {
            auto points = cell.point_ids();
            size_t n_p = points.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = points[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }

        for (auto& pt_id : point_to_cell)
        {
            auto bar = *std::next(msh.points_begin(), pt_id.first);
            cell_i = pt_id.second;
            auto cell = *std::next(msh.cells_begin(), cell_i);
            
            // vector evaluation
            {
                auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof + n_ten_cbs, 0, n_vec_cbs, 1);
                auto t_phi = cell_basis.eval_functions( bar );
                assert(t_phi.rows() == cell_basis.size());
                auto uh = disk::eval(vec_x_cell_dof, t_phi);
                approx_ux.push_back(uh(0,0));
                approx_uy.push_back(uh(1,0));
                approx_uz.push_back(uh(2,0));
            }
            
            // tensor evaluation
            {
                auto ten_basis = make_sym_matrix_monomial_basis(msh, cell, hho_di.grad_degree());
                Matrix<RealType, Dynamic, 1> ten_x_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_ten_cbs, 1);

                auto t_ten_phi = ten_basis.eval_functions( bar );
                assert(t_ten_phi.size() == ten_basis.size());
                auto sigma_h = disk::eval(ten_x_cell_dof, t_ten_phi);
                
                approx_sxx.push_back(sigma_h(0,0));
                approx_sxy.push_back(sigma_h(0,1));
                approx_syy.push_back(sigma_h(1,1));
                approx_sxz.push_back(sigma_h(0,2));
                approx_syz.push_back(sigma_h(1,2));
                approx_szz.push_back(sigma_h(2,2));
            }
        }

        disk::silo_database silo;
        silo_file_name += std::to_string(0) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_nodal_variable<double> vx_silo("ux", approx_ux);
        disk::silo_nodal_variable<double> vy_silo("uy", approx_uy);
        disk::silo_nodal_variable<double> vz_silo("uz", approx_uz);
        disk::silo_nodal_variable<double> sxx_silo("sxx", approx_sxx);
        disk::silo_nodal_variable<double> syy_silo("syy", approx_syy);
        disk::silo_nodal_variable<double> szz_silo("szz", approx_szz);
        disk::silo_nodal_variable<double> sxy_silo("sxy", approx_sxy);
        disk::silo_nodal_variable<double> sxz_silo("sxz", approx_sxz);
        disk::silo_nodal_variable<double> syz_silo("syz", approx_syz);
        silo.add_variable("mesh", vx_silo);
        silo.add_variable("mesh", vy_silo);
        silo.add_variable("mesh", vz_silo);
        silo.add_variable("mesh", sxx_silo);
        silo.add_variable("mesh", syy_silo);
        silo.add_variable("mesh", szz_silo);
        silo.add_variable("mesh", sxy_silo);
        silo.add_variable("mesh", sxz_silo);
        silo.add_variable("mesh", syz_silo);
        

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
        
    }
    
    // Write a silo file for mortar cell displacements
    static void write_silo_um_field(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof){
        
        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> approx_u;
        std::vector<RealType> approx_s;
        size_t n_ten_cbs = disk::vector_basis_size(hho_di.grad_degree(), dim, dim);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(),dim, dim);
        size_t cell_dof = n_ten_cbs + n_vec_cbs;

        approx_u.reserve( num_points );
        approx_s.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh)
        {
            auto points = cell.point_ids();
            size_t n_p = points.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = points[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }

        for (auto& pt_id : point_to_cell)
        {
            auto bar = *std::next(msh.points_begin(), pt_id.first);
            cell_i = pt_id.second;
            auto cell = *std::next(msh.cells_begin(), cell_i);
            
            // vector evaluation
            {
                auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof + n_ten_cbs, 0, n_vec_cbs, 1);
                auto t_phi = cell_basis.eval_functions( bar );
                assert(t_phi.rows() == cell_basis.size());
                auto uh = disk::eval(vec_x_cell_dof, t_phi);
                approx_u.push_back(uh);
            }
            
            // tensor evaluation
            {
                auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.reconstruction_degree());
                Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_ten_cbs, 1);
                auto t_dphi = cell_basis.eval_gradients( bar );
                
                Matrix<RealType, 1, 1> sigma_h = Matrix<RealType, 1, 1>::Zero();
                for (size_t i = 1; i < t_dphi.rows(); i++){
                    sigma_h = sigma_h + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 1);
                }
                
//                auto ten_basis = make_vector_monomial_basis(msh, cell, hho_di.grad_degree());
//                Matrix<RealType, Dynamic, 1> ten_x_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_ten_cbs, 1);
//
//                auto t_ten_phi = ten_basis.eval_functions( bar );
//                assert(t_ten_phi.size() == ten_basis.size());
//                auto sigma_h = disk::eval(ten_x_cell_dof, t_ten_phi);
                
                approx_s.push_back(sigma_h(0,0));
            }
        }

        disk::silo_database silo;
        silo_file_name += std::to_string(0) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_nodal_variable<double> v_silo("u", approx_u);
        disk::silo_nodal_variable<double> s_silo("s", approx_s);
        silo.add_variable("mesh", v_silo);
        silo.add_variable("mesh", s_silo);

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
        
    }
        
};


#endif /* postprocessor_hpp */
