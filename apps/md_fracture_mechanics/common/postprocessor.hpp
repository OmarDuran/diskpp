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
#include "../common/elastic_two_fields_assembler.hpp"

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
class postprocessor {
    
public:
    
    // Write a silo file for cell displacements
    static void write_silo_u_field(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof){
        
        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> approx_ux, approx_uy;
        size_t n_ten_cbs = disk::sym_matrix_basis_size(hho_di.grad_degree(), dim, dim);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(),dim, dim);
        size_t cell_dof = n_ten_cbs + n_vec_cbs;

        approx_ux.reserve( num_points );
        approx_uy.reserve( num_points );
        
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
        }

        disk::silo_database silo;
        silo_file_name += std::to_string(0) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_nodal_variable<double> vx_silo("ux", approx_ux);
        disk::silo_nodal_variable<double> vy_silo("uy", approx_uy);
        silo.add_variable("mesh", vx_silo);
        silo.add_variable("mesh", vy_silo);

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
        
    }
        
};


#endif /* postprocessor_hpp */
