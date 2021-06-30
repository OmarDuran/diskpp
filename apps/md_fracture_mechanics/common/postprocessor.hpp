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
#include "../common/elastic_two_fields_assembler_3d.hpp"


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
    
    // Write a pair of vtk files for skin variables
    static void write_skin_vtk_files(std::string vtk_file_name_l, std::string vtk_file_name_r, Mesh & msh, elastic_two_fields_assembler_3d<Mesh> & assembler, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof, size_t f_ind){
        
        using RealType = double;
        timecounter tc;
        tc.tic();
        
        std::ofstream file_l, file_r;
        file_l.open(vtk_file_name_l.c_str());
        file_l << "# vtk DataFile Version 1.0" << std::endl;
        file_l << "2D Unstructured Grid of Linear Triangles" << std::endl;
        file_l << "ASCII" << std::endl;

        file_r.open(vtk_file_name_r.c_str());
        file_r << "# vtk DataFile Version 1.0" << std::endl;
        file_r << "2D Unstructured Grid of Linear Triangles" << std::endl;
        file_r << "ASCII" << std::endl;
        
        // sigma n and t
        {
            fracture_3d<Mesh> f = assembler.fractures()[f_ind];
            auto storage = msh.backend_storage();
            size_t n_cells_dof = assembler.get_n_cells_dofs();
            size_t n_faces_dofs = assembler.get_n_faces_dofs();
            size_t n_hybrid_dofs = assembler.get_n_hybrid_dofs();
            size_t n_skins_dofs = assembler.get_n_skin_dof();
            size_t n_cells = f.m_pairs.size();
            size_t sigma_degree = hho_di.face_degree()-1;
            size_t n_f_sigma_bs = disk::scalar_basis_size(sigma_degree, Mesh::dimension - 1);
            size_t n_data = 3*n_cells;

            // scan for selected cells, common cells are discardable
            std::vector<std::pair<size_t,size_t>> point_to_cell_l;
            std::vector<std::pair<size_t,size_t>> point_to_cell_r;
            std::vector<size_t> point_map_l(msh.points_size(),-1);
            std::vector<size_t> point_map_r(msh.points_size(),-1);
            
            size_t cell_i = 0;
            size_t pt_l = 0;
            size_t pt_r = 0;
            for (auto chunk : f.m_pairs)
            {
                auto& face_l = storage->surfaces[chunk.first];
                auto& face_r = storage->surfaces[chunk.second];

                {
                    auto points_l = face_l.point_ids();
                    size_t n_p = points_l.size();
                    for (size_t l = 0; l < n_p; l++)
                    {
                        auto pt_id = points_l[l];
                        if (point_map_l[pt_id]==size_t(-1)) {
                            point_map_l[pt_id] = pt_l;
                            pt_l++;
                            point_to_cell_l.push_back(std::make_pair(pt_id, cell_i));
                        }
                    }
                }
                {
                    auto points_r = face_r.point_ids();
                    size_t n_p = points_r.size();
                    for (size_t l = 0; l < n_p; l++)
                    {
                        auto pt_id = points_r[l];
                        if (point_map_r[pt_id]==size_t(-1)) {
                            point_map_r[pt_id] = pt_r;
                            pt_r++;
                            point_to_cell_r.push_back(std::make_pair(pt_id, cell_i));
                        }
                    }
                }
                cell_i++;
            }

            // write points left
            file_l << "DATASET UNSTRUCTURED_GRID" << std::endl;
            file_l << "POINTS " << point_to_cell_l.size() << " float" << std::endl;
            for (auto& pt_id : point_to_cell_l){
                auto bar = *std::next(msh.points_begin(), pt_id.first);
                file_l << bar.x() << " " << bar.y() << " " << bar.z() << std::endl;

            }


            // write points right
            file_r << "DATASET UNSTRUCTURED_GRID" << std::endl;
            file_r << "POINTS " << point_to_cell_r.size() << " float" << std::endl;
            for (auto& pt_id : point_to_cell_r){
                auto bar = *std::next(msh.points_begin(), pt_id.first);
                file_r << bar.x() << " " << bar.y() << " " << bar.z() << std::endl;
            }


            // write cells left
            file_l << "CELLS " << f.m_pairs.size() << " " << f.m_pairs.size() * 4 << std::endl;
            file_r << "CELLS " << f.m_pairs.size() << " " << f.m_pairs.size() * 4 << std::endl;
            for (auto chunk : f.m_pairs)
            {
                auto& face_l = storage->surfaces[chunk.first];
                auto& face_r = storage->surfaces[chunk.second];

                {
                    auto points_l = face_l.point_ids();
                    size_t n_p = points_l.size();
                    file_l << n_p << " ";
                    for (size_t l = 0; l < n_p; l++)
                    {
                        auto pt_id = points_l[l];
                        file_l << point_map_l[pt_id] << " ";
                    }
                    file_l << std::endl;
                }
                {
                    auto points_r = face_r.point_ids();
                    size_t n_p = points_r.size();
                    file_r << n_p << " ";
                    for (size_t l = 0; l < n_p; l++)
                    {
                        auto pt_id = points_r[l];
                        file_r << point_map_r[pt_id] << " ";
                    }
                    file_r << std::endl;
                }

            }

            file_l << "CELL_TYPES " << f.m_pairs.size() << std::endl;
            file_r << "CELL_TYPES " << f.m_pairs.size() << std::endl;
            for (auto chunk : f.m_pairs)
            {
                file_l << "5" << std::endl;
                file_r << "5" << std::endl;

            }
            
            assert(point_to_cell_l.size() == point_to_cell_r.size());
            
            Matrix<RealType, Dynamic, 3> data_f_s_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_cells, 3);
            Matrix<RealType, Dynamic, 3> data_f_s_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_cells, 3);
            
            size_t cell_ind = 0;
            for (auto chunk : f.m_pairs) {

                auto& face_l = storage->surfaces[chunk.first];
                auto& face_r = storage->surfaces[chunk.second];

                auto bar_l = barycenter(msh,face_l);
                auto bar_r = barycenter(msh,face_r);
                
                // hybrid sigma evaluation
                {

                    auto face_basis = make_scalar_monomial_basis(msh, face_l, sigma_degree);
                    size_t offset_n = cell_ind*3*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_n_x_dof = x_dof.block(offset_n, 0, n_f_sigma_bs, 1);

                    size_t offset_t1 = cell_ind*3*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_f_sigma_bs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_t1_x_dof = x_dof.block(offset_t1, 0, n_f_sigma_bs, 1);
                    
                    size_t offset_t2 = cell_ind*3*n_f_sigma_bs + n_cells_dof + n_faces_dofs + 2.0 * n_f_sigma_bs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_t2_x_dof = x_dof.block(offset_t2, 0, n_f_sigma_bs, 1);

                    auto t_phi = face_basis.eval_functions( bar_l );
                    assert(t_phi.rows() == face_basis.size());

                    auto snh = disk::eval(sigma_n_x_dof, t_phi);
                    auto st1h = disk::eval(sigma_t1_x_dof, t_phi);
                    auto st2h = disk::eval(sigma_t2_x_dof, t_phi);

                    data_f_s_l(cell_ind,0) = snh;
                    data_f_s_l(cell_ind,1) = st1h;
                    data_f_s_l(cell_ind,2) = st2h;

                    data_f_s_r(cell_ind,0) = snh;
                    data_f_s_r(cell_ind,1) = st1h;
                    data_f_s_r(cell_ind,2) = st2h;
                    
                }
                cell_ind++;
            }
            
            size_t n_points = point_to_cell_l.size();
            
            Matrix<RealType, Dynamic, 3> data_u_l = Matrix<RealType, Dynamic, Dynamic>::Zero(n_points, 3);
            
            Matrix<RealType, Dynamic, 3> data_u_r = Matrix<RealType, Dynamic, Dynamic>::Zero(n_points, 3);
            
            for (auto& pt_id : point_to_cell_l)
            {
                auto cell_ind = pt_id.second;
                auto point = *std::next(msh.points_begin(), pt_id.first);
                auto chunk = f.m_pairs[cell_ind];
                
                size_t cell_ind_l = f.m_elements[cell_ind].first;
                size_t cell_ind_r = f.m_elements[cell_ind].second;
                auto& face_l = storage->surfaces[chunk.first];
                auto& face_r = storage->surfaces[chunk.second];
                auto& cell_l = storage->volumes[cell_ind_l];
                auto& cell_r = storage->volumes[cell_ind_r];
                
//                 hybrid sigma evaluation
                {

                    auto face_basis = make_scalar_monomial_basis(msh, face_l, sigma_degree);
                    size_t offset_n = cell_ind*3*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_n_x_dof = x_dof.block(offset_n, 0, n_f_sigma_bs, 1);

                    size_t offset_t1 = cell_ind*3*n_f_sigma_bs + n_cells_dof + n_faces_dofs + n_f_sigma_bs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_t1_x_dof = x_dof.block(offset_t1, 0, n_f_sigma_bs, 1);

                    size_t offset_t2 = cell_ind*3*n_f_sigma_bs + n_cells_dof + n_faces_dofs + 2.0 * n_f_sigma_bs + n_skins_dofs + assembler.compress_hybrid_indexes().at(f_ind);
                    Matrix<RealType, Dynamic, 1> sigma_t2_x_dof = x_dof.block(offset_t2, 0, n_f_sigma_bs, 1);

                    auto t_phi = face_basis.eval_functions( point );
                    assert(t_phi.rows() == face_basis.size());

                    auto snh = disk::eval(sigma_n_x_dof, t_phi);
                    auto st1h = disk::eval(sigma_t1_x_dof, t_phi);
                    auto st2h = disk::eval(sigma_t2_x_dof, t_phi);

//                    data_sigma_n(cell_ind,0) += (1.0/3.0)*point.x();
//                    data_sigma_n(cell_ind,1) += (1.0/3.0)*point.y();
//                    data_sigma_n(cell_ind,2) += (1.0/3.0)*point.z();
//                    data_sigma_n(cell_ind,3) += (1.0/3.0)*snh;
//
//                    data_sigma_t(cell_ind,0) += (1.0/3.0)*point.x();
//                    data_sigma_t(cell_ind,1) += (1.0/3.0)*point.y();
//                    data_sigma_t(cell_ind,2) += (1.0/3.0)*point.z();
//                    data_sigma_t(cell_ind,3) += (1.0/3.0)*st1h;
//                    data_sigma_t(cell_ind,4) += (1.0/3.0)*st2h;

                }

                // u evaluation
                {
                    size_t face_l_offset = assembler.get_n_cells_dofs() + assembler.compress_indexes().at(chunk.first);

//                    size_t face_r_offset = assembler.get_n_cells_dofs() + assembler.compress_indexes().at(chunk.second);

                    auto face_basis_l = make_vector_monomial_basis(msh, face_l, hho_di.face_degree());
//                    auto face_basis_r = make_vector_monomial_basis(msh, face_r, hho_di.face_degree());
                    size_t n_u_bs = disk::vector_basis_size(hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);

                    Matrix<RealType, Dynamic, 1> u_l_x_dof = x_dof.block(face_l_offset, 0, n_u_bs, 1);
//                    Matrix<RealType, Dynamic, 1> u_r_x_dof = x_dof.block(face_r_offset, 0, n_u_bs, 1);

                    auto t_phi_l = face_basis_l.eval_functions( point );
//                    auto t_phi_r = face_basis_r.eval_functions( point );
                    assert(t_phi_l.rows() == face_basis_l.size());
//                    assert(t_phi_r.rows() == face_basis_r.size());

                    auto ul = disk::eval(u_l_x_dof, t_phi_l);
//                    auto ur = disk::eval(u_r_x_dof, t_phi_r);

                    {
                        const auto n = disk::normal(msh, cell_l, face_l);
                        const auto t1 = disk::tanget(msh, cell_l, face_l).first;
                        const auto t2 = disk::tanget(msh, cell_l, face_l).second;
                        auto unl = ul.dot(n);
                        auto ut1l = ul.dot(t1);
                        auto ut2l = ul.dot(t2);
                        data_u_l(point_map_l[pt_id.first],0) = unl;
                        data_u_l(point_map_l[pt_id.first],1) = ut1l;
                        data_u_l(point_map_l[pt_id.first],2) = ut2l;
                    }
                    {
//                        const auto n = disk::normal(msh, cell_r, face_r);
//                        const auto t1 = disk::tanget(msh, cell_r, face_r).first;
//                        const auto t2 = disk::tanget(msh, cell_r, face_r).second;
//                        auto unr = ur.dot(n);
//                        auto ut1r = ur.dot(t1);
//                        auto ut2r = ur.dot(t2);
//                        data_u_r(3*cell_ind+ip,0) = bar.x();
//                        data_u_r(3*cell_ind+ip,1) = bar.y();
//                        data_u_r(3*cell_ind+ip,2) = bar.z();
//                        data_u_r(3*cell_ind+ip,3) = unr;
//                        data_u_r(3*cell_ind+ip,4) = ut1r;
//                        data_u_r(3*cell_ind+ip,5) = ut2r;
                    }
                }
                
            }
            
            // writing data
            
            file_l << "CELL_DATA " << n_cells << std::endl;
            file_l << "SCALARS " << "sigma_n" << " float" << std::endl;
            file_l << "LOOKUP_TABLE default " << std::endl;
            for (size_t i = 0; i < data_f_s_l.rows(); i++) {
                file_l << data_f_s_l(i,0) << std::endl;
            }
            
            file_l << "VECTORS " << "sigma_t" << " float" << std::endl;
            for (size_t i = 0; i < data_f_s_l.rows(); i++) {
                file_l << data_f_s_l(i,1) << " " << data_f_s_l(i,2) << " " << 0.0 << std::endl;
            }
            
            
            
            file_l << "POINT_DATA " << n_points << std::endl;
            file_l << "SCALARS " << "u_n" << " float" << std::endl;
            file_l << "LOOKUP_TABLE default " << std::endl;
            for (size_t i = 0; i < data_u_l.rows(); i++) {
                file_l << data_u_l(i,0) << std::endl;
            }
            
            file_l << "VECTORS " << "u_t" << " float" << std::endl;
            for (size_t i = 0; i < data_u_l.rows(); i++) {
                file_l << data_u_l(i,1) << " " << data_u_l(i,2) << " " << 0.0 << std::endl;
            }
            
//            for (auto& pt_id : point_to_cell_l){
//
//                auto bar = *std::next(msh.points_begin(), pt_id.first);
//                //            cell_i = pt_id.second;
//                //            auto cell = *std::next(msh.cells_begin(), cell_i);
//                size_t cell_ind_l = f.m_elements[cell_ind].first;
//                size_t cell_ind_r = f.m_elements[cell_ind].second;
//                auto& face_l = storage->surfaces[chunk.first];
//                auto& face_r = storage->surfaces[chunk.second];
//                auto& cell_l = storage->volumes[cell_ind_l];
//                auto& cell_r = storage->volumes[cell_ind_r];
//
//                auto points = face_l.point_ids();
//
//                for (size_t ip = 0; ip < points.size(); ip++) {
//
//                    auto pt_id = points[ip];
//                    auto bar = *std::next(msh.points_begin(), pt_id);

//
//
//
//                 }
//
//                cell_ind++;
//            }
//
//            size_t pre = 15;
            
        }
                
        file_l.close();
        file_r.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "VTK file rendered in : " << tc << " seconds" << reset << std::endl;
        
    }
        
};


#endif /* postprocessor_hpp */
