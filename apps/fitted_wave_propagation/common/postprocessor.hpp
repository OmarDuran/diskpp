//
//  postprocessor.hpp
//  acoustics
//
//  Created by Omar Durán on 4/10/20.
//

#pragma once
#ifndef postprocessor_hpp
#define postprocessor_hpp

#include "../common/acoustic_one_field_assembler.hpp"
#include "../common/acoustic_two_fields_assembler.hpp"
#include "../common/elastodynamic_one_field_assembler.hpp"
#include "../common/elastodynamic_three_fields_assembler.hpp"
#include "../common/elastoacoustic_two_fields_assembler.hpp"

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
class postprocessor {
    
public:
    
    /// Compute L2 and H1 errors for one field approximation
    static void compute_errors_one_field(Mesh & msh, disk::hho_degree_info & hho_di, acoustic_one_field_assembler<Mesh> & assembler, Matrix<double, Dynamic, 1> & x_dof,std::function<double(const typename Mesh::point_type& )> scal_fun, std::function<std::vector<double>(const typename Mesh::point_type& )> flux_fun, std::ostream & error_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        auto dim = Mesh::dimension;
        size_t cell_dof = disk::scalar_basis_size(hho_di.cell_degree(), dim);
        
        RealType scalar_l2_error = 0.0;
        RealType flux_l2_error = 0.0;
        size_t cell_i = 0;
        RealType h = 10.0;
        for (auto& cell : msh)
        {
            RealType h_l = diameter(msh, cell);
            if (h_l < h) {
                h = h_l;
            }

            Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);

            // scalar evaluation
            {
                auto cell_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, cell_basis, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, cell_basis, scal_fun);
                Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
                Matrix<RealType, Dynamic, 1> diff = real_dofs - scalar_cell_dof;
                scalar_l2_error += diff.dot(mass*diff);

            }

            // flux evaluation
            {
                auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
                auto rec_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.reconstruction_degree());
                auto gr = make_scalar_hho_laplacian(msh, cell, hho_di);
                Matrix<RealType, Dynamic, 1> all_dofs = assembler.gather_dof_data(msh, cell, x_dof);
                Matrix<RealType, Dynamic, 1> recdofs = gr.first * all_dofs;

                // Error integrals
                for (auto & point_pair : int_rule) {

                    RealType omega = point_pair.weight();
                    auto t_dphi = rec_basis.eval_gradients( point_pair.point() );
                    Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();

                    for (size_t i = 1; i < t_dphi.rows(); i++){
                        grad_uh = grad_uh + recdofs(i-1)*t_dphi.block(i, 0, 1, 2);
                    }

                    Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
                    grad_u_exact(0,0) =  flux_fun(point_pair.point())[0];
                    grad_u_exact(0,1) =  flux_fun(point_pair.point())[1];
                    flux_l2_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);

                }
            }

            cell_i++;
        }
        tc.toc();
        std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
        error_file << "Characteristic h size = " << h << std::endl;
        error_file << "L2-norm error = " << std::setprecision(16) << std::sqrt(scalar_l2_error) << std::endl;
        error_file << "H1-norm error = " << std::setprecision(16) << std::sqrt(flux_l2_error) << std::endl;
        error_file << std::endl;
        error_file.flush();


    }
    
    /// Compute L2 and H1 errors for two fields approximation
    static void compute_errors_two_fields(Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,std::function<double(const typename Mesh::point_type& )> scal_fun, std::function<std::vector<double>(const typename Mesh::point_type& )> flux_fun, std::ostream & error_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        size_t n_scal_dof = disk::scalar_basis_size(hho_di.cell_degree(), Mesh::dimension);
        size_t n_vec_dof = disk::scalar_basis_size(hho_di.reconstruction_degree(), Mesh::dimension)-1;
        size_t cell_dof = n_scal_dof + n_vec_dof;
        
        RealType scalar_l2_error = 0.0;
        RealType flux_l2_error = 0.0;
        size_t cell_i = 0;
        RealType h = 10.0;
        for (auto& cell : msh)
        {
             RealType h_l = diameter(msh, cell);
            if (h_l < h) {
                h = h_l;
            }

            // scalar evaluation
            {
                auto cell_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+n_vec_dof, 0, n_scal_dof, 1);
                Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, cell_basis, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, cell_basis, scal_fun);
                Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
                Matrix<RealType, Dynamic, 1> diff = real_dofs - scalar_cell_dof;
                scalar_l2_error += diff.dot(mass*diff);

            }

            // flux evaluation
            {
                auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
                auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.reconstruction_degree());
                Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_vec_dof, 1);
                
                // Error integrals
                for (auto & point_pair : int_rule) {

                    RealType omega = point_pair.weight();
                    auto t_dphi = cell_basis.eval_gradients( point_pair.point() );
                    
                    Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                    for (size_t i = 1; i < t_dphi.rows(); i++){
                      grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
                    }

                    Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
                    grad_u_exact(0,0) =  flux_fun(point_pair.point())[0];
                    grad_u_exact(0,1) =  flux_fun(point_pair.point())[1];
                    flux_l2_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);

                }
            }

            cell_i++;
        }
        tc.toc();
        
        std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
        error_file << "Characteristic h size = " << h << std::endl;
        error_file << "L2-norm error = " << std::setprecision(16) << std::sqrt(scalar_l2_error) << std::endl;
        error_file << "H1-norm error = " << std::setprecision(16) << std::sqrt(flux_l2_error) << std::endl;
        error_file << std::endl;
        error_file.flush();
        
    }
    
    // Write a silo file for one field approximation
    static void write_silo_one_field(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
    std::function<double(const typename Mesh::point_type& )> scal_fun, bool cell_centered_Q){

        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> exact_u, approx_u;
        size_t cell_dof = disk::scalar_basis_size(hho_di.cell_degree(), dim);
        
        if (cell_centered_Q) {
            exact_u.reserve( num_cells );
            approx_u.reserve( num_cells );

            size_t cell_i = 0;
            for (auto& cell : msh)
            {
                auto bar = barycenter(msh, cell);
                exact_u.push_back( scal_fun(bar) );
                
                // scalar evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    RealType uh = scalar_cell_dof.dot( t_phi );
                    approx_u.push_back(uh);
                }
                cell_i++;
            }

        }else{

            exact_u.reserve( num_points );
            approx_u.reserve( num_points );

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
                exact_u.push_back( scal_fun(bar) );

                cell_i = pt_id.second;
                auto cell = *std::next(msh.cells_begin(), cell_i);
                // scalar evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    RealType uh = scalar_cell_dof.dot( t_phi );
                    approx_u.push_back(uh);
                }
            }

        }

        disk::silo_database silo;
        silo_file_name += std::to_string(it) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        if (cell_centered_Q) {
            disk::silo_zonal_variable<double> v_silo("v", exact_u);
            disk::silo_zonal_variable<double> vh_silo("vh", approx_u);
            silo.add_variable("mesh", v_silo);
            silo.add_variable("mesh", vh_silo);
        }else{
            disk::silo_nodal_variable<double> v_silo("v", exact_u);
            disk::silo_nodal_variable<double> vh_silo("vh", approx_u);
            silo.add_variable("mesh", v_silo);
            silo.add_variable("mesh", vh_silo);
        }

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
    // Write a silo file for two fields approximation
    static void write_silo_two_fields(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
    std::function<double(const typename Mesh::point_type& )> scal_fun,  std::function<std::vector<double>(const typename Mesh::point_type& )> flux_fun, bool cell_centered_Q){

        timecounter tc;
        tc.tic();
        
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> exact_u, approx_u;
        std::vector<RealType> exact_dux, exact_duy, approx_dux, approx_duy;
        
        size_t n_scal_dof = disk::scalar_basis_size(hho_di.cell_degree(), Mesh::dimension);
        size_t n_vec_dof = disk::scalar_basis_size(hho_di.reconstruction_degree(), Mesh::dimension)-1;
        size_t cell_dof = n_scal_dof + n_vec_dof;
        
        if (cell_centered_Q) {
            exact_u.reserve( num_cells );
            approx_u.reserve( num_cells );
            exact_dux.reserve( num_cells );
            exact_duy.reserve( num_cells );
            approx_dux.reserve( num_cells );
            approx_duy.reserve( num_cells );

            size_t cell_i = 0;
            for (auto& cell : msh)
            {
                auto bar = barycenter(msh, cell);
                exact_u.push_back( scal_fun(bar) );
                exact_dux.push_back( flux_fun(bar)[0] );
                exact_duy.push_back( flux_fun(bar)[1] );
                
                // scalar evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+n_vec_dof, 0, n_scal_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    RealType uh = scalar_cell_dof.dot( t_phi );
                    approx_u.push_back(uh);
                }
                
                // flux evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.reconstruction_degree());
                    Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_vec_dof, 1);
                    auto t_dphi = cell_basis.eval_gradients( bar );
                    
                    Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                    for (size_t i = 1; i < t_dphi.rows(); i++){
                      grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
                    }

                    approx_dux.push_back(grad_uh(0,0));
                    approx_duy.push_back(grad_uh(0,1));
                }
                
                cell_i++;
            }

        }else{

            exact_u.reserve( num_points );
            approx_u.reserve( num_points );
            exact_dux.reserve( num_points );
            exact_duy.reserve( num_points );
            approx_dux.reserve( num_points );
            approx_duy.reserve( num_points );

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
                
                exact_u.push_back( scal_fun(bar) );
                exact_dux.push_back( flux_fun(bar)[0] );
                exact_duy.push_back( flux_fun(bar)[1] );
                
                // scalar evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+n_vec_dof, 0, n_scal_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    RealType uh = scalar_cell_dof.dot( t_phi );
                    approx_u.push_back(uh);
                }
                
                // flux evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.reconstruction_degree());
                    Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_vec_dof, 1);
                    auto t_dphi = cell_basis.eval_gradients( bar );
                    
                    Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                    for (size_t i = 1; i < t_dphi.rows(); i++){
                      grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
                    }

                    approx_dux.push_back(grad_uh(0,0));
                    approx_duy.push_back(grad_uh(0,1));
                }
            }

        }

        disk::silo_database silo;
        silo_file_name += std::to_string(it) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        if (cell_centered_Q) {
            disk::silo_zonal_variable<double> v_silo("v", exact_u);
            disk::silo_zonal_variable<double> qx_silo("qx", exact_dux);
            disk::silo_zonal_variable<double> qy_silo("qy", exact_duy);
            disk::silo_zonal_variable<double> vh_silo("vh", approx_u);
            disk::silo_zonal_variable<double> qhx_silo("qhx", approx_dux);
            disk::silo_zonal_variable<double> qhy_silo("qhy", approx_duy);
            silo.add_variable("mesh", v_silo);
            silo.add_variable("mesh", qx_silo);
            silo.add_variable("mesh", qy_silo);
            silo.add_variable("mesh", vh_silo);
            silo.add_variable("mesh", qhx_silo);
            silo.add_variable("mesh", qhy_silo);
        }else{
            disk::silo_nodal_variable<double> v_silo("v", exact_u);
            disk::silo_nodal_variable<double> qx_silo("qx", exact_dux);
            disk::silo_nodal_variable<double> qy_silo("qy", exact_duy);
            disk::silo_nodal_variable<double> vh_silo("vh", approx_u);
            disk::silo_nodal_variable<double> qhx_silo("qhx", approx_dux);
            disk::silo_nodal_variable<double> qhy_silo("qhy", approx_duy);
            silo.add_variable("mesh", v_silo);
            silo.add_variable("mesh", qx_silo);
            silo.add_variable("mesh", qy_silo);
            silo.add_variable("mesh", vh_silo);
            silo.add_variable("mesh", qhx_silo);
            silo.add_variable("mesh", qhy_silo);
        }

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
    /// Compute L2 and H1 errors for one field vectorial approximation
    static void compute_errors_one_field_vectorial(Mesh & msh, disk::hho_degree_info & hho_di, elastodynamic_one_field_assembler<Mesh> & assembler, Matrix<double, Dynamic, 1> & x_dof, std::function<static_vector<double, 2>(const typename Mesh::point_type& )> vec_fun, std::function<static_matrix<double, 2, 2>(const typename Mesh::point_type& )> flux_fun, std::ostream & error_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        auto dim = Mesh::dimension;
        size_t cell_dof = disk::vector_basis_size(hho_di.cell_degree(), dim, dim);
        
        RealType vector_l2_error = 0.0;
        RealType flux_l2_error = 0.0;
        RealType h = 10.0;
        size_t cell_ind = 0;
        for (auto& cell : msh)
        {
            RealType h_l = diameter(msh, cell);
            if (h_l < h) {
                h = h_l;
            }

            Matrix<RealType, Dynamic, 1> vec_cell_dof = x_dof.block(cell_ind*cell_dof, 0, cell_dof, 1);

            // scalar evaluation
            {
                auto cell_basis = disk::make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, cell_basis, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, cell_basis, vec_fun);
                Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
                Matrix<RealType, Dynamic, 1> diff = real_dofs - vec_cell_dof;
                vector_l2_error += diff.dot(mass*diff);

            }

            elastic_material_data<RealType> material = assembler.get_material_data()[cell_ind];
            RealType mu = material.rho()*material.vs()*material.vs();
            RealType lambda = material.rho()*material.vp()*material.vp() - 2.0*mu;
            
            // flux evaluation
            {
                auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
                Matrix<RealType, Dynamic, 1> all_dofs = assembler.gather_dof_data(msh, cell, x_dof);
                
                auto           sgr = make_vector_hho_symmetric_laplacian(msh, cell, hho_di);
                dynamic_vector<RealType> GTu = sgr.first * all_dofs;

                auto           dr   = make_hho_divergence_reconstruction(msh, cell, hho_di);
                dynamic_vector<RealType> divu = dr.first * all_dofs;

                auto cbas_v = disk::make_vector_monomial_basis(msh, cell, hho_di.reconstruction_degree());
                auto cbas_s = disk::make_scalar_monomial_basis(msh, cell, hho_di.face_degree());

                auto rec_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.reconstruction_degree());

                // Error integrals
                for (auto & point_pair : int_rule) {

                    RealType omega = point_pair.weight();
                    
                    auto t_dphi = rec_basis.eval_gradients( point_pair.point() );
                    auto gphi   = cbas_v.eval_sgradients(point_pair.point());
                    auto epsilon = disk::eval(GTu, gphi, dim);
                    auto divphi   = cbas_s.eval_functions(point_pair.point());
                    auto trace_epsilon = disk::eval(divu, divphi);
                    auto sigma = 2.0 * mu * epsilon + lambda * trace_epsilon * static_matrix<RealType, 2, 2>::Identity();
                    auto flux_diff = (flux_fun(point_pair.point()) - sigma).eval();
                    flux_l2_error += omega * flux_diff.squaredNorm();

                }
            }

            cell_ind++;
        }
        tc.toc();
        std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
        error_file << "Characteristic h size = " << h << std::endl;
        error_file << "L2-norm error = " << std::setprecision(16) << std::sqrt(vector_l2_error) << std::endl;
        error_file << "H1-norm error = " << std::setprecision(16) << std::sqrt(flux_l2_error) << std::endl;
        error_file << std::endl;
        error_file.flush();


    }
    
    /// Compute L2 and H1 errors for two fields vectorial approximation
    static void compute_errors_three_fields_vectorial(Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof, std::function<static_vector<double, 2>(const typename Mesh::point_type& )> vec_fun, std::function<static_matrix<double, 2, 2>(const typename Mesh::point_type& )> flux_fun, std::ostream & error_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        auto dim = Mesh::dimension;
        size_t n_ten_cbs = disk::sym_matrix_basis_size(hho_di.grad_degree(), dim, dim);
        size_t n_sca_cbs = disk::scalar_basis_size(hho_di.face_degree(), dim);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(), dim, dim);
        size_t cell_dof = n_ten_cbs + n_sca_cbs + n_vec_cbs;
        
        RealType vector_l2_error = 0.0;
        RealType flux_l2_error = 0.0;
        size_t cell_i = 0;
        RealType h = 10.0;
        for (auto& cell : msh)
        {
            RealType h_l = diameter(msh, cell);
            if (h_l < h) {
                h = h_l;
            }

            Matrix<RealType, Dynamic, 1> vec_cell_dof = x_dof.block(cell_i*cell_dof+n_ten_cbs+n_sca_cbs, 0, n_vec_cbs, 1);

            // vector evaluation
            {
                auto cell_basis = disk::make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, cell_basis, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, cell_basis, vec_fun);
                Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
                Matrix<RealType, Dynamic, 1> diff = real_dofs - vec_cell_dof;
                vector_l2_error += diff.dot(mass*diff);

            }

            // tensor evaluation
            {
                auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
                
                auto ten_basis = make_sym_matrix_monomial_basis(msh, cell, hho_di.grad_degree());
                Matrix<RealType, Dynamic, 1> ten_x_cell_dof = x_dof.block(cell_i*cell_dof, 0, n_ten_cbs, 1);

                auto sca_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.face_degree());
                Matrix<RealType, Dynamic, 1> sigma_v_x_cell_dof = x_dof.block(cell_i*cell_dof+n_ten_cbs, 0, n_sca_cbs, 1);

                // Error integrals
                for (auto & point_pair : int_rule) {

                    RealType omega = point_pair.weight();
                    
                    auto t_ten_phi = ten_basis.eval_functions( point_pair.point() );
                    assert(t_ten_phi.size() == ten_basis.size());
                    auto sigma_h = disk::eval(ten_x_cell_dof, t_ten_phi);
                    
                    auto t_sca_phi = sca_basis.eval_functions( point_pair.point() );
                    assert(t_sca_phi.size() == sca_basis.size());
                    auto sigma_v_h = disk::eval(sigma_v_x_cell_dof, t_sca_phi);
                    
                    sigma_h  += sigma_v_h * static_matrix<RealType, 2, 2>::Identity();
                    
                    auto flux_diff = (flux_fun(point_pair.point()) - sigma_h).eval();
                    flux_l2_error += omega * flux_diff.squaredNorm();

                }
                
            }

            cell_i++;
        }
        tc.toc();
        std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
        error_file << "Characteristic h size = " << h << std::endl;
        error_file << "L2-norm error = " << std::setprecision(16) << std::sqrt(vector_l2_error) << std::endl;
        error_file << "H1-norm error = " << std::setprecision(16) << std::sqrt(flux_l2_error) << std::endl;
        error_file << std::endl;
        error_file.flush();

    }

    // Write a silo file for one field vectorial approximation
    static void write_silo_one_field_vectorial(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
    std::function<static_vector<double, 2>(const typename Mesh::point_type& )> vec_fun, bool cell_centered_Q){

        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> exact_ux, exact_uy, approx_ux, approx_uy;
        size_t cell_dof = disk::vector_basis_size(hho_di.cell_degree(), dim, dim);
        
        if (cell_centered_Q) {
            exact_ux.reserve( num_cells );
            exact_uy.reserve( num_cells );
            approx_ux.reserve( num_cells );
            approx_uy.reserve( num_cells );

            size_t cell_i = 0;
            for (auto& cell : msh)
            {
                auto bar = barycenter(msh, cell);
                exact_ux.push_back( vec_fun(bar)(0,0) );
                exact_uy.push_back( vec_fun(bar)(1,0) );
                
                // vector evaluation
                {
                    auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    assert(t_phi.rows() == cell_basis.size());
                    auto uh = disk::eval(vec_x_cell_dof, t_phi);
                    approx_ux.push_back(uh(0,0));
                    approx_uy.push_back(uh(1,0));
                }
                cell_i++;
            }

        }else{

            exact_ux.reserve( num_points );
            exact_uy.reserve( num_points );
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
                
                exact_ux.push_back( vec_fun(bar)(0,0) );
                exact_uy.push_back( vec_fun(bar)(1,0) );
                
                // vector evaluation
                {
                    auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    assert(t_phi.rows() == cell_basis.size());
                    auto uh = disk::eval(vec_x_cell_dof, t_phi);
                    approx_ux.push_back(uh(0,0));
                    approx_uy.push_back(uh(1,0));
                }
            }

        }

        disk::silo_database silo;
        silo_file_name += std::to_string(it) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        if (cell_centered_Q) {
            disk::silo_zonal_variable<double> vx_silo("vx", exact_ux);
            disk::silo_zonal_variable<double> vy_silo("vy", exact_uy);
            disk::silo_zonal_variable<double> vhx_silo("vhx", approx_ux);
            disk::silo_zonal_variable<double> vhy_silo("vhy", approx_uy);
            silo.add_variable("mesh", vx_silo);
            silo.add_variable("mesh", vy_silo);
            silo.add_variable("mesh", vhx_silo);
            silo.add_variable("mesh", vhy_silo);
        }else{
            disk::silo_nodal_variable<double> vx_silo("vx", exact_ux);
            disk::silo_nodal_variable<double> vy_silo("vy", exact_uy);
            disk::silo_nodal_variable<double> vhx_silo("vhx", approx_ux);
            disk::silo_nodal_variable<double> vhy_silo("vhy", approx_uy);
            silo.add_variable("mesh", vx_silo);
            silo.add_variable("mesh", vy_silo);
            silo.add_variable("mesh", vhx_silo);
            silo.add_variable("mesh", vhy_silo);
        }

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
    // Write a silo file for three fields vectorial approximation
    static void write_silo_three_fields_vectorial(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,std::function<static_vector<double, 2>(const typename Mesh::point_type& )> vec_fun, std::function<static_matrix<double, 2, 2>(const typename Mesh::point_type& )> flux_fun, bool cell_centered_Q){

        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> exact_ux, exact_uy, approx_ux, approx_uy;
        std::vector<RealType> exact_sxx, exact_sxy, exact_syy;
        std::vector<RealType> approx_sxx, approx_sxy, approx_syy;
        size_t n_ten_cbs = disk::sym_matrix_basis_size(hho_di.grad_degree(), dim, dim);
        size_t n_sca_cbs = disk::scalar_basis_size(hho_di.face_degree(), dim);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(), dim, dim);
        size_t cell_dof = n_ten_cbs + n_sca_cbs + n_vec_cbs;
        
        if (cell_centered_Q) {
            exact_ux.reserve( num_cells );
            exact_uy.reserve( num_cells );
            approx_ux.reserve( num_cells );
            approx_uy.reserve( num_cells );
            
            exact_sxx.reserve( num_cells );
            exact_sxy.reserve( num_cells );
            exact_syy.reserve( num_cells );
            approx_sxx.reserve( num_cells );
            approx_sxy.reserve( num_cells );
            approx_syy.reserve( num_cells );

            size_t cell_i = 0;
            for (auto& cell : msh)
            {
                auto bar = barycenter(msh, cell);
                exact_ux.push_back( vec_fun(bar)(0,0) );
                exact_uy.push_back( vec_fun(bar)(1,0) );
                
                exact_sxx.push_back( flux_fun(bar)(0,0) );
                exact_sxy.push_back( flux_fun(bar)(0,1) );
                exact_syy.push_back( flux_fun(bar)(1,1) );
                
                // vector evaluation
                {
                    auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof + n_ten_cbs + n_sca_cbs, 0, n_vec_cbs, 1);
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

                    auto sca_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.face_degree());
                    Matrix<RealType, Dynamic, 1> sigma_v_x_cell_dof = x_dof.block(cell_i*cell_dof+n_ten_cbs, 0, n_sca_cbs, 1);

                    auto t_ten_phi = ten_basis.eval_functions( bar );
                    assert(t_ten_phi.size() == ten_basis.size());
                    auto sigma_h = disk::eval(ten_x_cell_dof, t_ten_phi);
                    
                    auto t_sca_phi = sca_basis.eval_functions( bar );
                    assert(t_sca_phi.size() == sca_basis.size());
                    auto sigma_v_h = disk::eval(sigma_v_x_cell_dof, t_sca_phi);
                
                    sigma_h  += sigma_v_h * static_matrix<RealType, 2, 2>::Identity();

                    approx_sxx.push_back(sigma_h(0,0));
                    approx_sxy.push_back(sigma_h(0,1));
                    approx_syy.push_back(sigma_h(1,1));
                }
                
                cell_i++;
            }

        }else{

            exact_ux.reserve( num_points );
            exact_uy.reserve( num_points );
            approx_ux.reserve( num_points );
            approx_uy.reserve( num_points );
            
            exact_sxx.reserve( num_points );
            exact_sxy.reserve( num_points );
            exact_syy.reserve( num_points );
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
                
                exact_ux.push_back( vec_fun(bar)(0,0) );
                exact_uy.push_back( vec_fun(bar)(1,0) );
                
                exact_sxx.push_back( flux_fun(bar)(0,0) );
                exact_sxy.push_back( flux_fun(bar)(0,1) );
                exact_syy.push_back( flux_fun(bar)(1,1) );
                
                // vector evaluation
                {
                    auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> vec_x_cell_dof = x_dof.block(cell_i*cell_dof + n_ten_cbs + n_sca_cbs, 0, n_vec_cbs, 1);
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

                    auto sca_basis = disk::make_scalar_monomial_basis(msh, cell, hho_di.face_degree());
                    Matrix<RealType, Dynamic, 1> sigma_v_x_cell_dof = x_dof.block(cell_i*cell_dof+n_ten_cbs, 0, n_sca_cbs, 1);

                    auto t_ten_phi = ten_basis.eval_functions( bar );
                    assert(t_ten_phi.size() == ten_basis.size());
                    auto sigma_h = disk::eval(ten_x_cell_dof, t_ten_phi);
                    
                    auto t_sca_phi = sca_basis.eval_functions( bar );
                    assert(t_sca_phi.size() == sca_basis.size());
                    auto sigma_v_h = disk::eval(sigma_v_x_cell_dof, t_sca_phi);
                
                    sigma_h  += sigma_v_h * static_matrix<RealType, 2, 2>::Identity();

                    approx_sxx.push_back(sigma_h(0,0));
                    approx_sxy.push_back(sigma_h(0,1));
                    approx_syy.push_back(sigma_h(1,1));
                }
                
            }

        }

        disk::silo_database silo;
        silo_file_name += std::to_string(it) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        if (cell_centered_Q) {
            disk::silo_zonal_variable<double> vx_silo("vx", exact_ux);
            disk::silo_zonal_variable<double> vy_silo("vy", exact_uy);
            disk::silo_zonal_variable<double> vhx_silo("vhx", approx_ux);
            disk::silo_zonal_variable<double> vhy_silo("vhy", approx_uy);
            
            disk::silo_nodal_variable<double> sxx_silo("sxx", exact_sxx);
            disk::silo_nodal_variable<double> sxy_silo("sxy", exact_sxy);
            disk::silo_nodal_variable<double> syy_silo("syy", exact_syy);
            disk::silo_nodal_variable<double> shxx_silo("shxx", approx_sxx);
            disk::silo_nodal_variable<double> shxy_silo("shxy", approx_sxy);
            disk::silo_nodal_variable<double> shyy_silo("shyy", approx_syy);
            
            silo.add_variable("mesh", vx_silo);
            silo.add_variable("mesh", vy_silo);
            silo.add_variable("mesh", vhx_silo);
            silo.add_variable("mesh", vhy_silo);
            silo.add_variable("mesh", sxx_silo);
            silo.add_variable("mesh", sxy_silo);
            silo.add_variable("mesh", syy_silo);
            silo.add_variable("mesh", shxx_silo);
            silo.add_variable("mesh", shxy_silo);
            silo.add_variable("mesh", shyy_silo);
        }else{
            disk::silo_nodal_variable<double> vx_silo("vx", exact_ux);
            disk::silo_nodal_variable<double> vy_silo("vy", exact_uy);
            disk::silo_nodal_variable<double> vhx_silo("vhx", approx_ux);
            disk::silo_nodal_variable<double> vhy_silo("vhy", approx_uy);
            
            disk::silo_nodal_variable<double> sxx_silo("sxx", exact_sxx);
            disk::silo_nodal_variable<double> sxy_silo("sxy", exact_sxy);
            disk::silo_nodal_variable<double> syy_silo("syy", exact_syy);
            disk::silo_nodal_variable<double> shxx_silo("shxx", approx_sxx);
            disk::silo_nodal_variable<double> shxy_silo("shxy", approx_sxy);
            disk::silo_nodal_variable<double> shyy_silo("shyy", approx_syy);
            
            silo.add_variable("mesh", vx_silo);
            silo.add_variable("mesh", vy_silo);
            silo.add_variable("mesh", vhx_silo);
            silo.add_variable("mesh", vhy_silo);
            silo.add_variable("mesh", sxx_silo);
            silo.add_variable("mesh", sxy_silo);
            silo.add_variable("mesh", syy_silo);
            silo.add_variable("mesh", shxx_silo);
            silo.add_variable("mesh", shxy_silo);
            silo.add_variable("mesh", shyy_silo);
            
        }

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
    // Write a silo file with acoustic properties as zonal variables
    static void write_silo_acoustic_property_map(std::string silo_file_name, Mesh & msh, std::vector< acoustic_material_data<double> > & material){

        timecounter tc;
        tc.tic();
        
        auto num_cells = msh.cells_size();
        std::vector<double> rho_data, vp_data;
        vp_data.reserve( num_cells );
        rho_data.reserve( num_cells );
        
        for (size_t cell_id = 0; cell_id < num_cells; cell_id++)
        {
            acoustic_material_data<double> acoustic_data = material[cell_id];
            rho_data.push_back( acoustic_data.rho() );
            vp_data.push_back( acoustic_data.vp() );
        }

        disk::silo_database silo;
        silo_file_name += ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_zonal_variable<double> rho_silo("rho", rho_data);
        disk::silo_zonal_variable<double> vp_silo("vp", vp_data);
        silo.add_variable("mesh", rho_silo);
        silo.add_variable("mesh", vp_silo);

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Properties file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
    /// Compute the discrete acoustic energy for one field approximation
    static double compute_acoustic_energy_one_field(Mesh & msh, disk::hho_degree_info & hho_di, acoustic_one_field_assembler<Mesh> & assembler, double & time, Matrix<double, Dynamic, 1> & p_dof, Matrix<double, Dynamic, 1> & v_dof, std::ostream & energy_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        auto dim = Mesh::dimension;
        size_t cell_dof = disk::scalar_basis_size(hho_di.cell_degree(), dim);
        
        std::vector<RealType> energy_vec(msh.cells_size());
        #ifdef HAVE_INTEL_TBB
                size_t n_cells = msh.cells_size();
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [&msh,&assembler,&energy_vec,&p_dof,&v_dof,&cell_dof] (size_t & cell_ind){
                            auto& cell = msh.backend_storage()->surfaces[cell_ind];
                            Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_v = v_dof.block(cell_ind*cell_dof, 0, cell_dof, 1);
                            Matrix<RealType, Dynamic, 1> cell_mass_tested = mass_matrix * cell_alpha_dof_n_v;
                            Matrix<RealType, 1, 1> term_1 = cell_alpha_dof_n_v.transpose() * cell_mass_tested;
                            
                            energy_vec[cell_ind] = term_1(0,0);
                
                            Matrix<RealType, Dynamic, Dynamic> laplacian_loc = assembler.laplacian_operator(cell_ind, msh, cell);
                            Matrix<RealType, Dynamic, 1> cell_p_dofs = assembler.gather_dof_data(msh, cell, p_dof);
                            Matrix<RealType, Dynamic, 1> cell_stiff_tested = laplacian_loc * cell_p_dofs;
                            Matrix<RealType, 1, 1> term_2 = cell_p_dofs.transpose() * cell_stiff_tested;
                
                            energy_vec[cell_ind] += term_2(0,0);
                }
            );
        #else
            for (size_t cell_ind = 0; cell_ind < msh.cells_size(); cell_ind++)
            {
                auto& cell = msh.backend_storage()->surfaces[cell_ind];
                
                Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_v = v_dof.block(cell_ind*cell_dof, 0, cell_dof, 1);
                Matrix<RealType, Dynamic, 1> cell_mass_tested = mass_matrix * cell_alpha_dof_n_v;
                Matrix<RealType, 1, 1> term_1 = cell_alpha_dof_n_v.transpose() * cell_mass_tested;
                
                energy_vec[cell_ind] = term_1(0,0);
    
                Matrix<RealType, Dynamic, Dynamic> laplacian_loc = assembler.laplacian_operator(cell_ind, msh, cell);
                Matrix<RealType, Dynamic, 1> cell_p_dofs = assembler.gather_dof_data(msh, cell, p_dof);
                Matrix<RealType, Dynamic, 1> cell_stiff_tested = laplacian_loc * cell_p_dofs;
                Matrix<RealType, 1, 1> term_2 = cell_p_dofs.transpose() * cell_stiff_tested;
    
                energy_vec[cell_ind] += term_2(0,0);
        
            }
        #endif
    
        RealType energy_h = std::accumulate(energy_vec.begin(), energy_vec.end(),0.0);
        energy_h *= 0.5;
        
        tc.toc();
        std::cout << bold << cyan << "Energy completed: " << tc << " seconds" << reset << std::endl;
        energy_file << time << "   " << std::setprecision(16) << energy_h << std::endl;
        return energy_h;
    }
    
    /// Compute the discrete acoustic energy for one field approximation
    static double compute_acoustic_energy_two_fields(Mesh & msh, disk::hho_degree_info & hho_di, acoustic_two_fields_assembler<Mesh> & assembler, double & time, Matrix<double, Dynamic, 1> & x_dof, std::ostream & energy_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        size_t n_scal_cbs = disk::scalar_basis_size(hho_di.cell_degree(), Mesh::dimension);
        size_t n_vec_cbs = disk::scalar_basis_size(hho_di.reconstruction_degree(), Mesh::dimension)-1;
        size_t n_cbs = n_scal_cbs + n_vec_cbs;
        
        std::vector<RealType> energy_vec(msh.cells_size());
        #ifdef HAVE_INTEL_TBB
                size_t n_cells = msh.cells_size();
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [&msh,&assembler,&energy_vec,&x_dof,&n_cbs] (size_t & cell_ind){
                            auto& cell = msh.backend_storage()->surfaces[cell_ind];
                            Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                            Matrix<RealType, Dynamic, 1> cell_dof = x_dof.block(cell_ind*n_cbs, 0, n_cbs, 1);
                            Matrix<RealType, Dynamic, 1> cell_mass_tested = mass_matrix * cell_dof;
                            Matrix<RealType, 1, 1> term = cell_dof.transpose() * cell_mass_tested;
                            energy_vec[cell_ind] = term(0,0);
                }
            );
        #else
            for (size_t cell_ind = 0; cell_ind < msh.cells_size(); cell_ind++)
            {
                auto& cell = msh.backend_storage()->surfaces[cell_ind];
                
                Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                Matrix<RealType, Dynamic, 1> cell_dof = x_dof.block(cell_ind*n_cbs, 0, n_cbs, 1);
                Matrix<RealType, Dynamic, 1> cell_mass_tested = mass_matrix * cell_dof;
                Matrix<RealType, 1, 1> term = cell_dof.transpose() * cell_mass_tested;
            
                energy_vec[cell_ind] = term(0,0);
        
            }
        #endif
    
        RealType energy_h = std::accumulate(energy_vec.begin(), energy_vec.end(),0.0);
        energy_h *= 0.5;
        
        tc.toc();
        std::cout << bold << cyan << "Energy completed: " << tc << " seconds" << reset << std::endl;
        energy_file << time << "   " << std::setprecision(16) << energy_h << std::endl;
        
        return energy_h;
    }
    
    /// Compute the discrete elastic energy for one field approximation
    static double compute_elastic_energy_one_field(Mesh & msh, disk::hho_degree_info & hho_di, elastodynamic_one_field_assembler<Mesh> & assembler, double & time, Matrix<double, Dynamic, 1> & u_dof, Matrix<double, Dynamic, 1> & v_dof, std::ostream & energy_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        size_t cell_dof = disk::vector_basis_size(hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        
        std::vector<RealType> energy_vec(msh.cells_size());
        #ifdef HAVE_INTEL_TBB
                size_t n_cells = msh.cells_size();
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [&msh,&assembler,&energy_vec,&u_dof,&v_dof,&cell_dof] (size_t & cell_ind){
                            auto& cell = msh.backend_storage()->surfaces[cell_ind];
                            Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_v = v_dof.block(cell_ind*cell_dof, 0, cell_dof, 1);
                            Matrix<RealType, Dynamic, 1> cell_mass_tested = mass_matrix * cell_alpha_dof_n_v;
                            Matrix<RealType, 1, 1> term_1 = cell_alpha_dof_n_v.transpose() * cell_mass_tested;
                            
                            energy_vec[cell_ind] = term_1(0,0);

                            Matrix<RealType, Dynamic, Dynamic> laplacian_loc = assembler.laplacian_operator(cell_ind, msh, cell);
                            Matrix<RealType, Dynamic, 1> cell_p_dofs = assembler.gather_dof_data(msh, cell, u_dof);
                            Matrix<RealType, Dynamic, 1> cell_stiff_tested = laplacian_loc * cell_p_dofs;
                            Matrix<RealType, 1, 1> term_2 = cell_p_dofs.transpose() * cell_stiff_tested;

                            energy_vec[cell_ind] += term_2(0,0);
                }
            );
        #else
            for (size_t cell_ind = 0; cell_ind < msh.cells_size(); cell_ind++)
            {
                auto& cell = msh.backend_storage()->surfaces[cell_ind];
                
                Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_v = v_dof.block(cell_ind*cell_dof, 0, cell_dof, 1);
                Matrix<RealType, Dynamic, 1> cell_mass_tested = mass_matrix * cell_alpha_dof_n_v;
                Matrix<RealType, 1, 1> term_1 = cell_alpha_dof_n_v.transpose() * cell_mass_tested;
                
                energy_vec[cell_ind] = term_1(0,0);

                Matrix<RealType, Dynamic, Dynamic> laplacian_loc = assembler.laplacian_operator(cell_ind, msh, cell);
                Matrix<RealType, Dynamic, 1> cell_p_dofs = assembler.gather_dof_data(msh, cell, u_dof);
                Matrix<RealType, Dynamic, 1> cell_stiff_tested = laplacian_loc * cell_p_dofs;
                Matrix<RealType, 1, 1> term_2 = cell_p_dofs.transpose() * cell_stiff_tested;

                energy_vec[cell_ind] += term_2(0,0);
        
            }
        #endif

        RealType energy_h = std::accumulate(energy_vec.begin(), energy_vec.end(),0.0);
        energy_h *= 0.5;
        
        tc.toc();
        std::cout << bold << cyan << "Energy completed: " << tc << " seconds" << reset << std::endl;
        energy_file << time << "   " << std::setprecision(16) << energy_h << std::endl;
        return energy_h;
    }
    
    /// Compute the discrete acoustic energy for one field approximation
    static double compute_elastic_energy_three_fields(Mesh & msh, disk::hho_degree_info & hho_di, elastodynamic_three_fields_assembler<Mesh> & assembler, double & time, Matrix<double, Dynamic, 1> & x_dof, std::ostream & energy_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        size_t n_ten_cbs = disk::sym_matrix_basis_size(hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_sca_cbs = disk::scalar_basis_size(hho_di.face_degree(), Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        
        std::vector<RealType> energy_vec(msh.cells_size());
        #ifdef HAVE_INTEL_TBB
                size_t n_cells = msh.cells_size();
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [&msh,&assembler,&energy_vec,&x_dof,&n_ten_cbs,&n_sca_cbs,&n_vec_cbs] (size_t & cell_ind){
                            auto& cell = msh.backend_storage()->surfaces[cell_ind];
                            Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                            Matrix<RealType, Dynamic, 1> x_dof_loc = assembler.gather_dof_data(msh, cell, x_dof);
                            
                            Matrix<RealType, Dynamic, Dynamic> mass_matrix_v = mass_matrix.block(n_ten_cbs + n_sca_cbs, n_ten_cbs + n_sca_cbs, n_vec_cbs, n_vec_cbs);
                            Matrix<RealType, Dynamic, 1> v_dof = x_dof_loc.block(n_ten_cbs + n_sca_cbs, 0, n_vec_cbs, 1);
                            Matrix<RealType, Dynamic, 1> v_mass_tested = mass_matrix_v * v_dof;
                            Matrix<RealType, 1, 1> term_1 = v_dof.transpose() * v_mass_tested;
                            energy_vec[cell_ind] = term_1(0,0);
                            
                            Matrix<RealType, Dynamic, Dynamic> mass_matrix_stress = mass_matrix.block(0, 0, n_ten_cbs + n_sca_cbs, n_ten_cbs + n_sca_cbs);
                            Matrix<RealType, Dynamic, 1> sigma_dof = x_dof_loc.block(0, 0, n_ten_cbs + n_sca_cbs, 1);
                            Matrix<RealType, Dynamic, 1> epsilon_mass = mass_matrix_stress * sigma_dof;
                            Matrix<RealType, 1, 1> term_2 = sigma_dof.transpose() * epsilon_mass;
                            energy_vec[cell_ind] += term_2(0,0);
                }
            );
        #else
            for (size_t cell_ind = 0; cell_ind < msh.cells_size(); cell_ind++)
            {
                auto& cell = msh.backend_storage()->surfaces[cell_ind];
                
                Matrix<RealType, Dynamic, Dynamic> mass_matrix = assembler.mass_operator(cell_ind, msh, cell);
                Matrix<RealType, Dynamic, 1> x_dof_loc = assembler.gather_dof_data(msh, cell, x_dof);
                
                Matrix<RealType, Dynamic, Dynamic> mass_matrix_v = mass_matrix.block(n_ten_cbs + n_sca_cbs, n_ten_cbs + n_sca_cbs, n_vec_cbs, n_vec_cbs);
                Matrix<RealType, Dynamic, 1> v_dof = x_dof_loc.block(n_ten_cbs + n_sca_cbs, 0, n_vec_cbs, 1);
                Matrix<RealType, Dynamic, 1> v_mass_tested = mass_matrix_v * v_dof;
                Matrix<RealType, 1, 1> term_1 = v_dof.transpose() * v_mass_tested;
                energy_vec[cell_ind] = term_1(0,0);
                
                Matrix<RealType, Dynamic, Dynamic> mass_matrix_stress = mass_matrix.block(0, 0, n_ten_cbs + n_sca_cbs, n_ten_cbs + n_sca_cbs);
                Matrix<RealType, Dynamic, 1> sigma_dof = x_dof_loc.block(0, 0, n_ten_cbs + n_sca_cbs, 1);
                Matrix<RealType, Dynamic, 1> epsilon_mass = mass_matrix_stress * sigma_dof;
                Matrix<RealType, 1, 1> term_2 = sigma_dof.transpose() * epsilon_mass;
                energy_vec[cell_ind] += term_2(0,0);
        
            }
        #endif
    
        RealType energy_h = std::accumulate(energy_vec.begin(), energy_vec.end(),0.0);
        energy_h *= 0.5;
        
        tc.toc();
        std::cout << bold << cyan << "Energy completed: " << tc << " seconds" << reset << std::endl;
        energy_file << time << "   " << std::setprecision(16) << energy_h << std::endl;
        return energy_h;
    }
    
    // Write a silo file with elastic properties as zonal variables
    static void write_silo_elastic_property_map(std::string silo_file_name, Mesh & msh, std::vector< elastic_material_data<double> > & material){

        timecounter tc;
        tc.tic();
        
        auto num_cells = msh.cells_size();
        std::vector<double> rho_data, vp_data, vs_data;
        rho_data.reserve( num_cells );
        vp_data.reserve( num_cells );
        vs_data.reserve( num_cells );

        for (size_t cell_id = 0; cell_id < num_cells; cell_id++)
        {
            elastic_material_data<double> elastic_data = material[cell_id];
            rho_data.push_back( elastic_data.rho() );
            vp_data.push_back( elastic_data.vp() );
            vs_data.push_back( elastic_data.vs() );
        }
        

        disk::silo_database silo;
        silo_file_name += ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_zonal_variable<double> rho_silo("rho", rho_data);
        disk::silo_zonal_variable<double> vp_silo("vp", vp_data);
        disk::silo_zonal_variable<double> vs_silo("vs", vs_data);
        silo.add_variable("mesh", rho_silo);
        silo.add_variable("mesh", vp_silo);
        silo.add_variable("mesh", vs_silo);

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Properties file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
};


#endif /* postprocessor_hpp */
