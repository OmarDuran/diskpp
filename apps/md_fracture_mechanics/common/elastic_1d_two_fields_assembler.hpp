//
//  elastic_1d_two_fields_assembler.hpp
//  elastodynamics
//
//  Created by Omar Durán on 22/03/21.
//

#pragma once
#ifndef elastic_1d_two_fields_assembler_hpp
#define elastic_1d_two_fields_assembler_hpp

#include "bases/bases.hpp"
#include "methods/hho"
#include "../common/assembly_index.hpp"
#include "../common/elastic_material_data.hpp"

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
class elastic_1d_two_fields_assembler
{
    
    
    typedef disk::BoundaryConditions<Mesh, true>    boundary_type;
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;
    using node_type = typename Mesh::node_type;
    using edge_type = typename Mesh::edge_type;

    std::vector<size_t>                 m_compress_indexes;
    std::vector<size_t>                 m_compress_sigma_indexes;
    std::vector<size_t>                 m_expand_indexes;

    disk::hho_degree_info               m_hho_di;
    boundary_type                       m_bnd;
    std::vector< Triplet<T> >           m_triplets;
    std::vector< elastic_material_data<T> > m_material;
    std::vector< size_t >               m_elements_with_bc_eges;

    size_t      m_n_edges;
    size_t      m_n_essential_edges;
    size_t      m_n_cells_dof;
    size_t      m_n_faces_dof;
    bool        m_hho_stabilization_Q;
    bool        m_scaled_stabilization_Q;
    
public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    elastic_1d_two_fields_assembler(const Mesh& msh, const disk::hho_degree_info& hho_di, const boundary_type& bnd)
        : m_hho_di(hho_di), m_bnd(bnd), m_hho_stabilization_Q(true), m_scaled_stabilization_Q(false)
    {
            
        auto is_dirichlet = [&](const typename Mesh::face& fc) -> bool {

            auto fc_id = msh.lookup(fc);
            return bnd.is_dirichlet_face(fc_id);
        };

        m_n_edges = msh.faces_size();
        m_n_essential_edges = std::count_if(msh.faces_begin(), msh.faces_end(), is_dirichlet);

        m_compress_indexes.resize( m_n_edges );
        m_expand_indexes.resize( m_n_edges - m_n_essential_edges );

        m_n_faces_dof = 0;
        
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        for (size_t face_id = 0; face_id < msh.faces_size(); face_id++)
        {
            m_compress_indexes.at(face_id) = m_n_faces_dof;
//                m_expand_indexes.at(compressed_offset) = face_id;
            auto non_essential_dofs = n_fbs - m_bnd.dirichlet_imposed_dofs(face_id, m_hho_di.face_degree());
            
            m_n_faces_dof += non_essential_dofs;
        }
        
        size_t n_ten_cbs = disk::vector_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        m_n_cells_dof = n_cbs * msh.cells_size();
            
        size_t system_size = m_n_cells_dof + m_n_faces_dof;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
        classify_cells(msh);
    }

    void scatter_data(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs,
             const Matrix<T, Dynamic, 1>& rhs)
    {
        auto fcs = faces(msh, cl);
        size_t n_ten_cbs = disk::vector_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs + n_fbs*fcs.size());

        auto cell_offset        = disk::priv::offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * n_cbs;

        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );
        
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = disk::priv::offset(msh, fc);
            auto fc_id = msh.lookup(fc);
            auto face_LHS_offset = n_cbs * msh.cells_size() + m_compress_indexes.at(fc_id);
            bool dirichlet = m_bnd.is_dirichlet_face(fc_id);

            if (dirichlet)
             {
                 switch (m_bnd.dirichlet_boundary_type(fc_id)) {
                    case disk::DIRICHLET: {
                        for (size_t i = 0; i < n_fbs; i++){
                            asm_map.push_back( assembly_index(face_LHS_offset+i, false) );
                        }
                        break;
                    }
                     default: {
                        throw std::logic_error("Unknown Dirichlet Conditions.");
                        break;
                     }
                 }
             }
            else{
                for (size_t i = 0; i < n_fbs; i++)
                    asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
            }
            
            bool neumann = m_bnd.is_neumann_face(fc_id);
            if (neumann) {
                auto bc_rhs = neumman_rhs(msh, fc, fc_id);
                for (size_t i = 0; i < bc_rhs.rows(); i++)
                {
                    RHS(face_LHS_offset+i) += bc_rhs(i);
                }
            }
            
            
            
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
            }
        }

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs(i);
        }

    }
            
    void scatter_bc_data(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, Dynamic>& lhs)
    {
        auto fcs = faces(msh, cl);
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs + n_fbs*fcs.size());

        auto cell_offset        = disk::priv::offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * n_cbs;

        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );
        
        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(n_cbs + fcs.size()*n_fbs);
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = disk::priv::offset(msh, fc);
            auto fc_id = msh.lookup(fc);
            auto face_LHS_offset = n_cbs * msh.cells_size() + m_compress_indexes.at(fc_id);
            bool dirichlet = m_bnd.is_dirichlet_face(fc_id);
            if (dirichlet)
             {
                 switch (m_bnd.dirichlet_boundary_type(fc_id)) {
                    case disk::DIRICHLET: {
                        for (size_t i = 0; i < n_fbs; i++){
                            asm_map.push_back( assembly_index(face_LHS_offset+i, false) );
                        }
                        break;
                    }
                    case disk::DX: {
                         for (size_t i = 0; i < n_fbs/Mesh::dimension; i++){
                             asm_map.push_back( assembly_index(face_LHS_offset+i, false) );
                             asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
                         }
                        break;
                    }
                    case disk::DY: {
                        for (size_t i = 0; i < n_fbs/Mesh::dimension; i++){
                            asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i, false) );
                        }
                        break;
                    }
                     default: {
                        throw std::logic_error("Unknown Dirichlet Conditions.");
                        break;
                     }
                 }
                 
                 auto fb = make_vector_monomial_basis(msh, fc, m_hho_di.face_degree());
                 auto dirichlet_fun  = m_bnd.dirichlet_boundary_func(fc_id);

                 Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, fb);
                 Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, fb, dirichlet_fun);
                 dirichlet_data.block(n_cbs + face_i*n_fbs, 0, n_fbs, 1) = mass.llt().solve(rhs);
             }
            else{
                for (size_t i = 0; i < n_fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true));
            }
            
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( !asm_map[j].assemble() )
                    RHS(asm_map[i]) -= lhs(i,j) * dirichlet_data(j);
            }
        }

    }
            
    void scatter_rhs_data(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& rhs)
    {
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs);

        auto cell_offset        = disk::priv::offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * n_cbs;

        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        assert( asm_map.size() == rhs.rows());

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs(i);
        }

    }

    void assemble(const Mesh& msh, std::function<double(const typename Mesh::point_type& )> rhs_fun){
        
        LHS.setZero();
        RHS.setZero();
        size_t cell_ind = 0;
        for (auto& cell : msh)
        {
            Matrix<T, Dynamic, Dynamic> mixed_operator_loc = mixed_operator(cell_ind,msh,cell);
            Matrix<T, Dynamic, 1> f_loc = mixed_rhs(msh, cell, rhs_fun);
            scatter_data(msh, cell, mixed_operator_loc, f_loc);
            cell_ind++;
        }
        
        finalize();
    }

    void apply_bc(const Mesh& msh){
        
        #ifdef HAVE_INTEL_TBB2
                size_t n_cells = m_elements_with_bc_eges.size();
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [this,&msh] (size_t & i){
                        size_t cell_ind = m_elements_with_bc_eges[i];
                        auto& cell = msh.backend_storage()->surfaces[cell_ind];
                        Matrix<T, Dynamic, Dynamic> mixed_operator_loc = mixed_operator(cell_ind, msh, cell);
                        scatter_bc_data(msh, cell, mixed_operator_loc);
                }
            );
        #else
            auto storage = msh.backend_storage();
            for (auto& cell_ind : m_elements_with_bc_eges)
            {
                auto& cell = storage->surfaces[cell_ind];
                Matrix<T, Dynamic, Dynamic> mixed_operator_loc = mixed_operator(cell_ind, msh, cell);
                scatter_bc_data(msh, cell, mixed_operator_loc);
            }
        #endif
        
    }
            
    void assemble_rhs(const Mesh& msh, std::function<static_vector<double, 2>(const typename Mesh::point_type& )>  rhs_fun){
        
        RHS.setZero();
         
    #ifdef HAVE_INTEL_TBB
            size_t n_cells = msh.cells_size();
            tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                [this,&msh,&rhs_fun] (size_t & cell_ind){
                    auto& cell = msh.backend_storage()->surfaces[cell_ind];
                    Matrix<T, Dynamic, 1> f_loc = this->mixed_rhs(msh, cell, rhs_fun);
                    this->scatter_rhs_data(msh, cell, f_loc);
            }
        );
    #else
        auto contribute = [this,&msh,&rhs_fun] (const typename Mesh::cell_type& cell){
            Matrix<T, Dynamic, 1> f_loc = this->mixed_rhs(msh, cell, rhs_fun);
            this->scatter_rhs_data(msh, cell, f_loc);
        };
        
        for (auto& cell : msh){
            contribute(cell);
        }
    #endif
        apply_bc(msh);
    }
            
    Matrix<T, Dynamic, Dynamic> mixed_operator(size_t & cell_ind, const Mesh& msh, const typename Mesh::cell_type& cell){
            
        elastic_material_data<T> & material = m_material[cell_ind];
        T rho = material.rho();
        T mu = material.mu();
        T lambda = material.l();
        
        auto reconstruction_operator   = mixed_scalar_reconstruction(msh, cell);
        Matrix<T, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        auto n_rows = R_operator.rows();
        auto n_cols = R_operator.cols();
        
        Matrix<T, Dynamic, Dynamic> S_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        auto stabilization_operator    = compute_scalar_hdg_stabilization(msh, cell, m_scaled_stabilization_Q);
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        Matrix<T, Dynamic, Dynamic> mass = mass_operator(cell_ind, msh, cell);
        Matrix<T, Dynamic, Dynamic> M_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        M_operator.block(0,0,mass.rows(),mass.cols()) = mass;
        
//        std::cout << "r = " << R_operator << std::endl;
//        std::cout << "s = " << S_operator << std::endl;
//        std::cout << "m = " << M_operator << std::endl;
        
        return M_operator + R_operator + 2.0*(rho*mu)*S_operator;
    }
    
    std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
                 Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
    mixed_scalar_reconstruction(const Mesh& msh, const typename Mesh::cell_type& cell)
    {
        using T = typename Mesh::coordinate_type;
        typedef Matrix<T, Dynamic, Dynamic> matrix_type;
        typedef Matrix<T, Dynamic, 1>       vector_type;

        const size_t DIM = Mesh::dimension;

        const auto recdeg = m_hho_di.reconstruction_degree();
        const auto celdeg = m_hho_di.cell_degree();
        const auto facdeg = m_hho_di.face_degree();

        auto cb = make_scalar_monomial_basis(msh, cell, recdeg);

        const auto rbs = disk::scalar_basis_size(recdeg, Mesh::dimension);
        const auto cbs = disk::scalar_basis_size(celdeg, Mesh::dimension);
        const auto fbs = disk::scalar_basis_size(facdeg, Mesh::dimension - 1);

        const auto num_faces = howmany_faces(msh, cell);

        const matrix_type stiff  = make_stiffness_matrix(msh, cell, cb);
        
        matrix_type gr_lhs = matrix_type::Zero(rbs-1, rbs-1);
        matrix_type gr_rhs = matrix_type::Zero(rbs-1, cbs + num_faces*fbs);
        
        gr_lhs = stiff.block(1, 1, rbs-1, rbs-1);
        gr_rhs.block(0, 0, rbs-1, cbs) = stiff.block(1, 0, rbs-1, cbs);

//        std::cout << " mass = " << stiff << std::endl;
//        std::cout << " mass rhs = " << stiff.block(1, 0, rbs-1, cbs) << std::endl;
        const auto fcs = faces(msh, cell);
        for (size_t i = 0; i < fcs.size(); i++)
        {
            const auto fc = fcs[i];
            const auto n  = normal(msh, cell, fc);
            auto fb = make_scalar_monomial_basis(msh, fc, facdeg);
            
            auto point = barycenter(msh,fc);
            T weight = 1.0;
            vector_type c_phi_tmp = cb.eval_functions(point);
//            std::cout << "c_phi_tmp = " << c_phi_tmp << std::endl;
            vector_type c_phi = c_phi_tmp.head(cbs);
            Matrix<T, Dynamic, DIM> c_dphi_tmp = cb.eval_gradients(point);
//            std::cout << "c_dphi_tmp = " << c_dphi_tmp << std::endl;
            Matrix<T, Dynamic, DIM> c_dphi = c_dphi_tmp.block(1, 0, rbs-1, DIM);
            vector_type f_phi = fb.eval_functions(point);
            gr_rhs.block(0, cbs+i*fbs, rbs-1, fbs) += weight * (c_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs-1, cbs) -= weight * (c_dphi * n) * c_phi.transpose();
        }

        auto vec_cell_size = gr_lhs.cols();
        auto nrows = gr_rhs.cols()+vec_cell_size;
        auto ncols = gr_rhs.cols()+vec_cell_size;

        // Shrinking data
        matrix_type data_mixed = matrix_type::Zero(nrows,ncols);
        data_mixed.block(0, vec_cell_size, vec_cell_size, ncols-vec_cell_size) = -gr_rhs;
        data_mixed.block(vec_cell_size, 0, nrows-vec_cell_size, vec_cell_size) = gr_rhs.transpose();

        matrix_type oper = gr_lhs.llt().solve(gr_rhs);
        return std::make_pair(oper, data_mixed);
    }
    
    Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
    compute_scalar_hdg_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl, bool scaled_Q = true)
    {
        using T = typename Mesh::coordinate_type;
        typedef Matrix<T, Dynamic, Dynamic> matrix_type;

        const auto celdeg = m_hho_di.cell_degree();
        const auto facdeg = m_hho_di.face_degree();

        
        const auto cbs = disk::vector_basis_size(celdeg, Mesh::dimension, Mesh::dimension);
        const auto fbs = disk::vector_basis_size(facdeg, Mesh::dimension - 1, Mesh::dimension);

        const auto num_faces = howmany_faces(msh, cl);
        const auto total_dofs = cbs + num_faces * fbs;

        matrix_type       data = matrix_type::Zero(total_dofs, total_dofs);
        const matrix_type If   = matrix_type::Identity(fbs, fbs);

        auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
        const auto fcs = faces(msh, cl);

        for (size_t i = 0; i < num_faces; i++)
        {
            const auto fc = fcs[i];
            const auto h  = diameter(msh, fc);
            auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

            matrix_type oper  = matrix_type::Zero(fbs, total_dofs);
            matrix_type tr    = matrix_type::Zero(fbs, total_dofs);
            matrix_type mass  = matrix_type::Identity(fbs, fbs); // hard coded
            matrix_type trace = matrix_type::Zero(fbs, cbs);

            oper.block(0, cbs + i  * fbs, fbs, fbs) = -If;

            auto point = barycenter(msh,fc);
            T weight = 1.0;
            const auto c_phi = cb.eval_functions(point);
            const auto f_phi = fb.eval_functions(point);

            assert(c_phi.rows() == cbs);
            assert(f_phi.rows() == fbs);
            assert(c_phi.cols() == f_phi.cols());

            trace += disk::priv::outer_product(disk::priv::inner_product(weight, f_phi), c_phi);

            tr.block(0, cbs + i * fbs, fbs, fbs) = -mass;
            tr.block(0, 0, fbs, cbs) = trace;

            oper.block(0, 0, fbs, cbs) = mass.ldlt().solve(trace);
//            std::cout << "mass = " << mass << std::endl;
//            std::cout << "tr = " << tr << std::endl;
//            std::cout << "oper.transpose() = " << oper.transpose() << std::endl;
//            std::cout << "oper.transpose() * tr = " << std::endl;
//            std::cout << oper.transpose() * tr << std::endl;
            if (scaled_Q) {
                data += oper.transpose() * tr / h;
            }else{
                data += oper.transpose() * tr;
            }
        }
//        std::cout << "data = " << data << std::endl;
        return data;
    }
            
    Matrix<T, Dynamic, Dynamic>
    symmetric_tensor_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cell)
    {
            
        size_t dim = Mesh::dimension;
        auto gradeg = m_hho_di.grad_degree();
        auto ten_b = make_vector_monomial_basis(msh, cell, gradeg);
        //make_sym_matrix_monomial_basis(msh, cell, gradeg);
//        auto ten_bs = disk::sym_matrix_basis_size(gradeg, dim, dim);
        auto ten_bs = disk::sym_matrix_basis_size(gradeg, dim, dim);
        Matrix<T, Dynamic, Dynamic> mass_matrix = Matrix<T, Dynamic, Dynamic>::Zero(ten_bs, ten_bs);
        
        auto qps = integrate(msh, cell, 2 * gradeg);

//        // number of tensor components
//        size_t dec = 0;
//         if (dim == 3)
//             dec = 6;
//         else if (dim == 2)
//             dec = 3;
//         else
//             std::logic_error("Expected 3 >= dim > 1");
//
//         for (auto& qp : qps)
//         {
//             auto phi = ten_b.eval_functions(qp.point());
//
//             for (size_t j = 0; j < ten_bs; j++)
//             {
//
//                auto qp_phi_j = disk::priv::inner_product(qp.weight(), phi[j]);
//                for (size_t i = j; i < ten_bs; i += dec){
//                         mass_matrix(i, j) += disk::priv::inner_product(phi[i], qp_phi_j);
//                }
//             }
//         }
//
//        for (size_t j = 0; j < ten_bs; j++){
//            for (size_t i = 0; i < j; i++){
//                 mass_matrix(i, j) = mass_matrix(j, i);
//            }
//        }
        
        return mass_matrix;
    }
    
    Matrix<T, Dynamic, Dynamic>
    symmetric_tensor_trace_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cell)
    {
            
        size_t dim = Mesh::dimension;
        auto gradeg = m_hho_di.grad_degree();
        auto ten_b = make_vector_monomial_basis(msh, cell, gradeg);
        //make_sym_matrix_monomial_basis(msh, cell, gradeg);
        auto ten_bs = disk::sym_matrix_basis_size(gradeg, dim, dim);
        Matrix<T, Dynamic, Dynamic> mass_matrix = Matrix<T, Dynamic, Dynamic>::Zero(ten_bs, ten_bs);
        
        auto qps = integrate(msh, cell, 2 * gradeg);

//        // number of tensor components
//        size_t dec = 0;
//         if (dim == 3)
//             dec = 6;
//         else if (dim == 2)
//             dec = 3;
//         else
//             std::logic_error("Expected 3 >= dim > 1");
//
//         for (auto& qp : qps)
//         {
//             auto phi = ten_b.eval_functions(qp.point());
//
//             for (size_t j = 0; j < ten_bs; j++)
//             {
//                auto identity = phi[j];
//                identity.setZero();
//                for(size_t d = 0; d < dim; d++){
//                    identity(d,d) = 1.0;
//                }
//                auto trace = phi[j].trace();
//                auto trace_phi_j = disk::priv::inner_product(phi[j].trace(), identity);
//                auto qp_phi_j = disk::priv::inner_product(qp.weight(), trace_phi_j);
//                for (size_t i = 0; i < ten_bs; i ++){
//                         mass_matrix(i, j) += disk::priv::inner_product(phi[i], qp_phi_j);
//                }
//             }
//         }
        
        return mass_matrix;
    }
            
    Matrix<T, Dynamic, Dynamic> mass_operator(size_t & cell_ind, const Mesh& msh, const typename Mesh::cell_type& cell){
            
        size_t n_ten_cbs = disk::vector_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        size_t rbs = disk::scalar_basis_size(m_hho_di.reconstruction_degree(), Mesh::dimension);
            
        elastic_material_data<T> & material = m_material[cell_ind];
        T rho = material.rho();
        T mu = material.mu();
        T lambda = material.l();
    
        Matrix<T, Dynamic, Dynamic> mass_matrix = Matrix<T, Dynamic, Dynamic>::Zero(n_cbs, n_cbs);
        
        auto scal_basis = disk::make_scalar_monomial_basis(msh, cell, m_hho_di.reconstruction_degree());
        Matrix<T, Dynamic, Dynamic> mass_matrix_sigma_full = disk::make_stiffness_matrix(msh, cell, scal_basis);
        Matrix<T, Dynamic, Dynamic> mass_matrix_sigma = mass_matrix_sigma_full.block(1, 1, rbs-1, rbs-1);
        
        // Constitutive relationship inverse
        mass_matrix_sigma *= (1.0/(lambda+2.0*mu));
        mass_matrix.block(0, 0, n_ten_cbs, n_ten_cbs) = mass_matrix_sigma;

        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh,cell,scal_basis);
//        std::cout << "mass 1d" << mass << std::endl;
        mass_matrix.block(n_ten_cbs,n_ten_cbs,n_vec_cbs,n_vec_cbs) = mass;
        
        // basis
//        {
//            auto pts = cell.point_ids();
//            auto pt_id = pts[0];
//            auto bar = *std::next(msh.points_begin(), pt_id);
//            auto t_phi = scal_basis.eval_functions(bar);
//            std::cout << "t_phi : " << std::endl;
//            std::cout << t_phi << std::endl;
//        }
        
        
        return mass_matrix;
    }
    
    void classify_cells(const Mesh& msh){

        m_elements_with_bc_eges.clear();
        size_t cell_ind = 0;
        for (auto& cell : msh)
        {
            auto face_list = faces(msh, cell);
            for (size_t face_i = 0; face_i < face_list.size(); face_i++)
            {
                auto fc = face_list[face_i];
                auto fc_id = msh.lookup(fc);
                bool is_dirichlet_Q = m_bnd.is_dirichlet_face(fc_id);
                if (is_dirichlet_Q)
                {
                    m_elements_with_bc_eges.push_back(cell_ind);
                    break;
                }
            }
            cell_ind++;
        }
    }
            
    void project_over_cells(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<double(const typename Mesh::point_type& )> vec_fun, std::function<double(const typename Mesh::point_type& )> ten_fun){
        size_t n_dof = LHS.rows();
        x_glob = Matrix<T, Dynamic, 1>::Zero(n_dof);
        size_t n_ten_cbs = disk::vector_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        for (auto& cell : msh)
        {
            Matrix<T, Dynamic, 1> x_proj_ten_dof = project_ten_function(msh, cell, ten_fun);
            Matrix<T, Dynamic, 1> x_proj_vec_dof = project_function(msh, cell, m_hho_di.cell_degree(), vec_fun);
            
            Matrix<T, Dynamic, 1> x_proj_dof = Matrix<T, Dynamic, 1>::Zero(n_cbs);
            x_proj_dof.block(0, 0, n_ten_cbs, 1)                    = x_proj_ten_dof;
            x_proj_dof.block(n_ten_cbs, 0, n_vec_cbs, 1)  = x_proj_vec_dof;
            scatter_cell_dof_data(msh, cell, x_glob, x_proj_dof);
        }
    }
    
    Matrix<T, Dynamic, 1> project_ten_function(const Mesh& msh, const typename Mesh::cell_type& cell,
                      std::function<double(const typename Mesh::point_type& )> ten_fun){
    
            auto recdeg = m_hho_di.reconstruction_degree();
            auto rec_basis = make_scalar_monomial_basis(msh, cell, recdeg);
            auto rbs = disk::scalar_basis_size(recdeg, Mesh::dimension);
            Matrix<T, Dynamic, Dynamic> mass_matrix_q_full  = make_stiffness_matrix(msh, cell, rec_basis);
            Matrix<T, Dynamic, Dynamic> mass_matrix_q = Matrix<T, Dynamic, Dynamic>::Zero(rbs-1, rbs-1);
            mass_matrix_q = mass_matrix_q_full.block(1, 1, rbs-1, rbs-1);

            Matrix<T, Dynamic, 1> rhs = Matrix<T, Dynamic, 1>::Zero(rbs-1);
            const auto qps = integrate(msh, cell, 2*recdeg);
            for (auto& qp : qps)
            {
              auto dphi = rec_basis.eval_gradients(qp.point());
              auto flux = ten_fun(qp.point());
              for (size_t i = 0; i < rbs-1; i++){
              Matrix<T, 1, 1> phi_i = dphi.block(i+1, 0, 1, 1).transpose();
                  int aka = 0;
                  rhs(i) = rhs(i) + (qp.weight() * flux * phi_i(0,0));
              }
                
            }
            Matrix<T, Dynamic, 1> x_dof = mass_matrix_q.llt().solve(rhs);
            return x_dof;
    }
    
    void project_over_faces(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<double(const typename Mesh::point_type& )> vec_fun){

        for (auto& cell : msh)
        {
            auto fcs = faces(msh, cell);
            for (size_t i = 0; i < fcs.size(); i++)
            {
                auto face = fcs[i];
                auto fc_id = msh.lookup(face);
                bool is_dirichlet_Q = m_bnd.is_dirichlet_face(fc_id);
                if (is_dirichlet_Q)
                {
                    continue;
                }
                Matrix<T, Dynamic, 1> x_proj_dof = Matrix<T, Dynamic, Dynamic>::Zero(1, 1);
                auto bar = barycenter(msh,face);
                x_proj_dof(0,0) = vec_fun(bar);
                scatter_face_dof_data(msh, face, x_glob, x_proj_dof);
            }
        }
    }
            
    void finalize(void)
    {
        LHS.setFromTriplets( m_triplets.begin(), m_triplets.end() );
        m_triplets.clear();
    }

    Matrix<T, Dynamic, 1>
    gather_dof_data(  const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& x_glob) const
    {
        auto num_faces = howmany_faces(msh, cl);
        auto cell_ofs = disk::priv::offset(msh, cl);
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_dof = n_cbs + num_faces * n_fbs;
            
        Matrix<T, Dynamic, 1> x_el(n_dof);
        x_el.block(0, 0, n_cbs, 1) = x_glob.block(cell_ofs * n_cbs, 0, n_cbs, 1);
        auto fcs = faces(msh, cl);
        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto fc = fcs[i];
            auto eid = find_element_id(msh.faces_begin(), msh.faces_end(), fc);
            if (!eid.first) throw std::invalid_argument("This is a bug: face not found");
            const auto face_id                  = eid.second;

            if (m_bnd.is_dirichlet_face( face_id))
            {
                auto fb = make_vector_monomial_basis(msh, fc, m_hho_di.face_degree());
                auto dirichlet_fun  = m_bnd.dirichlet_boundary_func(face_id);
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, fb);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, fb, dirichlet_fun);
                x_el.block(n_cbs + i * n_fbs, 0, n_fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_ofs = disk::priv::offset(msh, fc);
                auto global_ofs = n_cbs * msh.cells_size() + m_compress_indexes.at(face_ofs)*n_fbs;
                x_el.block(n_cbs + i*n_fbs, 0, n_fbs, 1) = x_glob.block(global_ofs, 0, n_fbs, 1);
            }
        }
        return x_el;
    }
            
    void scatter_cell_dof_data(  const Mesh& msh, const typename Mesh::cell_type& cell,
                    Matrix<T, Dynamic, 1>& x_glob, Matrix<T, Dynamic, 1> & x_proj_dof) const
    {
        auto cell_ofs = disk::priv::offset(msh, cell);
        size_t n_ten_cbs = disk::vector_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        x_glob.block(cell_ofs * n_cbs, 0, n_cbs, 1) = x_proj_dof;
    }
    
    void scatter_face_dof_data(  const Mesh& msh, const typename Mesh::face_type& face,
                    Matrix<T, Dynamic, 1>& x_glob, Matrix<T, Dynamic, 1> & x_proj_dof) const
    {
        size_t n_ten_cbs = disk::vector_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_cells = msh.cells_size();
        auto face_offset = disk::priv::offset(msh, face);
        auto glob_offset = n_cbs * n_cells + m_compress_indexes.at(face_offset)*n_fbs;
        x_glob.block(glob_offset, 0, n_fbs, 1) = x_proj_dof;
    }
                 
    std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
                 Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
    strain_tensor_reconstruction(const Mesh& msh, const typename Mesh::cell_type& cell)
    {

        using T        = typename Mesh::coordinate_type;
        typedef Matrix<T, Dynamic, Dynamic> matrix_type;

        const size_t N = Mesh::dimension;

        auto graddeg = m_hho_di.grad_degree();
        auto celdeg  = m_hho_di.cell_degree();
        auto facdeg  = m_hho_di.face_degree();

        auto ten_b = make_vector_monomial_basis(msh, cell, graddeg);
        auto vec_b = make_vector_monomial_basis(msh, cell, celdeg);

        auto ten_bs = disk::vector_basis_size(graddeg, Mesh::dimension, Mesh::dimension);
        auto vec_bs = disk::vector_basis_size(celdeg, Mesh::dimension, Mesh::dimension);
        auto fbs = disk::vector_basis_size(facdeg, Mesh::dimension - 1, Mesh::dimension);

        auto num_faces = howmany_faces(msh, cell);

        matrix_type gr_lhs = matrix_type::Zero(ten_bs, ten_bs);
        matrix_type gr_rhs = matrix_type::Zero(ten_bs, vec_bs + num_faces * fbs);
        
        const auto qps = integrate(msh, cell, 2 * graddeg);

        size_t dec = 0;
         if (N == 3)
             dec = 6;
         else if (N == 2)
             dec = 3;
         else
             std::logic_error("Expected 3 >= dim > 1");

         for (auto& qp : qps)
         {
             const auto gphi = ten_b.eval_functions(qp.point());

             for (size_t j = 0; j < ten_bs; j++)
             {
                 
//                auto qp_gphi_j = disk::priv::inner_product(qp.weight(), gphi[j]);
//                for (size_t i = j; i < ten_bs; i += dec){
//                         gr_lhs(i, j) += disk::priv::inner_product(gphi[i], qp_gphi_j);
//                }
             }
         }

         for (size_t j = 0; j < ten_bs; j++)
             for (size_t i = 0; i < j; i++)
                 gr_lhs(i, j) = gr_lhs(j, i);

         if (celdeg > 0)
         {
             const auto qpc = integrate(msh, cell, graddeg + celdeg - 1);
             for (auto& qp : qpc)
             {
                 const auto gphi    = ten_b.eval_functions(qp.point());
                 const auto dphi    = vec_b.eval_sgradients(qp.point());
                 const auto qp_dphi = disk::priv::inner_product(qp.weight(), dphi);

//                 gr_rhs.block(0, 0, ten_bs, vec_bs) += disk::priv::outer_product(gphi, qp_dphi);

             }
         }

         const auto fcs = faces(msh, cell);
         for (size_t i = 0; i < fcs.size(); i++)
         {
             const auto fc = fcs[i];
             const auto n  = normal(msh, cell, fc);
             const auto fb = make_vector_monomial_basis(msh, fc, facdeg);

             const auto qps_f = integrate(msh, fc, graddeg + std::max(celdeg, facdeg));
             for (auto& qp : qps_f)
             {
                 const auto gphi = ten_b.eval_functions(qp.point());
                 const auto cphi = vec_b.eval_functions(qp.point());
                 const auto fphi = fb.eval_functions(qp.point());

//                 const auto qp_gphi_n = disk::priv::inner_product(gphi, disk::priv::inner_product(qp.weight(), n));
//                 gr_rhs.block(0, vec_bs + i * fbs, ten_bs, fbs) += disk::priv::outer_product(qp_gphi_n, fphi);
//                 gr_rhs.block(0, 0, ten_bs, vec_bs) -= disk::priv::outer_product(qp_gphi_n, cphi);
             }
         }
            
        auto n_rows = gr_rhs.cols() + ten_bs;
        auto n_cols = gr_rhs.cols() + ten_bs;

        // Shrinking data
        matrix_type data_mixed = matrix_type::Zero(n_rows,n_cols);
        data_mixed.block(0, (ten_bs), ten_bs, n_cols-(ten_bs)) = -gr_rhs;
        data_mixed.block((ten_bs), 0, n_rows-(ten_bs), ten_bs) = gr_rhs.transpose();

        matrix_type oper = gr_lhs.llt().solve(gr_rhs);
        return std::make_pair(oper, data_mixed);
    }
            
    Matrix<typename Mesh::coordinate_type, Dynamic, 1>
    mixed_rhs(const Mesh& msh, const typename Mesh::cell_type& cell, std::function<double(const typename Mesh::point_type& )> & rhs_fun, size_t di = 0)
    {
        auto recdeg = m_hho_di.grad_degree();
        auto celdeg = m_hho_di.cell_degree();
        auto facdeg = m_hho_di.face_degree();

        auto ten_bs = disk::vector_basis_size(recdeg, Mesh::dimension, Mesh::dimension);
        auto vec_bs = disk::vector_basis_size(celdeg, Mesh::dimension, Mesh::dimension);
        size_t n_cbs = ten_bs + vec_bs;
        auto cell_basis   = make_vector_monomial_basis(msh, cell, celdeg);
        using T = typename Mesh::coordinate_type;

        Matrix<T, Dynamic, 1> ret_loc = Matrix<T, Dynamic, 1>::Zero(cell_basis.size());
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(n_cbs);

        const auto qps = integrate(msh, cell, 2 * (celdeg + di));

        for (auto& qp : qps)
        {
            const auto phi  = cell_basis.eval_functions(qp.point());
            const auto qp_f = disk::priv::inner_product(qp.weight(), rhs_fun(qp.point()));
            ret_loc += disk::priv::outer_product(phi, qp_f);
        }
        ret.block(ten_bs,0,vec_bs,1) = ret_loc;
        return ret;
    }
    
    Matrix<typename Mesh::coordinate_type, Dynamic, 1>
    neumman_rhs(const Mesh& msh, const typename Mesh::face_type& face, size_t face_id, size_t di = 0)
    {
        auto recdeg = m_hho_di.grad_degree();
        auto celdeg = m_hho_di.cell_degree();
        auto facdeg = m_hho_di.face_degree();
        
        auto n_fbs   = disk::vector_basis_size(facdeg, Mesh::dimension - 1, Mesh::dimension);
        auto face_basis   = make_vector_monomial_basis(msh, face, facdeg);
        using T = typename Mesh::coordinate_type;

        auto neumann_fun  = m_bnd.neumann_boundary_func(face_id);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(n_fbs);
        auto point = barycenter(msh,face);
        T weight = 1.0;
        const auto phi  = face_basis.eval_functions(point);
        const auto qp_f = disk::priv::inner_product(weight, neumann_fun(point));
        ret += disk::priv::outer_product(phi, qp_f);
        
        return ret;
    }
            
    void load_material_data(const Mesh& msh, elastic_material_data<T> material){
        m_material.clear();
        m_material.reserve(msh.cells_size());
        for (size_t cell_ind = 0; cell_ind < msh.cells_size(); cell_ind++)
        {
            m_material.push_back(material);
        }
    }
            
    void load_material_data(const Mesh& msh, std::function<std::vector<double>(const typename Mesh::point_type& )> elastic_mat_fun){
        m_material.clear();
        m_material.reserve(msh.cells_size());
        for (auto& cell : msh)
        {
            auto bar = barycenter(msh, cell);
            std::vector<double> mat_data = elastic_mat_fun(bar);
            T rho = mat_data[0];
            T vp = mat_data[1];
            T vs = mat_data[2];
            elastic_material_data<T> material(rho,vp,vs);
            m_material.push_back(material);
        }
    }
     
    void load_material_data(const Mesh& msh){
        m_material.clear();
        m_material.reserve(msh.cells_size());
        T rho = 1.0;
        T vp = sqrt(3.0);
        T vs = 1.0;
        elastic_material_data<T> material(rho,vp,vs);
        size_t cell_i = 0;
        for (auto& cell : msh)
        {
            m_material.push_back(material);
            cell_i++;
        }
    }
            
    void set_hdg_stabilization(){
        if(m_hho_di.cell_degree() > m_hho_di.face_degree())
        {
            m_hho_stabilization_Q = false;
            std::cout << "Proceeding with HDG stabilization cell degree is higher than face degree." << std::endl;
            std::cout << "cell degree = " << m_hho_di.cell_degree() << std::endl;
            std::cout << "face degree = " << m_hho_di.face_degree() << std::endl;
        }else{
            std::cout << "Proceeding with HHO stabilization cell and face degree are equal." << std::endl;
            std::cout << "cell degree = " << m_hho_di.cell_degree() << std::endl;
            std::cout << "face degree = " << m_hho_di.face_degree() << std::endl;
        }
    }
    
    void set_hho_stabilization(){
        m_hho_stabilization_Q = true;
    }
    
    void set_scaled_stabilization(){
        m_scaled_stabilization_Q = true;
    }
            
    boundary_type & get_bc_conditions(){
             return m_bnd;
    }
            
    std::vector< elastic_material_data<T> > & get_material_data(){
        return m_material;
    }
            
    size_t get_n_face_dof(){
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_face_dof = (m_n_edges - m_n_essential_edges) * n_fbs;
        return n_face_dof;
    }
    
    size_t get_n_faces(){
        return m_n_edges - m_n_essential_edges;
    }
    
    size_t get_face_basis_data(){
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        return n_fbs;
    }
    
    size_t get_cell_basis_data(){
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        return n_cbs;
    }
    
    size_t get_n_cells_dofs(){
        return m_n_cells_dof;
    }
    size_t get_n_faces_dofs(){
        return m_n_faces_dof;
    }
    std::vector<size_t> & compress_indexes(){
        return m_compress_indexes;
    }
    
};

#endif /* elastic_1d_two_fields_assembler_hpp */
