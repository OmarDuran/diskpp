//
//  elastic_two_fields_assembler.hpp
//  elastodynamics
//
//  Created by Omar Durán on 22/03/21.
//

#pragma once
#ifndef elastic_two_fields_assembler_hpp
#define elastic_two_fields_assembler_hpp

#include "bases/bases.hpp"
#include "methods/hho"
#include "../common/assembly_index.hpp"
#include "../common/elastic_material_data.hpp"

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
class elastic_two_fields_assembler
{
    
    
    typedef disk::BoundaryConditions<Mesh, false>    boundary_type;
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
    std::vector< Triplet<T> >           m_mass_triplets; // Candidate for deletion
    std::vector< elastic_material_data<T> > m_material;
    std::vector< size_t >               m_elements_with_bc_eges;
    std::vector<std::pair<size_t,size_t>> m_fracture_pairs;
    std::vector<std::pair<size_t,size_t>> m_elements_with_fractures_eges;
    std::vector<std::pair<size_t,size_t>> m_end_point_mortars;

    size_t      m_n_edges;
    size_t      m_n_essential_edges;
    size_t      m_n_cells_dof;
    size_t      m_n_faces_dof;
    size_t      m_n_hybrid_dof;
    size_t      m_sigma_degree;
    bool        m_hho_stabilization_Q;
    bool        m_scaled_stabilization_Q;
    
public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;
    SparseMatrix<T> m_skin_operator;

    elastic_two_fields_assembler(const Mesh& msh, const disk::hho_degree_info& hho_di, const boundary_type& bnd, const std::vector<std::pair<size_t,size_t>> & fracture_pairs, std::vector<std::pair<size_t,size_t>> & end_point_mortars, SparseMatrix<T> & skin_operator)
        : m_hho_di(hho_di), m_bnd(bnd), m_fracture_pairs(fracture_pairs), m_end_point_mortars(end_point_mortars), m_skin_operator(skin_operator), m_hho_stabilization_Q(true), m_scaled_stabilization_Q(false)
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
        m_n_hybrid_dof = 0;
        
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        m_sigma_degree = m_hho_di.face_degree()-1;
        size_t n_f_sigma_n_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_f_sigma_t_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);

        for (size_t face_id = 0; face_id < msh.faces_size(); face_id++)
        {
            m_compress_indexes.at(face_id) = m_n_faces_dof;
//                m_expand_indexes.at(compressed_offset) = face_id;
            auto non_essential_dofs = n_fbs - m_bnd.dirichlet_imposed_dofs(face_id, m_hho_di.face_degree());
            
            m_n_faces_dof += non_essential_dofs;
        }
        
        m_n_hybrid_dof = (n_f_sigma_n_bs + n_f_sigma_t_bs) * m_fracture_pairs.size() + 2.0*2.0*m_end_point_mortars.size();
        
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        m_n_cells_dof = n_cbs * msh.cells_size();
            
        size_t system_size = m_n_cells_dof + m_n_faces_dof + m_n_hybrid_dof;

        // skin data
        size_t skin_size = 2*m_skin_operator.rows();
        system_size += skin_size;
        
        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
            
        classify_cells(msh);
        classify_fracture_cells(msh);
    }

    void scatter_data(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs,
             const Matrix<T, Dynamic, 1>& rhs)
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
//                    RHS(face_LHS_offset+face_i*n_fbs+i) += bc_rhs(i);
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
    
    void scatter_mortar_data(const Mesh& msh, const size_t & face_id, const size_t & fracture_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_f_sigma_bs = 2.0*disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + fracture_ind*n_f_sigma_bs;
        
        for (size_t i = 0; i < n_f_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(frac_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j],asm_map_i[i], mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_mortar_skin_data(const Mesh& msh, const size_t & face_id, const size_t & fracture_ind, const size_t & skin_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_hho_di.cell_degree(), Mesh::dimension - 1);
        size_t n_skin_bs = m_skin_operator.rows();
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_n_hybrid_dof + skin_ind * n_skin_bs + fracture_ind*n_f_sigma_bs;
        
        for (size_t i = 0; i < n_f_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j],asm_map_i[i], mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_point_mortar_data(const Mesh& msh, const size_t & face_id, const size_t & point_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_f_sigma_bs = 2.0;
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_fracture_pairs.size()*n_f_sigma_bs;
        auto point_LHS_offset = frac_LHS_offset + n_f_sigma_bs*point_ind;
        
        for (size_t i = 0; i < n_f_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(point_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j],asm_map_i[i], mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_mortar_mass_data(const Mesh& msh, const size_t & fracture_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_f_sigma_bs = 2.0*disk::scalar_basis_size(m_sigma_degree, Mesh::dimension-1);
        
        std::vector<assembly_index> asm_map;
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + fracture_ind*n_f_sigma_bs;
        
        for (size_t i = 0; i < n_f_sigma_bs; i++)
        asm_map.push_back( assembly_index(frac_LHS_offset+i, true));
        
        assert( asm_map.size() == mortar_mat.rows() && asm_map.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j],mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_data(const Mesh& msh, const size_t & skin_ind)
    {
        size_t n_skin_dof = m_skin_operator.rows();
        
        std::vector<assembly_index> asm_map;
        auto LHS_offset = m_n_cells_dof + m_n_faces_dof + m_n_hybrid_dof;
        
        for (size_t i = 0; i < n_skin_dof; i++)
        asm_map.push_back( assembly_index(LHS_offset + skin_ind * n_skin_dof +i, true));
        
        assert( asm_map.size() == m_skin_operator.rows() && asm_map.size() == m_skin_operator.cols() );

        for (size_t i = 0; i < m_skin_operator.rows(); i++)
        {
            for (size_t j = 0; j < m_skin_operator.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j],m_skin_operator.coeff(i,j)) );
            }
        }
    
    }

    void assemble(const Mesh& msh, std::function<static_vector<double, 2>(const typename Mesh::point_type& )> rhs_fun){
        
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
        // mortars assemble
        assemble_mortars(msh);
        
        scatter_skin_data(msh, 0);
        scatter_skin_data(msh, 1);
        
        assemble_mortars_skin(msh);
        
        finalize();

    }

    void assemble_mortars(const Mesh& msh){
        auto storage = msh.backend_storage();
        
        size_t fracture_ind = 0;
        for (auto chunk : m_fracture_pairs) {
            
            size_t cell_ind_l = m_elements_with_fractures_eges[fracture_ind].first;
            size_t cell_ind_r = m_elements_with_fractures_eges[fracture_ind].second;
            auto& face_l = storage->edges[chunk.first];
            auto& face_r = storage->edges[chunk.second];
            auto& cell_l = storage->surfaces[cell_ind_l];
            auto& cell_r = storage->surfaces[cell_ind_r];
            
            Matrix<T, Dynamic, Dynamic> mortar_l = -1.0*mortar_coupling_matrix(msh,cell_l,face_l);
            Matrix<T, Dynamic, Dynamic> mortar_r = -1.0*mortar_coupling_matrix(msh,cell_r,face_r);
            
            scatter_mortar_data(msh,chunk.first,fracture_ind,mortar_l);
            scatter_mortar_data(msh,chunk.second,fracture_ind,mortar_r);
            
            Matrix<T, Dynamic, Dynamic> mass_matrix = sigma_mass_matrix(msh, face_l, face_r);
            scatter_mortar_mass_data(msh,fracture_ind,mass_matrix);
            
            
            fracture_ind++;
        }
        
        size_t point_mortar_ind = 0;
        for (auto p_chunk : m_end_point_mortars) {
            
            auto chunk = m_fracture_pairs[p_chunk.first];
            auto& node = storage->nodes[p_chunk.second];
            
            size_t cell_ind_l = m_elements_with_fractures_eges[p_chunk.first].first;
            size_t cell_ind_r = m_elements_with_fractures_eges[p_chunk.first].second;
            auto& face_l = storage->edges[chunk.first];
            auto& face_r = storage->edges[chunk.second];
            auto& cell_l = storage->surfaces[cell_ind_l];
            auto& cell_r = storage->surfaces[cell_ind_r];
            
            
            Matrix<T, Dynamic, Dynamic> mortar_l = -1.0*point_mortar_coupling_matrix(msh,cell_l,face_l,node);
            Matrix<T, Dynamic, Dynamic> mortar_r = -1.0*point_mortar_coupling_matrix(msh,cell_r,face_r,node);
            
            scatter_point_mortar_data(msh,chunk.first,point_mortar_ind,mortar_l);
            scatter_point_mortar_data(msh,chunk.second,point_mortar_ind,mortar_r);
            
            point_mortar_ind++;
        }
    }
    
    void assemble_mortars_skin(const Mesh& msh){
        auto storage = msh.backend_storage();
        
        size_t fracture_ind = 0;
        for (auto chunk : m_fracture_pairs) {
            
            size_t cell_ind_l = m_elements_with_fractures_eges[fracture_ind].first;
            size_t cell_ind_r = m_elements_with_fractures_eges[fracture_ind].second;
            auto& face_l = storage->edges[chunk.first];
            auto& face_r = storage->edges[chunk.second];
            auto& cell_l = storage->surfaces[cell_ind_l];
            auto& cell_r = storage->surfaces[cell_ind_r];
            
            Matrix<T, Dynamic, Dynamic> mortar_l = +1.0*mortar_coupling_matrix_skin(msh,cell_l,face_l);
            Matrix<T, Dynamic, Dynamic> mortar_r = +1.0*mortar_coupling_matrix_skin(msh,cell_r,face_r);
            
            scatter_mortar_skin_data(msh,chunk.first,fracture_ind,0,mortar_l);
            scatter_mortar_skin_data(msh,chunk.second,fracture_ind,1,mortar_r);
            
            fracture_ind++;
        }
        
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
        
        auto reconstruction_operator   = strain_tensor_reconstruction(msh, cell);
        Matrix<T, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        auto n_rows = R_operator.rows();
        auto n_cols = R_operator.cols();
        
        Matrix<T, Dynamic, Dynamic> S_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        if(m_hho_stabilization_Q)
        {
            auto rec_for_stab   = make_vector_hho_symmetric_laplacian(msh, cell, m_hho_di);
            auto stabilization_operator    = make_vector_hho_stabilization(msh, cell, rec_for_stab.first, m_hho_di, m_scaled_stabilization_Q);
            auto n_s_rows = stabilization_operator.rows();
            auto n_s_cols = stabilization_operator.cols();
            S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        }else{
            auto stabilization_operator    = make_vector_hdg_stabilization(msh, cell, m_hho_di, m_scaled_stabilization_Q);
            auto n_s_rows = stabilization_operator.rows();
            auto n_s_cols = stabilization_operator.cols();
            S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        }
        
        Matrix<T, Dynamic, Dynamic> mass = mass_operator(cell_ind, msh, cell, false);
        Matrix<T, Dynamic, Dynamic> M_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        M_operator.block(0,0,mass.rows(),mass.cols()) = mass;

        return M_operator + R_operator + (rho*mu)*S_operator;
    }
            
    Matrix<T, Dynamic, Dynamic>
    symmetric_tensor_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cell)
    {
            
        size_t dim = Mesh::dimension;
        auto gradeg = m_hho_di.grad_degree();
        auto ten_b = make_sym_matrix_monomial_basis(msh, cell, gradeg);
        auto ten_bs = disk::sym_matrix_basis_size(gradeg, dim, dim);
        Matrix<T, Dynamic, Dynamic> mass_matrix = Matrix<T, Dynamic, Dynamic>::Zero(ten_bs, ten_bs);
        
        auto qps = integrate(msh, cell, 2 * gradeg);

        // number of tensor components
        size_t dec = 0;
         if (dim == 3)
             dec = 6;
         else if (dim == 2)
             dec = 3;
         else
             std::logic_error("Expected 3 >= dim > 1");

         for (auto& qp : qps)
         {
             auto phi = ten_b.eval_functions(qp.point());

             for (size_t j = 0; j < ten_bs; j++)
             {
                 
                auto qp_phi_j = disk::priv::inner_product(qp.weight(), phi[j]);
                for (size_t i = j; i < ten_bs; i += dec){
                         mass_matrix(i, j) += disk::priv::inner_product(phi[i], qp_phi_j);
                }
             }
         }

        for (size_t j = 0; j < ten_bs; j++){
            for (size_t i = 0; i < j; i++){
                 mass_matrix(i, j) = mass_matrix(j, i);
            }
        }
        
        return mass_matrix;
    }
    
    Matrix<T, Dynamic, Dynamic>
    symmetric_tensor_trace_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cell)
    {
            
        size_t dim = Mesh::dimension;
        auto gradeg = m_hho_di.grad_degree();
        auto ten_b = make_sym_matrix_monomial_basis(msh, cell, gradeg);
        auto ten_bs = disk::sym_matrix_basis_size(gradeg, dim, dim);
        Matrix<T, Dynamic, Dynamic> mass_matrix = Matrix<T, Dynamic, Dynamic>::Zero(ten_bs, ten_bs);
        
        auto qps = integrate(msh, cell, 2 * gradeg);

        // number of tensor components
        size_t dec = 0;
         if (dim == 3)
             dec = 6;
         else if (dim == 2)
             dec = 3;
         else
             std::logic_error("Expected 3 >= dim > 1");

         for (auto& qp : qps)
         {
             auto phi = ten_b.eval_functions(qp.point());

             for (size_t j = 0; j < ten_bs; j++)
             {
                auto identity = phi[j];
                identity.setZero();
                for(size_t d = 0; d < dim; d++){
                    identity(d,d) = 1.0;
                }
                auto trace = phi[j].trace();
                auto trace_phi_j = disk::priv::inner_product(phi[j].trace(), identity);
                auto qp_phi_j = disk::priv::inner_product(qp.weight(), trace_phi_j);
                for (size_t i = 0; i < ten_bs; i ++){
                         mass_matrix(i, j) += disk::priv::inner_product(phi[i], qp_phi_j);
                }
             }
         }
        
        return mass_matrix;
    }
            
    Matrix<T, Dynamic, Dynamic> mass_operator(size_t & cell_ind, const Mesh& msh, const typename Mesh::cell_type& cell, bool add_vector_mass_Q = true){
            
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
            
        elastic_material_data<T> & material = m_material[cell_ind];
        T rho = material.rho();
        T mu = material.mu();
        T lambda = material.l();
    
        Matrix<T, Dynamic, Dynamic> mass_matrix = Matrix<T, Dynamic, Dynamic>::Zero(n_cbs, n_cbs);
        
        // Symmetric stress tensor mass block
        
        // Stress tensor
        Matrix<T, Dynamic, Dynamic> mass_matrix_sigma  = symmetric_tensor_mass_matrix(msh, cell);
        
        // Tensor trace
        Matrix<T, Dynamic, Dynamic> mass_matrix_trace_sigma  = symmetric_tensor_trace_mass_matrix(msh, cell);
        
        // Constitutive relationship inverse
        mass_matrix_trace_sigma *= (lambda/(2.0*mu+2.0*lambda));
        mass_matrix_sigma -= mass_matrix_trace_sigma;
        mass_matrix_sigma *= (1.0/(2.0*mu));
        mass_matrix.block(0, 0, n_ten_cbs, n_ten_cbs) = mass_matrix_sigma;
        
        if (add_vector_mass_Q) {
            // vector velocity mass mass block
            auto vec_basis = disk::make_vector_monomial_basis(msh, cell, m_hho_di.cell_degree());
            Matrix<T, Dynamic, Dynamic> mass_matrix_v = disk::make_mass_matrix(msh, cell, vec_basis);
            mass_matrix_v *= rho;
            mass_matrix.block(n_ten_cbs, n_ten_cbs, n_vec_cbs, n_vec_cbs) = mass_matrix_v;
        }

        return mass_matrix;
    }
    
    Matrix<T, Dynamic, Dynamic> mortar_coupling_matrix(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, m_hho_di.face_degree());
        auto sn_basis = disk::make_scalar_monomial_basis(msh, face, m_sigma_degree);
        auto st_basis = disk::make_scalar_monomial_basis(msh, face, m_sigma_degree);
        
        size_t n_s_basis = sn_basis.size() + st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());

        const auto qps = integrate(msh, face, 2 * (degree+di));
        const auto n = disk::normal(msh, cell, face);
        const auto t = disk::tanget(msh, cell, face);
        for (auto& qp : qps)
        {
            const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
            const auto sn_f_phi = sn_basis.eval_functions(qp.point());
            const auto st_f_phi = st_basis.eval_functions(qp.point());
            
            const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
            const auto s_n_opt = disk::priv::outer_product(sn_f_phi, w_n_dot_u_f_phi);
            
            const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_t_dot_u_f_phi);

            ret.block(0,0,sn_basis.size(),vec_u_basis.size()) += s_n_opt;
            ret.block(sn_basis.size(),0,st_basis.size(),vec_u_basis.size()) += s_t_opt;
        }

        return ret;
    }
    
    Matrix<T, Dynamic, Dynamic> mortar_coupling_matrix_skin(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, m_hho_di.face_degree());
        auto st_basis = disk::make_scalar_monomial_basis(msh, face, m_hho_di.cell_degree());
        
        size_t n_s_basis = st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());

        const auto qps = integrate(msh, face, 2 * (degree+di));
        const auto n = disk::normal(msh, cell, face);
        const auto t = disk::tanget(msh, cell, face);
        for (auto& qp : qps)
        {
            const auto u_f_phi = vec_u_basis.eval_functions(qp.point());

            const auto st_f_phi = st_basis.eval_functions(qp.point());
                        
            const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_t_dot_u_f_phi);

            ret.block(0,0,n_s_basis,vec_u_basis.size()) += s_t_opt;
        }

        return ret;
    }
    
    Matrix<T, Dynamic, Dynamic> sigma_mass_matrix(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, size_t di = 0)
    {
        const auto degree     = m_sigma_degree;
        auto sn_basis = disk::make_scalar_monomial_basis(msh, face_l, m_sigma_degree);
        auto st_basis = disk::make_scalar_monomial_basis(msh, face_r, m_sigma_degree);
        
        size_t n_s_basis = sn_basis.size() + st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, n_s_basis);

        T c_perp = 1000.0;
        const auto qps_l = integrate(msh, face_l, 2 * (degree+di));
        for (auto& qp : qps_l)
        {
            const auto sn_f_phi = sn_basis.eval_functions(qp.point());
            const auto w_sn_f_phi = disk::priv::inner_product(qp.weight(), sn_f_phi);
            const auto s_n_opt = disk::priv::outer_product(sn_f_phi, w_sn_f_phi);
            ret.block(0,0,sn_basis.size(),sn_basis.size()) += c_perp * s_n_opt;
        }
        
        T c_para =  1000.0;
        const auto qps_r = integrate(msh, face_r, 2 * (degree+di));
        for (auto& qp : qps_r)
        {
            const auto st_f_phi = st_basis.eval_functions(qp.point());
            const auto w_st_f_phi = disk::priv::inner_product(qp.weight(), st_f_phi);
            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_st_f_phi);
            ret.block(sn_basis.size(),sn_basis.size(),st_basis.size(),st_basis.size()) += c_para * s_t_opt;
        }

        return ret;
    }
    
    Matrix<T, Dynamic, Dynamic> point_mortar_coupling_matrix(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, const typename Mesh::node_type& node, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, m_hho_di.face_degree());
        
        size_t sigma_degree = 0;
        auto sn_basis = disk::make_scalar_monomial_basis(msh, face, sigma_degree);
        auto st_basis = disk::make_scalar_monomial_basis(msh, face, sigma_degree);
        
        size_t n_s_basis = sn_basis.size() + st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());

        const auto n = disk::normal(msh, cell, face);
        const auto t = disk::tanget(msh, cell, face);
        typename Mesh::point_type point = barycenter(msh, node);
        {
            const auto u_f_phi = vec_u_basis.eval_functions(point);
            const auto sn_f_phi = sn_basis.eval_functions(point);
            const auto st_f_phi = st_basis.eval_functions(point);
            
            const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(1.0, n));
            const auto s_n_opt = disk::priv::outer_product(sn_f_phi, w_n_dot_u_f_phi);
            
            const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(1.0, t));
            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_t_dot_u_f_phi);

            ret.block(0,0,sn_basis.size(),vec_u_basis.size()) += s_n_opt;
            ret.block(sn_basis.size(),0,st_basis.size(),vec_u_basis.size()) += s_t_opt;
        }
        return ret;
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
    
    void classify_fracture_cells(const Mesh& msh){

        m_elements_with_fractures_eges.clear();
        size_t cell_l, cell_r;
        for (auto chunk : m_fracture_pairs) {
            size_t cell_ind = 0;
            for (auto& cell : msh)
            {
                auto face_list = faces(msh, cell);
                for (size_t face_i = 0; face_i < face_list.size(); face_i++)
                {
                    
                    auto fc = face_list[face_i];
                    auto fc_id = msh.lookup(fc);
                    
                    bool is_left_fracture_Q = fc_id == chunk.first;
                    if (is_left_fracture_Q)
                    {
                        cell_l = cell_ind;
                        break;
                    }
                    
                    bool is_right_fracture_Q = fc_id == chunk.second;
                    if (is_right_fracture_Q)
                    {
                        cell_r = cell_ind;
                        break;
                    }
                    
                }
                cell_ind++;
            }
            m_elements_with_fractures_eges.push_back(std::make_pair(cell_l,cell_r));
        }
    }
            
    void project_over_cells(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<static_vector<double, 2>(const typename Mesh::point_type& )> vec_fun, std::function<static_matrix<double, 2,2>(const typename Mesh::point_type& )> ten_fun){
        size_t n_dof = LHS.rows();
        x_glob = Matrix<T, Dynamic, 1>::Zero(n_dof);
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
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
                      std::function<static_matrix<T, 2,2>(const typename Mesh::point_type& )> ten_fun){
    
        Matrix<T, Dynamic, Dynamic> mass_matrix  = symmetric_tensor_mass_matrix(msh, cell);
        size_t dim = Mesh::dimension;
        auto gradeg = m_hho_di.grad_degree();
        auto ten_bs = disk::sym_matrix_basis_size(gradeg, dim, dim);
        auto ten_b = make_sym_matrix_monomial_basis(msh, cell, gradeg);
        Matrix<T, Dynamic, 1> rhs = Matrix<T, Dynamic, 1>::Zero(ten_bs);

        const auto qps = integrate(msh, cell, 2 * gradeg);
        for (auto& qp : qps)
        {
            auto phi = ten_b.eval_functions(qp.point());
            static_matrix<T, 2,2> sigma = ten_fun(qp.point());
            for (size_t i = 0; i < ten_bs; i++){
                auto qp_phi_i = disk::priv::inner_product(qp.weight(), phi[i]);
                rhs(i,0) += disk::priv::inner_product(qp_phi_i,sigma);
            }
        }
        Matrix<T, Dynamic, 1> x_dof = mass_matrix.llt().solve(rhs);
        return x_dof;
    }
            
    void project_over_faces(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<static_vector<double, 2>(const typename Mesh::point_type& )> vec_fun){

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
                Matrix<T, Dynamic, 1> x_proj_dof = project_function(msh, face, m_hho_di.face_degree(), vec_fun);
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
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        x_glob.block(cell_ofs * n_cbs, 0, n_cbs, 1) = x_proj_dof;
    }
    
    void scatter_face_dof_data(  const Mesh& msh, const typename Mesh::face_type& face,
                    Matrix<T, Dynamic, 1>& x_glob, Matrix<T, Dynamic, 1> & x_proj_dof) const
    {
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
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

        auto ten_b = make_sym_matrix_monomial_basis(msh, cell, graddeg);
        auto vec_b = make_vector_monomial_basis(msh, cell, celdeg);

        auto ten_bs = disk::sym_matrix_basis_size(graddeg, Mesh::dimension, Mesh::dimension);
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
                 
                auto qp_gphi_j = disk::priv::inner_product(qp.weight(), gphi[j]);
                for (size_t i = j; i < ten_bs; i += dec){
                         gr_lhs(i, j) += disk::priv::inner_product(gphi[i], qp_gphi_j);
                }
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

                 gr_rhs.block(0, 0, ten_bs, vec_bs) += disk::priv::outer_product(gphi, qp_dphi);

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

                 const auto qp_gphi_n = disk::priv::inner_product(gphi, disk::priv::inner_product(qp.weight(), n));
                 gr_rhs.block(0, vec_bs + i * fbs, ten_bs, fbs) += disk::priv::outer_product(qp_gphi_n, fphi);
                 gr_rhs.block(0, 0, ten_bs, vec_bs) -= disk::priv::outer_product(qp_gphi_n, cphi);
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
    mixed_rhs(const Mesh& msh, const typename Mesh::cell_type& cell, std::function<static_vector<double, 2>(const typename Mesh::point_type& )> & rhs_fun, size_t di = 0)
    {
        auto recdeg = m_hho_di.grad_degree();
        auto celdeg = m_hho_di.cell_degree();
        auto facdeg = m_hho_di.face_degree();

        auto ten_bs = disk::sym_matrix_basis_size(recdeg, Mesh::dimension, Mesh::dimension);
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
        const auto qps = integrate(msh, face, 2 * (facdeg + di));

        for (auto& qp : qps)
        {
            const auto phi  = face_basis.eval_functions(qp.point());
            const auto qp_f = disk::priv::inner_product(qp.weight(), neumann_fun(qp.point()));
            ret += disk::priv::outer_product(phi, qp_f);
        }
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
    size_t get_n_hybrid_dofs(){
        return m_n_hybrid_dof;
    }

    std::vector<size_t> & compress_indexes(){
        return m_compress_indexes;
    }
    
};

#endif /* elastic_two_fields_assembler_hpp */
