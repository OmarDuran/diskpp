//
//  elastoacoustic_two_fields_assembler.hpp
//  acoustics
//
//  Created by Omar Durán on 6/3/20.
//

#pragma once
#ifndef elastoacoustic_two_fields_assembler_hpp
#define elastoacoustic_two_fields_assembler_hpp

#include "bases/bases.hpp"
#include "methods/hho"
#include "../common/assembly_index.hpp"
#include "../common/acoustic_material_data.hpp"
#include "../common/elastic_material_data.hpp"
#include <map>

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
class elastoacoustic_two_fields_assembler
{
    
    typedef disk::BoundaryConditions<Mesh, false>    e_boundary_type;
    typedef disk::BoundaryConditions<Mesh, true>     a_boundary_type;
    using T = typename Mesh::coordinate_type;

    std::vector<size_t>                 m_e_compress_indexes;
    std::vector<size_t>                 m_e_expand_indexes;
    
    std::vector<size_t>                 m_a_compress_indexes;
    std::vector<size_t>                 m_a_expand_indexes;

    disk::hho_degree_info               m_hho_di;
    e_boundary_type                     m_e_bnd;
    a_boundary_type                     m_a_bnd;
    std::vector< Triplet<T> >           m_triplets;
    std::vector< Triplet<T> >           m_mass_triplets;
    std::map<size_t,elastic_material_data<T>> m_e_material;
    std::map<size_t,acoustic_material_data<T>> m_a_material;
    std::vector< size_t >               m_elements_with_bc_eges;
    std::set<size_t>                    m_interface_face_indexes;

    size_t      m_n_edges;
    size_t      m_n_essential_edges;
    bool        m_hho_stabilization_Q;
    size_t      m_n_elastic_cell_dof;
    size_t      m_n_acoustic_cell_dof;
    size_t      m_n_elastic_face_dof;
    size_t      m_n_acoustic_face_dof;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;
    SparseMatrix<T>         MASS;

    elastoacoustic_two_fields_assembler(const Mesh& msh, const disk::hho_degree_info& hho_di, const e_boundary_type& e_bnd, const a_boundary_type& a_bnd, std::map<size_t,elastic_material_data<T>> e_material, std::map<size_t,acoustic_material_data<T>> a_material)
        : m_hho_di(hho_di), m_e_bnd(e_bnd), m_a_bnd(a_bnd), m_e_material(e_material), m_a_material(a_material), m_hho_stabilization_Q(true)
    {
            
        auto storage = msh.backend_storage();
        auto is_e_dirichlet = [&](const typename Mesh::face& fc) -> bool {

            auto fc_id = msh.lookup(fc);
            return e_bnd.is_dirichlet_face(fc_id);
        };
        
        auto is_a_dirichlet = [&](const typename Mesh::face& fc) -> bool {

            auto fc_id = msh.lookup(fc);
            return a_bnd.is_dirichlet_face(fc_id);
        };

        size_t n_e_essential_edges = std::count_if(msh.faces_begin(), msh.faces_end(), is_e_dirichlet);
        size_t n_a_essential_edges = std::count_if(msh.faces_begin(), msh.faces_end(), is_a_dirichlet);
        
        
        std::set<size_t> e_egdes;
        for (auto &chunk : m_e_material) {
            size_t cell_i = chunk.first;
            auto& cell = storage->surfaces[cell_i];
            auto cell_faces = faces(msh,cell);
            for (auto &face : cell_faces) {
                if (!is_e_dirichlet(face)) {
                    auto fc_id = msh.lookup(face);
                    e_egdes.insert(fc_id);
                }
            }
        }
        size_t n_e_edges = e_egdes.size();

        std::set<size_t> a_egdes;
        for (auto &chunk : m_a_material) {
            size_t cell_i = chunk.first;
            auto& cell = storage->surfaces[cell_i];
            auto cell_faces = faces(msh,cell);
            for (auto &face : cell_faces) {
                if (!is_a_dirichlet(face)) {
                    auto fc_id = msh.lookup(face);
                    a_egdes.insert(fc_id);
                }
            }
        }
        size_t n_a_edges = a_egdes.size();
        
        m_n_edges = msh.faces_size();
        m_n_essential_edges = n_e_essential_edges + n_a_essential_edges;

        m_e_compress_indexes.resize( m_n_edges );
        m_e_expand_indexes.resize( m_n_edges - m_n_essential_edges );
        
        m_a_compress_indexes.resize( m_n_edges );
        m_a_expand_indexes.resize( m_n_edges - m_n_essential_edges );

        
        
        size_t e_compressed_offset = 0;
        for (auto face_id : e_egdes)
        {
            m_e_compress_indexes.at(face_id) = e_compressed_offset;
            m_e_expand_indexes.at(e_compressed_offset) = face_id;
            e_compressed_offset++;
        }
        
        size_t a_compressed_offset = 0;
        for (auto face_id : a_egdes)
        {
            m_a_compress_indexes.at(face_id) = a_compressed_offset;
            m_a_expand_indexes.at(a_compressed_offset) = face_id;
            a_compressed_offset++;
        }

        size_t n_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        size_t n_s_cbs = disk::scalar_basis_size(m_hho_di.cell_degree(), Mesh::dimension);
        size_t n_s_fbs = disk::scalar_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1);

        m_n_elastic_cell_dof = (n_cbs * m_e_material.size());
        m_n_acoustic_cell_dof = (n_s_cbs * m_a_material.size());
        
        m_n_elastic_face_dof = (n_fbs * n_e_edges);
        m_n_acoustic_face_dof = (n_s_fbs * n_a_edges);
        size_t system_size = m_n_elastic_cell_dof + m_n_acoustic_cell_dof + m_n_elastic_face_dof + m_n_acoustic_face_dof;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
        MASS = SparseMatrix<T>( system_size, system_size );
            
//        classify_cells(msh);
    }
    
    void scatter_e_data(size_t e_cell_ind, const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs,
             const Matrix<T, Dynamic, 1>& rhs)
    {
        auto fcs = faces(msh, cl);
        size_t n_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs + n_fbs*fcs.size());

        auto cell_LHS_offset    = e_cell_ind * n_cbs;

        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );
            
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            auto fc = fcs[face_i];
            auto fc_id = msh.lookup(fc);
            auto face_LHS_offset = m_n_elastic_cell_dof + m_n_acoustic_cell_dof + m_e_compress_indexes.at(fc_id)*n_fbs;
            bool dirichlet = m_e_bnd.is_dirichlet_face(fc_id);

            for (size_t i = 0; i < n_fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
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
    
    void scatter_a_data(size_t a_cell_ind, const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, Dynamic>& lhs,
    const Matrix<T, Dynamic, 1>& rhs)
    {
        auto fcs = faces(msh, cl);
        size_t n_cbs = disk::scalar_basis_size(m_hho_di.cell_degree(), Mesh::dimension);
        size_t n_fbs = disk::scalar_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1);
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs + n_fbs*fcs.size());

        auto cell_LHS_offset    = a_cell_ind * n_cbs + m_n_elastic_cell_dof;

        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );
        
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            auto fc = fcs[face_i];
            auto fc_id = msh.lookup(fc);
            auto face_LHS_offset = m_n_elastic_cell_dof + m_n_acoustic_cell_dof + m_n_elastic_face_dof + m_a_compress_indexes.at(fc_id)*n_fbs;

            bool dirichlet = m_a_bnd.is_dirichlet_face(fc_id);

            for (size_t i = 0; i < n_fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
            
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
    
    void scatter_e_mass_data(size_t e_cell_ind, const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& mass_matrix)
    {
        size_t n_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs);
        
        auto cell_LHS_offset    = e_cell_ind * n_cbs;

        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        assert( asm_map.size() == mass_matrix.rows() && asm_map.size() == mass_matrix.cols() );

        for (size_t i = 0; i < mass_matrix.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < mass_matrix.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    m_mass_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], mass_matrix(i,j)) );
            }
        }

    }
    
    void scatter_a_mass_data(size_t a_cell_ind, const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& mass_matrix)
    {
        size_t n_cbs = disk::scalar_basis_size(m_hho_di.cell_degree(), Mesh::dimension);
        std::vector<assembly_index> asm_map;
        asm_map.reserve(n_cbs);

        auto cell_LHS_offset    = a_cell_ind * n_cbs + m_n_elastic_cell_dof;
        
        for (size_t i = 0; i < n_cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        assert( asm_map.size() == mass_matrix.rows() && asm_map.size() == mass_matrix.cols() );

        for (size_t i = 0; i < mass_matrix.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < mass_matrix.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    m_mass_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], mass_matrix(i,j)) );
            }
        }

    }
    
    void assemble(const Mesh& msh, std::function<static_vector<T, 2>(const typename Mesh::point_type& )> e_rhs_fun, std::function<T(const typename Mesh::point_type& )> a_rhs_fun){
        
        auto storage = msh.backend_storage();
        LHS.setZero();
        RHS.setZero();
        
        // elastic block
        size_t e_cell_ind = 0;
        for (auto e_chunk : m_e_material) {
            auto& cell = storage->surfaces[e_chunk.first];
            Matrix<T, Dynamic, Dynamic> vectorial_laplacian_operator_loc = e_laplacian_operator(e_chunk.second,msh,cell);
            auto cell_basis   = make_vector_monomial_basis(msh, cell, m_hho_di.cell_degree());
            Matrix<T, Dynamic, 1> f_loc = make_rhs(msh, cell, cell_basis, e_rhs_fun);
            scatter_e_data(e_cell_ind, msh, cell, vectorial_laplacian_operator_loc, f_loc);
            e_cell_ind++;
        }
        
        // acoustic block
        size_t a_cell_ind = 0;
        for (auto a_chunk : m_a_material) {
            auto& cell = storage->surfaces[a_chunk.first];
            
            Matrix<T, Dynamic, Dynamic> laplacian_operator_loc = a_laplacian_operator(a_chunk.second, msh, cell);
            auto cell_basis   = make_scalar_monomial_basis(msh, cell, m_hho_di.cell_degree());
            Matrix<T, Dynamic, 1> f_loc = make_rhs(msh, cell, cell_basis, a_rhs_fun);
            scatter_a_data(a_cell_ind, msh, cell, laplacian_operator_loc, f_loc);
            a_cell_ind++;
        }

        finalize();
    }

    void assemble_mass(const Mesh& msh){
        
        auto storage = msh.backend_storage();
        MASS.setZero();
        
        // elastic block
        size_t e_cell_ind = 0;
        for (auto e_chunk : m_e_material) {
            auto& cell = storage->surfaces[e_chunk.first];
            Matrix<T, Dynamic, Dynamic> mass_matrix = e_mass_operator(e_chunk.second,msh, cell);
            scatter_e_mass_data(e_cell_ind,msh, cell, mass_matrix);
            e_cell_ind++;
        }
        
        // acoustic block
        size_t  a_cell_ind = 0;
        for (auto a_chunk : m_a_material) {
            auto& cell = storage->surfaces[a_chunk.first];
            Matrix<T, Dynamic, Dynamic> mass_matrix = a_mass_operator(a_chunk.second,msh, cell);
            scatter_a_mass_data(a_cell_ind,msh, cell, mass_matrix);
            a_cell_ind++;
        }
        
        finalize_mass();
    }
    
    Matrix<T, Dynamic, Dynamic> e_mass_operator(elastic_material_data<T> & material, const Mesh& msh, const typename Mesh::cell_type& cell){
        auto vec_basis = disk::make_vector_monomial_basis(msh, cell, m_hho_di.cell_degree());
        Matrix<T, Dynamic, Dynamic> mass_matrix = disk::make_mass_matrix(msh, cell, vec_basis);
        mass_matrix *= (material.rho());
        return mass_matrix;
    }
    
    Matrix<T, Dynamic, Dynamic> a_mass_operator(acoustic_material_data<T> & material, const Mesh& msh, const typename Mesh::cell_type& cell){

        auto scal_basis = disk::make_scalar_monomial_basis(msh, cell, m_hho_di.cell_degree());
        Matrix<T, Dynamic, Dynamic> mass_matrix = disk::make_mass_matrix(msh, cell, scal_basis);
        mass_matrix *= (1.0/(material.rho()*material.vp()*material.vp()));
        return mass_matrix;
    }
    
    
    Matrix<T, Dynamic, Dynamic> e_laplacian_operator(elastic_material_data<T> & material, const Mesh& msh, const typename Mesh::cell_type& cell){
           
        T mu = material.rho()*material.vs()*material.vs();
        T lambda = material.rho()*material.vp()*material.vp() - 2.0*mu;
        auto reconstruction_operator   = make_matrix_symmetric_gradrec(msh, cell, m_hho_di);
        auto rec_for_stab   = make_vector_hho_symmetric_laplacian(msh, cell, m_hho_di);
        auto divergence_operator = make_hho_divergence_reconstruction(msh, cell, m_hho_di);

        Matrix<T, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        Matrix<T, Dynamic, Dynamic> S_operator;
        if(m_hho_stabilization_Q)
        {
            auto stabilization_operator    = make_vector_hho_stabilization(msh, cell, rec_for_stab.first, m_hho_di);
            S_operator = stabilization_operator;
        }else{
            auto stabilization_operator    = make_vector_hdg_stabilization(msh, cell, m_hho_di);
            S_operator = stabilization_operator;
        }
        return 2.0*mu*(R_operator + S_operator)+lambda * divergence_operator.second;
    }
    
    Matrix<T, Dynamic, Dynamic> a_laplacian_operator(acoustic_material_data<T> & material, const Mesh& msh, const typename Mesh::cell_type& cell){

        auto reconstruction_operator   = make_scalar_hho_laplacian(msh, cell, m_hho_di);
        Matrix<T, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        
        Matrix<T, Dynamic, Dynamic> S_operator;
        if(m_hho_stabilization_Q)
        {
            auto stabilization_operator    = make_scalar_hho_stabilization(msh, cell, reconstruction_operator.first, m_hho_di);
            S_operator = stabilization_operator;
        }else{
            auto stabilization_operator    = make_scalar_hdg_stabilization(msh, cell, m_hho_di);
            S_operator = stabilization_operator;
        }
        return (1.0/material.rho())*(R_operator + S_operator);
    }
            
    void project_over_cells(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<static_vector<T, 2>(const typename Mesh::point_type& )> vec_fun, std::function<T(const typename Mesh::point_type& )> scal_fun){
        
        auto storage = msh.backend_storage();
        size_t n_dof = MASS.rows();
        x_glob = Matrix<T, Dynamic, 1>::Zero(n_dof);

        // elastic block
        size_t e_cell_ind = 0;
        for (auto e_chunk : m_e_material) {
            auto& cell = storage->surfaces[e_chunk.first];
            Matrix<T, Dynamic, 1> x_proj_dof = project_function(msh, cell, m_hho_di.cell_degree(), vec_fun);
            scatter_e_cell_dof_data(e_cell_ind, msh, cell, x_glob, x_proj_dof);
            e_cell_ind++;
        }
        
        // acoustic block
        size_t a_cell_ind = 0;
        for (auto a_chunk : m_a_material) {
            auto& cell = storage->surfaces[a_chunk.first];
            Matrix<T, Dynamic, 1> x_proj_dof = project_function(msh, cell, m_hho_di.cell_degree(), scal_fun);
            scatter_a_cell_dof_data(a_cell_ind, msh, cell, x_glob, x_proj_dof);
            a_cell_ind++;
        }
    
    }
    
    void
    scatter_e_cell_dof_data(size_t e_cell_ind, const Mesh& msh, const typename Mesh::cell_type& cell,
                    Matrix<T, Dynamic, 1>& x_glob, Matrix<T, Dynamic, 1> x_proj_dof) const
    {
        size_t n_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        auto cell_ofs = e_cell_ind * n_cbs;
        x_glob.block(cell_ofs, 0, n_cbs, 1) = x_proj_dof;
    }
    
    void
    scatter_a_cell_dof_data(size_t a_cell_ind, const Mesh& msh, const typename Mesh::cell_type& cell,
                    Matrix<T, Dynamic, 1>& x_glob, Matrix<T, Dynamic, 1> x_proj_dof) const
    {
        size_t n_cbs = disk::scalar_basis_size(m_hho_di.cell_degree(), Mesh::dimension);
        auto cell_ofs = a_cell_ind * n_cbs + m_n_elastic_cell_dof;
        x_glob.block(cell_ofs, 0, n_cbs, 1) = x_proj_dof;
    }
            
    void finalize(void)
    {
        LHS.setFromTriplets( m_triplets.begin(), m_triplets.end() );
        m_triplets.clear();
    }
            
    void finalize_mass(void)
    {
        MASS.setFromTriplets( m_mass_triplets.begin(), m_mass_triplets.end() );
        m_mass_triplets.clear();
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
    
    void set_interface_face_indexes(std::set<size_t> & interface_face_indexes){
        m_interface_face_indexes = interface_face_indexes;
    }
            
    void set_hho_stabilization(){
        m_hho_stabilization_Q = true;
    }
            
    e_boundary_type & get_e_bc_conditions(){
             return m_e_bnd;
    }
    
    a_boundary_type & get_a_bc_conditions(){
             return m_a_bnd;
    }
            
    size_t get_n_face_dof(){
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_face_dof = (m_n_edges - m_n_essential_edges) * n_fbs;
        return n_face_dof;
    }
    
    size_t get_cell_basis_data(){
        size_t n_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        return n_cbs;
    }
            
};

#endif /* elastoacoustic_two_fields_assembler_hpp */
