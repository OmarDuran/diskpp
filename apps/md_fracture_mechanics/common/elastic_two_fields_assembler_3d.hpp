//
//  elastic_two_fields_assembler_3d.hpp
//  elastodynamics
//
//  Created by Omar Durán on 22/03/21.
//

#pragma once
#ifndef elastic_two_fields_assembler_3d_hpp
#define elastic_two_fields_assembler_3d_hpp

#include "bases/bases.hpp"
#include "methods/hho"
#include "../common/assembly_index.hpp"
#include "../common/elastic_material_data.hpp"

#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename Mesh>
struct fracture_3d {
    
    using T = typename Mesh::coordinate_type;
    
    size_t m_bl_index;
    size_t m_el_index;
    
    size_t m_br_index;
    size_t m_er_index;
    
    size_t m_skin_bs;
    
    std::pair<size_t,size_t> m_bc_type = {0,0}; // (0 -> none, 1 -> D, 2 -> N)
    std::pair<std::vector<T>,std::vector<T>> m_bc_data = {{0,0},{0,0}}; // (left and right val)

    std::vector<std::pair<size_t,size_t>> m_pairs;
    
    std::vector<bool>  m_flips_l, m_flips_r;
    
    std::vector<std::pair<size_t,size_t>> m_elements;
    
    fracture_3d(){
        
    }
    
    fracture_3d(const fracture_3d &other){
        m_bl_index       = other.m_bl_index;
        m_el_index       = other.m_el_index;
        m_br_index       = other.m_br_index;
        m_er_index       = other.m_er_index;
        m_pairs         = other.m_pairs;
        m_flips_l       = other.m_flips_l;
        m_flips_r       = other.m_flips_r;
        m_elements      = other.m_elements;
        m_skin_bs       = other.m_skin_bs;
        m_bc_type       = other.m_bc_type;
        m_bc_data       = other.m_bc_data;
    }
         
    fracture_3d& operator = (const fracture_3d &other){
    
        m_bl_index       = other.m_bl_index;
        m_el_index       = other.m_el_index;
        m_br_index       = other.m_br_index;
        m_er_index       = other.m_er_index;
        m_pairs         = other.m_pairs;
        m_flips_l       = other.m_flips_l;
        m_flips_r       = other.m_flips_r;
        m_elements      = other.m_elements;
        m_skin_bs       = other.m_skin_bs;
        m_bc_type       = other.m_bc_type;
        m_bc_data       = other.m_bc_data;
        return *this;
    }
    
    void build(const Mesh& msh){
//        build_mesh(msh);
        build_elements(msh);
        m_skin_bs = 1*(4 * m_pairs.size() + 1);
    }
    
    void build_elements(const Mesh& msh){
        m_elements.clear();
        size_t cell_l, cell_r;
        for (auto chunk : m_pairs) {
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
            m_elements.push_back(std::make_pair(cell_l,cell_r));
        }
    }
    
    void build_mesh(const Mesh& msh){
       
        auto storage = msh.backend_storage();
                
        std::set<size_t> set_l, set_r;
        for (auto chunk : m_pairs) {
            set_l.insert(chunk.first);
            set_r.insert(chunk.second);
        }
        

        std::map<size_t,size_t> node_map_l;
        std::vector<size_t> frac_indexes_l;
        frac_indexes_l.reserve(set_l.size());
        {   // build connectivity map on left side
            size_t node_index_b = m_bl_index;
            size_t node_index_e = m_el_index;
            size_t node_index = node_index_b;
            
            size_t node_c = 1;
            node_map_l[node_index] = node_c;
            node_c++;
            while (node_index_e != node_index) {
                for (auto id : set_l) {
                    auto& face = storage->edges[id];
                    auto points = face.point_ids();
                    bool check_Q = points[0] == node_index || points[1] == node_index;
                    if (check_Q) {
                        frac_indexes_l.push_back(id);
                        set_l.erase(id);
                        if (points[0] == node_index) {
                            node_index = points[1];
                        }else{
                            node_index = points[0];
                        }
                        node_map_l[node_index] = node_c;
                        node_c++;
                        break;
                    }
                }
            }
        }
        
        std::map<size_t,size_t> node_map_r;
        std::vector<size_t> frac_indexes_r;
        frac_indexes_r.reserve(set_r.size());
        {   // build connectivity map on right side
            size_t node_index_b = m_br_index;
            size_t node_index_e = m_er_index;
            size_t node_index = node_index_b;
            
            size_t node_c = 1;
            node_map_r[node_index] = node_c;
            node_c++;
            while (node_index_e != node_index) {
                for (auto id : set_r) {
                    auto& face = storage->edges[id];
                    auto points = face.point_ids();
                    bool check_Q = points[0] == node_index || points[1] == node_index;
                    if (check_Q) {
                        frac_indexes_r.push_back(id);
                        set_r.erase(id);
                        if (points[0] == node_index) {
                            node_index = points[1];
                        }else{
                            node_index = points[0];
                        }
                        node_map_r[node_index] = node_c;
                        node_c++;
                        break;
                    }
                }
            }
        }
                
        // renumbering fracture pairs
        m_pairs.clear();
        assert(frac_indexes_l.size()==frac_indexes_r.size());
        for (size_t i = 0; i < frac_indexes_l.size(); i++) {
            m_pairs.push_back(std::make_pair(frac_indexes_l[i], frac_indexes_r[i]));
        }
        
        m_flips_l.reserve(m_pairs.size());
        m_flips_r.reserve(m_pairs.size());
        for (auto chunk : m_pairs) {
            
            auto& face_l = storage->edges[chunk.first];
            auto& face_r = storage->edges[chunk.second];
            
            auto points_l = face_l.point_ids();
            auto points_r = face_r.point_ids();
            
            if (node_map_l[points_l[0]] < node_map_l[points_l[1]]) {
                m_flips_l.push_back(false);
            }else{
                m_flips_l.push_back(true);
            }
            
            if (node_map_r[points_r[0]] < node_map_r[points_r[1]]) {
                m_flips_r.push_back(false);
            }else{
                m_flips_r.push_back(true);
            }
            
        }

    }
};

struct restriction_3d {
    
    std::pair<size_t,size_t> m_f_index;
    std::pair<size_t,size_t> m_p_index;
    std::pair<size_t,size_t> m_s_index;
    
    restriction_3d(){
        
    }
    
    restriction_3d(const restriction_3d &other){
        m_f_index       = other.m_f_index;
        m_p_index       = other.m_p_index;
        m_s_index       = other.m_s_index;
    }
         
    restriction_3d& operator = (const restriction_3d &other){
    
        m_f_index       = other.m_f_index;
        m_p_index       = other.m_p_index;
        m_s_index       = other.m_s_index;
        return *this;
    }
};

template<typename Mesh>
class elastic_two_fields_assembler_3d
{
    
    
    typedef disk::BoundaryConditions<Mesh, false>    boundary_type;
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;
    using node_type = typename Mesh::node_type;
    using edge_type = typename Mesh::edge_type;

    std::vector<size_t>                 m_compress_indexes;
    std::vector<size_t>                 m_expand_indexes;
    
    std::vector<size_t>                 m_dof_dest_l, m_dof_dest_r;
    std::vector<bool>                   m_flip_dest_l, m_flip_dest_r;

    disk::hho_degree_info               m_hho_di;
    boundary_type                       m_bnd;
    std::vector< Triplet<T> >           m_triplets;
    std::vector< elastic_material_data<T> > m_material;
    std::vector< size_t >               m_elements_with_bc_eges;
    std::vector<fracture_3d<Mesh> >        m_fractures;
    std::vector<restriction_3d >           m_restrictions;
    
    std::vector<size_t>                 m_compress_sigma_indexes;
    std::vector<size_t>                 m_compress_fracture_indexes;
    std::vector<size_t>                 m_compress_hybrid_indexes;
    std::vector<size_t>                 m_compress_up_mortar_indexes;
    
    std::vector<std::pair<size_t,size_t>> m_fracture_pairs;
    std::vector<std::pair<size_t,size_t>> m_elements_with_fractures_eges;
    std::vector<std::pair<size_t,size_t>> m_end_point_mortars;

    size_t      m_n_edges;
    size_t      m_n_essential_edges;
    size_t      m_n_cells_dof;
    size_t      m_n_faces_dof;
    size_t      m_n_hybrid_dof;
    size_t      m_n_f_hybrid_dof;
    size_t      m_n_skin_dof;
    size_t      m_sigma_degree;
    size_t      m_n_up_mortar_dof;
    size_t      m_n_mortar_points;
    bool        m_hho_stabilization_Q;
    bool        m_scaled_stabilization_Q;
    T           m_l_f = 0.001;
    
public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    elastic_two_fields_assembler_3d(const Mesh& msh, const disk::hho_degree_info& hho_di, const boundary_type& bnd, const std::vector<fracture_3d<Mesh> > & fractures/*, std::vector<std::pair<size_t,size_t>> & end_point_mortars, std::vector<fracture_3d<Mesh> > & fractures, std::vector<restriction > & restrictions*/)
        : m_hho_di(hho_di), m_bnd(bnd), m_fractures(fractures)/*, m_end_point_mortars(end_point_mortars),  m_restrictions(restrictions)*/, m_hho_stabilization_Q(true), m_scaled_stabilization_Q(false)
    {
            
        auto is_dirichlet = [&](const typename Mesh::face& fc) -> bool {

            auto fc_id = msh.lookup(fc);
            bool check_Q = bnd.is_dirichlet_face(fc_id);
            return check_Q;
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
            auto non_essential_dofs = n_fbs - m_bnd.dirichlet_imposed_dofs(face_id, m_hho_di.face_degree());
            
            m_n_faces_dof += non_essential_dofs;
        }
        
        m_n_hybrid_dof = 0;
        m_n_f_hybrid_dof = 0;
        m_n_skin_dof = 0;
        m_compress_fracture_indexes.resize(m_fractures.size());
        m_compress_hybrid_indexes.resize(m_fractures.size());
        
        size_t frac_c = 0;
        for (auto f : m_fractures) {
            m_compress_hybrid_indexes.at(frac_c) = m_n_f_hybrid_dof;
            m_compress_fracture_indexes.at(frac_c) = m_n_skin_dof;
            
            m_n_f_hybrid_dof += (n_f_sigma_n_bs + 2*n_f_sigma_t_bs) * f.m_pairs.size();
            m_n_hybrid_dof += (n_f_sigma_n_bs + 2*n_f_sigma_t_bs) * f.m_pairs.size();
//            m_n_hybrid_dof += 2 * 2;
//            m_n_skin_dof += 4 * f.m_skin_bs;
            frac_c++;
        }
        
        m_n_mortar_points = 0;
        m_n_up_mortar_dof = 0;
        m_compress_up_mortar_indexes.resize(m_n_mortar_points);
        for (size_t i = 0; i < m_n_mortar_points; i++) {
            m_compress_up_mortar_indexes.at(i) = m_n_up_mortar_dof;
            m_n_up_mortar_dof += 2;
        }
        
        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        m_n_cells_dof = n_cbs * msh.cells_size();
            
        size_t system_size = m_n_cells_dof + m_n_faces_dof + m_n_hybrid_dof;
//        system_size += m_n_up_mortar_dof;
        
        // skin data
        for (auto f : m_fractures) {
//            system_size += 4 * f.m_skin_bs;
        }
        
        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
        classify_cells(msh);

    }

    void scatter_data(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs,
             const Matrix<T, Dynamic, 1>& rhs)
    {
        const auto dim = Mesh::dimension;
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
                        for (size_t i = 0; i < (dim-1)*(n_fbs/dim); i += dim-1){
                            asm_map.push_back( assembly_index(face_LHS_offset+i-0, false) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i+0, true) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i+1, true) );
                        }
                        break;
                    }
                    case disk::DY: {
                        for (size_t i = 0; i < (dim-1)*(n_fbs/dim); i += dim-1){
                            asm_map.push_back( assembly_index(face_LHS_offset+i+0, true) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i-0, false) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i+1, true) );
                        }
                        break;
                    }
                    case disk::DZ: {
                     for (size_t i = 0; i < (dim-1)*(n_fbs/dim); i += dim-1){
                         asm_map.push_back( assembly_index(face_LHS_offset+i+0, true) );
                         asm_map.push_back( assembly_index(face_LHS_offset+i+1, true) );
                         asm_map.push_back( assembly_index(face_LHS_offset+i-0, false) );
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
        const auto dim = Mesh::dimension;
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
                        for (size_t i = 0; i < (dim-1)*(n_fbs/dim); i += dim-1){
                            asm_map.push_back( assembly_index(face_LHS_offset+i-0, false) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i+0, true) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i+1, true) );
                        }
                        break;
                    }
                    case disk::DY: {
                        for (size_t i = 0; i < (dim-1)*(n_fbs/dim); i += dim-1){
                            asm_map.push_back( assembly_index(face_LHS_offset+i+0, true) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i-0, false) );
                            asm_map.push_back( assembly_index(face_LHS_offset+i+1, true) );
                        }
                        break;
                    }
                     case disk::DZ: {
                         for (size_t i = 0; i < (dim-1)*(n_fbs/dim); i += dim-1){
                             asm_map.push_back( assembly_index(face_LHS_offset+i+0, true) );
                             asm_map.push_back( assembly_index(face_LHS_offset+i+1, true) );
                             asm_map.push_back( assembly_index(face_LHS_offset+i-0, false) );
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
    
    void scatter_mortar_data(const Mesh& msh, const size_t & face_id, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_f_sigma_bs = 3.0*disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);

        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof + cell_ind*n_f_sigma_bs;
        frac_LHS_offset += m_compress_hybrid_indexes.at(fracture_ind);
    
        
        for (size_t i = 0; i < n_f_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(frac_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j], mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j],asm_map_i[i], mortar_mat(i,j)) );
            }
        }
    
    }
        
    void scatter_point_mortar_data(const Mesh& msh, const size_t & face_id, const size_t & point_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_f_sigma_bs = 2.0;
        size_t n_skin_bs = 2 * m_fracture_pairs.size() + 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_fracture_pairs.size()*n_f_sigma_bs;
        auto point_LHS_offset = frac_LHS_offset  + 4.0 * n_skin_bs + n_f_sigma_bs*point_ind;
        
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
    
    void scatter_skins_point_mortar_ul_n_data(const Mesh& msh, const size_t & face_id, const size_t & frac_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = 2 * m_fracture_pairs.size() + 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_dof_dest_l.at(frac_ind);
        
        for (size_t i = 0; i < n_skin_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j], +1.0*mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j], asm_map_i[i], -0.0*mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_ul_t_data(const Mesh& msh, const size_t & face_id, const size_t & frac_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = 2 * m_fracture_pairs.size() + 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_dof_dest_l.at(frac_ind) + n_skin_bs;
        
        for (size_t i = 0; i < n_skin_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j], +1.0*mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j],asm_map_i[i], -0.0*mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_ur_n_data(const Mesh& msh, const size_t & face_id, const size_t & frac_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = 2 * m_fracture_pairs.size() + 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_dof_dest_r.at(frac_ind) + 2*n_skin_bs;
        
        for (size_t i = 0; i < n_skin_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],+1.0*mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j], asm_map_i[i],-0.0*mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_ur_t_data(const Mesh& msh, const size_t & face_id, const size_t & frac_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = 2 * m_fracture_pairs.size() + 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_dof_dest_r.at(frac_ind) + 3*n_skin_bs;
        
        for (size_t i = 0; i < n_skin_sigma_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_fbs; i++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mortar_mat.rows() && asm_map_j.size() == mortar_mat.cols() );

        for (size_t i = 0; i < mortar_mat.rows(); i++)
        {
            for (size_t j = 0; j < mortar_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],+1.0*mortar_mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j],asm_map_i[i], -0.0*mortar_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_u_n_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const Matrix<T, Dynamic, Dynamic>& mat)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        size_t n_fractures = m_fractures.size();
        size_t n_0d_bc_bs = 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_l_j, asm_map_r_j;
        auto base_i = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof + m_n_f_hybrid_dof;
        base_i += fracture_ind * 4 * n_0d_bc_bs;
        auto base_j = m_n_cells_dof + m_n_faces_dof;
        base_j += m_compress_fracture_indexes.at(fracture_ind);
        
        auto s_point_l_LHS_offset = base_i;
        auto skin_l_LHS_offset = base_j + n_skin_sigma_bs * n_cells;
        auto skin_r_LHS_offset = base_j + n_skin_sigma_bs * n_cells + 2 * n_skin_bs;
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map_i.push_back( assembly_index(s_point_l_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map_l_j.push_back( assembly_index(skin_l_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map_r_j.push_back( assembly_index(skin_r_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mat.rows() && asm_map_l_j.size() == mat.cols() );
        assert( asm_map_i.size() == mat.rows() && asm_map_r_j.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_l_j[j], +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j], asm_map_i[i], -1.0*mat(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_r_j[j], +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j], asm_map_i[i], -1.0*mat(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+1, asm_map_l_j[j]+n_cells, -1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j]+n_cells, asm_map_i[i]+1, +1.0*mat(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+1, asm_map_r_j[j]+n_cells, -1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j]+n_cells, asm_map_i[i]+1, +1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_u_t_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const Matrix<T, Dynamic, Dynamic>& mat)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        size_t n_fractures = m_fractures.size();
        size_t n_0d_bc_bs = 1;
        
        std::vector<assembly_index> asm_map_i, asm_map_l_j, asm_map_r_j;
        auto base_i = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof + m_n_f_hybrid_dof +  2*n_0d_bc_bs;
        base_i += fracture_ind * 4 * n_0d_bc_bs;
        auto base_j = m_n_cells_dof + m_n_faces_dof + n_skin_bs;
        base_j += m_compress_fracture_indexes.at(fracture_ind);
        
        auto s_point_l_LHS_offset = base_i;
        auto skin_l_LHS_offset = base_j + n_skin_sigma_bs * n_cells;
        auto skin_r_LHS_offset = base_j + n_skin_sigma_bs * n_cells + 2 * n_skin_bs;
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map_i.push_back( assembly_index(s_point_l_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map_l_j.push_back( assembly_index(skin_l_LHS_offset+i, true));
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map_r_j.push_back( assembly_index(skin_r_LHS_offset+i, true));
        
        assert( asm_map_i.size() == mat.rows() && asm_map_l_j.size() == mat.cols() );
        assert( asm_map_i.size() == mat.rows() && asm_map_r_j.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_l_j[j], +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j], asm_map_i[i], +1.0*mat(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_r_j[j], +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j], asm_map_i[i], +1.0*mat(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+1, asm_map_l_j[j]+n_cells, +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j]+n_cells, asm_map_i[i]+1, +1.0*mat(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+1, asm_map_r_j[j]+n_cells, +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j]+n_cells, asm_map_i[i]+1, +1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_mass_n_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const Matrix<T, Dynamic, Dynamic>& mat)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        size_t n_fractures = m_fractures.size();
        size_t n_0d_bc_bs = 1;
        
        std::vector<assembly_index> asm_map;
        auto base = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof + m_n_f_hybrid_dof;
        base += fracture_ind * 4 * n_0d_bc_bs;
        auto s_point_l_LHS_offset = base;

        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(s_point_l_LHS_offset+i, true));
        
        assert( asm_map.size() == mat.rows() && asm_map.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map[i]+1, asm_map[j]+1, +1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_mass_t_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const Matrix<T, Dynamic, Dynamic>& mat)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        size_t n_fractures = m_fractures.size();
        size_t n_0d_bc_bs = 1;
        
        std::vector<assembly_index> asm_map;
        auto base = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof + m_n_f_hybrid_dof + 2*n_0d_bc_bs;
        base += fracture_ind * 4 * n_0d_bc_bs;
        auto s_point_l_LHS_offset = base;

        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(s_point_l_LHS_offset+i, true));
        
        assert( asm_map.size() == mat.rows() && asm_map.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], +1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map[i]+1, asm_map[j]+1, +1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skins_point_mortar_rhs_t_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const Matrix<T, Dynamic, Dynamic>& rhs)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        size_t n_fractures = m_fractures.size();
        size_t n_0d_bc_bs = 1;
        
        std::vector<assembly_index> asm_map;
        auto base = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof + m_n_f_hybrid_dof + 2*n_0d_bc_bs;
        base += fracture_ind * 4 * n_0d_bc_bs;
        auto s_point_l_LHS_offset = base;

        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(s_point_l_LHS_offset+i, true));
        
        assert( asm_map.size() == rhs.rows() );

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            RHS(asm_map[i]) += rhs(i);
            RHS(asm_map[i]+1) += rhs(i);
        }
    
    }

    void scatter_skins_bc_u_t_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, size_t s_ind)
    {
 
        auto storage = msh.backend_storage();
        
        Matrix<T, Dynamic, Dynamic> mat = Matrix<T, Dynamic, Dynamic>::Zero(1,1);
        T beta = 1.0e+10;
        mat(0,0) = +1.0*beta;
        
        size_t cell_ind = 0;
        if(f.m_bc_type.first == 1){
            cell_ind = 0;
        }else{
            cell_ind = f.m_pairs.size() - 1;
        }
        

        
        auto chunk = f.m_pairs[cell_ind];
        size_t cell_ind_l = f.m_elements[cell_ind].first;
        size_t cell_ind_r = f.m_elements[cell_ind].second;
        auto& face_l = storage->edges[chunk.first];
        auto& face_r = storage->edges[chunk.second];
        auto& cell_l = storage->surfaces[cell_ind_l];
        auto& cell_r = storage->surfaces[cell_ind_r];
        
        const auto n_l = disk::normal(msh, cell_l, face_l);
        const auto t_l = disk::tanget(msh, cell_l, face_l);
        
        const auto n_r = disk::normal(msh, cell_r, face_r);
        const auto t_r = disk::tanget(msh, cell_r, face_r);
        
        // scattering data
        
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        size_t n_skin_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        size_t n_fractures = m_fractures.size();
        size_t n_0d_bc_bs = 1;
        
        // rhs value
        T uD_val;
        size_t shift = 0;
        static_vector<T, 2> uD;
        if(s_ind==0){
            uD = {f.m_bc_data.first[0],f.m_bc_data.first[1]};
            uD_val = uD.dot(t_l);
        }else{
            uD = {f.m_bc_data.second[0],f.m_bc_data.second[1]};
            uD_val = uD.dot(t_r);
            shift += 2 * n_skin_bs;
        }
        
        

        
        std::vector<assembly_index> asm_map;
        auto base = m_n_cells_dof + m_n_faces_dof + n_skin_bs;
        base += m_compress_fracture_indexes.at(fracture_ind) + shift;
        
        auto u_LHS_offset = base + n_skin_sigma_bs * n_cells + cell_ind * n_cells;
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(u_LHS_offset+i, true));
        
        assert( asm_map.size() == mat.rows() && asm_map.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], +1.0*mat(i,j)) );
            }
            RHS(asm_map[i]) += uD_val * beta;
        }
        
    }
    
    void scatter_skins_point_restriction_u_n_data(const Mesh& msh, restriction_3d & r, const Matrix<T, Dynamic, Dynamic>& mat)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        
        auto f_ind_f = r.m_f_index.first;
        auto p_ind_f = r.m_p_index.first;
        auto s_ind_f = r.m_s_index.first;
        
        auto f_ind_s = r.m_f_index.second;
        auto p_ind_s = r.m_p_index.second;
        auto s_ind_s = r.m_s_index.second;
        
        
        auto f_f = m_fractures[f_ind_f];
        auto f_s = m_fractures[f_ind_s];
        

        size_t shift_f = 0;
        size_t n_skin_bs_f = f_f.m_skin_bs;
        size_t n_cells_f = f_f.m_pairs.size();
        if(s_ind_f){
            shift_f += 2 * n_skin_bs_f;
        }
        
        size_t shift_s = 0;
        size_t n_skin_bs_s = f_s.m_skin_bs;
        size_t n_cells_s = f_s.m_pairs.size();
        if(s_ind_s){
            shift_s += 2 * n_skin_bs_s;
        }
        
        size_t n_0d_bc_bs = 1;
        size_t n_0d_up_bs = 1;
        
        std::vector<assembly_index> asm_map;
        auto base_f = m_n_cells_dof + m_n_faces_dof + m_compress_fracture_indexes.at(f_ind_f);
        auto base_s = m_n_cells_dof + m_n_faces_dof + m_compress_fracture_indexes.at(f_ind_s);
        auto skin_f_LHS_offset = base_f + n_skin_sigma_bs * n_cells_f + p_ind_f * n_cells_f + shift_f;
        auto skin_s_LHS_offset = base_s + n_skin_sigma_bs * n_cells_s + p_ind_s * n_cells_s + shift_s;
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(skin_f_LHS_offset+i, true));

        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(skin_s_LHS_offset+i, true));
                
        assert( asm_map.size() == mat.rows() && asm_map.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], +1.0*mat(i,j)) );
            }
        }
    }
    
    void scatter_skins_point_restriction_u_t_data(const Mesh& msh, restriction_3d & r, const Matrix<T, Dynamic, Dynamic>& mat)
    {
 
        size_t n_f_sigma_bs = disk::scalar_basis_size(m_sigma_degree, Mesh::dimension - 1);
        size_t n_skin_sigma_bs = 3.0;
        
        auto f_ind_f = r.m_f_index.first;
        auto p_ind_f = r.m_p_index.first;
        auto s_ind_f = r.m_s_index.first;
        
        auto f_ind_s = r.m_f_index.second;
        auto p_ind_s = r.m_p_index.second;
        auto s_ind_s = r.m_s_index.second;
        
        
        auto f_f = m_fractures[f_ind_f];
        auto f_s = m_fractures[f_ind_s];
        

        size_t shift_f = 0;
        size_t n_skin_bs_f = f_f.m_skin_bs;
        size_t n_cells_f = f_f.m_pairs.size();
        if(s_ind_f){
            shift_f += 2 * n_skin_bs_f;
        }
        
        size_t shift_s = 0;
        size_t n_skin_bs_s = f_s.m_skin_bs;
        size_t n_cells_s = f_s.m_pairs.size();
        if(s_ind_s){
            shift_s += 2 * n_skin_bs_s;
        }
        
        size_t n_0d_bc_bs = 1;
        size_t n_0d_up_bs = 1;
        
        std::vector<assembly_index> asm_map;
        auto base_f = m_n_cells_dof + m_n_faces_dof + n_skin_bs_f + m_compress_fracture_indexes.at(f_ind_f);
        auto base_s = m_n_cells_dof + m_n_faces_dof + n_skin_bs_s + m_compress_fracture_indexes.at(f_ind_s);
        auto skin_f_LHS_offset = base_f + n_skin_sigma_bs * n_cells_f + p_ind_f * n_cells_f + shift_f;
        auto skin_s_LHS_offset = base_s + n_skin_sigma_bs * n_cells_s + p_ind_s * n_cells_s + shift_s;
        
        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(skin_f_LHS_offset+i, true));

        for (size_t i = 0; i < n_0d_bc_bs; i++)
        asm_map.push_back( assembly_index(skin_s_LHS_offset+i, true));
                
        assert( asm_map.size() == mat.rows() && asm_map.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], +1.0*mat(i,j)) );
            }
        }
    }
    
    void scatter_mortar_mass_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mortar_mat)
    {
        size_t n_f_sigma_bs = 3.0*disk::scalar_basis_size(m_sigma_degree, Mesh::dimension-1);
        
        std::vector<assembly_index> asm_map;
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof +  cell_ind*n_f_sigma_bs;
        frac_LHS_offset += m_compress_hybrid_indexes.at(fracture_ind);
        
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
    
    void scatter_mortar_rhs_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mortar_rhs)
    {
        size_t n_f_sigma_bs = 2.0*disk::scalar_basis_size(m_sigma_degree, Mesh::dimension-1);
        
        std::vector<assembly_index> asm_map;
        auto frac_LHS_offset = m_n_cells_dof + m_n_faces_dof + m_n_skin_dof +  cell_ind*n_f_sigma_bs;
        frac_LHS_offset += m_compress_hybrid_indexes.at(fracture_ind);
        
        for (size_t i = 0; i < n_f_sigma_bs; i++)
        asm_map.push_back( assembly_index(frac_LHS_offset+i, true));
        
        assert( asm_map.size() == mortar_rhs.rows() );

        for (size_t i = 0; i < mortar_rhs.rows(); i++)
        {
            RHS(asm_map[i]) += mortar_rhs(i);
        }
    
    }
    
    void scatter_skin_weighted_mass_l_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mass_mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_sigma_skin_l_bs = f.m_skin_bs;
        std::vector<assembly_index> asm_map;
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + cell_ind * n_sigma_skin_bs;
        skin_LHS_offset += m_compress_fracture_indexes.at(fracture_ind);
        
        if(skin_LHS_offset + n_sigma_skin_l_bs >= 12978){
            int aka = 0;
        }
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map.push_back( assembly_index(skin_LHS_offset+i, true));
        
        assert( asm_map.size() == mass_mat.rows() && asm_map.size() == mass_mat.cols() );

        for (size_t i = 0; i < mass_mat.rows(); i++)
        {
            for (size_t j = 0; j < mass_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j],mass_mat(i,j)) );
            }
        }
        
        for (size_t i = 0; i < mass_mat.rows(); i++)
        {
            for (size_t j = 0; j < mass_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i]+n_sigma_skin_l_bs, asm_map[j]+n_sigma_skin_l_bs,mass_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_weighted_mass_r_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mass_mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_sigma_skin_l_bs = f.m_skin_bs;
        std::vector<assembly_index> asm_map;
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + cell_ind * n_sigma_skin_bs;
        skin_LHS_offset += m_compress_fracture_indexes.at(fracture_ind);
        skin_LHS_offset += 2.0 * n_sigma_skin_l_bs;
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map.push_back( assembly_index(skin_LHS_offset+i, true));
        
        assert( asm_map.size() == mass_mat.rows() && asm_map.size() == mass_mat.cols() );

        for (size_t i = 0; i < mass_mat.rows(); i++)
        {
            for (size_t j = 0; j < mass_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i], asm_map[j],mass_mat(i,j)) );
            }
        }
        
        for (size_t i = 0; i < mass_mat.rows(); i++)
        {
            for (size_t j = 0; j < mass_mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map[i]+n_sigma_skin_l_bs, asm_map[j]+n_sigma_skin_l_bs,mass_mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_weighted_ul_n_data(const Mesh& msh, const size_t & face_id, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + cell_ind * n_sigma_skin_bs;
        skin_LHS_offset += m_compress_fracture_indexes.at(fracture_ind);
        
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < n_fbs; j++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+j, true));
        
        assert( asm_map_i.size() == mat.rows() && asm_map_j.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],+1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j], asm_map_i[i],-1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_weighted_ul_t_data(const Mesh& msh, const size_t & face_id, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_sigma_skin_l_bs = f.m_skin_bs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + n_sigma_skin_l_bs + cell_ind*n_sigma_skin_bs;
        skin_LHS_offset += m_compress_fracture_indexes.at(fracture_ind);
        
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < n_fbs; j++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+j, true));
        
        assert( asm_map_i.size() == mat.rows() && asm_map_j.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],+1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j], asm_map_i[i],-1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_weighted_ur_n_data(const Mesh& msh, const size_t & face_id, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_sigma_skin_r_bs = f.m_skin_bs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + 2 * n_sigma_skin_r_bs + cell_ind*n_sigma_skin_bs;
        skin_LHS_offset += m_compress_fracture_indexes.at(fracture_ind);
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < n_fbs; j++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+j, true));
        
        assert( asm_map_i.size() == mat.rows() && asm_map_j.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],+1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j], asm_map_i[i],-1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_weighted_ur_t_data(const Mesh& msh, const size_t & face_id, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_sigma_skin_l_bs = f.m_skin_bs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        std::vector<assembly_index> asm_map_i, asm_map_j;
        auto skin_LHS_offset = m_n_cells_dof + m_n_faces_dof + 3 * n_sigma_skin_l_bs + cell_ind*n_sigma_skin_bs;
        skin_LHS_offset += m_compress_fracture_indexes.at(fracture_ind);
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < n_fbs; j++)
        asm_map_j.push_back( assembly_index(face_LHS_offset+j, true));
        
        assert( asm_map_i.size() == mat.rows() && asm_map_j.size() == mat.cols() );

        for (size_t i = 0; i < mat.rows(); i++)
        {
            for (size_t j = 0; j < mat.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_j[j],+1.0*mat(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_j[j], asm_map_i[i],-1.0*mat(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_hybrid_l_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_skin_l_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        std::vector<assembly_index> asm_map_i, asm_map_l_j, asm_map_r_j;
        auto base = m_n_cells_dof + m_n_faces_dof + m_compress_fracture_indexes.at(fracture_ind);
        auto skin_LHS_offset = base + cell_ind * n_sigma_skin_bs;
        auto hybrid_l_LHS_offset = base + n_cells * n_sigma_skin_bs + cell_ind;
        auto hybrid_r_LHS_offset = base + n_cells * n_sigma_skin_bs + cell_ind + 1;
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < 1; j++)
        asm_map_l_j.push_back( assembly_index(hybrid_l_LHS_offset+j, true));
        
        for (size_t j = 0; j < 1; j++)
        asm_map_r_j.push_back( assembly_index(hybrid_r_LHS_offset+j, true));
        
        Matrix<T, Dynamic, Dynamic> mat_l = mat.block(0,0,3,1);
        Matrix<T, Dynamic, Dynamic> mat_r = mat.block(0,1,3,1);
        assert( asm_map_i.size() == mat_l.rows() && asm_map_l_j.size() == mat_l.cols() );
        assert( asm_map_i.size() == mat_r.rows() && asm_map_r_j.size() == mat_r.cols() );

        for (size_t i = 0; i < mat_l.rows(); i++)
        {
            for (size_t j = 0; j < mat_l.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_l_j[j],+1.0*mat_l(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j], asm_map_i[i],+1.0*mat_l(i,j)) );

                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_r_j[j],+1.0*mat_r(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j], asm_map_i[i],+1.0*mat_r(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+n_skin_l_bs, asm_map_l_j[j]+n_skin_l_bs,+1.0*mat_l(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j]+n_skin_l_bs, asm_map_i[i]+n_skin_l_bs,+1.0*mat_l(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+n_skin_l_bs, asm_map_r_j[j]+n_skin_l_bs,+1.0*mat_r(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j]+n_skin_l_bs, asm_map_i[i]+n_skin_l_bs,+1.0*mat_r(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_hybrid_r_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_skin_l_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        std::vector<assembly_index> asm_map_i, asm_map_l_j, asm_map_r_j;
        auto base = m_n_cells_dof + m_n_faces_dof + 2 * n_skin_l_bs + m_compress_fracture_indexes.at(fracture_ind);
        auto skin_LHS_offset = base + cell_ind * n_sigma_skin_bs;
        auto hybrid_l_LHS_offset = base + n_cells * n_sigma_skin_bs + cell_ind;
        auto hybrid_r_LHS_offset = base + n_cells * n_sigma_skin_bs + cell_ind + 1;
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < 1; j++)
        asm_map_l_j.push_back( assembly_index(hybrid_l_LHS_offset+j, true));
        
        for (size_t j = 0; j < 1; j++)
        asm_map_r_j.push_back( assembly_index(hybrid_r_LHS_offset+j, true));
        
        Matrix<T, Dynamic, Dynamic> mat_l = mat.block(0,0,3,1);
        Matrix<T, Dynamic, Dynamic> mat_r = mat.block(0,1,3,1);
        assert( asm_map_i.size() == mat_l.rows() && asm_map_l_j.size() == mat_l.cols() );
        assert( asm_map_i.size() == mat_r.rows() && asm_map_r_j.size() == mat_r.cols() );

        for (size_t i = 0; i < mat_l.rows(); i++)
        {
            for (size_t j = 0; j < mat_l.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_l_j[j],+1.0*mat_l(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j], asm_map_i[i],+1.0*mat_l(i,j)) );

                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_r_j[j],+1.0*mat_r(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j], asm_map_i[i],+1.0*mat_r(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+n_skin_l_bs, asm_map_l_j[j]+n_skin_l_bs,+1.0*mat_l(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j]+n_skin_l_bs, asm_map_i[i]+n_skin_l_bs,+1.0*mat_l(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+n_skin_l_bs, asm_map_r_j[j]+n_skin_l_bs,+1.0*mat_r(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j]+n_skin_l_bs, asm_map_i[i]+n_skin_l_bs,+1.0*mat_r(i,j)) );
            }
        }
    
    }
    
    void scatter_skin_hybrid_nodewise_l_data(const Mesh& msh, size_t fracture_ind, fracture_3d<Mesh> & f, const size_t & cell_ind, const Matrix<T, Dynamic, Dynamic>& mat)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_skin_l_bs = f.m_skin_bs;
        size_t n_cells = f.m_pairs.size();
        std::vector<assembly_index> asm_map_i, asm_map_l_j, asm_map_r_j;
        auto base = m_n_cells_dof + m_n_faces_dof + m_compress_fracture_indexes.at(fracture_ind);
        auto skin_LHS_offset = base + cell_ind * n_sigma_skin_bs;
        auto hybrid_l_LHS_offset = base + n_cells * n_sigma_skin_bs + cell_ind;
        auto hybrid_r_LHS_offset = base + n_cells * n_sigma_skin_bs + cell_ind + 1;
        
        for (size_t i = 0; i < n_sigma_skin_bs; i++)
        asm_map_i.push_back( assembly_index(skin_LHS_offset+i, true));

        for (size_t j = 0; j < 1; j++)
        asm_map_l_j.push_back( assembly_index(hybrid_l_LHS_offset+j, true));
        
        for (size_t j = 0; j < 1; j++)
        asm_map_r_j.push_back( assembly_index(hybrid_r_LHS_offset+j, true));
        
        Matrix<T, Dynamic, Dynamic> mat_l = mat.block(0,0,3,1);
        Matrix<T, Dynamic, Dynamic> mat_r = mat.block(0,1,3,1);
        assert( asm_map_i.size() == mat_l.rows() && asm_map_l_j.size() == mat_l.cols() );
        assert( asm_map_i.size() == mat_r.rows() && asm_map_r_j.size() == mat_r.cols() );

        for (size_t i = 0; i < mat_l.rows(); i++)
        {
            for (size_t j = 0; j < mat_l.cols(); j++)
            {
                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_l_j[j],+1.0*mat_l(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j], asm_map_i[i],+1.0*mat_l(i,j)) );

                m_triplets.push_back( Triplet<T>(asm_map_i[i], asm_map_r_j[j],+1.0*mat_r(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j], asm_map_i[i],+1.0*mat_r(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+n_skin_l_bs, asm_map_l_j[j]+n_skin_l_bs,+1.0*mat_l(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_l_j[j]+n_skin_l_bs, asm_map_i[i]+n_skin_l_bs,+1.0*mat_l(i,j)) );
                
                m_triplets.push_back( Triplet<T>(asm_map_i[i]+n_skin_l_bs, asm_map_r_j[j]+n_skin_l_bs,+1.0*mat_r(i,j)) );
                m_triplets.push_back( Triplet<T>(asm_map_r_j[j]+n_skin_l_bs, asm_map_i[i]+n_skin_l_bs,+1.0*mat_r(i,j)) );
            }
        }
    
    }
    
    void scatter_rhs_skin_weighted_u_data(const Mesh& msh, const size_t & face_id, const size_t & fracture_ind, const Matrix<T, Dynamic, 1>& rhs)
    {
        size_t n_sigma_skin_bs = 3;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        std::vector<assembly_index> asm_map_i;
        auto face_LHS_offset = m_n_cells_dof + m_compress_indexes.at(face_id);

        for (size_t i = 0; i < n_fbs; i++)
        asm_map_i.push_back( assembly_index(face_LHS_offset+i, true));
        
        assert( asm_map_i.size() == rhs.rows() );

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            RHS(asm_map_i[i]) += rhs(i);
        }
    
    }
    
    auto mortar_coupling_matrix_skin_u(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
        {
            const auto degree     = m_hho_di.face_degree();
            
            auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, m_hho_di.face_degree());
            auto s_basis = disk::make_scalar_monomial_basis(msh, face, m_hho_di.face_degree());
            
            size_t n_s_basis = s_basis.size();
            Matrix<T, Dynamic, Dynamic> ret_n = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
            
            Matrix<T, Dynamic, Dynamic> ret_t = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
            
            const auto qps = integrate(msh, face, 2 * (degree+di));
            const auto n = disk::normal(msh, cell, face);
            const auto t = disk::tanget(msh, cell, face);

            for (auto& qp : qps)
            {
                const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
                const auto s_f_phi = s_basis.eval_functions(qp.point());
                            
                const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
                const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
                
                const auto s_n_opt = disk::priv::outer_product(s_f_phi, w_n_dot_u_f_phi);
                const auto s_t_opt = disk::priv::outer_product(s_f_phi, w_t_dot_u_f_phi);

                ret_n += s_n_opt;
                ret_t += s_t_opt;
            }

            return std::make_pair(ret_n, ret_t);
        }

    void assemble(const Mesh& msh, std::function<static_vector<double, 3>(const typename Mesh::point_type& )> rhs_fun){
        
        LHS.setZero();
        RHS.setZero();
 
        // rock mass hho assemble
        assemble_rock_mass(msh,rhs_fun);
        
        // mortars assemble
        assemble_mortars(msh);
        
        // skins assemble
//        assemble_skins(msh);
    
        finalize();

    }
    
    void assemble_rock_mass(const Mesh& msh, std::function<static_vector<double, 3>(const typename Mesh::point_type& )> rhs_fun){
        
        size_t cell_ind = 0;
        for (auto& cell : msh)
        {
            Matrix<T, Dynamic, Dynamic> mixed_operator_loc = mixed_operator(cell_ind,msh,cell);
            Matrix<T, Dynamic, 1> f_loc = mixed_rhs(msh, cell, rhs_fun);
            scatter_data(msh, cell, mixed_operator_loc, f_loc);
            cell_ind++;
        }

    }

    void assemble_mortars(const Mesh& msh){
        auto storage = msh.backend_storage();
        
        size_t f_ind = 0;
        for (auto f : m_fractures) {
            size_t cell_ind = 0;
            for (auto chunk : f.m_pairs) {
                
                size_t cell_ind_l = f.m_elements[cell_ind].first;
                size_t cell_ind_r = f.m_elements[cell_ind].second;
                auto& face_l = storage->surfaces[chunk.first];
                auto& face_r = storage->surfaces[chunk.second];
                auto& cell_l = storage->volumes[cell_ind_l];
                auto& cell_r = storage->volumes[cell_ind_r];
                
                Matrix<T, Dynamic, Dynamic> mortar_l = -1.0*mortar_coupling_matrix(msh,cell_l,face_l,1);
                Matrix<T, Dynamic, Dynamic> mortar_r = -1.0*mortar_coupling_matrix(msh,cell_r,face_r,-1);
                            
                scatter_mortar_data(msh,chunk.first,f_ind,f,cell_ind,mortar_l);
                scatter_mortar_data(msh,chunk.second,f_ind,f,cell_ind,mortar_r);
                
                Matrix<T, Dynamic, Dynamic> mass_matrix = sigma_mass_matrix(msh, face_l, face_r);
                scatter_mortar_mass_data(msh,f_ind,f,cell_ind,mass_matrix);
                
//                Matrix<T, Dynamic, Dynamic> rhs = sigma_rhs(msh, face_l, face_r);
//                scatter_mortar_rhs_data(msh,f_ind,f,cell_ind,rhs);
                
                cell_ind++;
            }
            f_ind++;
        }
    }
    
    void assemble_skins(const Mesh& msh){

        auto storage = msh.backend_storage();
        
        size_t f_ind = 0;
        for (auto f : m_fractures) {
            size_t cell_ind = 0;
            for (auto chunk : f.m_pairs) {
                
                size_t cell_ind_l = f.m_elements[cell_ind].first;
                size_t cell_ind_r = f.m_elements[cell_ind].second;
                auto& face_l = storage->surfaces[chunk.first];
                auto& face_r = storage->surfaces[chunk.second];
                auto& cell_l = storage->volumes[cell_ind_l];
                auto& cell_r = storage->volumes[cell_ind_r];
                
                
                // mass matrix
                auto mass_matrix = skin_weighted_mass_matrix(msh, face_l, face_r, f, cell_ind);
                scatter_skin_weighted_mass_l_data(msh, f_ind, f, cell_ind, mass_matrix.first);
                scatter_skin_weighted_mass_r_data(msh, f_ind, f, cell_ind, mass_matrix.second);
                
                auto ul_div_phi = skin_coupling_matrix_ul(msh, cell_l, face_l, f, cell_ind);
                auto ur_div_phi = skin_coupling_matrix_ur(msh, cell_r, face_r, f, cell_ind);
                
//                scatter_skin_weighted_ul_n_data(msh, chunk.first, f_ind, f, cell_ind, ul_div_phi.first);
                scatter_skin_weighted_ul_t_data(msh, chunk.first, f_ind, f, cell_ind, ul_div_phi.second);
//                scatter_skin_weighted_ur_n_data(msh, chunk.second, f_ind, f, cell_ind, ur_div_phi.first);
                scatter_skin_weighted_ur_t_data(msh, chunk.second, f_ind, f, cell_ind, ur_div_phi.second);
                
                auto hybrid_matrix = skin_hybrid_matrix(msh, face_l, face_r, f, cell_ind);
                scatter_skin_hybrid_l_data(msh, f_ind, f, cell_ind, hybrid_matrix.first);
                scatter_skin_hybrid_r_data(msh, f_ind, f, cell_ind, hybrid_matrix.second);

                cell_ind++;
            }
            
            bool point_mortars_Q = true;
            if(point_mortars_Q){ // apply mortar
                
                Matrix<T, Dynamic, Dynamic> mortar = Matrix<T, Dynamic, Dynamic>::Zero(1,1);
                mortar(0,0) = 1.0;
                
                scatter_skins_point_mortar_u_n_data(msh,f_ind,f,mortar);
                scatter_skins_point_mortar_u_t_data(msh,f_ind,f,mortar);
                
                T alpha = 0.0e+0;
                if (f_ind == 0) {
                    alpha = 1.0;
                }
                mortar(0,0) = alpha;
                scatter_skins_point_mortar_mass_n_data(msh,f_ind,f,mortar);
                mortar(0,0) = alpha;
                scatter_skins_point_mortar_mass_t_data(msh,f_ind,f,mortar);
                
//                Matrix<T, Dynamic, Dynamic> rhs = Matrix<T, Dynamic, Dynamic>::Zero(1,1);
//                T beta = 0.0;
//                rhs(0,0) = (+3.0/200.0)*beta;
//                scatter_skins_point_mortar_rhs_t_data(msh,f_ind,f,rhs);
                    
            }
            
            if(f.m_bc_type.first == 1 || f.m_bc_type.second == 1){
                if (cell_ind == 0 || cell_ind == f.m_pairs.size()) {
                    scatter_skins_bc_u_t_data(msh, f_ind, f , 0);
                    scatter_skins_bc_u_t_data(msh, f_ind, f , 1);
                }

            }
            
            f_ind++;
        }
        
        // apply mortar restrictions
        
        bool point_restrictions_Q = true;
        if(point_restrictions_Q){ // apply restrictions

            Matrix<T, Dynamic, Dynamic> un_restriction = Matrix<T, Dynamic, Dynamic>::Zero(2,2);
            Matrix<T, Dynamic, Dynamic> ut_restriction = Matrix<T, Dynamic, Dynamic>::Zero(2,2);
            T beta = 1.0e+10;
            for (auto r : m_restrictions) {
                
                un_restriction(0,0) = +1.0*beta;
                un_restriction(1,1) = +1.0*beta;
                ut_restriction(0,0) = +1.0*beta;
                ut_restriction(1,1) = +1.0*beta;
                
                auto f_ind_f = r.m_f_index.first;
                auto f_ind_s = r.m_f_index.second;
                auto f_f = m_fractures[f_ind_f];
                auto f_s = m_fractures[f_ind_s];
                size_t n_cells_f = f_f.m_pairs.size();
                size_t n_cells_s = f_s.m_pairs.size();
                
                size_t face_ind_f,cell_ind_f;
                if (r.m_p_index.first == 0) {
                    if(r.m_s_index.first == 0){
                        cell_ind_f = f_f.m_elements[0].first;
                        face_ind_f = f_f.m_pairs[0].first;
                    }else{
                        cell_ind_f = f_f.m_elements[0].second;
                        face_ind_f = f_f.m_pairs[0].second;
                    }
                }else{
                    if(r.m_s_index.first == 0){
                        cell_ind_f = f_f.m_elements[n_cells_f-1].first;
                        face_ind_f = f_f.m_pairs[n_cells_f-1].first;
                    }else{
                        cell_ind_f = f_f.m_elements[n_cells_f-1].second;
                        face_ind_f = f_f.m_pairs[n_cells_f-1].second;
                    }
                }
                
                size_t face_ind_s,cell_ind_s;
                if (r.m_p_index.second == 0) {
                    if(r.m_s_index.second == 0){
                        cell_ind_s = f_s.m_elements[0].first;
                        face_ind_s = f_s.m_pairs[0].first;
                    }else{
                        cell_ind_s = f_s.m_elements[0].second;
                        face_ind_s = f_s.m_pairs[0].second;
                    }
                }else{
                    if(r.m_s_index.second == 0){
                        cell_ind_s = f_s.m_elements[n_cells_s-1].first;
                        face_ind_s = f_s.m_pairs[n_cells_s-1].first;
                    }else{
                        cell_ind_s = f_s.m_elements[n_cells_s-1].second;
                        face_ind_s = f_s.m_pairs[n_cells_s-1].second;
                    }
                }
                
                auto& face_f = storage->surfaces[face_ind_f];
                auto& cell_f = storage->volumes[cell_ind_f];
                auto& face_s = storage->surfaces[face_ind_s];
                auto& cell_s = storage->volumes[cell_ind_s];
                
                const auto n_f = disk::normal(msh, cell_f, face_f);
                const auto t_f = disk::tanget(msh, cell_f, face_f).first;
                assert(false);
                
                const auto n_s = disk::normal(msh, cell_s, face_s);
                const auto t_s = disk::tanget(msh, cell_s, face_s).first;
                assert(false);
                
                // simple because normal data is being neglected
                auto p_f_s = t_f.dot(t_s);
                auto p_s_f = t_s.dot(t_f);
                if(f_ind_f == f_ind_s){
                    ut_restriction(0,1) = -1.0*beta * (p_f_s);
                    ut_restriction(1,0) = -1.0*beta * (p_s_f);
                }else{
                    ut_restriction(0,1) = +1.0*beta * (p_f_s);
                    ut_restriction(1,0) = +1.0*beta * (p_s_f);
                }
                
                scatter_skins_point_restriction_u_n_data(msh,r,ut_restriction);
                scatter_skins_point_restriction_u_t_data(msh,r,ut_restriction);
            }
            
        }
        
    }
        
    void apply_bc(const Mesh& msh){
        
        #ifdef HAVE_INTEL_TBB2
                size_t n_cells = m_elements_with_bc_eges.size();
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [this,&msh] (size_t & i){
                        size_t cell_ind = m_elements_with_bc_eges[i];
                        auto& cell = msh.backend_storage()->volumes[cell_ind];
                        Matrix<T, Dynamic, Dynamic> mixed_operator_loc = mixed_operator(cell_ind, msh, cell);
                        scatter_bc_data(msh, cell, mixed_operator_loc);
                }
            );
        #else
            auto storage = msh.backend_storage();
            for (auto& cell_ind : m_elements_with_bc_eges)
            {
                auto& cell = storage->volumes[cell_ind];
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
        
        auto qps = integrate(msh, cell, 2 * gradeg + 1);

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
        mass_matrix_trace_sigma *= (lambda/(2.0*mu+3.0*lambda));
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
    
    Matrix<T, Dynamic, Dynamic> mortar_coupling_matrix(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, int sign = 1, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, m_hho_di.face_degree());
        auto sn_basis = disk::make_scalar_monomial_basis(msh, face, m_sigma_degree);
        auto st_basis = disk::make_scalar_monomial_basis(msh, face, m_sigma_degree);
        
        size_t n_s_basis = sn_basis.size() + st_basis.size() + st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());

        const auto qps = integrate(msh, face, 2 * (degree+ 1 + di));
        const auto n = disk::normal(msh, cell, face);
        const auto t1 = disk::tanget(msh, cell, face, sign).first;
        const auto t2 = disk::tanget(msh, cell, face).second;

        for (auto& qp : qps)
        {
            const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
            const auto sn_f_phi = sn_basis.eval_functions(qp.point());
            const auto st_f_phi = st_basis.eval_functions(qp.point());

            const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
            const auto s_n_opt = disk::priv::outer_product(sn_f_phi, w_n_dot_u_f_phi);

            const auto w_t1_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t1));
            const auto s_t1_opt = disk::priv::outer_product(st_f_phi, w_t1_dot_u_f_phi);
            
            const auto w_t2_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t2));
            const auto s_t2_opt = disk::priv::outer_product(st_f_phi, w_t2_dot_u_f_phi);

            ret.block(0,0,sn_basis.size(),vec_u_basis.size()) += s_n_opt;
            ret.block(sn_basis.size(),0,st_basis.size(),vec_u_basis.size()) += s_t1_opt;
            ret.block(sn_basis.size()+st_basis.size(),0,st_basis.size(),vec_u_basis.size()) += s_t2_opt;
        }

        return ret;
    }
    
    auto mortar_coupling_matrix_skin(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, m_hho_di.face_degree());
        auto s_basis = disk::make_scalar_monomial_basis(msh, face, m_hho_di.cell_degree());
        
        size_t n_s_basis = s_basis.size();
        Matrix<T, Dynamic, Dynamic> ret_n = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
        
        Matrix<T, Dynamic, Dynamic> ret_t = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
        
        const auto qps = integrate(msh, face, 2 * (degree+ 1 + di));
        const auto n = disk::normal(msh, cell, face);
        const auto t = disk::tanget(msh, cell, face);

        for (auto& qp : qps)
        {
            const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
            const auto s_f_phi = s_basis.eval_functions(qp.point());
                        
            const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
            const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
            
            const auto s_n_opt = disk::priv::outer_product(s_f_phi, w_n_dot_u_f_phi);
            const auto s_t_opt = disk::priv::outer_product(s_f_phi, w_t_dot_u_f_phi);

            ret_n += -1.0*s_n_opt;
            ret_t += -1.0*s_t_opt;
        }

        return std::make_pair(ret_n, ret_t);
    }
    
    auto mortar_coupling_matrix_skin_sigma(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto u_basis = disk::make_scalar_monomial_basis(msh, face, m_hho_di.cell_degree());
        auto s_basis = disk::make_scalar_monomial_basis(msh, face, m_sigma_degree);
        
        Matrix<T, Dynamic, Dynamic> ret_n = Matrix<T, Dynamic, Dynamic>::Zero(u_basis.size(),s_basis.size());
        
        Matrix<T, Dynamic, Dynamic> ret_t = Matrix<T, Dynamic, Dynamic>::Zero(u_basis.size(),s_basis.size());
        
        const auto qps = integrate(msh, face, 2 * (degree+di));
        const auto n = disk::normal(msh, cell, face);
        const auto t = disk::tanget(msh, cell, face);

        for (auto& qp : qps)
        {
            const auto u_f_phi = u_basis.eval_functions(qp.point());
            const auto s_f_phi = s_basis.eval_functions(qp.point());
                        
            const auto w_dot_u_f_phi = disk::priv::inner_product(qp.weight(),u_f_phi);
            
            const auto s_n_opt = disk::priv::outer_product(w_dot_u_f_phi,s_f_phi);
            const auto s_t_opt = disk::priv::outer_product(w_dot_u_f_phi,s_f_phi);

            ret_n += s_n_opt;
            ret_t += s_t_opt;
        }

        return std::make_pair(ret_n, ret_t);
    }
    
    Matrix<T, Dynamic, Dynamic> mortar_coupling_matrix_skin_jumps(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
    {
        const auto degree     = m_hho_di.face_degree();
        
        auto u_basis = disk::make_scalar_monomial_basis(msh, face, m_hho_di.cell_degree());
        auto s_basis = disk::make_scalar_monomial_basis(msh, face, m_sigma_degree);
        
        size_t n_s_basis = s_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, u_basis.size());

        const auto qps = integrate(msh, face, 2 * (degree+di));
        
        for (auto& qp : qps)
        {
            const auto u_f_phi = u_basis.eval_functions(qp.point());

            const auto s_f_phi = s_basis.eval_functions(qp.point());
                        
            const auto w_u_f_phi = disk::priv::inner_product(qp.weight(),u_f_phi);
            const auto s_opt = disk::priv::outer_product(s_f_phi, w_u_f_phi);

            ret += s_opt;
        }

        return ret;
    }
    
    Matrix<T, Dynamic, Dynamic> sigma_mass_matrix(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, size_t di = 0)
    {
        const auto degree     = m_sigma_degree;
        auto sn_basis = disk::make_scalar_monomial_basis(msh, face_l, m_sigma_degree);
        auto st_basis = disk::make_scalar_monomial_basis(msh, face_r, m_sigma_degree);
        
        size_t n_s_basis = sn_basis.size() + 2 * st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, n_s_basis);

        T c_perp = 0.0;
        const auto qps_l = integrate(msh, face_l, 2 * (degree+di+1));
        for (auto& qp : qps_l)
        {
            const auto sn_f_phi = sn_basis.eval_functions(qp.point());
            const auto w_sn_f_phi = disk::priv::inner_product(qp.weight(), sn_f_phi);
            const auto s_n_opt = disk::priv::outer_product(sn_f_phi, w_sn_f_phi);
            ret.block(0,0,sn_basis.size(),sn_basis.size()) += c_perp * s_n_opt;
        }
        
        T c_para = 0.0;
        const auto qps_r1 = integrate(msh, face_r, 2 * (degree+di));
        for (auto& qp : qps_r1)
        {
            const auto st_f_phi = st_basis.eval_functions(qp.point());
            const auto w_st_f_phi = disk::priv::inner_product(qp.weight(), st_f_phi);
            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_st_f_phi);
            ret.block(sn_basis.size(),sn_basis.size(),st_basis.size(),st_basis.size()) += c_para * s_t_opt;
        }
        
        const auto qps_r2 = integrate(msh, face_r, 2 * (degree+di));
        for (auto& qp : qps_r2)
        {
            const auto st_f_phi = st_basis.eval_functions(qp.point());
            const auto w_st_f_phi = disk::priv::inner_product(qp.weight(), st_f_phi);
            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_st_f_phi);
            ret.block(sn_basis.size()+st_basis.size(),sn_basis.size()+st_basis.size(),st_basis.size(),st_basis.size()) += c_para * s_t_opt;
        }

        return ret;
    }
    
    Matrix<T, Dynamic, Dynamic> sigma_rhs(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, size_t di = 0)
    {
        const auto degree     = m_sigma_degree;
        auto sn_basis = disk::make_scalar_monomial_basis(msh, face_l, m_sigma_degree);
        auto st_basis = disk::make_scalar_monomial_basis(msh, face_r, m_sigma_degree);
        
        size_t n_s_basis = sn_basis.size() + st_basis.size();
        Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, 1);

        T v_perp = -1.0/50;
        const auto qps_l = integrate(msh, face_l, 2 * (degree+di+1));
        for (auto& qp : qps_l)
        {
//            const auto sn_f_phi = sn_basis.eval_functions(qp.point());
//            const auto w_sn_f_phi = disk::priv::inner_product(qp.weight(), v_perp);
//            const auto s_n_opt = disk::priv::outer_product(sn_f_phi, w_sn_f_phi);
//            ret.block(0,0,sn_basis.size(),1) += s_n_opt;
        }
        
        T v_para = 0.0;
        const auto qps_r = integrate(msh, face_r, 2 * (degree+di));
        for (auto& qp : qps_r)
        {
//            const auto st_f_phi = st_basis.eval_functions(qp.point());
//            const auto w_st_f_phi = disk::priv::inner_product(qp.weight(), v_para);
//            const auto s_t_opt = disk::priv::outer_product(st_f_phi, w_st_f_phi);
//            ret.block(sn_basis.size(),0,st_basis.size(),1) += s_t_opt;
        }

        return ret;
    }
    
    auto skin_weighted_mass_matrix(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, fracture_3d<Mesh> & f, const size_t & cell_ind, size_t di = 0)
        {

            elastic_material_data<T> & material = m_material[0];
            T rho = material.rho();
            T mu = material.mu();
            T lambda = material.l();
            
            auto degree = m_hho_di.face_degree();
            auto sl_basis = disk::make_scalar_monomial_basis(msh, face_l, degree);
            auto sr_basis = disk::make_scalar_monomial_basis(msh, face_r, degree);
            if(f.m_flips_l.at(cell_ind)){
//                sl_basis.swap_nodes();
            }
            if(f.m_flips_r.at(cell_ind)){
//                sr_basis.swap_nodes();
            }
            assert(false);
            
            Matrix<T, Dynamic, Dynamic> ret_l = Matrix<T, Dynamic, Dynamic>::Zero(3, 3);
            Matrix<T, Dynamic, Dynamic> ret_r = Matrix<T, Dynamic, Dynamic>::Zero(3, 3);

            T c_l = (1.0e0)*(1.0/(lambda+2.0*mu));
            const auto qps_l = integrate(msh, face_l, 2 * (degree + 2 + di));
            for (auto& qp : qps_l)
            {
//
//                const auto sl_f_phi = sl_basis.eval_flux_functions(qp.point());
//                const auto w_sl_f_phi = disk::priv::inner_product(qp.weight(), sl_f_phi);
//                const auto s_opt_l = disk::priv::outer_product(sl_f_phi, w_sl_f_phi);
//                ret_l += c_l * s_opt_l;
            }
            
            T c_r = (1.0e0)*(1.0/(lambda+2.0*mu));
            const auto qps_r = integrate(msh, face_r, 2 * (degree + 2 + di));
            for (auto& qp : qps_r)
            {
                
//                const auto sr_f_phi = sr_basis.eval_flux_functions(qp.point());
//                const auto w_sr_f_phi = disk::priv::inner_product(qp.weight(), sr_f_phi);
//                const auto s_opt_r = disk::priv::outer_product(sr_f_phi, w_sr_f_phi);
//                ret_r += c_r * s_opt_r;
            }

            return std::make_pair(ret_l,ret_r);
        }
    
    auto skin_mass_matrix(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, fracture_3d<Mesh> & f, const size_t & cell_ind, size_t di = 0)
        {

            auto degree = m_hho_di.face_degree();
            auto sl_basis = disk::make_scalar_monomial_basis(msh, face_l, degree);
            auto sr_basis = disk::make_scalar_monomial_basis(msh, face_r, degree);
            if(f.m_flips_l.at(cell_ind)){
//                sl_basis.swap_nodes();
            }
            if(f.m_flips_r.at(cell_ind)){
//                sr_basis.swap_nodes();
            }
            assert(false);
            
            Matrix<T, Dynamic, Dynamic> ret_l = Matrix<T, Dynamic, Dynamic>::Zero(3, 3);
            Matrix<T, Dynamic, Dynamic> ret_r = Matrix<T, Dynamic, Dynamic>::Zero(3, 3);

            const auto qps_l = integrate(msh, face_l, 2 * (degree + 1 + di));
            for (auto& qp : qps_l)
            {
                
//                const auto sl_f_phi = sl_basis.eval_flux_functions(qp.point());
//                const auto w_sl_f_phi = disk::priv::inner_product(qp.weight(), sl_f_phi);
//                const auto s_opt_l = disk::priv::outer_product(sl_f_phi, w_sl_f_phi);
//                ret_l += s_opt_l;
            }
            
            const auto qps_r = integrate(msh, face_r, 2 * (degree+di));
            for (auto& qp : qps_r)
            {
                
//                const auto sr_f_phi = sr_basis.eval_flux_functions(qp.point());
//                const auto w_sr_f_phi = disk::priv::inner_product(qp.weight(), sr_f_phi);
//                const auto s_opt_r = disk::priv::outer_product(sr_f_phi, w_sr_f_phi);
//                ret_r += s_opt_r;
            }

            return std::make_pair(ret_l,ret_r);
        }
    
    auto skin_rhs(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, fracture_3d<Mesh> & f, const size_t & cell_ind, size_t di = 0)
        {
            
            auto degree = m_hho_di.face_degree();
            auto sl_basis = disk::make_scalar_monomial_basis(msh, face_l, degree);
            auto sr_basis = disk::make_scalar_monomial_basis(msh, face_r, degree);
            if(f.m_flips_l.at(cell_ind)){
                sl_basis.swap_nodes();
            }
            if(f.m_flips_r.at(cell_ind)){
                sr_basis.swap_nodes();
            }
            
            Matrix<T, Dynamic, Dynamic> rhs_l = Matrix<T, Dynamic, Dynamic>::Zero(3, 1);
            Matrix<T, Dynamic, Dynamic> rhs_r = Matrix<T, Dynamic, Dynamic>::Zero(3, 1);
            
            T val = 1.0;//-1.0/600.0;

            const auto qps_l = integrate(msh, face_l, 2 * (degree+di));
            for (auto& qp : qps_l)
            {
                
//                const auto sl_f_phi = sl_basis.eval_flux_functions(qp.point());
//                const auto w_sl_f_phi = disk::priv::inner_product(qp.weight(), val);
//                const auto s_opt_l = disk::priv::outer_product(sl_f_phi,w_sl_f_phi);
//                rhs_l += s_opt_l;
            }
            
            const auto qps_r = integrate(msh, face_r, 2 * (degree+di));
            for (auto& qp : qps_r)
            {
//
//                const auto sr_f_phi = sr_basis.eval_flux_functions(qp.point());
//                const auto w_sr_f_phi = disk::priv::inner_product(qp.weight(), val);
//                const auto s_opt_r = disk::priv::outer_product(sr_f_phi, w_sr_f_phi);
//                rhs_r += s_opt_r;
            }

            return std::make_pair(rhs_l,rhs_r);
        }
        
    auto skin_hybrid_matrix(const Mesh& msh, const typename Mesh::face_type& face_l, const typename Mesh::face_type& face_r, fracture_3d<Mesh> & f, const size_t & cell_ind, size_t di = 0)
        {

            elastic_material_data<T> & material = m_material[0];
            T rho = material.rho();
            T mu = material.mu();
            T lambda = material.l();
            
            auto degree = m_hho_di.face_degree();
            auto sl_basis = disk::make_scalar_monomial_basis(msh, face_l, degree);
            auto sr_basis = disk::make_scalar_monomial_basis(msh, face_r, degree);
//            if(f.m_flips_l.at(cell_ind)){
//                sl_basis.swap_nodes();
//            }
//            if(f.m_flips_r.at(cell_ind)){
//                sr_basis.swap_nodes();
//            }
            
            assert(false);
            
            Matrix<T, Dynamic, Dynamic> ret_l = Matrix<T, Dynamic, Dynamic>::Zero(3, 2);
            auto nodes_l = sl_basis.nodes();
            
            typename Mesh::point_type point_ll = nodes_l[0];
            typename Mesh::point_type point_lr = nodes_l[1];
            typename Mesh::point_type vl = point_lr - point_ll;
            auto nll = inplane_normal(msh,face_l,point_ll,vl);
            auto nlr = inplane_normal(msh,face_l,point_lr,vl);
            {
//                const auto s_phi = sl_basis.eval_flux_functions(point_ll);
//                const auto mu_f_phi = sl_basis.eval_lambda_functions(point_ll);
//                const auto w_mu_f_phi = disk::priv::inner_product(1.0, mu_f_phi);
//                const auto s_opt = disk::priv::outer_product(s_phi, w_mu_f_phi);
//
//                ret_l.block(0,0,3,1) += -1.0 * nll * s_opt;
            }
            {
//                const auto s_phi = sl_basis.eval_flux_functions(point_lr);
//                const auto mu_f_phi = sl_basis.eval_lambda_functions(point_lr);
//                const auto w_mu_f_phi = disk::priv::inner_product(1.0, mu_f_phi);
//                const auto s_opt = disk::priv::outer_product(s_phi, w_mu_f_phi);
//
//                ret_l.block(0,1,3,1) += -1.0 * nlr * s_opt;
            }
            
            Matrix<T, Dynamic, Dynamic> ret_r = Matrix<T, Dynamic, Dynamic>::Zero(3, 2);
            auto nodes_r = sr_basis.nodes();
            typename Mesh::point_type point_rl = nodes_r[0];
            typename Mesh::point_type point_rr = nodes_r[1];
            typename Mesh::point_type vr = point_rr - point_rl;
            auto nrl = inplane_normal(msh,face_r,point_rl,vr);
            auto nrr = inplane_normal(msh,face_r,point_rr,vr);
            
            {
//                const auto s_phi = sl_basis.eval_flux_functions(point_rl);
//                const auto mu_f_phi = sl_basis.eval_lambda_functions(point_rl);
//                const auto w_mu_f_phi = disk::priv::inner_product(1.0, mu_f_phi);
//                const auto s_opt = disk::priv::outer_product(s_phi, w_mu_f_phi);
//
//                ret_r.block(0,0,3,1) += -1.0 * nrl * s_opt;
            }
            {
//                const auto s_phi = sl_basis.eval_flux_functions(point_rr);
//                const auto mu_f_phi = sl_basis.eval_lambda_functions(point_rr);
//                const auto w_mu_f_phi = disk::priv::inner_product(1.0, mu_f_phi);
//                const auto s_opt = disk::priv::outer_product(s_phi, w_mu_f_phi);
//
//                ret_r.block(0,1,3,1) += -1.0 * nrr * s_opt;
            }
            return std::make_pair(ret_l,ret_r);
        }
    
    auto skin_coupling_matrix_ul(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, fracture_3d<Mesh> & f, const size_t & cell_ind, size_t di = 0)
        {
            const auto degree     = m_hho_di.face_degree();
            
            auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, degree);
            auto div_s_basis = disk::make_scalar_monomial_basis(msh, face, degree);
            if(f.m_flips_l.at(cell_ind)){
                div_s_basis.swap_nodes();
            }
            
            size_t n_s_basis = 3;
            Matrix<T, Dynamic, Dynamic> ret_n = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
            
            Matrix<T, Dynamic, Dynamic> ret_t = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
            
            const auto qps = integrate(msh, face, 2 * (degree + 2 + di));
            const auto n = disk::normal(msh, cell, face);
            const auto t = disk::tanget(msh, cell, face).first;
            assert(false);

            for (auto& qp : qps)
            {
//                const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
//                const auto s_f_phi = div_s_basis.eval_div_flux_functions(qp.point());
//
//                const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
//                const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
//
//                const auto s_n_opt = disk::priv::outer_product(s_f_phi, w_n_dot_u_f_phi);
//                const auto s_t_opt = disk::priv::outer_product(s_f_phi, w_t_dot_u_f_phi);
//
//                ret_n += s_n_opt;
//                ret_t += s_t_opt;
            }

            return std::make_pair(ret_n, ret_t);
        }
    
    auto skin_coupling_matrix_ur(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, fracture_3d<Mesh> & f, const size_t & cell_ind, size_t di = 0)
        {
            const auto degree     = m_hho_di.face_degree();
            
            auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, degree);
            auto div_s_basis = disk::make_scalar_monomial_basis(msh, face, degree);
            if(f.m_flips_r.at(cell_ind)){
                div_s_basis.swap_nodes();
            }
            
            size_t n_s_basis = 3;
            Matrix<T, Dynamic, Dynamic> ret_n = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
            
            Matrix<T, Dynamic, Dynamic> ret_t = Matrix<T, Dynamic, Dynamic>::Zero(n_s_basis, vec_u_basis.size());
            
            const auto qps = integrate(msh, face, 2 * (degree + 2 + di));
            const auto n = disk::normal(msh, cell, face);
            const auto t = disk::tanget(msh, cell, face).first;
            assert(false);

            for (auto& qp : qps)
            {
//                const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
//                const auto s_f_phi = div_s_basis.eval_div_flux_functions(qp.point());
//
//                const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
//                const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
//
//                const auto s_n_opt = disk::priv::outer_product(s_f_phi, w_n_dot_u_f_phi);
//                const auto s_t_opt = disk::priv::outer_product(s_f_phi, w_t_dot_u_f_phi);
//
//                ret_n += s_n_opt;
//                ret_t += s_t_opt;
            }

            return std::make_pair(ret_n, ret_t);
        }
    
    auto skin_coupling_rhs_u(const Mesh& msh, const typename Mesh::cell_type& cell, const typename Mesh::face_type& face, size_t di = 0)
        {
            const auto degree     = m_hho_di.face_degree();
            auto vec_u_basis = disk::make_vector_monomial_basis(msh, face, degree);
            Matrix<T, Dynamic, Dynamic> ret_n = Matrix<T, Dynamic, Dynamic>::Zero(vec_u_basis.size(),1);
            
            Matrix<T, Dynamic, Dynamic> ret_t = Matrix<T, Dynamic, Dynamic>::Zero(vec_u_basis.size(),1);
            
            const auto qps = integrate(msh, face, 2 * (degree+di));
            const auto n = disk::normal(msh, cell, face);
            const auto t = disk::tanget(msh, cell, face);

            T c_n = 1.0;
            T c_t = 1.0;
            for (auto& qp : qps)
            {
//                const auto u_f_phi = vec_u_basis.eval_functions(qp.point());
//
//                const auto w_n_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), n));
//                const auto w_t_dot_u_f_phi = disk::priv::inner_product(u_f_phi,disk::priv::inner_product(qp.weight(), t));
//
//                ret_n += c_n*w_n_dot_u_f_phi;
//                ret_t += c_t*w_t_dot_u_f_phi;
            }

            return std::make_pair(ret_n, ret_t);
        }
    
    
    
    auto inplane_normal(const Mesh& msh, const typename Mesh::face_type& face, typename Mesh::point_type &point, typename Mesh::point_type & v){
        // line normal
        auto bar = barycenter(msh,face);
        auto nd = (point - bar).to_vector();
        auto vt = (v).to_vector();
        auto vd = vt/vt.norm();
        auto np = nd/nd.norm();
        auto npv = np.dot(vd);
        return npv;
    }
    
    void skin_connected_cells(const Mesh& msh){
       
        auto storage = msh.backend_storage();
                
        std::set<size_t> set_l, set_r;
        for (auto chunk : m_fracture_pairs) {
            set_l.insert(chunk.first);
            set_r.insert(chunk.second);
        }
        

        std::map<size_t,size_t> node_map_l;
        std::vector<size_t> frac_indexes_l;
        frac_indexes_l.reserve(set_l.size());
        {   // build connectivity map on left side
            size_t index = m_end_point_mortars[0].first;
            size_t node_index_b = m_end_point_mortars[0].second;
            size_t node_index_e = m_end_point_mortars[1].second;
            size_t node_index = node_index_b;
            
            size_t node_c = 1;
            node_map_l[node_index] = node_c;
            node_c++;
            while (node_index_e != node_index) {
                for (auto id : set_l) {
                    auto& face = storage->edges[id];
                    auto points = face.point_ids();
                    bool check_Q = points[0] == node_index || points[1] == node_index;
                    if (check_Q) {
                        frac_indexes_l.push_back(id);
                        set_l.erase(id);
                        if (points[0] == node_index) {
                            node_index = points[1];
                        }else{
                            node_index = points[0];
                        }
                        node_map_l[node_index] = node_c;
                        node_c++;
                        break;
                    }
                }
            }
        }
        
        std::map<size_t,size_t> node_map_r;
        std::vector<size_t> frac_indexes_r;
        frac_indexes_r.reserve(set_r.size());
        {   // build connectivity map on right side
            size_t index = m_end_point_mortars[0].first;
            size_t node_index_b = m_end_point_mortars[0].second;
            size_t node_index_e = m_end_point_mortars[1].second;
            size_t node_index = node_index_b;
            
            size_t node_c = 1;
            node_map_r[node_index] = node_c;
            node_c++;
            while (node_index_e != node_index) {
                for (auto id : set_r) {
                    auto& face = storage->edges[id];
                    auto points = face.point_ids();
                    bool check_Q = points[0] == node_index || points[1] == node_index;
                    if (check_Q) {
                        frac_indexes_r.push_back(id);
                        set_r.erase(id);
                        if (points[0] == node_index) {
                            node_index = points[1];
                        }else{
                            node_index = points[0];
                        }
                        node_map_r[node_index] = node_c;
                        node_c++;
                        break;
                    }
                }
            }
        }
        
        m_dof_dest_l.reserve(m_fracture_pairs.size());
        m_dof_dest_r.reserve(m_fracture_pairs.size());
        m_flip_dest_l.reserve(m_fracture_pairs.size());
        m_flip_dest_r.reserve(m_fracture_pairs.size());
        
        auto dest_index = [](const size_t & id) -> size_t {
            size_t dest = 2 * (id - 1);
            return dest;
        };
        
        // renumbering fracture pairs
        m_fracture_pairs.clear();
        assert(frac_indexes_l.size()==frac_indexes_r.size());
        for (size_t i = 0; i < frac_indexes_l.size(); i++) {
            m_fracture_pairs.push_back(std::make_pair(frac_indexes_l[i], frac_indexes_r[i]));
        }
        
        for (auto chunk : m_fracture_pairs) {
            
            auto& face_l = storage->edges[chunk.first];
            auto& face_r = storage->edges[chunk.second];
            
            auto points_l = face_l.point_ids();
            auto points_r = face_r.point_ids();
            
            auto& point0 = storage->points[points_l[0]];
            auto& point1 = storage->points[points_l[1]];
            
            if (node_map_l[points_l[0]] < node_map_l[points_l[1]]) {
                size_t id = node_map_l[points_l[0]];
                m_dof_dest_l.push_back(dest_index(id));
                m_flip_dest_l.push_back(false);
            }else{
                size_t id = node_map_l[points_l[1]];
                m_dof_dest_l.push_back(dest_index(id));
                m_flip_dest_l.push_back(true);
            }
            
            if (node_map_r[points_r[0]] < node_map_r[points_r[1]]) {
                size_t id = node_map_r[points_r[0]];
                m_dof_dest_r.push_back(dest_index(id));
                m_flip_dest_r.push_back(false);
            }else{
                size_t id = node_map_r[points_r[1]];
                m_dof_dest_r.push_back(dest_index(id));
                m_flip_dest_r.push_back(true);
            }
            
        }

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
            
    void project_over_cells(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<static_vector<T, 3>(const typename Mesh::point_type& )> vec_fun, std::function<static_matrix<T, 3,3>(const typename Mesh::point_type& )> ten_fun){
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
                      std::function<static_matrix<T, 3,3>(const typename Mesh::point_type& )> ten_fun){
    
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
            static_matrix<T, 3,3> sigma = ten_fun(qp.point());
            for (size_t i = 0; i < ten_bs; i++){
                auto qp_phi_i = disk::priv::inner_product(qp.weight(), phi[i]);
                rhs(i,0) += disk::priv::inner_product(qp_phi_i,sigma);
            }
        }
        Matrix<T, Dynamic, 1> x_dof = mass_matrix.llt().solve(rhs);
        return x_dof;
    }
            
    void project_over_faces(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob, std::function<static_vector<double, 3>(const typename Mesh::point_type& )> vec_fun){

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
    
    void cells_residuals(const Mesh& msh, std::function<static_vector<T, 3>(const typename Mesh::point_type& )> vec_fun, std::function<static_matrix<T, 3,3>(const typename Mesh::point_type& )> ten_fun){

        size_t n_ten_cbs = disk::sym_matrix_basis_size(m_hho_di.grad_degree(), Mesh::dimension, Mesh::dimension);
        size_t n_vec_cbs = disk::vector_basis_size(m_hho_di.cell_degree(),Mesh::dimension, Mesh::dimension);
        size_t n_cbs = n_ten_cbs + n_vec_cbs;
        size_t n_fbs = disk::vector_basis_size(m_hho_di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        
        size_t cell_ind = 0;
        for (auto& cell : msh)
        {
            auto fcs = faces(msh, cell);
            size_t n_bs = n_cbs + n_fbs * fcs.size();
            Matrix<T, Dynamic, 1> x_proj_dof = Matrix<T, Dynamic, 1>::Zero(n_bs);
            
            Matrix<T, Dynamic, 1> x_proj_ten_dof = project_ten_function(msh, cell, ten_fun);
            Matrix<T, Dynamic, 1> x_proj_vec_dof = project_function(msh, cell, m_hho_di.cell_degree(), vec_fun);
            
            x_proj_dof.block(0, 0, n_ten_cbs, 1)                    = x_proj_ten_dof;
            x_proj_dof.block(n_ten_cbs, 0, n_vec_cbs, 1)  = x_proj_vec_dof;
            
            for (size_t i = 0; i < fcs.size(); i++)
            {
                auto face = fcs[i];
                auto fc_id = msh.lookup(face);
                Matrix<T, Dynamic, 1> x_proj_vec_dof = project_function(msh, face, m_hho_di.face_degree(), vec_fun);
                x_proj_dof.block(n_ten_cbs + n_vec_cbs + i * n_fbs, 0, n_fbs, 1)  = x_proj_vec_dof;
            }
            
            Matrix<T, Dynamic, Dynamic> mixed_operator_loc = mixed_operator(cell_ind,msh,cell);
            Matrix<T, Dynamic, 1> r_loc = mixed_operator_loc*x_proj_dof;
            std::cout << "cell res norm = "<< r_loc.head(n_ten_cbs+n_vec_cbs).norm() << std::endl;
            std::cout << "res sum = "<< r_loc.sum() << std::endl;
            
            cell_ind++;
        }
    }
    
    void project_over_skin_cells(const Mesh& msh, Matrix<T, Dynamic, 1> & x_glob){
        
        auto storage = msh.backend_storage();
        
//        size_t f_ind = 0;
//        for (auto f : m_fractures) {
//            size_t cell_ind = 0;
//            for (auto chunk : f.m_pairs) {
//                
//                size_t cell_ind_l = f.m_elements[cell_ind].first;
//                size_t cell_ind_r = f.m_elements[cell_ind].second;
//                auto& face_l = storage->edges[chunk.first];
//                auto& face_r = storage->edges[chunk.second];
//                auto& cell_l = storage->surfaces[cell_ind_l];
//                auto& cell_r = storage->surfaces[cell_ind_r];
//                
//                
//                // mass matrix
//                auto mass_matrix = skin_mass_matrix(msh, face_l, face_r, f, cell_ind, 2);
//                auto rhs = skin_rhs(msh, face_l, face_r, f, cell_ind, 2);
//                
//
//                Matrix<T, Dynamic, 1> x_l = mass_matrix.first.llt().solve(rhs.first);
//                Matrix<T, Dynamic, 1> x_r = mass_matrix.second.llt().solve(rhs.second);
//                
////                std::cout << "xl = " << x_l << std::endl;
////                std::cout << "xr = " << x_r << std::endl;
//                
//                if(0){
//                    auto bar = barycenter(msh,face_l);
//                    auto face_basis_l = make_scalar_monomial_basis(msh, face_l, m_hho_di.face_degree());
//                    auto face_basis_r = make_scalar_monomial_basis(msh, face_r, m_hho_di.face_degree());
//                    if (f.m_flips_l.at(cell_ind)) {
//                        face_basis_l.swap_nodes();
//                    }
//                    if (f.m_flips_r.at(cell_ind)) {
//                        face_basis_r.swap_nodes();
//                    }
//                    
//                    auto t_phi_l = face_basis_l.eval_flux_functions( bar );
//                    auto t_phi_r = face_basis_r.eval_flux_functions( bar );
//                    
//                    auto s_l = disk::eval(x_l, t_phi_l);
//                    auto s_r = disk::eval(x_r, t_phi_r);
//                    int aka = 0;
//                }
//
//                cell_ind++;
//            }
//                    
//            f_ind++;
//        }
        
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
        
        const auto qps = integrate(msh, cell, 2 * graddeg + 1);

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
    mixed_rhs(const Mesh& msh, const typename Mesh::cell_type& cell, std::function<static_vector<double, 3>(const typename Mesh::point_type& )> & rhs_fun, size_t di = 0)
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
    
    size_t get_n_f_hybrid_dofs(){
        return m_n_f_hybrid_dof;
    }
    
    size_t get_n_skin_dof(){
        return m_n_skin_dof;
    }

    std::vector<size_t> & compress_indexes(){
        return m_compress_indexes;
    }
    
    std::vector<size_t> & compress_fracture_indexes(){
        return m_compress_fracture_indexes;
    }
    
    std::vector<size_t> & compress_hybrid_indexes(){
        return m_compress_hybrid_indexes;
    }

    std::vector<size_t> & dof_dest_l(){
        return m_dof_dest_l;
    }
    
    std::vector<size_t> & dof_dest_r(){
        return m_dof_dest_r;
    }
    
    std::vector<bool> & flip_dest_l(){
        return m_flip_dest_l;
    }
    
    std::vector<bool> & flip_dest_r(){
        return m_flip_dest_r;
    }
    
    std::vector<std::pair<size_t,size_t>> & elements_with_fractures_eges(){
        return m_elements_with_fractures_eges;
    }
    
    std::vector<std::pair<size_t,size_t>> & fracture_pairs(){
        return m_fracture_pairs;
    }
    
};

#endif /* elastic_two_fields_assembler_3d_hpp */
