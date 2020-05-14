//
//  linear_solver.hpp
//  acoustics
//
//  Created by Omar Durán on 5/2/20.
//

#pragma once
#ifndef linear_solver_hpp
#define linear_solver_hpp


#ifdef HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#endif

template<typename T>
class linear_solver
{
    
    private:

    SparseMatrix<T> m_Kcc;
    SparseMatrix<T> m_Kcf;
    SparseMatrix<T> m_Kfc;
    SparseMatrix<T> m_Kff;
    Matrix<T, Dynamic, 1> m_Fc;
    Matrix<T, Dynamic, 1> m_Ff;
    
    SparseMatrix<T> m_Kcc_inv;
    SparseMatrix<T> m_K;
    Matrix<T, Dynamic, 1> m_F;
    
    #ifdef HAVE_INTEL_MKL
        PardisoLU<Eigen::SparseMatrix<T>>  m_analysis;
    #else
        SparseLU<SparseMatrix<T>> m_analysis;
    #endif

    size_t m_n_c_dof;
    size_t m_n_f_dof;
    bool m_global_sc_Q;
    bool m_is_decomposed_Q;
    
    
    void DecomposeK(){
        m_analysis.analyzePattern(m_K);
        m_analysis.factorize(m_K);
    }
    
    Matrix<T, Dynamic, 1> solve_global(Matrix<T, Dynamic, 1> & Fg){
        Matrix<T, Dynamic, 1> x_dof = m_analysis.solve(Fg);
        return x_dof;
    }
    
    void scatter_segments(Matrix<T, Dynamic, 1> & Fg){
        assert(m_n_c_dof + m_n_f_dof == Fg.rows());
        m_Fc = Fg.block(0, 0, m_n_c_dof, 1);
        m_Ff = Fg.block(m_n_c_dof, 0, m_n_f_dof, 1);
    }
    
    Matrix<T, Dynamic, 1> solve_sc(Matrix<T, Dynamic, 1> & Fg){
        
        scatter_segments(Fg);
        Matrix<T, Dynamic, 1> delta_c = m_Kcc_inv*m_Fc;
        m_F = m_Ff - m_Kfc*delta_c;
        Matrix<T, Dynamic, 1> x_n_f_dof = m_analysis.solve(m_F);
        
        Matrix<T, Dynamic, 1> Kcf_x_f_dof = m_Kcf*x_n_f_dof;
        Matrix<T, Dynamic, 1> delta_f = m_Kcc_inv*Kcf_x_f_dof;
        Matrix<T, Dynamic, 1> x_n_c_dof = delta_c - delta_f;
        
        // Composing global solution
        Matrix<T, Dynamic, 1> x_dof = Matrix<T, Dynamic, 1>::Zero(m_n_c_dof+m_n_f_dof,1);
        x_dof.block(0, 0, m_n_c_dof, 1) = x_n_c_dof;
        x_dof.block(m_n_c_dof, 0, m_n_f_dof, 1) = x_n_f_dof;
        return x_dof;

    }
    
    void scatter_blocks(SparseMatrix<T> & Kg){
        
        m_n_c_dof = Kg.rows() - m_n_f_dof;
        
        // scattering matrix blocks
        m_Kcc = Kg.block(0, 0, m_n_c_dof, m_n_c_dof);
        m_Kcf = Kg.block(0, m_n_c_dof, m_n_c_dof, m_n_f_dof);
        m_Kfc = Kg.block(m_n_c_dof,0, m_n_f_dof, m_n_c_dof);
        m_Kff = Kg.block(m_n_c_dof,m_n_c_dof, m_n_f_dof, m_n_f_dof);
        m_is_decomposed_Q = false;
    }
    
    public:
    
    linear_solver() : m_global_sc_Q(false), m_is_decomposed_Q(false)  {

    }
    
    linear_solver(SparseMatrix<T> & Kg) : m_K(Kg), m_global_sc_Q(false), m_is_decomposed_Q(false)  {

    }
    
    linear_solver(SparseMatrix<T> & Kg, size_t n_f_dof) :
        m_n_f_dof(n_f_dof),
        m_global_sc_Q(true),
        m_is_decomposed_Q(false) {
        scatter_blocks(Kg);
    }
    
    void SetKg(SparseMatrix<T> & Kg){
        m_global_sc_Q = false;
        m_K = Kg;
    }

    void SetKg(SparseMatrix<T> & Kg, size_t n_f_dof){
        m_global_sc_Q = true;
        m_n_f_dof = n_f_dof;
        scatter_blocks(Kg);
    }
    
    SparseMatrix<T> & Kcc(){
        return m_Kcc;
    }
    
    SparseMatrix<T> & Kcf(){
        return m_Kcf;
    }
    
    SparseMatrix<T> & Kfc(){
        return m_Kfc;
    }

    Matrix<T, Dynamic, 1> & Fc(){
        return m_Fc;
    }
    
    void condense_equations(std::pair<size_t,size_t> cell_basis_data){
        
        if (!m_global_sc_Q) {
            return;
        }
        
        size_t n_cells = cell_basis_data.first;
        size_t n_cbs   = cell_basis_data.second;
        size_t nnz_cc = n_cbs*n_cbs*n_cells;
        std::vector< Triplet<T> > triplets_cc;
        triplets_cc.resize(nnz_cc);
        m_Kcc_inv = SparseMatrix<T>( m_n_c_dof, m_n_c_dof );
        #ifdef HAVE_INTEL_TBB
                tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                    [this,&triplets_cc,&n_cbs] (size_t & cell_ind){
                    
                    size_t stride_eq = cell_ind * n_cbs;
                    size_t stride_l = cell_ind * n_cbs * n_cbs;

                    SparseMatrix<T> K_cc_loc = m_Kcc.block(stride_eq, stride_eq, n_cbs, n_cbs);
                    SparseLU<SparseMatrix<T>> analysis_cc;
                    analysis_cc.analyzePattern(K_cc_loc);
                    analysis_cc.factorize(K_cc_loc);
                    Matrix<T, Dynamic, Dynamic> K_cc_inv_loc = analysis_cc.solve(Matrix<T, Dynamic, Dynamic>::Identity(n_cbs, n_cbs));
            
                    size_t l = 0;
                    for (size_t i = 0; i < K_cc_inv_loc.rows(); i++)
                    {
                        for (size_t j = 0; j < K_cc_inv_loc.cols(); j++)
                        {
                            triplets_cc[stride_l+l] = Triplet<T>(stride_eq+i, stride_eq+j, K_cc_inv_loc(i,j));
                            l++;
                        }
                    }
                }
            );
        #else

            for (size_t cell_ind = 0; cell_ind < n_cells; cell_ind++)
            {
                size_t stride_eq = cell_ind * n_cbs;
                size_t stride_l = cell_ind * n_cbs * n_cbs;
                
                SparseMatrix<T> K_cc_loc = m_Kcc.block(stride_eq, stride_eq, n_cbs, n_cbs);
                SparseLU<SparseMatrix<T>> analysis_cc;
                analysis_cc.analyzePattern(K_cc_loc);
                analysis_cc.factorize(K_cc_loc);
                Matrix<T, Dynamic, Dynamic> K_cc_inv_loc = analysis_cc.solve(Matrix<T, Dynamic, Dynamic>::Identity(n_cbs, n_cbs));
        
                size_t l = 0;
                for (size_t i = 0; i < K_cc_inv_loc.rows(); i++)
                {
                    for (size_t j = 0; j < K_cc_inv_loc.cols(); j++)
                    {
                        triplets_cc[stride_l+l] = Triplet<T>(stride_eq+i, stride_eq+j, K_cc_inv_loc(i,j));
                        l++;
                    }
                }

            }
        #endif
        
        m_Kcc_inv.setFromTriplets( triplets_cc.begin(), triplets_cc.end() );
        triplets_cc.clear();
        m_K = m_Kff - m_Kfc*m_Kcc_inv*m_Kcf;
        m_is_decomposed_Q = false;
        return;

    }
        
    void factorize(){
        if (m_is_decomposed_Q) {
            return;
        }
        DecomposeK();
        m_is_decomposed_Q = true;
    }
        
    Matrix<T, Dynamic, 1> solve(Matrix<T, Dynamic, 1> & Fg){
        if (m_global_sc_Q) {
            return solve_sc(Fg);
        }else{
            return solve_global(Fg);
        }
    }
    
    
    size_t n_equations(){
        size_t n_equations = m_K.rows();
        return n_equations;
    }
        
};

#endif /* linear_solver_hpp */
