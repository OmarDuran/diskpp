//
//  linear_solver.hpp
//  acoustics
//
//  Created by Omar Durán on 5/2/20.
//

#pragma once
#ifndef linear_solver_hpp
#define linear_solver_hpp

#include <stdio.h>

class linear_solver
{
    private:

    SparseMatrix<double> m_Kcc;
    SparseMatrix<double> m_Kcf;
    SparseMatrix<double> m_Kfc;
    Matrix<double, Dynamic, 1> m_Fc;
    Matrix<double, Dynamic, 1> m_Ff;
    
    SparseMatrix<double> m_K;
    Matrix<double, Dynamic, 1> m_F;
    
    #ifdef HAVE_INTEL_MKL
        PardisoLU<Eigen::SparseMatrix<double>>  m_analysis_c;
        PardisoLU<Eigen::SparseMatrix<double>>  m_analysis;
    #else
        SparseLU<SparseMatrix<double>> m_analysis_c;
        SparseLU<SparseMatrix<double>> m_analysis;
    #endif

    size_t m_n_c_dof;
    size_t m_n_f_dof;
    bool m_static_condensation_Q;
    bool m_is_decomposed_Q;
    
    void DecomposeCellTerm(){
        m_analysis_c.analyzePattern(m_Kcc);
        m_analysis_c.factorize(m_Kcc);
    }
    
    void DecomposeK(){
        m_analysis.analyzePattern(m_K);
        m_analysis.factorize(m_K);
    }
    
    Matrix<double, Dynamic, 1> solve_global(Matrix<double, Dynamic, 1> & Fg){
        Matrix<double, Dynamic, 1> x_dof = m_analysis.solve(Fg);
        return x_dof;
    }
    
    void SetFg(Matrix<double, Dynamic, 1> & Fg){
        assert(m_n_c_dof + m_n_f_dof == Fg.rows());
        m_Fc = Fg.block(0, 0, m_n_c_dof, 1);
        m_Ff = Fg.block(m_n_c_dof, 0, m_n_f_dof, 1);
    }
    
    Matrix<double, Dynamic, 1> solve_sc(Matrix<double, Dynamic, 1> & Fg){
        
        SetFg(Fg);
        Matrix<double, Dynamic, 1> delta_c = m_analysis_c.solve(m_Fc);
        m_F = m_Ff - m_Kfc*delta_c;
        
        Matrix<double, Dynamic, 1> x_n_f_dof = m_analysis.solve(m_F);
        
        Matrix<double, Dynamic, 1> Kcf_x_f_dof = m_Kcf*x_n_f_dof;
        Matrix<double, Dynamic, 1> delta_f = m_analysis_c.solve(Kcf_x_f_dof);
        Matrix<double, Dynamic, 1> x_n_c_dof = delta_c - delta_f;
        
        // Composing global solution
        Matrix<double, Dynamic, 1> x_dof = Matrix<double, Dynamic, 1>::Zero(m_n_c_dof+m_n_f_dof,1);
        x_dof.block(0, 0, m_n_c_dof, 1) = x_n_c_dof;
        x_dof.block(m_n_c_dof, 0, m_n_f_dof, 1) = x_n_f_dof;
        return x_dof;
    }
    
    void factorize_sc(){
        if (m_is_decomposed_Q) {
            return;
        }
        DecomposeCellTerm();
        DecomposeK();
        m_is_decomposed_Q = true;
    }
    
    void factorize_global(){
        if (m_is_decomposed_Q) {
            return;
        }
        DecomposeK();
        m_is_decomposed_Q = true;
    }
    
    public:
    
    linear_solver(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg) : m_static_condensation_Q(false), m_is_decomposed_Q(false) {
        SetK(Kg);
    }
    
    linear_solver(SparseMatrix<double> & K, SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, size_t n_f_dof) : m_static_condensation_Q(true), m_is_decomposed_Q(false) {
        SetK(K);
        SetKg(Kg,n_f_dof);
        SetFg(Fg);
    }
        
    SparseMatrix<double> & Kcc(){
        return m_Kcc;
    }
    
    SparseMatrix<double> & Kcf(){
        return m_Kcf;
    }
    
    SparseMatrix<double> & Kfc(){
        return m_Kfc;
    }

    Matrix<double, Dynamic, 1> & Fc(){
        return m_Fc;
    }
    
    void SetKg(SparseMatrix<double> & Kg, size_t n_f_dof){
        m_n_c_dof = Kg.rows() - n_f_dof;
        m_n_f_dof = n_f_dof;
        
        // scattering matrix blocks
        m_Kcc = Kg.block(0, 0, m_n_c_dof, m_n_c_dof);
        m_Kcf = Kg.block(0, m_n_c_dof, m_n_c_dof, n_f_dof);
        m_Kfc = Kg.block(m_n_c_dof,0, n_f_dof, m_n_c_dof);
        m_is_decomposed_Q = false;
    }
    
    void SetK(SparseMatrix<double> & K){
        m_K = K;
        m_is_decomposed_Q = false;
    }
    
    void factorize(){
        if (m_static_condensation_Q) {
            return factorize_sc();
        }else{
            return factorize_global();
        }
    }
        
    Matrix<double, Dynamic, 1> solve(Matrix<double, Dynamic, 1> & Fg){
        if (m_static_condensation_Q) {
            return solve_sc(Fg);
        }else{
            return solve_global(Fg);
        }
    }
        
};

#endif /* linear_solver_hpp */
