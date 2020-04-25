//
//  ssprk_hho_scheme.hpp
//  acoustics
//
//  Created by Omar Durán on 4/21/20.
//

#pragma once
#ifndef ssprk_hho_scheme_hpp
#define ssprk_hho_scheme_hpp

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

class ssprk_hho_scheme
{
    private:

    SparseMatrix<double> m_Mc;
    SparseMatrix<double> m_Kc;
    SparseMatrix<double> m_Kcf;
    SparseMatrix<double> m_Kfc;
    SparseMatrix<double> m_Sff;
    Matrix<double, Dynamic, 1> m_Fc;
    SparseLU<SparseMatrix<double>> m_analysis_c;
    SparseLU<SparseMatrix<double>> m_analysis_f;
    size_t m_n_c_dof;
    size_t m_n_f_dof;
    
    public:
    
    ssprk_hho_scheme(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg, size_t n_f_dof){
        
        
        m_n_c_dof = Kg.rows() - n_f_dof;
        m_n_f_dof = n_f_dof;
        
        // Building blocks
        m_Mc = Mg.block(0, 0, m_n_c_dof, m_n_c_dof);
        m_Kc = Kg.block(0, 0, m_n_c_dof, m_n_c_dof);
        m_Kcf = Kg.block(0, m_n_c_dof, m_n_c_dof, n_f_dof);
        m_Kfc = Kg.block(m_n_c_dof,0, n_f_dof, m_n_c_dof);
        m_Sff = Kg.block(m_n_c_dof,m_n_c_dof, n_f_dof, n_f_dof);
        m_Fc = Fg.block(0, 0, m_n_c_dof, 1);
        DecomposeMassTerm();
        DecomposeFaceTerm();
    }
    
    void DecomposeMassTerm(){
        m_analysis_c.analyzePattern(m_Mc);
        m_analysis_c.factorize(m_Mc);
    }
    
    void DecomposeFaceTerm(){
        m_analysis_f.analyzePattern(m_Sff);
        m_analysis_f.factorize(m_Sff);
    }
    
    SparseLU<SparseMatrix<double>> & CellsAnalysis(){
        return m_analysis_c;
    }
    
    SparseLU<SparseMatrix<double>> & FacesAnalysis(){
        return m_analysis_f;
    }
    
    SparseMatrix<double> & Mc(){
        return m_Mc;
    }

    SparseMatrix<double> & Kc(){
        return m_Kc;
    }
    
    SparseMatrix<double> & Kcf(){
        return m_Kcf;
    }
    
    SparseMatrix<double> & Kfc(){
        return m_Kfc;
    }
    
    SparseMatrix<double> & Sff(){
        return m_Sff;
    }
    
    Matrix<double, Dynamic, 1> & Fc(){
        return m_Fc;
    }
    
    void explicit_rk_weight(Matrix<double, Dynamic, 1> & x_dof, Matrix<double, Dynamic, 1> & x_dof_n, double dt, double a, double b){
        
        Matrix<double, Dynamic, 1> x_c_dof = x_dof.block(0, 0, m_n_c_dof, 1);
        Matrix<double, Dynamic, 1> x_f_dof = x_dof.block(m_n_c_dof, 0, m_n_f_dof, 1);
    
        // Faces update (last state)
        {
            Matrix<double, Dynamic, 1> RHSf = Kfc()*x_c_dof;
            x_f_dof = -FacesAnalysis().solve(RHSf);
        }
    
        // Cells update
        Matrix<double, Dynamic, 1> RHSc = Fc() - Kc()*x_c_dof - Kcf()*x_f_dof;
        Matrix<double, Dynamic, 1> delta_x_c_dof = CellsAnalysis().solve(RHSc);
        Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
    
        // Faces update
        Matrix<double, Dynamic, 1> RHSf = Kfc()*x_n_c_dof;
        Matrix<double, Dynamic, 1> x_n_f_dof = -FacesAnalysis().solve(RHSf); // new state
    
        // Composing global solution
        x_dof_n = x_dof;
        x_dof_n.block(0, 0, m_n_c_dof, 1) = x_n_c_dof;
        x_dof_n.block(m_n_c_dof, 0, m_n_f_dof, 1) = x_n_f_dof;
    
    }
    
};

#endif /* ssprk_hho_scheme_hpp */
