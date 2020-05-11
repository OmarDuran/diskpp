//
//  erk_hho_scheme.hpp
//  acoustics
//
//  Created by Omar Durán on 5/8/20.
//

#ifndef erk_hho_scheme_hpp
#define erk_hho_scheme_hpp

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

class erk_hho_scheme
{
    private:

    SparseMatrix<double> m_Mc;
    SparseMatrix<double> m_Kcc;
    SparseMatrix<double> m_Kcf;
    SparseMatrix<double> m_Kfc;
    SparseMatrix<double> m_Sff;
    Matrix<double, Dynamic, 1> m_Fc;

    #ifdef HAVE_INTEL_MKL2
        PardisoLU<Eigen::SparseMatrix<double>>  m_analysis_c;
        PardisoLU<Eigen::SparseMatrix<double>>  m_analysis_f;
    #else
        SparseLU<SparseMatrix<double>> m_analysis_c;
        SparseLU<SparseMatrix<double>> m_analysis_f;
    #endif
    size_t m_n_c_dof;
    size_t m_n_f_dof;
    
    public:
    
    erk_hho_scheme(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, SparseMatrix<double> & Mg, size_t n_f_dof){
        
        
        m_n_c_dof = Kg.rows() - n_f_dof;
        m_n_f_dof = n_f_dof;
        
        // Building blocks
        m_Mc = Mg.block(0, 0, m_n_c_dof, m_n_c_dof);
        m_Kcc = Kg.block(0, 0, m_n_c_dof, m_n_c_dof);
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
    

    #ifdef HAVE_INTEL_MKL2
        PardisoLU<Eigen::SparseMatrix<double>> & CellsAnalysis(){
            return m_analysis_c;
        }
        
        PardisoLU<Eigen::SparseMatrix<double>> & FacesAnalysis(){
            return m_analysis_f;
        }
    #else
        SparseLU<SparseMatrix<double>> & CellsAnalysis(){
            return m_analysis_c;
        }
        
        SparseLU<SparseMatrix<double>> & FacesAnalysis(){
            return m_analysis_f;
        }
    #endif
    
    SparseMatrix<double> & Mc(){
        return m_Mc;
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
    
    SparseMatrix<double> & Sff(){
        return m_Sff;
    }
    
    Matrix<double, Dynamic, 1> & Fc(){
        return m_Fc;
    }
    
    void SetFg(Matrix<double, Dynamic, 1> & Fg){
        m_Fc = Fg.block(0, 0, m_n_c_dof, 1);
    }
    
    void erk_weight(Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k){
        
        Matrix<double, Dynamic, 1> y_c_dof = y.block(0, 0, m_n_c_dof, 1);
        Matrix<double, Dynamic, 1> y_f_dof = y.block(m_n_c_dof, 0, m_n_f_dof, 1);
    
        // Faces update (last state)
        {
            Matrix<double, Dynamic, 1> RHSf = Kfc()*y_c_dof;
            y_f_dof = -FacesAnalysis().solve(RHSf);
        }
        
        // Cells update
        Matrix<double, Dynamic, 1> RHSc = Fc() - Kcc()*y_c_dof - Kcf()*y_f_dof;
        Matrix<double, Dynamic, 1> delta_y_c_dof = CellsAnalysis().solve(RHSc);
        Matrix<double, Dynamic, 1> k_c_dof = delta_y_c_dof; // new state
    
        // Faces update
        Matrix<double, Dynamic, 1> RHSf = Kfc()*k_c_dof;
        Matrix<double, Dynamic, 1> k_f_dof = -FacesAnalysis().solve(RHSf); // new state
    
        // Composing the rk weight
        k = y;
        k.block(0, 0, m_n_c_dof, 1) = k_c_dof;
        k.block(m_n_c_dof, 0, m_n_f_dof, 1) = k_f_dof;
    
    }
    
};

#endif /* erk_hho_scheme_hpp */
