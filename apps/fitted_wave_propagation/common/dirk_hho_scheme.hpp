//
//  dirk_hho_scheme.hpp
//  acoustics
//
//  Created by Omar Durán on 4/21/20.
//

#ifndef dirk_hho_scheme_hpp
#define dirk_hho_scheme_hpp

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

class dirk_hho_scheme
{
    
    private:

    double m_scale;
    SparseMatrix<double> m_Mg;
    SparseMatrix<double> m_Kg;
    Matrix<double, Dynamic, 1> m_Fg;
#ifdef HAVE_INTEL_MKL_FADE
    PardisoLU<Eigen::SparseMatrix<double>>  m_analysis;
#else
    SparseLU<Eigen::SparseMatrix<double>>   m_analysis;
#endif
    
    public:
    
    dirk_hho_scheme(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg){
        
        m_Mg = Mg;
        m_Kg = Kg;
        m_Fg = Fg;
        m_scale = 0.0;
    }
    
    dirk_hho_scheme(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg, double scale){
        
        m_Mg = Mg;
        m_Kg = Kg;
        m_Fg = Fg;
        m_scale = scale;
        DecomposeMatrix();
    }
    
    void DecomposeMatrix(){
        SparseMatrix<double> K = m_Mg + m_scale * m_Kg;
        m_analysis.analyzePattern(K);
        m_analysis.factorize(K);
    }
    

    #ifdef HAVE_INTEL_MKL_FADE
        PardisoLU<Eigen::SparseMatrix<double>> & DirkAnalysis(){
            return m_analysis;
        }
    #else
        SparseLU<SparseMatrix<double>> & DirkAnalysis(){
            return m_analysis;
        }
    #endif
    
    SparseMatrix<double> & Mg(){
        return m_Mg;
    }
    
    SparseMatrix<double> & Kg(){
        return m_Kg;
    }
    
    Matrix<double, Dynamic, 1> & Fg(){
        return m_Fg;
    }
    
    void SetScale(double & scale){
        m_scale = scale;
    }
    
    void SetFg(Matrix<double, Dynamic, 1> & Fg){
        m_Fg = Fg;
    }
    
    void irk_weight(Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a, bool is_sdirk_Q){
    
        Matrix<double, Dynamic, 1> Fg = this->Fg();
        Fg -= Kg()*y;
        
        if (is_sdirk_Q) {
            k = DirkAnalysis().solve(Fg);
        }else{
            double scale = a * dt;
            SetScale(scale);
            DecomposeMatrix();
            k = DirkAnalysis().solve(Fg);
        }
    }
    
};

#endif /* dirk_hho_scheme_hpp */
