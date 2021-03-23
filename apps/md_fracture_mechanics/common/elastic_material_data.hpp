//
//  elastic_material_data.hpp
//  acoustics
//
//  Created by Omar Durán on 22/03/21.
//

#ifndef elastic_material_data_hpp
#define elastic_material_data_hpp

#include <stdio.h>

template<typename T = double>
class elastic_material_data {
        
    /// Fluid density
    T m_rho;
    
    /// First lame parameter
    T m_l;
    
    /// Second lame parameter
    T m_mu;
    
public:
    
    /// Default constructor
    elastic_material_data(T rho, T l, T mu){
        m_rho = rho;
        m_l = l;
        m_mu = mu;
    }
    
    /// Copy constructor
    elastic_material_data(const elastic_material_data & other){
        m_rho       = other.m_rho;
        m_l         = other.m_l;
        m_mu        = other.m_mu;
    }
    
    /// Assignement constructor
    const elastic_material_data & operator=(const elastic_material_data & other){
        
        // check for self-assignment
        if(&other == this){
            return *this;
        }
        
        m_rho       = other.m_rho;
        m_l         = other.m_l;
        m_mu        = other.m_mu;
        return *this;
        
    }
    
    /// Desconstructor
    virtual ~elastic_material_data(){
        
    }
    
    /// Print class attributes
    virtual void Print(std::ostream &out = std::cout) const{
        out << "\n density = " << m_rho;
        out << "\n first lame paremeter = " << m_l;
        out << "\n second lame paremeter = " << m_mu;
    }
    
    /// Print class attributes
    friend std::ostream & operator<<( std::ostream& out, const elastic_material_data & material ){
        material.Print(out);
        return out;
    }
    
    void Set_rho(T rho)
    {
        m_rho = rho;
    }
    
    T rho()
    {
        return m_rho;
    }
    
    
    void Set_l(T l)
    {
        m_l = l;
    }
    
    T l()
    {
        return m_l;
    }
    
    void Set_mu(T mu)
    {
        m_mu = mu;
    }
    
    T mu()
    {
        return m_mu;
    }
    
};

#endif /* elastic_material_data_hpp */
