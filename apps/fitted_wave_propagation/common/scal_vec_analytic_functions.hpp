//
//  scal_vec_analytic_functions.hpp
//  acoustics
//
//  Created by Omar Durán on 6/3/20.
//

#pragma once
#ifndef scal_vec_analytic_functions_hpp
#define scal_vec_analytic_functions_hpp

class scal_vec_analytic_functions
{
    public:
    
    /// Enumerate defining the function type
    enum EFunctionType { EFunctionNonPolynomial = 0, EFunctionQuadraticInTime = 1, EFunctionQuadraticInSpace = 2};
    
    
    scal_vec_analytic_functions(){
        m_function_type = EFunctionNonPolynomial;
    }
    
    ~scal_vec_analytic_functions(){
        
    }
    
    void set_function_type(EFunctionType function_type){
        m_function_type = function_type;
    }

    std::function<static_vector<double, 2>(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_u(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,ux,uy;
                            x = pt.x();
                            y = pt.y();
                            ux = x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y);
                            uy = x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y);
                            static_vector<double, 2> u{ux,uy};
                            return u;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,ux,uy;
                            x = pt.x();
                            y = pt.y();
                            ux = x*t*t*std::sin(M_PI*x)*std::sin(M_PI*y);
                            uy = x*t*t*std::sin(M_PI*x)*std::sin(M_PI*y);
                            static_vector<double, 2> u{ux,uy};
                            return u;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,ux,uy;
                            x = pt.x();
                            y = pt.y();
                            ux = (1 + x)*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                            uy = (1 + x)*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                            static_vector<double, 2> u{ux,uy};
                            return u;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                        static_vector<double, 2> u;
                        return u;
                    };
            }
                break;
        }
        
    }
    
    std::function<static_vector<double, 2>(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_v(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,vx,vy;
                            x = pt.x();
                            y = pt.y();
                            vx = -(std::sqrt(2.0)*M_PI*x*x*std::cos((M_PI*x)/2.)*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*y));
                            vy = -(std::sqrt(2.0)*M_PI*x*x*std::cos((M_PI*x)/2.)*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*y));
                            static_vector<double, 2> v{vx,vy};
                            return v;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,vx,vy;
                            x = pt.x();
                            y = pt.y();
                            vx = 2.0*t*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                            vy = 2.0*t*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                            static_vector<double, 2> v{vx,vy};
                            return v;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,vx,vy;
                            x = pt.x();
                            y = pt.y();
                            vx = std::sqrt(2.0)*M_PI*(1 + x)*x*x*(1 - y)*y*std::cos(std::sqrt(2.0)*M_PI*t);
                            vy = std::sqrt(2.0)*M_PI*(1 + x)*x*x*(1 - y)*y*std::cos(std::sqrt(2.0)*M_PI*t);
                            static_vector<double, 2> v{vx,vy};
                            return v;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                        double x,y;
                        x = pt.x();
                        y = pt.y();
                        static_vector<double, 2> v;
                        return v;
                    };
            }
                break;
        }
        
    }
    
    std::function<static_vector<double, 2>(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_a(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,ax,ay;
                             x = pt.x();
                             y = pt.y();
                             ax = -2*M_PI*M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y);
                             ay = -2*M_PI*M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y);
                             static_vector<double, 2> a{ax,ay};
                            return a;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,ax,ay;
                            x = pt.x();
                            y = pt.y();
                            ax = 2.0*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                            ay = 2.0*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                            static_vector<double, 2> a{ax,ay};
                            return a;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,ax,ay;
                            x = pt.x();
                            y = pt.y();
                            ax = -2*M_PI*M_PI*(1 + x)*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                            ay = -2*M_PI*M_PI*(1 + x)*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                            static_vector<double, 2> a{ax,ay};
                            return a;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                        static_vector<double, 2> f;
                        return f;
                    };
            }
                break;
        }
        
    }
    
    std::function<static_vector<double, 2>(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_f(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,fx,fy;
                            x = pt.x();
                            y = pt.y();
                            fx = -(std::cos(std::sqrt(2.0)*M_PI*t)*(-4*M_PI*x*std::sin((M_PI*x)/2.)*(M_PI*x*std::cos(M_PI*y) + 6*std::sin(M_PI*y)) + std::cos((M_PI*x)/2.)*(16*M_PI*x*std::cos(M_PI*y) + (24 + M_PI*M_PI*x*x)*std::sin(M_PI*y))))/4.0;
                            fy = (std::cos(std::sqrt(2.0)*M_PI*t)*(4*M_PI*x*std::sin((M_PI*x)/2.)*(M_PI*x*std::cos(M_PI*y) + 2*std::sin(M_PI*y)) + std::cos((M_PI*x)/2.)*(-16*M_PI*x*std::cos(M_PI*y) + (-8 + 5*M_PI*M_PI*x*x)*std::sin(M_PI*y))))/4.;
                            static_vector<double, 2> f{fx,fy};
                            return f;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,fx,fy;
                            x = pt.x();
                            y = pt.y();
                            fx = -2*(M_PI*t*t*std::cos(M_PI*x)*(M_PI*x*std::cos(M_PI*y) + 3*std::sin(M_PI*y)) +
                                 std::sin(M_PI*x)*(M_PI*t*t*std::cos(M_PI*y) - (1 + 2*M_PI*M_PI*t*t)*x*std::sin(M_PI*y)));
                            fy = (1 + M_PI*M_PI*t*t)*x*std::cos(M_PI*(x - y)) - (1 + 3*M_PI*M_PI*t*t)*x*std::cos(M_PI*(x + y)) - 2*M_PI*t*t*std::sin(M_PI*(x + y));
                            static_vector<double, 2> f{fx,fy};
                            return f;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                            double x,y,fx,fy;
                            x = pt.x();
                            y = pt.y();
                            fx = 2*(x*(-2 + (-2 + x)*x) - 3*y - x*(5 + x*(-6 + M_PI*M_PI*(1 + x)))*y + (3 + x*(9 + M_PI*M_PI*x*(1 + x)))*y*y)*
                            std::sin(std::sqrt(2.0)*M_PI*t);
                            fy = 2*(x*x*(6 + M_PI*M_PI*(-1 + y))*y + (-1 + y)*y + x*x*x*(3 + M_PI*M_PI*(-1 + y)*y) + x*(-2 + y + 3*y*y))*
                            std::sin(std::sqrt(2.0)*M_PI*t);
                            static_vector<double, 2> f{fx,fy};
                            return f;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_vector<double, 2> {
                    static_vector<double, 2> f;
                    return f;
                    };
            }
                break;
        }
        
    }
    
    
    std::function<static_matrix<double,2,2>(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_sigma(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_matrix<double,2,2> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            static_matrix<double,2,2> sigma = static_matrix<double,2,2>::Zero(2,2);
                            sigma(0,0) = M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::cos(M_PI*y) + 4*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y) -
                            M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::sin((M_PI*x)/2.)*std::sin(M_PI*y) +
                            (4*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y) - M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::sin((M_PI*x)/2.)*std::sin(M_PI*y))/2.0;
                        
                            sigma(0,1) = M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::cos(M_PI*y) + 2*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y) -
                            (M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::sin((M_PI*x)/2.)*std::sin(M_PI*y))/2.;
                            sigma(1,0) = sigma(0,1);
                        
                            sigma(1,1) = 3*M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::cos(M_PI*y) +
                            (4*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::cos((M_PI*x)/2.)*std::sin(M_PI*y) - M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::sin((M_PI*x)/2.)*std::sin(M_PI*y))/2.;
                            return sigma;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_matrix<double,2,2> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            static_matrix<double,2,2> sigma = static_matrix<double,2,2>::Zero(2,2);
                            sigma(0,0) = t*t*(M_PI*x*std::cos(M_PI*y)*std::sin(M_PI*x) + 3*(M_PI*x*std::cos(M_PI*x) + std::sin(M_PI*x))*std::sin(M_PI*y));
                        
                            sigma(0,1) = t*t*(M_PI*x*std::cos(M_PI*y)*std::sin(M_PI*x) + (M_PI*x*std::cos(M_PI*x) + std::sin(M_PI*x))*std::sin(M_PI*y));
                        
                            sigma(1,0) = sigma(0,1);
                        
                            sigma(1,1) = t*t*(3*M_PI*x*std::cos(M_PI*y)*std::sin(M_PI*x) + (M_PI*x*std::cos(M_PI*x) + std::sin(M_PI*x))*std::sin(M_PI*y));
                            return sigma;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
            {
                return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_matrix<double,2,2> {
                        double x,y;
                        x = pt.x();
                        y = pt.y();
                        static_matrix<double,2,2> sigma = static_matrix<double,2,2>::Zero(2,2);
                        sigma(0,0) = 2*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) + 4*x*(1 + x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) +
                        (2*x*x*(1 + x)*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - 2*x*x*(1 + x)*y*std::sin(std::sqrt(2.0)*M_PI*t))/2.0 +
                        (2*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) + 4*x*(1 + x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t))/2.0;
                    
                        sigma(0,1) = x*x*(1 + x)*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - x*x*(1 + x)*y*std::sin(std::sqrt(2.0)*M_PI*t) + x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) +
                        2*x*(1 + x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                    
                        sigma(1,0) = sigma(0,1);
                    
                        sigma(1,1) = 2*x*x*(1 + x)*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - 2*x*x*(1 + x)*y*std::sin(std::sqrt(2.0)*M_PI*t) +
                        (2*x*x*(1 + x)*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - 2*x*x*(1 + x)*y*std::sin(std::sqrt(2.0)*M_PI*t))/2.0 +
                        (2*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) + 4*x*(1 + x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t))/2.0;
                        return sigma;
                    };
            }
            break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> static_matrix<double,2,2> {
                        static_matrix<double,2,2> sigma(2,2);
                         return sigma;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_s_u(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return x*x*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*x)*std::sin(M_PI*y);
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return t*t*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return (1 - x)*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_s_v(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return std::sqrt(2.0)*M_PI*x*x*std::cos(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*x)*std::sin(M_PI*y);
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return 2*t*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return std::sqrt(2.0)*M_PI*(1 - x)*x*x*(1 - y)*y*std::cos(std::sqrt(2.0)*M_PI*t);
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_s_a(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return -2*M_PI*M_PI*x*x*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*x)*std::sin(M_PI*y);
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return 2*x*std::sin(M_PI*x)*std::sin(M_PI*y);
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            return -2*M_PI*M_PI*(1 - x)*x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_s_f(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y,f;
                            x = pt.x();
                            y = pt.y();
                            f = -2*std::sin(std::sqrt(2.0)*M_PI*t)*(2*M_PI*x*std::cos(M_PI*x) + std::sin(M_PI*x))*std::sin(M_PI*y);
                            return f;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y,f;
                            x = pt.x();
                            y = pt.y();
                            f = 2*(-(M_PI*t*t*std::cos(M_PI*x)) + (1 + M_PI*M_PI*t*t)*x*std::sin(M_PI*x))*std::sin(M_PI*y);
                            return f;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                            double x,y,f;
                            x = pt.x();
                            y = pt.y();
                            f = 2*((-1 + y)*y - 3*x*(-1 + y)*y + x*x*x*(-1 - M_PI*M_PI*(-1 + y)*y) + x*x*(1 + M_PI*M_PI*(-1 + y)*y))*std::sin(std::sqrt(2.0)*M_PI*t);
                            return f;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    
    std::function<std::vector<double>(const typename disk::generic_mesh<double, 2>::point_type& )> Evaluate_s_q(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = M_PI*x*x*std::cos(M_PI*x)*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*y) + 2*x*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*x)*std::sin(M_PI*y);
                            flux[1] = M_PI*x*x*std::cos(M_PI*y)*std::sin(std::sqrt(2.0)*M_PI*t)*std::sin(M_PI*x);
                            return flux;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = M_PI*t*t*x*std::cos(M_PI*x)*std::sin(M_PI*y) + t*t*std::sin(M_PI*x)*std::sin(M_PI*y);
                            flux[1] = M_PI*t*t*x*std::cos(M_PI*y)*std::sin(M_PI*x);
                        
                            return flux;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename disk::generic_mesh<double, 2>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = 2*(1 - x)*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) - x*x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                            flux[1] = (1 - x)*x*x*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - (1 - x)*x*x*y*std::sin(std::sqrt(2.0)*M_PI*t);
                            return flux;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename disk::generic_mesh<double, 2>::point_type& pt) -> std::vector<double> {
                        std::vector<double> f;
                        return f;
                    };
            }
                break;
        }
        
    }
    
    private:
    
    EFunctionType m_function_type;
  
};

#endif /* scal_vec_analytic_functions_hpp */
