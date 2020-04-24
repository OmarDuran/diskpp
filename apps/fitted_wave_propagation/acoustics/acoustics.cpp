//
//  acoustics.cpp
//  acoustics
//
//  Created by Omar Durán on 4/8/20.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <list>
#include <getopt.h>


#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
using namespace Eigen;

#include "timecounter.h"
#include "methods/hho"
#include "geometry/geometry.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "output/silo.hpp"

#include "loaders/loader.hpp" // candidate to delete

// application common sources
#include "../common/display_settings.hpp"
#include "../common/fitted_geometry_builders.hpp"
#include "../common/acoustic_material_data.hpp"
#include "../common/scal_analytic_functions.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

// implicit RK schemes
#include "../common/dirk_hho_scheme.hpp"
#include "../common/dirk_butcher_tableau.hpp"


#define fancy_stabilization_Q
#define compute_energy_Q
#define spatial_errors_Q
#define InhomogeneousQ

//double ComputeEnergySecondOrder(disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, Matrix<double, Dynamic, 1> & p, Matrix<double, Dynamic, 1> & v);
//
//double ComputeEnergyFirstOrder(disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
//std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> scal_fun,
//                               std::function<std::vector<double>(const typename disk::generic_mesh<double, 2>::point_type& )> flux_fun);
//
//void RenderSiloFileTwoFields(std::string silo_file_name, size_t it, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
//std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> scal_fun,
//std::function<std::vector<double>(const typename disk::generic_mesh<double, 2>::point_type& )> flux_fun, bool cell_centered_Q = true);
//

//template<typename Mesh>
//void RenderSiloFileScalarField(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
//                               std::function<double(const typename Mesh::point_type& )> scal_fun, bool cell_centered_Q = true);

//
//void ComputeL2ErrorTwoFields(disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
//std::function<double(const typename disk::generic_mesh<double, 2>::point_type& )> scal_fun,
//std::function<std::vector<double>(const typename disk::generic_mesh<double, 2>::point_type& )> flux_fun);
//
//void ComputeKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions);
//
//void ComputeFG(Matrix<double, Dynamic, 1> & Fg, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions);
//
//void ComputeInhomogeneousKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions);
//
//void ComputeKGFGSecondOrder(disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions);
//
//void DIRKStep(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n);
//
//void IRKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a);
//
//void DIRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, TDIRKHHOAnalyses & dirk_an, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, bool is_sdirk_Q = false);
//
//void IRKWeightOpt(TDIRKHHOAnalyses & dirk_an, Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a, bool is_sdirk_Q = false);
//
//void SSPRKStep(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof);
//
//void ERKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_n_dof, Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof, double dt, double a, double b);
//
//void SSPRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, TSSPRKHHOAnalyses & ssprk_an, double & dt, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof);
//
//void ERKWeightOpt(TSSPRKHHOAnalyses & ssprk_an,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b);
//
//void FaceDoFUpdate(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof);
//

//
//void EHHOFirstOrder(int argc, char **argv);
//
//
//void HeterogeneousEHHOFirstOrder(int argc, char **argv);

void HeterogeneousIHHOFirstOrder(int argc, char **argv);

void IHHOFirstOrder(int argc, char **argv);

void HeterogeneousIHHOSecondOrder(int argc, char **argv);

void IHHOSecondOrder(int argc, char **argv);

void HHOSecondOrderExample(int argc, char **argv);

void HHOFirstOrderExample(int argc, char **argv);

int main(int argc, char **argv)
{

//    HeterogeneousEHHOFirstOrder(argc, argv);
//    EHHOFirstOrder(argc, argv);

    HeterogeneousIHHOFirstOrder(argc, argv);
//    HeterogeneousIHHOSecondOrder(argc, argv);
    

    
//    IHHOFirstOrder(argc, argv);
//    IHHOSecondOrder(argc, argv);

    
//    // Examples solving the laplacian with optimal HHO convergence properties
//    HHOFirstOrderExample(argc, argv);
//    HHOSecondOrderExample(argc, argv);
    
    return 0;
}

//void HeterogeneousEHHOFirstOrder(int argc, char **argv){
//
//    bool render_silo_files_Q = true;
//    bool render_zonal_vars_Q = false;
//    using RealType = double;
//    size_t k_degree = 0;
//    size_t n_divs   = 0;
//
//    // Final time value 0.5
//    std::vector<size_t> nt_v = {5,10,20,40,80,160,320,640,1280,2560,5120,10240};
//    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.000048828125};
//
//    int tref = 9;
//    int s = 5;
//
//    size_t nt       = nt_v[tref];
//    RealType dt     = dt_v[tref];
//    RealType ti = 0.0;
//
//    int opt;
//    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
//    {
//        switch(opt)
//        {
//            case 'k':
//            {
//                k_degree = atoi(optarg);
//            }
//                break;
//            case 'l':
//            {
//                n_divs = atoi(optarg);
//            }
//                break;
//            case '?':
//            default:
//                std::cout << "wrong arguments" << std::endl;
//                exit(1);
//        }
//    }
//
//    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
//    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
//    std::cout << bold << red << "nt : " << nt << reset << std::endl;
//    std::cout << bold << red << "dt : " << dt << reset << std::endl;
//
//    // The mesh in ProtoN seems like is always 2D
//     mesh_init_params<RealType> mip;
//     mip.Nx = 10;
//     mip.Ny = 1;
//     mip.max_y = 0.1;
//
//    for (size_t i = 0; i < n_divs; i++) {
//        mip.Nx *= 2;
//    }
//
//    timecounter tc;
//
//    // Building the cartesian mesh
//    tc.tic();
//    poly_mesh<RealType> msh(mip);
//    tc.toc();
//
//    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
//
//    // Projection of initial data
//
//
//    // Creating HHO approximation spaces and corresponding linear operator
//    disk::hho_degree_info hho_di(k_degree,k_degree);
//
//    // Solving a HDG/HHO mixed problem
//    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
//    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
//
//
//    auto is_dirichlet = [&](const typename poly_mesh<RealType>::face_type& fc) -> bool {
//        return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
//    };
//    auto num_all_faces = msh.faces.size();
//    auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
//    auto num_other_faces = num_all_faces - num_dirichlet_faces;
//    auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
//    size_t n_f_dof = num_other_faces * fbs;
//
//    TAnalyticalFunction functions;
//    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionInhomogeneousInSpace);
//    RealType t = ti;
//    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
//    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
//    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
//    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
//
//    tc.tic();
//    // Projecting initial state(flux and velocity)
//    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS; // probably not needed
//    {
//        size_t cell_i = 0;
//        for (auto& cell : msh.cells)
//        {
//            { /// global mass
//
//                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//                auto n_rows = reconstruction_operator.second.rows();
//                auto n_cols = reconstruction_operator.second.cols();
//
//                auto n_s_rows = stabilization_operator.rows();
//                auto n_s_cols = stabilization_operator.cols();
//
//                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
//
//                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
//
//                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
//                size_t cell_dof_c;
//                {
//                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                    cell_dof_c = cell_basis.size();
//                }
//                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
//                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
//
//                // Compossing objects
//                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
//                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);
//
//            }
//
//            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
//            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
//            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
//            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
//
//            size_t cell_dof;
//            size_t cell_rec_dof;
//            {
//                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                cell_dof = cell_basis.size();
//
//            }
//            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
//            cell_rec_dof = cell_basis.size()-1;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
//            cell_i++;
//            // Initial projection of face unknows is not implemented yet
//        }
//    }
//    mass_assembler.finalize();
//    tc.toc();
//
//    size_t it = 0;
//    if (render_silo_files_Q) {
//        std::string silo_file_name = "e_scalar_inhomogeneous_wave_";
//        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//    }
//
//#ifdef compute_energy_Q
//    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
//    tc.tic();
//    energy_h_values(0,0) = 0.0;
//    energy_h_values(0,1) = 1.0;
//    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
//    tc.toc();
//    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
//#endif
//
//    Matrix<double, Dynamic, Dynamic> alpha;
//    Matrix<double, Dynamic, Dynamic> beta;
//    TSSPRKSchemes::OSSPRKSS(s, alpha, beta);
//
//    Matrix<double, Dynamic, 1> x_dof_n;
//    // Transient problem
//    bool optimized_Q = true;
//
//    if (optimized_Q) {
//
//        double tv = 0.0;
//        SparseMatrix<double> Kg;
//        Matrix<double, Dynamic, 1> Fg;
//        tc.tic();
//        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, tv, functions);
//        TSSPRKHHOAnalyses ssprk_an(Kg,Fg,mass_assembler.LHS,n_f_dof);
//        tc.toc();
//        std::cout << bold << cyan << "Linear transformations completed: " << tc << " seconds" << reset << std::endl;
//
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            SSPRKStepOpt(s, alpha, beta, ssprk_an, dt, x_dof_n_m, x_dof_n, n_f_dof);
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "e_scalar_inhomogeneous_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//#ifdef compute_energy_Q
//            {
//                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                energy_h_values(it,0) = t;
//                energy_h_values(it,1) = energy_h/energy_h0;
//            }
//#endif
//
//            if(it == nt){
//
//                std::string silo_file_name = "e_scalar_inhomogeneous_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//            x_dof_n_m = x_dof_n;
//        }
//    }else{
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            SSPRKStep(s, alpha, beta, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n, n_f_dof);
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "e_scalar_inhomogeneous_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun,render_zonal_vars_Q);
//            }
//
//#ifdef compute_energy_Q
//            {
//                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                energy_h_values(it,0) = t;
//                energy_h_values(it,1) = energy_h/energy_h0;
//            }
//#endif
//
//            if(it == nt){
//
//                std::string silo_file_name = "e_scalar_inhomogeneous_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun,render_zonal_vars_Q);
//
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//            x_dof_n_m = x_dof_n;
//        }
//    }
//
//#ifdef compute_energy_Q
//    std::ofstream energy_file ("EHHO_energy.txt");
//    if (energy_file.is_open())
//    {
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
//      energy_file.close();
//    }
//#endif
//
//}
//
//void EHHOFirstOrder(int argc, char **argv){
//
//    bool render_silo_files_Q = true;
//    bool render_zonal_vars_Q = false;
//    using RealType = double;
//    size_t k_degree = 0;
//    size_t n_divs   = 0;
//
//    // Final time value 1.0
//    std::vector<size_t> nt_v = {10,20,40,80,160,320,640,1280,2560,5120,10240,20480,40960,81920}; //13
//    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.000048828125,0.0000244140625,0.00001220703125};
//
//    int tref = 5 + 0;
//    int s = 5; // order s - 1
//
////    size_t nt       = nt_v[tref];
////    RealType dt     = dt_v[tref];
////    RealType ti = 0.0;
//
//    int opt;
//    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
//    {
//        switch(opt)
//        {
//            case 'k':
//            {
//                k_degree = atoi(optarg);
//            }
//                break;
//            case 'l':
//            {
//                n_divs = atoi(optarg);
//            }
//                break;
//            case '?':
//            default:
//                std::cout << "wrong arguments" << std::endl;
//                exit(1);
//        }
//    }
//
//    tref = n_divs + 5;
//    size_t nt       = nt_v[tref];
//    RealType dt     = dt_v[tref];
//    RealType ti = 0.0;
//
//    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
//    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
//    std::cout << bold << red << "nt : " << nt << reset << std::endl;
//    std::cout << bold << red << "dt : " << dt << reset << std::endl;
//
//    // The mesh in ProtoN seems like is always 2D
//     mesh_init_params<RealType> mip;
//     mip.Nx = 1;
//     mip.Ny = 1;
//
//    for (size_t i = 0; i < n_divs; i++) {
//        mip.Nx *= 2;
//        mip.Ny *= 2;
//    }
//
//    timecounter tc;
//
//    // Building the cartesian mesh
//    tc.tic();
//    poly_mesh<RealType> msh(mip);
//    tc.toc();
//
//    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
//
//    // Projection of initial data
//
//
//    // Creating HHO approximation spaces and corresponding linear operator
//    disk::hho_degree_info hho_di(k_degree,k_degree);
//
//    // Solving a HDG/HHO mixed problem
//    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
//    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
//
//
//    auto is_dirichlet = [&](const typename poly_mesh<RealType>::face_type& fc) -> bool {
//        return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
//    };
//    auto num_all_faces = msh.faces.size();
//    auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
//    auto num_other_faces = num_all_faces - num_dirichlet_faces;
//    auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
//    size_t n_f_dof = num_other_faces * fbs;
//
//    TAnalyticalFunction functions;
//    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionNonPolynomial);
//    RealType t = 0.0;
//    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
//    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
//    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
//    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
//
//    tc.tic();
//    // Projecting initial state(flux and velocity)
//    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS; // probably not needed
//    {
//        size_t cell_i = 0;
//        for (auto& cell : msh.cells)
//        {
//            { /// global mass
//
//                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//                auto n_rows = reconstruction_operator.second.rows();
//                auto n_cols = reconstruction_operator.second.cols();
//
//                auto n_s_rows = stabilization_operator.rows();
//                auto n_s_cols = stabilization_operator.cols();
//
//                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
//
//                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
//
//                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
//                size_t cell_dof_c;
//                {
//                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                    cell_dof_c = cell_basis.size();
//                }
//                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
//                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
//
//                // Compossing objects
//                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
//                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);
//
//            }
//
//            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
//            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
//            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
//            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
//
//            size_t cell_dof;
//            size_t cell_rec_dof;
//            {
//                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                cell_dof = cell_basis.size();
//
//            }
//            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
//            cell_rec_dof = cell_basis.size()-1;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
//            cell_i++;
//            // Initial projection of face unknows is not implemented yet
//        }
//    }
//    mass_assembler.finalize();
//    tc.toc();
//
//    size_t it = 0;
//    if (render_silo_files_Q) {
//        std::string silo_file_name = "scalar_wave_";
//        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//    }
//
//#ifdef compute_energy_Q
//    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
//    tc.tic();
//    energy_h_values(0,0) = 0.0;
//    energy_h_values(0,1) = 1.0;
//    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
//    tc.toc();
//    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
//#endif
//
//    Matrix<double, Dynamic, Dynamic> alpha;
//    Matrix<double, Dynamic, Dynamic> beta;
//    TSSPRKSchemes::OSSPRKSS(s, alpha, beta);
//
//    Matrix<double, Dynamic, 1> x_dof_n;
//    // Transient problem
//    bool optimized_Q = true;
//
//    if (optimized_Q) {
//
//        double tv = 0.0;
//        SparseMatrix<double> Kg;
//        Matrix<double, Dynamic, 1> Fg;
//        tc.tic();
//        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, tv, functions);
//        TSSPRKHHOAnalyses ssprk_an(Kg,Fg,mass_assembler.LHS,n_f_dof);
//        tc.toc();
//        std::cout << bold << cyan << "Linear transformations completed: " << tc << " seconds" << reset << std::endl;
//
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            SSPRKStepOpt(s, alpha, beta, ssprk_an, dt, x_dof_n_m, x_dof_n, n_f_dof);
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "scalar_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//#ifdef compute_energy_Q
//            {
//                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                energy_h_values(it,0) = t;
//                energy_h_values(it,1) = energy_h/energy_h0;
//            }
//#endif
//
//            if(it == nt){
//
//                std::string silo_file_name = "scalar_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//            x_dof_n_m = x_dof_n;
//        }
//    }else{
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            SSPRKStep(s, alpha, beta, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n, n_f_dof);
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "scalar_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//#ifdef compute_energy_Q
//            {
//                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                energy_h_values(it,0) = t;
//                energy_h_values(it,1) = energy_h/energy_h0;
//            }
//#endif
//
//            if(it == nt){
//
//                std::string silo_file_name = "scalar_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//            x_dof_n_m = x_dof_n;
//        }
//    }
//
//#ifdef compute_energy_Q
//    std::ofstream energy_file ("EHHO_energy.txt");
//    if (energy_file.is_open())
//    {
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
//      energy_file.close();
//    }
//#endif
//
//}
//
//void SSPRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, TSSPRKHHOAnalyses & ssprk_an, double & dt, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof){
//
//    size_t n_dof = x_dof_n_m.rows();
//    Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);
//
//    Matrix<double, Dynamic, 1> yn, ysi, yj;
//    ys.block(0, 0, n_dof, 1) = x_dof_n_m;
//    for (int i = 0; i < s; i++) {
//
//        ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
//        for (int j = 0; j <= i; j++) {
//            yn = ys.block(0, j, n_dof, 1);
//            ERKWeightOpt(ssprk_an, yn, yj, n_f_dof, dt, alpha(i,j), beta(i,j));
//            ysi += yj;
//        }
//        ys.block(0, i+1, n_dof, 1) = ysi;
//    }
//
//    x_dof_n = ys.block(0, s, n_dof, 1);
//
//}
//
//void SSPRKStep(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof){
//
//    size_t n_dof = x_dof_n_m.rows();
//    Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);
//    SparseMatrix<double> Kg;
//    Matrix<double, Dynamic, 1> Fg;
//
//    double t = tn + 0.0*dt;
//    ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
//
//    Matrix<double, Dynamic, 1> yn, ysi, yj;
//    ys.block(0, 0, n_dof, 1) = x_dof_n_m;
//    for (int i = 0; i < s; i++) {
//
//        ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
//        for (int j = 0; j <= i; j++) {
//            yn = ys.block(0, j, n_dof, 1);
//            ERKWeight(Kg, Fg, Mg, yn, yj, n_f_dof, dt, alpha(i,j), beta(i,j));
//            ysi += yj;
//        }
//        ys.block(0, i+1, n_dof, 1) = ysi;
//    }
//
//    x_dof_n = ys.block(0, s, n_dof, 1);
//
//}
//
//void HeterogeneousIHHOFirstOrder(int argc, char **argv){
//
//    bool render_silo_files_Q = true;
//    bool render_zonal_vars_Q = false;
//    using RealType = double;
//    size_t k_degree = 0;
//    size_t n_divs   = 0;
//
//    // Final time value 0.5
//    std::vector<size_t> nt_v = {5,10,20,40,80,160,320,640,1280,2560,5120};
//    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625};
//
//    int tref = 10;
//    int s = 3;
//
//    size_t nt       = nt_v[tref];
//    RealType dt     = dt_v[tref];
//    RealType ti = 0.0;
//
//    int opt;
//    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
//    {
//        switch(opt)
//        {
//            case 'k':
//            {
//                k_degree = atoi(optarg);
//            }
//                break;
//            case 'l':
//            {
//                n_divs = atoi(optarg);
//            }
//                break;
//            case '?':
//            default:
//                std::cout << "wrong arguments" << std::endl;
//                exit(1);
//        }
//    }
//
//    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
//    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
//    std::cout << bold << red << "nt : " << nt << reset << std::endl;
//    std::cout << bold << red << "dt : " << dt << reset << std::endl;
//
//    // The mesh in ProtoN seems like is always 2D
//     mesh_init_params<RealType> mip;
//     mip.Nx = 10;
//     mip.Ny = 1;
//     mip.max_y = 0.1;
//    for (size_t i = 0; i < n_divs; i++) {
//        mip.Nx *= 2;
//    }
//
//    timecounter tc;
//
//    // Building the cartesian mesh
//    tc.tic();
//    poly_mesh<RealType> msh(mip);
//    tc.toc();
//
//    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
//
//    // Projection of initial data
//
//
//    // Creating HHO approximation spaces and corresponding linear operator
//    disk::hho_degree_info hho_di(k_degree,k_degree);
//
//    // Solving a HDG/HHO mixed problem
//    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
//    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
//
//    TAnalyticalFunction functions;
//    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionInhomogeneousInSpace);
//    RealType t = ti;
//    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
//    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
//    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
//    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
//
//    tc.tic();
//    // Projecting initial state(flux and velocity)
//    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS;
//    {
//        size_t cell_i = 0;
//        for (auto& cell : msh.cells)
//        {
//            { /// global mass
//
//                double c = 1.0;
//                auto bar = barycenter(msh, cell);
//                double x = bar.x();
//                if (x < 0.5) {
//                    c *= contrast;
//                }
//
//                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//                auto n_rows = reconstruction_operator.second.rows();
//                auto n_cols = reconstruction_operator.second.cols();
//
//                auto n_s_rows = stabilization_operator.rows();
//                auto n_s_cols = stabilization_operator.cols();
//
//                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
//
//                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
//
//                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
//                size_t cell_dof_c;
//                {
//                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                    cell_dof_c = cell_basis.size();
//                }
//                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
//                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = (1.0/(c*c))*M_v;
//
//                // Compossing objects
//                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
//                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);
//
//            }
//
//
//            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
//            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
//            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
//            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
//
//            size_t cell_dof;
//            size_t cell_rec_dof;
//            {
//                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                cell_dof = cell_basis.size();
//
//            }
//            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
//            cell_rec_dof = cell_basis.size()-1;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
//            cell_i++;
//        }
//    }
//    mass_assembler.finalize();
//
////    // face update
////    if(0){
////        auto is_dirichlet = [&](const typename poly_mesh<RealType>::face_type& fc) -> bool {
////            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
////        };
////        auto num_all_faces = msh.faces.size();
////        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
////        auto num_other_faces = num_all_faces - num_dirichlet_faces;
////        auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
////        size_t n_f_dof = num_other_faces * fbs;
////
////        SparseMatrix<double> Kg;
////        Matrix<double, Dynamic, 1> Fg;
////        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
////        TSSPRKHHOAnalyses ssprk_an(Kg, Fg, mass_assembler.LHS, n_f_dof);
////
////        size_t n_c_dof = x_dof_n_m.rows() - n_f_dof;
////        Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
////        Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
////
////        // Faces update (last state)
////        {
////            Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_c_dof;
////            x_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf);
////        }
////        x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1) = x_f_dof;
////
////    }
//
//    tc.toc();
//    std::cout << bold << cyan << "Initial state computed: " << tc << " seconds" << reset << std::endl;
//    size_t it = 0;
//    if (render_silo_files_Q) {
//        std::string silo_file_name = "scalar_inhomogeneous_wave_";
//        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//    }
//
//#ifdef compute_energy_Q
//    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
//    tc.tic();
//    energy_h_values(0,0) = 0.0;
//    energy_h_values(0,1) = 1.0;
//    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
//    tc.toc();
//    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
//#endif
//
//    // Solving a HDG/HHO mixed problem
//    Matrix<RealType, Dynamic, Dynamic> a;
//    Matrix<RealType, Dynamic, 1> b;
//    Matrix<RealType, Dynamic, 1> c;
//
//    // DIRK(s) schemes
//    bool is_sdirk_Q = true;
//
//    if (is_sdirk_Q) {
//        TDIRKSchemes::SDIRKSchemes(s, a, b, c);
//    }else{
//        TDIRKSchemes::DIRKSchemesSS(s, a, b, c);
//    }
//
//    Matrix<double, Dynamic, 1> x_dof_n;
//    bool optimized_Q = true;
//
//    if (optimized_Q) {
//        // Transient problem
//
//#ifdef InhomogeneousQ
//        ComputeInhomogeneousKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
//#else
//        ComputeKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions); // Fixed boundary data
//#endif
//        TDIRKHHOAnalyses dirk_an(assembler.LHS,assembler.RHS,mass_assembler.LHS);
//
//        if (is_sdirk_Q) {
//            double scale = a(0,0) * dt;
//            dirk_an.SetScale(scale);
//            dirk_an.DecomposeMatrix();
//        }
//
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            DIRKStepOpt(s, a, b, c, msh, hho_di, assembler, dirk_an, tn, dt, functions, x_dof_n_m, x_dof_n, is_sdirk_Q);
//            tc.toc();
//
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "scalar_inhomogeneous_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//
//    #ifdef compute_energy_Q
//                {
//                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                    energy_h_values(it,0) = t;
//                    energy_h_values(it,1) = energy_h/energy_h0;
//                }
//    #endif
//
//    #ifdef spatial_errors_Q
//            if(it == nt){
//                std::string silo_file_name = "scalar_inhomogeneous_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//    #endif
//            x_dof_n_m = x_dof_n;
//        }
//    }else{
//        // Transient problem
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            DIRKStep(s, a, b, c, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n);
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "scalar_inhomogeneous_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//
//    #ifdef compute_energy_Q
//                {
//                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                    energy_h_values(it,0) = t;
//                    energy_h_values(it,1) = energy_h/energy_h0;
//                }
//    #endif
//
//    #ifdef spatial_errors_Q
//            if(it == nt){
//                std::string silo_file_name = "scalar_inhomogeneous_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//    #endif
//            x_dof_n_m = x_dof_n;
//        }
//    }
//
//#ifdef compute_energy_Q
//    std::ofstream energy_file ("IHHO_energy.txt");
//    if (energy_file.is_open())
//    {
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
//      energy_file.close();
//    }
//#endif
//
//}
//
//void IHHOFirstOrder(int argc, char **argv){
//
//    bool render_silo_files_Q = true;
//    bool render_zonal_vars_Q = false;
//    using RealType = double;
//    size_t k_degree = 0;
//    size_t n_divs   = 0;
//
//    // Final time value 1.0
//    std::vector<size_t> nt_v = {10,20,40,80,160,320,640,1280,2560,5120,10240,20480};
//    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.00009765625/2};
//
//    int tref = 5;
//    int s = 2;
//
//    size_t nt       = nt_v[tref];
//    RealType dt     = dt_v[tref];
//    RealType ti = 0.0;
//
//    int opt;
//    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
//    {
//        switch(opt)
//        {
//            case 'k':
//            {
//                k_degree = atoi(optarg);
//            }
//                break;
//            case 'l':
//            {
//                n_divs = atoi(optarg); //3
//            }
//                break;
//            case '?':
//            default:
//                std::cout << "wrong arguments" << std::endl;
//                exit(1);
//        }
//    }
//
//    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
//    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
//    std::cout << bold << red << "nt : " << nt << reset << std::endl;
//    std::cout << bold << red << "dt : " << dt << reset << std::endl;
//
//     mesh_init_params<RealType> mip;
//     mip.Nx = 1;
//     mip.Ny = 1;
//
//    for (size_t i = 0; i < n_divs; i++) {
//        mip.Nx *= 2;
//        mip.Ny *= 2;
//    }
//
//    timecounter tc;
//
//    // Building the cartesian mesh
//    tc.tic();
//    poly_mesh<RealType> msh(mip);
//    tc.toc();
//
//    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
//
//    // Projection of initial data
//
//
//    // Creating HHO approximation spaces and corresponding linear operator
//    disk::hho_degree_info hho_di(k_degree,k_degree);
//
//    // Solving a HDG/HHO mixed problem
//    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
//    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
//
//    TAnalyticalFunction functions;
//    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionNonPolynomial);
//    RealType t = 0.0;
//    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
//    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
//    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
//    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
//
//    tc.tic();
//    // Projecting initial state(flux and velocity)
//    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS;
//    {
//        size_t cell_i = 0;
//        for (auto& cell : msh.cells)
//        {
//            { /// global mass
//
//                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//                auto n_rows = reconstruction_operator.second.rows();
//                auto n_cols = reconstruction_operator.second.cols();
//
//                auto n_s_rows = stabilization_operator.rows();
//                auto n_s_cols = stabilization_operator.cols();
//
//                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
//
//                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
//
//                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
//                size_t cell_dof_c;
//                {
//                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                    cell_dof_c = cell_basis.size();
//                }
//                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
//                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
//
//                // Compossing objects
//                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
//                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);
//
//            }
//
//
//            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
//            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
//            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
//            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
//
//            size_t cell_dof;
//            size_t cell_rec_dof;
//            {
//                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//                cell_dof = cell_basis.size();
//
//            }
//            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
//            cell_rec_dof = cell_basis.size()-1;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
//            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
//            cell_i++;
//        }
//    }
//    mass_assembler.finalize();
//    tc.toc();
//    std::cout << bold << cyan << "Initial state computed: " << tc << " seconds" << reset << std::endl;
//    size_t it = 0;
//    if (render_silo_files_Q) {
//        std::string silo_file_name = "scalar_wave_";
//        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//    }
//
//#ifdef compute_energy_Q
//    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
//    tc.tic();
//    energy_h_values(0,0) = 0.0;
//    energy_h_values(0,1) = 1.0;
//    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
//    tc.toc();
//    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
//#endif
//
//    // Solving a HDG/HHO mixed problem
//    Matrix<RealType, Dynamic, Dynamic> a;
//    Matrix<RealType, Dynamic, 1> b;
//    Matrix<RealType, Dynamic, 1> c;
//
//    // DIRK(s) schemes
//    bool is_sdirk_Q = true;
//
//    if (is_sdirk_Q) {
//        TDIRKSchemes::SDIRKSchemes(s, a, b, c);
//    }else{
//        TDIRKSchemes::DIRKSchemesSS(s, a, b, c);
//    }
//
//    Matrix<double, Dynamic, 1> x_dof_n;
//    bool optimized_Q = false;
//
//    if (optimized_Q) {
//        // Transient problem
//
//#ifdef InhomogeneousQ
//        ComputeInhomogeneousKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
//#else
//        ComputeKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
//#endif
//        TDIRKHHOAnalyses dirk_an(assembler.LHS,assembler.RHS,mass_assembler.LHS);
//
//        if (is_sdirk_Q) {
//            double scale = a(0,0) * dt;
//            dirk_an.SetScale(scale);
//            dirk_an.DecomposeMatrix();
//        }
//
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            DIRKStepOpt(s, a, b, c, msh, hho_di, assembler, dirk_an, tn, dt, functions, x_dof_n_m, x_dof_n, is_sdirk_Q);
//            tc.toc();
//
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "scalar_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//
//    #ifdef compute_energy_Q
//                {
//                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                    energy_h_values(it,0) = t;
//                    energy_h_values(it,1) = energy_h/energy_h0;
//                }
//    #endif
//
//    #ifdef spatial_errors_Q
//            if(it == nt){
//                std::string silo_file_name = "scalar_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//    #endif
//            x_dof_n_m = x_dof_n;
//        }
//    }else{
//        // Transient problem
//        for(size_t it = 1; it <= nt; it++){
//
//            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//            RealType tn = dt*(it-1)+ti;
//            tc.tic();
//            DIRKStep(s, a, b, c, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n);
//            tc.toc();
//            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//            RealType t = tn + dt;
//            auto exact_vel_sol_fun = functions.Evaluate_v(t);
//            auto exact_flux_sol_fun = functions.Evaluate_q(t);
//
//            if (render_silo_files_Q) {
//                std::string silo_file_name = "scalar_wave_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//            }
//
//
//    #ifdef compute_energy_Q
//                {
//                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//                    energy_h_values(it,0) = t;
//                    energy_h_values(it,1) = energy_h/energy_h0;
//                }
//    #endif
//
//    #ifdef spatial_errors_Q
//            if(it == nt){
//                std::string silo_file_name = "scalar_wave_at_tf_";
//                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
//                std::cout << green << "dt size = " << std::endl << dt << std::endl;
//                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
//                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
//            }
//    #endif
//            x_dof_n_m = x_dof_n;
//        }
//    }
//
//#ifdef compute_energy_Q
//    std::ofstream energy_file ("IHHO_energy.txt");
//    if (energy_file.is_open())
//    {
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
//      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
//      energy_file.close();
//    }
//#endif
//
//}
//
//void DIRKStep(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n){
//
//    size_t n_dof = x_dof_n_m.rows();
//    Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
//    SparseMatrix<double> Kg;
//    Matrix<double, Dynamic, 1> Fg;
//
//    double t;
//    Matrix<double, Dynamic, 1> yn, ki;
//
//    x_dof_n = x_dof_n_m;
//    for (int i = 0; i < s; i++) {
//
//        yn = x_dof_n_m;
//        for (int j = 0; j < s - 1; j++) {
//            yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
//        }
//
//        t = tn + c(i,0) * dt;
//#ifdef InhomogeneousQ
//        ComputeInhomogeneousKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
//#else
//        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
//#endif
//
//        IRKWeight(Kg, Fg, Mg, yn, ki, dt, a(i,i));
//
//        // Accumulated solution
//        x_dof_n += dt*b(i,0)*ki;
//        k.block(0, i, n_dof, 1) = ki;
//    }
//
//}
//
//
//void ComputeInhomogeneousKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions){
//    using RealType = double;
//
//
//    auto exact_vel_sol_fun = functions.Evaluate_v(t);
//    auto rhs_fun = functions.Evaluate_f(t);
//
//    assembler.LHS *= 0.0;
//    assembler.RHS *= 0.0;
//    for (auto& cell : msh.cells)
//    {
//
//        double c = 1.0;
//        auto bar = barycenter(msh, cell);
//        double x = bar.x();
//        if (x < 0.5) {
//            c *= contrast;
//        }
//
//        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//        auto n_rows = reconstruction_operator.second.rows();
//        auto n_cols = reconstruction_operator.second.cols();
//
//        auto n_s_rows = stabilization_operator.rows();
//        auto n_s_cols = stabilization_operator.cols();
//
//        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
//
//        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
//        // Compossing objects
//        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + (1.0/(c))*S_operator;
//        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
//        assembler.assemble_mixed(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
//    }
//    assembler.finalize();
//    Kg = assembler.LHS;
//    Fg = assembler.RHS;
//
//
//}
//
//void ComputeKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions){
//    using RealType = double;
//
//
//    auto exact_vel_sol_fun = functions.Evaluate_v(t);
//    auto rhs_fun = functions.Evaluate_f(t);
//
//    assembler.LHS *= 0.0;
//    assembler.RHS *= 0.0;
//    for (auto& cell : msh.cells)
//    {
//        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//        auto n_rows = reconstruction_operator.second.rows();
//        auto n_cols = reconstruction_operator.second.cols();
//
//        auto n_s_rows = stabilization_operator.rows();
//        auto n_s_cols = stabilization_operator.cols();
//
//        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
//
//        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
//        // Compossing objects
//        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + S_operator;
//        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
//        assembler.assemble_mixed(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
//    }
//    assembler.finalize();
//    Kg = assembler.LHS;
//    Fg = assembler.RHS;
//
//
//}
//
//void ComputeFG(Matrix<double, Dynamic, 1> & Fg, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, double & t, TAnalyticalFunction & functions){
//
//    using RealType = double;
//    auto exact_vel_sol_fun = functions.Evaluate_v(t);
//    auto rhs_fun = functions.Evaluate_f(t);
//
//    assembler.LHS *= 0.0;
//    assembler.RHS *= 0.0;
//
//    for (auto& cell : msh.cells)
//    {
//
//        double c = 1.0;
//        auto bar = barycenter(msh, cell);
//        double x = bar.x();
//        if (x < 0.5) {
//            c *= contrast;
//        }
//
//        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
//#ifdef fancy_stabilization_Q
//        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
//#else
//        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//#endif
//
//        auto n_rows = reconstruction_operator.second.rows();
//        auto n_cols = reconstruction_operator.second.cols();
//
//        auto n_s_rows = stabilization_operator.rows();
//        auto n_s_cols = stabilization_operator.cols();
//
//        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
//        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
//
//        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
//        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
//        // Compossing objects
//        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + (1.0/c)*S_operator;
//        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
//        assembler.assemble_mixed_RHS(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
//    }
//    assembler.finalize();
//    Fg = assembler.RHS;
//
//}
//
//void ERKWeightOpt(TSSPRKHHOAnalyses & ssprk_an,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b){
//
//    timecounter tc;
//    tc.tic();
//
//    size_t n_c_dof = x_dof_n_m.rows() - n_f_dof;
//    Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
//    Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
//
//    // Faces update (last state)
//    {
//        Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_c_dof;
//        x_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf);
//    }
//
//    // Cells update
//    Matrix<double, Dynamic, 1> RHSc = ssprk_an.Fc() - ssprk_an.Kc()*x_c_dof - ssprk_an.Kcf()*x_f_dof;
//    Matrix<double, Dynamic, 1> delta_x_c_dof = ssprk_an.CellsAnalysis().solve(RHSc);
//    Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
//
//    // Faces update
//    Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_n_c_dof;
//    Matrix<double, Dynamic, 1> x_n_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf); // new state
//
//    // Composing global solution
//    x_dof_n = x_dof_n_m;
//    x_dof_n.block(0, 0, n_c_dof, 1) = x_n_c_dof;
//    x_dof_n.block(n_c_dof, 0, n_f_dof, 1) = x_n_f_dof;
//    tc.toc();
//
//}
//
//void ERKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b){
//
//    using RealType = double;
//    timecounter tc;
//    tc.tic();
//
//    size_t n_c_dof = Kg.rows() - n_f_dof;
//
//    // Composing objects
//    SparseMatrix<RealType> Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
//    SparseMatrix<RealType> Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
//    SparseMatrix<RealType> Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
//    SparseMatrix<RealType> Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
//    SparseMatrix<RealType> Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
//    Matrix<double, Dynamic, 1> Fc = Fg.block(0, 0, n_c_dof, 1);
//    Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
//    Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
//
//
//    SparseLU<SparseMatrix<RealType>> analysis_f;
//    analysis_f.analyzePattern(Sff);
//    analysis_f.factorize(Sff);
//    {
//        // Faces update (last state)
//        Matrix<double, Dynamic, 1> RHSf = Kfc*x_c_dof;
//        Matrix<double, Dynamic, 1> x_f_dof_c = -analysis_f.solve(RHSf);
//        x_f_dof = x_f_dof_c;
//    }
//
//    // Cells update
//    SparseLU<SparseMatrix<RealType>> analysis_c;
//    analysis_c.analyzePattern(Mc);
//    analysis_c.factorize(Mc);
//    Matrix<double, Dynamic, 1> RHSc = Fc - Kc*x_c_dof - Kcf*x_f_dof;
//    Matrix<double, Dynamic, 1> delta_x_c_dof = analysis_c.solve(RHSc);
//    Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
//
//    // Faces update
//    Matrix<double, Dynamic, 1> RHSf = Kfc*x_n_c_dof;
//    Matrix<double, Dynamic, 1> x_n_f_dof = -analysis_f.solve(RHSf); // new state
//
//    // Composing global solution
//    x_dof_n = x_dof_n_m;
//    x_dof_n.block(0, 0, n_c_dof, 1) = x_n_c_dof;
//    x_dof_n.block(n_c_dof, 0, n_f_dof, 1) = x_n_f_dof;
//    tc.toc();
//
//}
//
//void FaceDoFUpdate(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof){
//
//    using RealType = double;
//    timecounter tc;
//    tc.tic();
//
//    size_t n_c_dof = x_dof.rows() - n_f_dof;
//
//    // Composing objects
//    SparseMatrix<RealType> Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
//    SparseMatrix<RealType> Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
//    SparseMatrix<RealType> Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
//    SparseMatrix<RealType> Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
//    SparseMatrix<RealType> Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
//    Matrix<double, Dynamic, 1> x_c_dof = x_dof.block(0, 0, n_c_dof, 1);
//    Matrix<double, Dynamic, 1> x_f_dof = x_dof.block(n_c_dof, 0, n_f_dof, 1);
//
//    std::cout << "x_f_dof = " << x_f_dof << std::endl;
//
//    // Faces update (last state)
//    Matrix<double, Dynamic, 1> RHSf = Kfc*x_c_dof;
//    SparseLU<SparseMatrix<RealType>> analysis_f;
//    analysis_f.analyzePattern(Sff);
//    analysis_f.factorize(Sff);
//    Matrix<double, Dynamic, 1> x_f_dof_c = -analysis_f.solve(RHSf); // new state
//    std::cout << "x_f_dof_c = " << x_f_dof_c << std::endl;
//    x_f_dof = x_f_dof_c;
//    x_dof.block(n_c_dof, 0, n_f_dof, 1) = x_f_dof;
//
//    tc.toc();
//
//}
//
//void IRKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a){
//
//    using RealType = double;
//    timecounter tc;
//    tc.tic();
//
//    Fg -= Kg*y;
//    Kg *= (a*dt);
//    Kg += Mg;
//
//    SparseLU<SparseMatrix<RealType>> analysis_t;
//    analysis_t.analyzePattern(Kg);
//    analysis_t.factorize(Kg);
//    k = analysis_t.solve(Fg);
//    tc.toc();
//
//}
//
//void DIRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, disk::generic_mesh<double, 2> & msh, disk::hho_degree_info & hho_di, assembler<disk::generic_mesh<double, 2>> & assembler, TDIRKHHOAnalyses & dirk_an, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, bool is_sdirk_Q){
//
//    size_t n_dof = x_dof_n_m.rows();
//    Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
//    Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
//    xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
//
//    double t;
//    Matrix<double, Dynamic, 1> yn, ki;
//
//    x_dof_n = x_dof_n_m;
//    for (int i = 0; i < s; i++) {
//
//        yn = x_dof_n_m;
//        for (int j = 0; j < s - 1; j++) {
//            yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
//        }
//
//        t = tn + c(i,0) * dt;
//        ComputeFG(Fg, msh, hho_di, assembler, t, functions);
//        Fg_c = dirk_an.Kg()*xd;
//        dirk_an.SetFg(Fg);
//
//        IRKWeightOpt(dirk_an, yn, ki, dt, a(i,i),is_sdirk_Q);
//
//        // Accumulated solution
//        x_dof_n += dt*b(i,0)*ki;
//        k.block(0, i, n_dof, 1) = ki;
//    }
//
//}
//
//void IRKWeightOpt(TDIRKHHOAnalyses & dirk_an, Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a, bool is_sdirk_Q){
//
//    timecounter tc;
//    tc.tic();
//
//    Matrix<double, Dynamic, 1> Fg = dirk_an.Fg();
//    Fg -= dirk_an.Kg()*y;
//
//    if (is_sdirk_Q) {
//        k = dirk_an.DirkAnalysis().solve(Fg);
//    }else{
//        double scale = a * dt;
//        dirk_an.SetScale(scale);
//        dirk_an.DecomposeMatrix();
//        k = dirk_an.DirkAnalysis().solve(Fg);
//    }
//    tc.toc();
//}

void HHOFirstOrderExample(int argc, char **argv){

    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 1.0;
    size_t nx = 2;
    size_t ny = 2;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);

    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Manufactured solution
#ifdef quadratic_space_solution_Q

    auto exact_scal_fun = [](const mesh_type::point_type& pt) -> RealType {
        return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
    };

    auto exact_flux_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
        flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };

    auto rhs_fun = [](const typename mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return -2.0*((x - 1)*x + (y - 1)*y);
    };

#else

    auto exact_scal_fun = [](const mesh_type::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };

    auto exact_flux_fun = [](const mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
        flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };

    auto rhs_fun = [](const mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };

#endif

    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);


    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_scal_fun);
    tc.tic();
    auto assembler = two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(assembler.LHS);
    analysis_t.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> x_dof = analysis_t.solve(assembler.RHS); // new state
    tc.toc();
    std::cout << bold << cyan << "Number of equations : " << assembler.RHS.rows() << reset << std::endl;
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Computing errors
    postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_scal_fun, exact_flux_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_mixed_";
    postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_scal_fun, exact_flux_fun, false);
}

void HHOSecondOrderExample(int argc, char **argv){

    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();
    
    RealType lx = 1.0;
    RealType ly = 1.0;
    size_t nx = 2;
    size_t ny = 2;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;
    
    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Manufactured solution
#ifdef quadratic_space_solution_Q

    auto exact_scal_fun = [](const mesh_type::point_type& pt) -> RealType {
        return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
    };

    auto exact_flux_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
        flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };

    auto rhs_fun = [](const typename mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return -2.0*((x - 1)*x + (y - 1)*y);
    };

#else

    auto exact_scal_fun = [](const mesh_type::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };

    auto exact_flux_fun = [](const mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
        flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };

    auto rhs_fun = [](const mesh_type::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };

#endif

    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);


    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_scal_fun);
    tc.tic();
    auto assembler = one_field_assembler<mesh_type>(msh, hho_di, bnd);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    assembler.assemble(msh, rhs_fun);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(assembler.LHS);
    analysis_t.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> x_dof = analysis_t.solve(assembler.RHS); // new state
    tc.toc();
    std::cout << bold << cyan << "Number of equations : " << assembler.RHS.rows() << reset << std::endl;
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Computing errors
    postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler, x_dof, exact_scal_fun, exact_flux_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_";
    postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, x_dof, exact_scal_fun, false);
}

void IHHOSecondOrder(int argc, char **argv){

    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 1.0;
    size_t nx = 2;
    size_t ny = 2;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt     = tf/nt;
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
    RealType t = ti;
    auto exact_scal_fun     = functions.Evaluate_u(t);
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_accel_fun    = functions.Evaluate_a(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_scal_fun); // easy because boundary assumes zero every where any time.
    tc.tic();
    auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
    assembler.load_material_data(msh);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, p_dof_n, exact_scal_fun);
    assembler.project_over_cells(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_cells(msh, a_dof_n, exact_accel_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_";
    postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
    
    bool standar_Q = true;
    // Newmark process
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;

        RealType beta = 0.25;
        RealType gamma = 0.5;
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 1.5;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_fun     = functions.Evaluate_u(t);
            auto exact_flux_fun     = functions.Evaluate_q(t);
            auto rhs_fun            = functions.Evaluate_f(t);

            assembler.get_bc_conditions().updateDirichletFunction(exact_scal_fun, 0);
            assembler.assemble(msh, rhs_fun);

            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*p_dof_n;

            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += assembler.MASS;
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;

            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
            
            std::string silo_file_name = "scalar_";
            postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
            
            if(it == nt){
                auto assembler_c = one_field_assembler<mesh_type>(msh, hho_di, bnd);
                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler_c, p_dof_n, exact_scal_fun, exact_flux_fun);
            }
            
        }
        std::cout << green << "Number of time steps =  " << nt << reset << std::endl;
        std::cout << green << "Step size =  " << dt << reset << std::endl;
    }
}

void HeterogeneousIHHOSecondOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.1;
    size_t nx = 10;
    size_t ny = 1;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh_x_direction(sim_data.m_n_divs);
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);

    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.5
    size_t nt = 5;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 0.5;
    RealType dt     = tf/nt;

    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_scal_fun     = functions.Evaluate_u(t);
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_accel_fun    = functions.Evaluate_a(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_scal_fun); // easy because boundary assumes zero every where any time.
    tc.tic();
    auto assembler = acoustic_one_field_assembler<mesh_type>(msh, hho_di, bnd);
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        rho = 1.0;
        if (x < 0.5) {
            vp = 2.0;
        }else{
            vp = 1.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    assembler.project_over_cells(msh, p_dof_n, exact_scal_fun);
    assembler.project_over_faces(msh, p_dof_n, exact_scal_fun);
    assembler.project_over_cells(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_faces(msh, v_dof_n, exact_vel_fun);
    assembler.project_over_cells(msh, a_dof_n, exact_accel_fun);
    assembler.project_over_faces(msh, a_dof_n, exact_accel_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_";
    postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
    
    // Newmark process
    bool standar_Q = false;
    {
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;
        RealType beta = 0.25;
        RealType gamma = 0.5;
        
        if (!standar_Q) {
            RealType kappa = 0.25;
            gamma = 1.5;
            beta = kappa*(gamma+0.5)*(gamma+0.5);
        }
        
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;

            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_fun     = functions.Evaluate_u(t);
            auto exact_flux_fun     = functions.Evaluate_q(t);
            auto rhs_fun            = functions.Evaluate_f(t);

            assembler.get_bc_conditions().updateDirichletFunction(exact_scal_fun, 0);
            assembler.assemble(msh, rhs_fun);

            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*p_dof_n;

            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += assembler.MASS;
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;

            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
            
            std::string silo_file_name = "scalar_";
            postprocessor<mesh_type>::write_silo_one_field(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_fun, false);
            
            if(it == nt){
                auto assembler_c = one_field_assembler<mesh_type>(msh, hho_di, bnd);
                postprocessor<mesh_type>::compute_errors_one_field(msh, hho_di, assembler_c, p_dof_n, exact_scal_fun, exact_flux_fun);
            }
            
        }
        std::cout << green << "Number of time steps =  " << nt << reset << std::endl;
        std::cout << green << "Step size =  " << dt << reset << std::endl;
    }
}

void IHHOFirstOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 1.0;
    size_t nx = 2;
    size_t ny = 2;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt     = tf/nt;
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionQuadraticInSpace);
    RealType t = ti;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vel_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    assembler.load_material_data(msh);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, exact_vel_fun, exact_flux_fun);
    assembler.project_over_faces(msh, x_dof, exact_vel_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    
    // Solving a first order equation HDG/HHO propagation problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    int s = 2;
    bool is_sdirk_Q = true;
    
    if (is_sdirk_Q) {
        dirk_butcher_tableau::sdirk_tables(s, a, b, c);
    }else{
        dirk_butcher_tableau::dirk_tables(s, a, b, c);
    }
    
    assembler.assemble(msh, rhs_fun);
    dirk_hho_scheme dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (is_sdirk_Q) {
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        dirk_an.DecomposeMatrix();
    }
    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        RealType tn = dt*(it-1)+ti;
        
        // DIRK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
            
            RealType t;
            Matrix<double, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {
                
                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }
                
                t = tn + c(i,0) * dt;
                auto exact_vel_fun      = functions.Evaluate_v(t);
                auto rhs_fun            = functions.Evaluate_f(t);
                assembler.get_bc_conditions().updateDirichletFunction(exact_vel_fun, 0);
                assembler.assemble(msh, rhs_fun);
                dirk_an.SetFg(assembler.RHS);
                dirk_an.irk_weight(yn, ki, dt, a(i,i),is_sdirk_Q);

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "DIRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;
        
        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        
        std::string silo_file_name = "scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        
        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun);
        }

    }
    std::cout << green << "Number of DIRK steps   =  " << s << reset << std::endl;
    std::cout << green << "Number of time steps =  " << nt << reset << std::endl;
    std::cout << green << "Step size =  " << dt << reset << std::endl;
    
}

void HeterogeneousIHHOFirstOrder(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();
    
    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    RealType lx = 1.0;
    RealType ly = 0.1;
    size_t nx = 10;
    size_t ny = 1;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, true> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh_x_direction(sim_data.m_n_divs);
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Time controls : Final time value 0.5
    size_t nt = 5;
    for (unsigned int i = 0; i < sim_data.m_nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 0.5;
    RealType dt     = tf/nt;
    
    scal_analytic_functions functions;
    functions.set_function_type(scal_analytic_functions::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_vel_fun      = functions.Evaluate_v(t);
    auto exact_flux_fun     = functions.Evaluate_q(t);
    auto rhs_fun            = functions.Evaluate_f(t);
    
    // Creating HHO approximation spaces and corresponding linear operator
    size_t cell_k_degree = sim_data.m_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,sim_data.m_k_degree);

    // Solving a primal HHO mixed problem
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(exact_vel_fun);
    tc.tic();
    auto assembler = acoustic_two_fields_assembler<mesh_type>(msh, hho_di, bnd);
    
    auto acoustic_mat_fun = [](const typename mesh_type::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> mat_data(2);
        RealType rho, vp;
        rho = 1.0;
        if (x < 0.5) {
            vp = 10.0;
        }else{
            vp = 1.0;
        }
        mat_data[0] = rho; // rho
        mat_data[1] = vp; // seismic compressional velocity vp
        return mat_data;
    };
    
    assembler.load_material_data(msh,acoustic_mat_fun);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    tc.toc();
    std::cout << bold << cyan << "Assembler generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    std::string silo_file_props_name = "properties_map";
    postprocessor<mesh_type>::write_silo_acoustic_property_map(silo_file_props_name, msh, assembler.get_material_data());
    
    tc.tic();
    assembler.assemble_mass(msh);
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial data
    Matrix<RealType, Dynamic, 1> x_dof;
    assembler.project_over_cells(msh, x_dof, exact_vel_fun, exact_flux_fun);
    assembler.project_over_faces(msh, x_dof, exact_vel_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_mixed_";
        postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
    
    // Solving a first order equation HDG/HHO propagation problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    int s = 3;
    bool is_sdirk_Q = false;
    
    if (is_sdirk_Q) {
        dirk_butcher_tableau::sdirk_tables(s, a, b, c);
    }else{
        dirk_butcher_tableau::odirk_tables(s, a, b, c);
    }
    
    assembler.assemble(msh, rhs_fun);
    dirk_hho_scheme dirk_an(assembler.LHS,assembler.RHS,assembler.MASS);
    
    if (is_sdirk_Q) {
        double scale = a(0,0) * dt;
        dirk_an.SetScale(scale);
        dirk_an.DecomposeMatrix();
    }
    Matrix<double, Dynamic, 1> x_dof_n;
    for(size_t it = 1; it <= nt; it++){

        std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        RealType tn = dt*(it-1)+ti;
        
        // DIRK step
        tc.tic();
        {
            size_t n_dof = x_dof.rows();
            Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
            Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
            xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
            
            ;
            Matrix<double, Dynamic, 1> yn, ki;

            x_dof_n = x_dof;
            for (int i = 0; i < s; i++) {
                
                yn = x_dof;
                for (int j = 0; j < s - 1; j++) {
                    yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
                }
                
                {
                    RealType t = tn + c(i,0) * dt;
                    auto exact_vel_fun      = functions.Evaluate_v(t);
                    auto rhs_fun            = functions.Evaluate_f(t);
                    assembler.get_bc_conditions().updateDirichletFunction(exact_vel_fun, 0);
                    assembler.assemble(msh, rhs_fun);
                }
                dirk_an.SetFg(assembler.RHS);
                dirk_an.irk_weight(yn, ki, dt, a(i,i),is_sdirk_Q);

                // Accumulated solution
                x_dof_n += dt*b(i,0)*ki;
                k.block(0, i, n_dof, 1) = ki;
            }
        }
        tc.toc();
        std::cout << bold << cyan << "DIRK step completed: " << tc << " seconds" << reset << std::endl;
        x_dof = x_dof_n;
        
        t = tn + dt;
        auto exact_vel_fun = functions.Evaluate_v(t);
        auto exact_flux_fun = functions.Evaluate_q(t);
        std::string silo_file_name = "scalar_mixed_";
            postprocessor<mesh_type>::write_silo_two_fields(silo_file_name, it, msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun, false);
        
        if(it == nt){
            // Computing errors
            postprocessor<mesh_type>::compute_errors_two_fields(msh, hho_di, x_dof, exact_vel_fun, exact_flux_fun);
        }

    }
    std::cout << green << "Number of DIRK steps   =  " << s << reset << std::endl;
    std::cout << green << "Number of time steps =  " << nt << reset << std::endl;
    std::cout << green << "Step size =  " << dt << reset << std::endl;
    
}
