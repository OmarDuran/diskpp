//
//  md_small_strain.cpp
//  md_small_strain
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

// application common sources
#include "../common/display_settings.hpp"
#include "../common/fitted_geometry_builders.hpp"
#include "../common/linear_solver.hpp"
#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"

// ----- common data types ------------------------------
using RealType = double;
typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
typedef disk::BoundaryConditions<mesh_type, false> boundary_type;

void CrackExample(int argc, char **argv);

int main(int argc, char **argv)
{
//    CrackExample(argc, argv);
    
    simulation_data sim_data = preprocessor::process_convergence_test_args(argc, argv);
    sim_data.print_simulation_data();
    
    timecounter tc;
    
    // Reading the polygonal mesh
    tc.tic();
    mesh_type msh;
    polygon_2d_mesh_reader<RealType> mesh_builder;
    
    // Reading the polygonal mesh
//    std::string mesh_file = "meshes/simple_mesh_single_crack_nel_2.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_nel_4.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_4.txt";
//    std::string mesh_file = "meshes/simple_mesh_single_crack_duplicated_nodes_nel_42.txt";
//    std::string mesh_file = "meshes/base_polymesh_nel_215.txt";
    
//    std::string mesh_file = "meshes/base_polymesh_fracture_nel_215.txt";
//    std::string mesh_file = "meshes/base_polymesh_fracture_nel_831.txt";
    std::string mesh_file = "meshes/base_polymesh_tilted_fracture_nel_986.txt";
    
    mesh_builder.set_poly_mesh_file(mesh_file);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    
    tc.tic();
    std::vector<std::pair<size_t,size_t>> fracture_pairs;
    std::vector<mesh_type::point_type> fracture_bars;
    fracture_bars.reserve(msh.faces_size());
    auto storage = msh.backend_storage();
    for (size_t face_id = 0; face_id < msh.faces_size(); face_id++)
    {
        auto& egde = storage->edges[face_id];
        mesh_type::point_type bar = barycenter(msh, egde);
        fracture_bars.push_back(bar);
    }
    
    auto are_equal_Q = [](const mesh_type::point_type& a, const mesh_type::point_type& b)-> bool {
        bool check_Q = fabs(a.x() - b.x()) <= 1.0e-4 && fabs(a.y() - b.y()) <= 1.0e-4;
        return check_Q;
    };

    for (size_t i = 0; i < fracture_bars.size(); i++) {
        mesh_type::point_type bar_i = fracture_bars.at(i);
        for (size_t j = i+1; j < fracture_bars.size(); j++) {
            mesh_type::point_type bar_j = fracture_bars.at(j);
             if (are_equal_Q(bar_i,bar_j)) {
                 fracture_pairs.push_back(std::make_pair(i, j));
             }
        }
    }
    
    tc.toc();
    std::cout << bold << cyan << "Fracture mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    // Constant elastic properties
    RealType rho,l,mu;
    rho = 1.0;
    l = 1.0;//2000.0;
    mu = 1.0;//2000.0;
    elastic_material_data<RealType> material(rho,l,mu);

    // Creating HHO approximation spaces and corresponding linear operator
    size_t face_k_degree = sim_data.m_k_degree;
    size_t cell_k_degree = face_k_degree;
    if(sim_data.m_hdg_stabilization_Q){
        cell_k_degree++;
    }
    disk::hho_degree_info hho_di(cell_k_degree,face_k_degree);

    // Solving a scalar primal HHO problem
    auto null_v_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType ux = 0.0;
        RealType uy = 0.0;
        return static_vector<RealType, 2>{ux, uy};
    };
    auto u_top_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType ux = -0.0;
        RealType uy = -0.1;
        return static_vector<RealType, 2>{ux, uy};
    };
    
    auto rhs_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
        RealType x,y;
        x = pt.x();
        y = pt.y();
        RealType rx = 0.0;
        RealType ry = 0.0;
        return static_vector<RealType, 2>{rx, ry};
    };
    
    boundary_type bnd(msh);
//    RealType minlx = 0.0;
//    RealType minly = 0.0;
//    RealType maxlx = 2.0;
//    RealType maxly = 4.0;
    RealType minlx = -10.0;
    RealType minly = -10.0;
    RealType maxlx = 10.0;
    RealType maxly = 10.0;
    // defining boundary conditions
    {
        size_t bc_D_bot_id = 0;
        size_t bc_N_right_id = 1;
        size_t bc_D_top_id = 2;
        size_t bc_N_left_id = 3;
        RealType eps = 1.0e-4;
        
        for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
        {
            auto face = *face_it;
            mesh_type::point_type bar = barycenter(msh, face);
            auto fc_id = msh.lookup(face);
            if (std::fabs(bar.y()-minly) < eps) {
                disk::bnd_info bi{bc_D_bot_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(std::fabs(bar.x()-maxlx) < eps){
                disk::bnd_info bi{bc_N_right_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.y()-maxly) < eps) {
                disk::bnd_info bi{bc_D_top_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.x()-minlx) < eps) {
                disk::bnd_info bi{bc_N_left_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
        }   
        
        bnd.addDirichletBC(disk::DY, bc_D_bot_id, null_v_fun);
        bnd.addDirichletBC(disk::DX, bc_N_right_id, null_v_fun);
        bnd.addDirichletBC(disk::DY, bc_D_top_id, u_top_fun);
        bnd.addDirichletBC(disk::DX, bc_N_left_id, null_v_fun);
    }

    tc.tic();
    auto assembler = elastic_two_fields_assembler<mesh_type>(msh, hho_di, bnd, fracture_pairs);
    if(sim_data.m_hdg_stabilization_Q){
        assembler.set_hdg_stabilization();
    }
    if(sim_data.m_scaled_stabilization_Q){
        assembler.set_scaled_stabilization();
    }
    assembler.load_material_data(msh,material);
    assembler.assemble(msh, rhs_fun);
    assembler.apply_bc(msh);
    tc.toc();
    std::cout << bold << cyan << "Assemble in : " << tc.to_double() << " seconds" << reset << std::endl;
    
//    std::cout << "k = " << assembler.LHS.toDense() <<  std::endl;
    
    // Solving LS
    Matrix<RealType, Dynamic, 1> x_dof;
    tc.tic();
    linear_solver<RealType> analysis(assembler.LHS);
    analysis.set_direct_solver(true);
    tc.toc();
    std::cout << bold << cyan << "Create analysis in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    analysis.factorize();
    tc.toc();
    std::cout << bold << cyan << "Factorized in : " << tc.to_double() << " seconds" << reset << std::endl;
    
    tc.tic();
    x_dof = analysis.solve(assembler.RHS);
    tc.toc();
    std::cout << bold << cyan << "Linear Solve in : " << tc.to_double() << " seconds" << reset << std::endl;
    std::cout << bold << cyan << "Number of equations : " << analysis.n_equations() << reset << std::endl;
    
    // render silo
    size_t it = 0;
    std::string silo_file_name = "single_fracture";
    postprocessor<mesh_type>::write_silo_u_field(silo_file_name, it, msh, hho_di, x_dof);
    
    return 0;
}


/// ongoing by force

struct assembly_info
{
    size_t linear_system_size;
    double time_gradrec, time_statcond, time_stab, time_assembly, time_divrec;
};

struct solver_info
{
    double time_solver;
};

struct postprocess_info
{
    double time_postprocess;
};

struct ElasticityParameters
{
    double lambda;
    double mu;
};

template<typename Mesh>
class linear_elasticity_solver_c
{
    typedef Mesh                                mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::face            face_type;

    typedef disk::BoundaryConditions<mesh_type, false> bnd_type;

    typedef dynamic_matrix<scalar_type> matrix_dynamic;
    typedef dynamic_vector<scalar_type> vector_dynamic;

    typedef disk::assembler_mechanics<mesh_type> assembler_type;

    size_t m_cell_degree, m_face_degree;

    const static size_t dimension = mesh_type::dimension;

    const bnd_type&                      m_bnd;
    const mesh_type&                     m_msh;
    typename disk::hho_degree_info m_hdi;
    assembler_type                       m_assembler;

    std::vector<vector_dynamic> m_bL;
    std::vector<matrix_dynamic> m_AL;

    bool m_verbose;

    ElasticityParameters m_elas_parameters;

  public:
    
    std::vector<vector_dynamic> m_solution_data;
    vector_dynamic              m_system_solution;
    
    linear_elasticity_solver_c(const mesh_type&           msh,
                             const bnd_type&            bnd,
                             const ElasticityParameters data,
                             int                        degree,
                             int                        l = 0) :
      m_msh(msh),
      m_verbose(false), m_bnd(bnd)
    {
        if (l < -1 or l > 1)
        {
            std::cout << "'l' should be -1, 0 or 1. Reverting to 0." << std::endl;
            l = 0;
        }

        if (degree == 0 && l == -1)
        {
            std::cout << "'l' should be 0 or 1. Reverting to 0." << std::endl;
            l = 0;
        }

        m_cell_degree = degree + l;
        m_face_degree = degree;

        m_elas_parameters.mu     = data.mu;
        m_elas_parameters.lambda = data.lambda;

        m_hdi       = disk::hho_degree_info(m_cell_degree, m_face_degree);
        m_assembler = disk::make_mechanics_assembler(m_msh, m_hdi, m_bnd);
        m_AL.clear();
        m_AL.reserve(m_msh.cells_size());

        m_bL.clear();
        m_bL.reserve(m_msh.cells_size());

        if (m_verbose)
            m_hdi.info_degree();
    }

    void
    changeElasticityParameters(const ElasticityParameters data)
    {
        m_elas_parameters.mu     = data.mu;
        m_elas_parameters.lambda = data.lambda;
    }

    bool
    verbose(void) const
    {
        return m_verbose;
    }
    void
    verbose(bool v)
    {
        m_verbose = v;
    }

    size_t
    getDofs()
    {
        return m_msh.faces_size() * disk::vector_basis_size(m_hdi.face_degree(), dimension - 1, dimension);
    }

    template<typename LoadFunction>
    assembly_info
    assemble(const LoadFunction& lf)
    {
        assembly_info ai;
        bzero(&ai, sizeof(ai));

        timecounter tc;

        for (auto& cl : m_msh)
        {
            tc.tic();
            const auto sgr = make_vector_hho_symmetric_laplacian(m_msh, cl, m_hdi);
            const auto sg  = make_matrix_symmetric_gradrec(m_msh, cl, m_hdi);
            tc.toc();
            ai.time_gradrec += tc.to_double();

            tc.tic();
            const auto dr = make_hho_divergence_reconstruction(m_msh, cl, m_hdi);
            tc.toc();
            ai.time_divrec += tc.to_double();

            tc.tic();
            matrix_dynamic stab;
            if (m_hdi.cell_degree() == (m_hdi.face_degree() + 1))
            {
                stab = make_vector_hdg_stabilization(m_msh, cl, m_hdi);
            }
            else
            {
                stab = make_vector_hho_stabilization(m_msh, cl, sgr.first, m_hdi);
            }
            tc.toc();
            ai.time_stab += tc.to_double();

            tc.tic();
            auto                 cb       = disk::make_vector_monomial_basis(m_msh, cl, m_hdi.cell_degree());
            const auto           cell_rhs = make_rhs(m_msh, cl, cb, lf, 1);
            const matrix_dynamic loc =
              2.0 * m_elas_parameters.mu * (sg.second + stab) + m_elas_parameters.lambda * dr.second;
            const auto scnp = make_vector_static_condensation_withMatrix(m_msh, cl, m_hdi, loc, cell_rhs);

            m_AL.push_back(std::get<1>(scnp));
            m_bL.push_back(std::get<2>(scnp));
            tc.toc();
            ai.time_statcond += tc.to_double();

            m_assembler.assemble(m_msh, cl, m_bnd, std::get<0>(scnp), 2);
        }

        m_assembler.impose_neumann_boundary_conditions(m_msh, m_bnd);
        m_assembler.finalize();

        ai.linear_system_size = m_assembler.LHS.rows();
        ai.time_assembly      = ai.time_gradrec + ai.time_divrec + ai.time_stab + ai.time_statcond;
        return ai;
    }

    solver_info
    solve(void)
    {
        solver_info si;

        size_t systsz = m_assembler.LHS.rows();
        size_t nnz    = m_assembler.LHS.nonZeros();

        if (verbose())
        {
            std::cout << "Starting linear solver..." << std::endl;
            std::cout << " * Solving for " << systsz << " unknowns." << std::endl;
            std::cout << " * Matrix fill: " << 100.0 * double(nnz) / (systsz * systsz) << "%" << std::endl;
        }

        timecounter tc;

        tc.tic();
        m_system_solution = vector_dynamic::Zero(systsz);

        disk::solvers::pardiso_params<scalar_type> pparams;
        mkl_pardiso(pparams, m_assembler.LHS, m_assembler.RHS, m_system_solution);

        tc.toc();
        si.time_solver = tc.to_double();

        return si;
    }

    template<typename LoadFunction>
    postprocess_info
    postprocess(const LoadFunction& lf)
    {
        const auto   fbs = disk::vector_basis_size(m_hdi.face_degree(), dimension - 1, dimension);
        const auto   cbs = disk::vector_basis_size(m_hdi.cell_degree(), dimension, dimension);

        postprocess_info pi;

        m_solution_data.reserve(m_msh.cells_size());

        const auto solF = m_assembler.expand_solution(m_msh, m_bnd, m_system_solution, 2);

        timecounter tc;
        tc.tic();
        size_t cell_i = 0;
        for (auto& cl : m_msh)
        {
            const auto fcs        = faces(m_msh, cl);
            const auto num_faces  = fcs.size();
            const auto total_dofs = cbs;// + num_faces * fbs;

            vector_dynamic xFs = vector_dynamic::Zero(num_faces * fbs);
            vector_dynamic x   = vector_dynamic::Zero(total_dofs);

            for (size_t face_i = 0; face_i < num_faces; face_i++)
            {
                const auto fc  = fcs[face_i];
                const auto eid = find_element_id(m_msh.faces_begin(), m_msh.faces_end(), fc);
                if (!eid.first)
                    throw std::invalid_argument("This is a bug: face not found");

                const auto face_id             = eid.second;
                xFs.segment(face_i * fbs, fbs) = solF.segment(face_id * fbs, fbs);
            }

            const vector_dynamic xT          = m_bL.at(cell_i) - m_AL.at(cell_i) * xFs;
            x.segment(0, cbs)                = xT;
//            x.segment(cbs, total_dofs - cbs) = xFs;
            m_solution_data.push_back(x);

            cell_i++;
        }
        tc.toc();
        pi.time_postprocess = tc.to_double();

        return pi;
    }

    template<typename AnalyticalSolution>
    scalar_type
    compute_l2_displacement_error(const AnalyticalSolution& as)
    {
        scalar_type err_dof = 0;

        const size_t cbs      = disk::vector_basis_size(m_hdi.cell_degree(), dimension, dimension);
        const int    diff_deg = m_hdi.face_degree() - m_hdi.cell_degree();
        const int    di       = std::max(diff_deg, 1);

        size_t cell_i = 0;

        for (auto& cl : m_msh)
        {
            const auto x = m_solution_data.at(cell_i++);

            const vector_dynamic true_dof = disk::project_function(m_msh, cl, m_hdi.cell_degree(), as, di);

            auto                 cb   = disk::make_vector_monomial_basis(m_msh, cl, m_hdi.cell_degree());
            const matrix_dynamic mass = disk::make_mass_matrix(m_msh, cl, cb);

            const vector_dynamic comp_dof = x.head(cbs);
            const vector_dynamic diff_dof = (true_dof - comp_dof);
            assert(comp_dof.size() == true_dof.size());
            err_dof += diff_dof.dot(mass * diff_dof);
        }

        return sqrt(err_dof);
    }

    template<typename AnalyticalSolution>
    scalar_type
    compute_l2_stress_error(const AnalyticalSolution& stress) const
    {
        const auto face_degree = m_hdi.face_degree();
        const auto cell_degree = m_hdi.cell_degree();
        const auto rec_degree  = m_hdi.reconstruction_degree();

        size_t      cell_i = 0;
        scalar_type error_stress(0.0);

        for (auto& cl : m_msh)
        {
            const auto           x   = m_solution_data.at(cell_i++);
            const auto           sgr = make_vector_hho_symmetric_laplacian(m_msh, cl, m_hdi);
            const vector_dynamic GTu = sgr.first * x;

            const auto           dr   = make_hho_divergence_reconstruction(m_msh, cl, m_hdi);
            const vector_dynamic divu = dr.first * x;

            auto cbas_v = disk::make_vector_monomial_basis(m_msh, cl, rec_degree);
            auto cbas_s = disk::make_scalar_monomial_basis(m_msh, cl, face_degree);

            auto qps = disk::integrate(m_msh, cl, 2 * rec_degree);
            for (auto& qp : qps)
            {
                const auto gphi   = cbas_v.eval_sgradients(qp.point());
                const auto GT_iqn = disk::eval(GTu, gphi, dimension);

                const auto divphi   = cbas_s.eval_functions(qp.point());
                const auto divu_iqn = disk::eval(divu, divphi);

                const auto sigma =
                  2.0 * m_elas_parameters.mu * GT_iqn +
                  m_elas_parameters.lambda * divu_iqn * static_matrix<scalar_type, dimension, dimension>::Identity();

                const auto stress_diff = (stress(qp.point()) - sigma).eval();

                error_stress += qp.weight() * stress_diff.squaredNorm();
            }
        }

        return sqrt(error_stress);
    }

    
};


void CrackExample(int argc, char **argv){
    
    using RealType = double;
    simulation_data sim_data = preprocessor::process_args(argc, argv);
    sim_data.print_simulation_data();

    // Building a cartesian mesh
    timecounter tc;
    tc.tic();

    sim_data.m_n_divs = 6;
    sim_data.m_k_degree = 0;
    RealType lx = 3.0;
    RealType ly = 3.0;
    size_t nx = 3;
    size_t ny = 3;
    typedef disk::mesh<RealType, 2, disk::generic_mesh_storage<RealType, 2>>  mesh_type;
    typedef disk::BoundaryConditions<mesh_type, false> boundary_type;
    mesh_type msh;

    cartesian_2d_mesh_builder<RealType> mesh_builder(lx,ly,nx,ny);
    mesh_builder.refine_mesh(sim_data.m_n_divs);
    mesh_builder.set_translation_data(0.0, 0.0);
    mesh_builder.build_mesh();
    mesh_builder.move_to_mesh_storage(msh);
    tc.toc();
    std::cout << bold << cyan << "Mesh generation: " << tc.to_double() << " seconds" << reset << std::endl;
    
    disk::hho_degree_info hho_di(sim_data.m_k_degree+1,sim_data.m_k_degree);
    boundary_type bnd(msh);
    
    auto null_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
            RealType x,y;
            x = pt.x();
            y = pt.y();
            static_vector<RealType, 2> f{0,0};
            return f;
    };
        

    
    
    // Elasticity Parameters
    ElasticityParameters material_data;
    material_data.mu     = 1;
    material_data.lambda = 1;
    size_t degree = sim_data.m_k_degree;

    size_t bc_Neumann_id = 0;
    size_t bc_Dirichlet_id = 1;
    size_t bc_Dirichlet_x_id = 2;
    size_t bc_Dirichlet_y_id = 3;
    RealType eps = 1.0e-8;
    bool p_one_Q = true;
    
    if (p_one_Q) {
        
        auto t_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
                RealType tx,ty;
                tx = 0.0;
                ty = 1.0;
                static_vector<RealType, 2> t{tx,ty};
                return t;
        };
        
        for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
        {
            auto face = *face_it;
            mesh_type::point_type bar = barycenter(msh, face);
            auto fc_id = msh.lookup(face);
            if (bar.x() < 1.0 && (std::fabs(bar.y()) < eps)) {
                disk::bnd_info bi{bc_Neumann_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if(bar.x() > 1.0 && (std::fabs(bar.y()) < eps)){
                disk::bnd_info bi{bc_Dirichlet_y_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.x()) < eps) {
                disk::bnd_info bi{bc_Dirichlet_x_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.x()-lx) < eps || std::fabs(bar.y()-ly) < eps) {
                disk::bnd_info bi{bc_Dirichlet_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
        }
        bnd.addDirichletBC(disk::DIRICHLET, bc_Dirichlet_id, null_fun);
        bnd.addDirichletBC(disk::DX, bc_Dirichlet_x_id, null_fun);
        bnd.addDirichletBC(disk::DY, bc_Dirichlet_y_id, null_fun);
        bnd.addNeumannBC(disk::NEUMANN, bc_Neumann_id, t_fun);
    }else{
        
        auto t_fun = [](const mesh_type::point_type& pt) -> static_vector<RealType, 2> {
                RealType tx,ty;
                tx = 0.0;
                ty = 1.0;
                static_vector<RealType, 2> t{tx,ty};
                return t;
        };
        
        for (auto face_it = msh.boundary_faces_begin(); face_it != msh.boundary_faces_end(); face_it++)
        {
            auto face = *face_it;
            mesh_type::point_type bar = barycenter(msh, face);
            auto fc_id = msh.lookup(face);
//            if (bar.x() < 1.0 && (std::fabs(bar.y()) < eps)) {
//                disk::bnd_info bi{bc_Neumann_id, true};
//                msh.backend_storage()->boundary_info.at(fc_id) = bi;
//                continue;
//            }
            
            if(bar.x() > 1.0 && (std::fabs(bar.y()) < eps)){
                disk::bnd_info bi{bc_Dirichlet_y_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.x()) < eps) {
                disk::bnd_info bi{bc_Dirichlet_x_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
            if (std::fabs(bar.y()-ly) < eps) {
                disk::bnd_info bi{bc_Neumann_id, true};
                msh.backend_storage()->boundary_info.at(fc_id) = bi;
                continue;
            }
            
        }
//        bnd.addDirichletBC(disk::DIRICHLET, bc_Dirichlet_id, null_fun);
        bnd.addDirichletBC(disk::DX, bc_Dirichlet_x_id, null_fun);
        bnd.addDirichletBC(disk::DY, bc_Dirichlet_y_id, null_fun);
        bnd.addNeumannBC(disk::NEUMANN, bc_Neumann_id, t_fun);
    }
    
    

    linear_elasticity_solver_c<mesh_type> le(msh, bnd, material_data, degree, 1);
    le.verbose(true);
    
    le.changeElasticityParameters(material_data);
    assembly_info    assembling_info = le.assemble(null_fun);
    solver_info      solve_info      = le.solve();
    le.postprocess(null_fun);
    
    // render silo
    {
        std::string silo_file_name = "single_fracture";
        
        timecounter tc;
        tc.tic();
        
        auto dim = mesh_type::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> approx_ux, approx_uy;
        size_t cell_dof = disk::vector_basis_size(hho_di.cell_degree(), dim, dim);

        approx_ux.reserve( num_points );
        approx_uy.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh)
        {
            auto points = cell.point_ids();
            size_t n_p = points.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = points[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }

        for (auto& pt_id : point_to_cell)
        {
            auto bar = *std::next(msh.points_begin(), pt_id.first);
            cell_i = pt_id.second;
            auto cell = *std::next(msh.cells_begin(), cell_i);
            
            // vector evaluation
            {
                auto cell_basis = make_vector_monomial_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> vec_x_cell_dof = le.m_solution_data.at(cell_i);
                auto t_phi = cell_basis.eval_functions( bar );
                assert(t_phi.rows() == cell_basis.size());
                auto uh = disk::eval(vec_x_cell_dof, t_phi);
                approx_ux.push_back(uh(0,0));
                approx_uy.push_back(uh(1,0));
            }
        }

        disk::silo_database silo;
        silo_file_name += std::to_string(0) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        disk::silo_nodal_variable<double> vx_silo("ux", approx_ux);
        disk::silo_nodal_variable<double> vy_silo("uy", approx_uy);
        silo.add_variable("mesh", vx_silo);
        silo.add_variable("mesh", vy_silo);

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
}
