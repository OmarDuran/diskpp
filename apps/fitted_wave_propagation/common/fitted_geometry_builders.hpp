//
//  fitted_geometry_builder.hpp
//  acoustics
//
//  Created by Omar Durán on 4/8/20.
//

#pragma once
#ifndef fitted_geometry_builder_hpp
#define fitted_geometry_builder_hpp

#include <vector>
#include <array>
#include <fstream>
#include <cassert>
#include <thread>
#include <set>
#include "geometry/geometry.hpp"

template<typename MESH>
class fitted_geometry_builder {
    
protected:
    
    std::string m_log_file = "mesh_log.txt";
    
    size_t m_dimension = 0;
    
    size_t m_n_elements = 0;
    
    MESH    m_mesh;
    
public:
    
    fitted_geometry_builder() {
        
    }
    
    // set the log file name
    void set_log_file(std::string log_file) {
        m_log_file = log_file;
    }
    
    // get the geometry dimension
    size_t get_dimension() {
        return m_dimension;
    }
    
    // get the number of elements
    size_t get_n_elements() {
        return m_n_elements;
    }
    
    // get the mesh
    MESH & get_mesh() {
        return m_mesh;
    }
    
    // build the mesh
    virtual bool build_mesh()  = 0;
    
    // Print in log file relevant mesh information
    virtual void print_log_file()    = 0;
    
    virtual ~ fitted_geometry_builder() {
        
    }
    
};

template<typename T>
class cartesian_2d_mesh_builder : public fitted_geometry_builder<disk::generic_mesh<T, 2>>
{
    typedef disk::generic_mesh<T,2>                 mesh_type;
    typedef typename mesh_type::point_type          point_type;
    typedef typename mesh_type::node_type           node_type;
    typedef typename mesh_type::edge_type           edge_type;
    typedef typename mesh_type::surface_type        surface_type;
    
    mesh_type                                       msh;
    std::vector<point_type>                         points;
    std::vector<node_type>                          vertices;
    std::vector<std::pair<size_t,edge_type>>        edges;
    std::vector<std::pair<size_t,edge_type>>        skeleton_edges;
    std::vector<std::pair<size_t,edge_type>>        boundary_edges;
    std::vector<std::pair<size_t,surface_type>>     surfaces;

    T m_lx = 0.0;
    T m_ly = 0.0;
    
    size_t m_nx = 0;
    size_t m_ny = 0;
    
    void reserve_storage(){
        size_t n_points = (m_nx + 1) * (m_ny + 1);
        points.reserve(n_points);
        vertices.reserve(n_points);
        
        size_t n_edges = 2*m_nx*m_ny + m_nx + m_ny;
        size_t n_skel_edges = 2*m_nx*m_ny + m_nx + m_ny;
        size_t n_bc_edges = n_edges - n_skel_edges;
        skeleton_edges.reserve(n_skel_edges);
        boundary_edges.reserve(n_bc_edges);
        
        size_t n_surfaces = m_nx * m_ny;
        surfaces.reserve(n_surfaces);
    }
    
public:

    cartesian_2d_mesh_builder(T lx, T ly, size_t nx, size_t ny) : fitted_geometry_builder<mesh_type>()
    {
        m_lx = lx;
        m_ly = ly;
        m_nx = nx;
        m_ny = ny;
        fitted_geometry_builder<mesh_type>::m_dimension = 2;
    }
    
    // uniform refinement x-direction
    void refine_mesh_x_direction(size_t n_refinements){
        for (unsigned int i = 0; i < n_refinements; i++) {
            m_nx *= 2;
        }
    }
    
    // uniform refinement y-direction
    void refine_mesh_y_direction(size_t n_refinements){
        for (unsigned int i = 0; i < n_refinements; i++) {
            m_ny *= 2;
        }
    }
    
    // uniform refinement
    void refine_mesh(size_t n_refinements){
        refine_mesh_x_direction(n_refinements);
        refine_mesh_y_direction(n_refinements);
    }
    
    // build the mesh
    bool build_mesh(){
        
        std::vector<T> range_x(m_nx+1);
        T dx = m_lx/T(m_nx);
        for (unsigned int i = 0; i <= m_nx; i++) {
            range_x[i] = i*dx;
        }
        
        std::vector<T> range_y(m_ny+1);
        T dy = m_ly/T(m_ny);
        for (unsigned int i = 0; i <= m_ny; i++) {
            range_y[i] = i*dy;
        }
        
        size_t node_id = 0;
        for (unsigned int j = 0; j <= m_ny; j++) {
            T yv = range_y[j];
            for (unsigned int i = 0; i <= m_nx; i++) {
                T xv = range_y[i];
                point_type point(xv, yv);
                points.push_back(point);
                vertices.push_back(node_type(point_identifier<2>(node_id)));
                node_id++;
            }
        }
        
        size_t edge_id = 0;
        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {
                
                size_t id_0 = i + (j - 1) * (m_nx + 1);
                size_t id_1 = id_0 + 1;
                size_t id_2 = i + (m_nx + 1) + 1 + (j - 1)* (m_nx + 1);
                size_t id_3 = id_2 - 1;
                
                // Adding edges: Cases to avoid edge duplicity
                if (i==0 && j==0) {
                    auto e0 = edge_type({
                        typename node_type::id_type(id_0),
                        typename node_type::id_type(id_1)});
                    edges.push_back( std::make_pair(edge_id, e0) );
                    edge_id++;
                    
                    auto e1 = edge_type({
                        typename node_type::id_type(id_1),
                        typename node_type::id_type(id_2)});
                    edges.push_back( std::make_pair(edge_id, e1) );
                    edge_id++;
                    
                    auto e2 = edge_type({
                        typename node_type::id_type(id_2),
                        typename node_type::id_type(id_3)});
                    edges.push_back( std::make_pair(edge_id, e2) );
                    edge_id++;
                    
                    auto e3 = edge_type({
                        typename node_type::id_type(id_3),
                        typename node_type::id_type(id_1)});
                    edges.push_back( std::make_pair(edge_id, e3) );
                    edge_id++;
                }
                
                if ((i>=1 && i < m_nx) && j==0) {
                    auto e0 = edge_type({
                        typename node_type::id_type(id_0),
                        typename node_type::id_type(id_1)});
                    edges.push_back( std::make_pair(edge_id, e0) );
                    edge_id++;
                    
                    auto e1 = edge_type({
                        typename node_type::id_type(id_1),
                        typename node_type::id_type(id_2)});
                    edges.push_back( std::make_pair(edge_id, e1) );
                    edge_id++;
                    
                    auto e2 = edge_type({
                        typename node_type::id_type(id_2),
                        typename node_type::id_type(id_3)});
                    edges.push_back( std::make_pair(edge_id, e2) );
                    edge_id++;
                }
                if (i==1 && j>=1) {
                    
                    auto e1 = edge_type({
                        typename node_type::id_type(id_1),
                        typename node_type::id_type(id_2)});
                    edges.push_back( std::make_pair(edge_id, e1) );
                    edge_id++;
                    
                    auto e2 = edge_type({
                        typename node_type::id_type(id_2),
                        typename node_type::id_type(id_3)});
                    edges.push_back( std::make_pair(edge_id, e2) );
                    edge_id++;
                    
                    auto e3 = edge_type({
                        typename node_type::id_type(id_3),
                        typename node_type::id_type(id_1)});
                    edges.push_back( std::make_pair(edge_id, e3) );
                    edge_id++;
                }
                
                if (i>=1 && j>=1) {
                    
                    auto e1 = edge_type({
                        typename node_type::id_type(id_1),
                        typename node_type::id_type(id_2)});
                    edges.push_back( std::make_pair(edge_id, e1) );
                    edge_id++;
                    
                    auto e2 = edge_type({
                        typename node_type::id_type(id_2),
                        typename node_type::id_type(id_3)});
                    edges.push_back( std::make_pair(edge_id, e2) );
                    edge_id++;

                }
            }
        }
        
        size_t surface_id = 0;
        for (size_t j = 0; j <= m_ny; j++) {
            for (size_t i = 0; i <= m_nx; i++) {
                
                size_t id_0 = i + (j - 1) * (m_nx + 1);
                size_t id_1 = id_0 + 1;
                size_t id_2 = i + (m_nx + 1) + 1 + (j - 1)* (m_nx + 1);
                size_t id_3 = id_2 - 1;
                
                auto surface_edges = {
                    typename edge_type::id_type(id_0),
                    typename edge_type::id_type(id_1),
                    typename edge_type::id_type(id_2),
                    typename edge_type::id_type(id_3)};
                auto surface = surface_type(surface_edges);
                surfaces.push_back( std::make_pair(surface_id, surface) );
                surface_id++;
                
            }
        }
        
        return true;
    }
    
    // Print in log file relevant mesh information
    void print_log_file(){
        fitted_geometry_builder<mesh_type>::m_n_elements = surfaces.size();
        std::ofstream file;
        file.open (fitted_geometry_builder<mesh_type>::m_log_file.c_str());
        file << "Number of surfaces : " << surfaces.size() << std::endl;
        file << "Number of skeleton edges : " << skeleton_edges.size() << std::endl;
        file << "Number of boundary edges : " << boundary_edges.size() << std::endl;
        file << "Number of vertices : " << vertices.size() << std::endl;
        file.close();
    }
    
};

#endif /* fitted_geometry_builder_hpp */



