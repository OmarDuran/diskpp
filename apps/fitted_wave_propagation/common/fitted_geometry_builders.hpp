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
#include <sstream>
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
    
    // build the mesh
    virtual bool build_mesh()  = 0;
    
    // move generated mesh data to an external mesh storage
    virtual void move_to_mesh_storage(MESH& msh) = 0;
    
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
    
    struct polygon_2d
    {
        std::vector<size_t>                 m_member_nodes;
        std::set<std::array<size_t, 2>>     m_member_edges;
        bool operator<(const polygon_2d & other) {
            return m_member_nodes < other.m_member_nodes;
        }
    };
    
    std::vector<point_type>                         points;
    std::vector<node_type>                          vertices;
    std::vector<std::array<size_t, 2>>              facets;
    std::vector<std::array<size_t, 2>>              skeleton_edges;
    std::vector<std::array<size_t, 2>>              boundary_edges;
    std::vector<polygon_2d>                         polygons;
    

    T m_lx = 0.0;
    T m_ly = 0.0;
            
    T m_x_t = 0.0;
    T m_y_t = 0.0;
    
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
        
        size_t n_polygons = m_nx * m_ny;
        polygons.reserve(n_polygons);
    }
            
    void validate_edge(std::array<size_t, 2> & edge){
        assert(edge[0] != edge[1]);
        if (edge[0] > edge[1]){
            std::swap(edge[0], edge[1]);
        }
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
        
        reserve_storage();
        
        std::vector<T> range_x(m_nx+1,0.0);
        T dx = m_lx/T(m_nx);
        for (unsigned int i = 0; i <= m_nx; i++) {
            range_x[i] = i*dx;
        }
        
        std::vector<T> range_y(m_ny+1,0.0);
        T dy = m_ly/T(m_ny);
        for (unsigned int i = 0; i <= m_ny; i++) {
            range_y[i] = i*dy;
        }
        
        size_t node_id = 0;
        for (unsigned int j = 0; j <= m_ny; j++) {
            T yv = range_y[j] + m_y_t;
            for (unsigned int i = 0; i <= m_nx; i++) {
                T xv = range_x[i] + m_x_t;
                point_type point(xv, yv);
                points.push_back(point);
                vertices.push_back(node_type(point_identifier<2>(node_id)));
                node_id++;
            }
        }
        
        size_t edge_id = 0;
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                
                size_t id_0 = i + j * (m_nx + 1);
                size_t id_1 = id_0 + 1;
                size_t id_2 = i + (m_nx + 1) + 1 + j* (m_nx + 1);
                size_t id_3 = id_2 - 1;
                
                // Adding edges: Cases to avoid edge duplicity
                if (i==0 && j==0) {
            
                    std::array<size_t, 2> e0 = {id_0,id_1};
                    validate_edge(e0);
                    facets.push_back( e0 );
                    boundary_edges.push_back( e0 );
                    edge_id++;
                    
                    std::array<size_t, 2> e1 = {id_1,id_2};
                    validate_edge(e1);
                    facets.push_back( e1 );
                    if(j == m_ny - 1) boundary_edges.push_back( e1 );
                    edge_id++;
                    
                    std::array<size_t, 2> e2 = {id_2,id_3};
                    validate_edge(e2);
                    facets.push_back( e2 );
                    edge_id++;
                    
                    std::array<size_t, 2> e3 = {id_3,id_0};
                    validate_edge(e3);
                    facets.push_back( e3 );
                    boundary_edges.push_back( e3 );
                    edge_id++;
                }
                
                if ((i>0 && i < m_nx) && j==0) {
                    std::array<size_t, 2> e0 = {id_0,id_1};
                    validate_edge(e0);
                    facets.push_back( e0 );
                    boundary_edges.push_back( e0 );
                    edge_id++;
                    
                    std::array<size_t, 2> e1 = {id_1,id_2};
                    validate_edge(e1);
                    facets.push_back( e1 );
                    if(i == m_nx - 1) boundary_edges.push_back( e1 );
                    edge_id++;
                    
                    std::array<size_t, 2> e2 = {id_2,id_3};
                    validate_edge(e2);
                    facets.push_back( e2 );
                    if(j == m_ny - 1) boundary_edges.push_back( e2 );
                    edge_id++;
                }
                if (i==0 && j>0) {
                    
                    std::array<size_t, 2> e1 = {id_1,id_2};
                    validate_edge(e1);
                    facets.push_back( e1 );
                    edge_id++;
                    
                    std::array<size_t, 2> e2 = {id_2,id_3};
                    validate_edge(e2);
                    facets.push_back( e2 );
                    if(j == m_ny - 1) boundary_edges.push_back( e2 );
                    edge_id++;
                    
                    std::array<size_t, 2> e3 = {id_3,id_0};
                    validate_edge(e3);
                    boundary_edges.push_back( e3 );
                    facets.push_back( e3 );
                    edge_id++;
                }
                
                if (i>0 && j>0) {
                    
                    std::array<size_t, 2> e1 = {id_1,id_2};
                    validate_edge(e1);
                    facets.push_back( e1 );
                    if(i == m_nx - 1) boundary_edges.push_back( e1 );
                    edge_id++;
                    
                    std::array<size_t, 2> e2 = {id_2,id_3};
                    validate_edge(e2);
                    facets.push_back( e2 );
                    if(j == m_ny - 1) boundary_edges.push_back( e2 );
                    edge_id++;

                }
            }
        }
        
        size_t surface_id = 0;
        for (size_t j = 0; j < m_ny; j++) {
            for (size_t i = 0; i < m_nx; i++) {
                
                size_t id_0 = i + j * (m_nx + 1);
                size_t id_1 = id_0 + 1;
                size_t id_2 = i + (m_nx + 1) + 1 + j* (m_nx + 1);
                size_t id_3 = id_2 - 1;
            
                polygon_2d polygon;
                polygon.m_member_nodes = {id_0,id_1,id_2,id_3};
                std::array<size_t, 2> e0 = {id_0,id_1};
                validate_edge(e0);
                std::array<size_t, 2> e1 = {id_1,id_2};
                validate_edge(e1);
                std::array<size_t, 2> e2 = {id_2,id_3};
                validate_edge(e2);
                std::array<size_t, 2> e3 = {id_3,id_0};
                validate_edge(e3);
            
                polygon.m_member_edges = {e0,e1,e2,e3};
                polygons.push_back( polygon );
                surface_id++;
                
            }
        }
            
        return true;
    }


    
    void move_to_mesh_storage(mesh_type& msh){
        
        auto storage = msh.backend_storage();
        storage->points = std::move(points);
        storage->nodes = std::move(vertices);
        
        std::vector<edge_type> edges;
        edges.reserve(facets.size());
        for (size_t i = 0; i < facets.size(); i++)
        {
            assert(facets[i][0] < facets[i][1]);
            auto node1 = typename node_type::id_type(facets[i][0]);
            auto node2 = typename node_type::id_type(facets[i][1]);

            auto e = edge_type{{node1, node2}};

            e.set_point_ids(facets[i].begin(), facets[i].end());
            edges.push_back(e);
        }
        std::sort(edges.begin(), edges.end());
            
        storage->boundary_info.resize(edges.size());
        for (size_t i = 0; i < boundary_edges.size(); i++)
        {
            assert(boundary_edges[i][0] < boundary_edges[i][1]);
            auto node1 = typename node_type::id_type(boundary_edges[i][0]);
            auto node2 = typename node_type::id_type(boundary_edges[i][1]);

            auto e = edge_type{{node1, node2}};

            auto position = find_element_id(edges.begin(), edges.end(), e);

            if (position.first == false)
            {
                std::cout << "Bad bug at " << __FILE__ << "("
                          << __LINE__ << ")" << std::endl;
                return;
            }

                disk::bnd_info bi{0, true};
                storage->boundary_info.at(position.second) = bi;
        }
            
        storage->edges = std::move(edges);
            
        std::vector<surface_type> surfaces;
        surfaces.reserve( polygons.size() );

        for (auto& p : polygons)
        {
            std::vector<typename edge_type::id_type> surface_edges;
            for (auto& e : p.m_member_edges)
            {
                assert(e[0] < e[1]);
                auto n1 = typename node_type::id_type(e[0]);
                auto n2 = typename node_type::id_type(e[1]);

                edge_type edge{{n1, n2}};
                auto edge_id = find_element_id(storage->edges.begin(),
                                               storage->edges.end(), edge);
                if (!edge_id.first)
                {
                    std::cout << "Bad bug at " << __FILE__ << "("
                              << __LINE__ << ")" << std::endl;
                    return;
                }

                surface_edges.push_back(edge_id.second);
            }
            auto surface = surface_type(surface_edges);
            surface.set_point_ids(p.m_member_nodes.begin(), p.m_member_nodes.end());
            surfaces.push_back( surface );
        }

        std::sort(surfaces.begin(), surfaces.end());
        storage->surfaces = std::move(surfaces);
        
    }
    
    void set_translation_data(T x_t, T y_t){
        m_x_t = x_t;
        m_y_t = y_t;
    }
            
    size_t get_nx(){
            return m_nx;
    }
            
    size_t get_ny(){
            return m_ny;
    }
            
    // Print in log file relevant mesh information
    void print_log_file(){
        fitted_geometry_builder<mesh_type>::m_n_elements = polygons.size();
        std::ofstream file;
        file.open (fitted_geometry_builder<mesh_type>::m_log_file.c_str());
        file << "Number of surfaces : " << polygons.size() << std::endl;
        file << "Number of skeleton edges : " << skeleton_edges.size() << std::endl;
        file << "Number of boundary edges : " << boundary_edges.size() << std::endl;
        file << "Number of vertices : " << vertices.size() << std::endl;
        file.close();
    }
    
};


template<typename T>
class polygon_2d_mesh_reader : public fitted_geometry_builder<disk::generic_mesh<T, 2>>
{
    typedef disk::generic_mesh<T,2>                 mesh_type;
    typedef typename mesh_type::point_type          point_type;
    typedef typename mesh_type::node_type           node_type;
    typedef typename mesh_type::edge_type           edge_type;
    typedef typename mesh_type::surface_type        surface_type;
    
    struct polygon_2d
    {
        std::vector<size_t>                 m_member_nodes;
        std::set<std::array<size_t, 2>>     m_member_edges;
        bool operator<(const polygon_2d & other) {
            return m_member_nodes < other.m_member_nodes;
        }
    };
    
    std::vector<point_type>                         points;
    std::vector<node_type>                          vertices;
    std::vector<std::array<size_t, 2>>              facets;
    std::vector<std::array<size_t, 2>>              skeleton_edges;
    std::vector<std::array<size_t, 2>>              boundary_edges;
    std::vector<polygon_2d>                         polygons;
    std::string poly_mesh_file;
    std::set<size_t> bc_points;
    
    void clear_storage() {
        points.clear();
        vertices.clear();
        skeleton_edges.clear();
        boundary_edges.clear();
        polygons.clear();
        bc_points.clear();
    }
    
    void reserve_storage(){
        
        std::ifstream input;
        input.open(poly_mesh_file.c_str());
        
        size_t n_points, n_polygons, n_bc_curves, n_bc_edges, n_edges;
        if (input.is_open()) {
            std::string line;
            std::getline(input, line);
            std::stringstream(line) >> n_points >> n_polygons >> n_bc_curves;
            for(size_t id = 0; id < n_points; id++){
                if(std::getline(input, line)){

                }
                else{
                    break;
                }
            }
            
            n_edges = 0;
            n_bc_edges = 0;
            size_t n_polygon_vertices;
            for(size_t surface_id=0; surface_id < n_polygons; surface_id++)
            {
                if(std::getline(input, line)){
                    std::stringstream(line) >> n_polygon_vertices;
                    n_edges += n_polygon_vertices;
                  }
                  else{
                      break;
                  }
            }
            
            bc_points.clear();
            size_t bc_point_id;
            for(size_t bc_id=0; bc_id < n_bc_curves; bc_id++)
            {
                if(std::getline(input, line)){
                    std::stringstream input_line(line);
                    while(!input_line.eof()){
                        input_line >> bc_point_id;
                        bc_point_id--;
                        bc_points.insert(bc_point_id);
                    }
                }
                else{
                  break;
                }
            }
            n_bc_edges = bc_points.size();
            
            points.reserve(n_points);
            vertices.reserve(n_points);
            
            size_t n_skel_edges = n_edges - n_bc_edges;
            skeleton_edges.reserve(n_skel_edges);
            boundary_edges.reserve(n_bc_edges);
            polygons.reserve(n_polygons);
            
        }
    }
            
    void validate_edge(std::array<size_t, 2> & edge){
        assert(edge[0] != edge[1]);
        if (edge[0] > edge[1]){
            std::swap(edge[0], edge[1]);
        }
    }
    
public:

    polygon_2d_mesh_reader() : fitted_geometry_builder<mesh_type>()
    {
        fitted_geometry_builder<mesh_type>::m_dimension = 2;
    }
    
    void set_poly_mesh_file(std::string mesh_file){
        poly_mesh_file = mesh_file;
    }
    
    // build the mesh
    bool build_mesh(){
        
        clear_storage();
        reserve_storage();
        
        std::ifstream input;
        input.open(poly_mesh_file.c_str());
        
        size_t n_points, n_polygons, n_bc_curves;
        if (input.is_open()) {
            std::string line;
            std::getline(input, line);
            std::stringstream(line) >> n_points >> n_polygons >> n_bc_curves;
            
            T xv, yv;
            for(size_t id = 0; id < n_points; id++){
                if(std::getline(input, line)){
                    std::stringstream(line) >> xv >> yv;
                    point_type point(xv, yv);
                    points.push_back(point);
                    vertices.push_back(node_type(point_identifier<2>(id)));
                }
                else{
                    break;
                }
            }
            
            size_t n_polygon_vertices, id;
            for(size_t surface_id=0; surface_id < n_polygons; surface_id++)
            {
                if(std::getline(input, line)){
                    std::stringstream input_line(line);
                    input_line >> n_polygon_vertices;
                    
                    polygon_2d polygon;
                    std::vector<size_t> member_nodes;
                    for (size_t i = 0; i < n_polygon_vertices; i++) {
                        input_line >> id;
                        id--;
                        member_nodes.push_back(id);
                    }
                    polygon.m_member_nodes = member_nodes;
                    
                    assert(member_nodes.size() == n_polygon_vertices);
                    
                    std::set< std::array<size_t, 2> > member_edges;
                    std::array<size_t, 2> edge;
                    for (size_t i = 0; i < member_nodes.size(); i++) {
                        
                        if (i == n_polygon_vertices - 1) {
                            edge = {member_nodes[i],member_nodes[0]};
                        }else{
                            edge = {member_nodes[i],member_nodes[i+1]};
                        }
                        
                        validate_edge(edge);
                        facets.push_back( edge );
                        member_edges.insert(edge);
                        
                        bool is_bc_point_l_Q = bc_points.find(edge.at(0)) != bc_points.end();
                        bool is_bc_point_r_Q = bc_points.find(edge.at(1)) != bc_points.end();
                        if (is_bc_point_l_Q && is_bc_point_r_Q) {
                            boundary_edges.push_back( edge );
                        }
                    }
                    
                    polygon.m_member_edges = member_edges;
                    polygons.push_back( polygon );
                    
                  }
                  else{
                      break;
                  }
            }
        }
        
        return true;
    }
    
    void move_to_mesh_storage(mesh_type& msh){
        
        auto storage = msh.backend_storage();
        storage->points = std::move(points);
        storage->nodes = std::move(vertices);
        
        std::vector<edge_type> edges;
        edges.reserve(facets.size());
        for (size_t i = 0; i < facets.size(); i++)
        {
            assert(facets[i][0] < facets[i][1]);
            auto node1 = typename node_type::id_type(facets[i][0]);
            auto node2 = typename node_type::id_type(facets[i][1]);

            auto e = edge_type{{node1, node2}};

            e.set_point_ids(facets[i].begin(), facets[i].end());
            edges.push_back(e);
        }
        std::sort(edges.begin(), edges.end());
            
        storage->boundary_info.resize(edges.size());
        for (size_t i = 0; i < boundary_edges.size(); i++)
        {
            assert(boundary_edges[i][0] < boundary_edges[i][1]);
            auto node1 = typename node_type::id_type(boundary_edges[i][0]);
            auto node2 = typename node_type::id_type(boundary_edges[i][1]);

            auto e = edge_type{{node1, node2}};

            auto position = find_element_id(edges.begin(), edges.end(), e);

            if (position.first == false)
            {
                std::cout << "Bad bug at " << __FILE__ << "("
                          << __LINE__ << ")" << std::endl;
                return;
            }

                disk::bnd_info bi{0, true};
                storage->boundary_info.at(position.second) = bi;
        }
            
        storage->edges = std::move(edges);
            
        std::vector<surface_type> surfaces;
        surfaces.reserve( polygons.size() );

        for (auto& p : polygons)
        {
            std::vector<typename edge_type::id_type> surface_edges;
            for (auto& e : p.m_member_edges)
            {
                assert(e[0] < e[1]);
                auto n1 = typename node_type::id_type(e[0]);
                auto n2 = typename node_type::id_type(e[1]);

                edge_type edge{{n1, n2}};
                auto edge_id = find_element_id(storage->edges.begin(),
                                               storage->edges.end(), edge);
                if (!edge_id.first)
                {
                    std::cout << "Bad bug at " << __FILE__ << "("
                              << __LINE__ << ")" << std::endl;
                    return;
                }

                surface_edges.push_back(edge_id.second);
            }
            auto surface = surface_type(surface_edges);
            surface.set_point_ids(p.m_member_nodes.begin(), p.m_member_nodes.end());
            surfaces.push_back( surface );
        }

        std::sort(surfaces.begin(), surfaces.end());
        storage->surfaces = std::move(surfaces);
        
    }
            
    // Print in log file relevant mesh information
    void print_log_file(){
        fitted_geometry_builder<mesh_type>::m_n_elements = polygons.size();
        std::ofstream file;
        file.open (fitted_geometry_builder<mesh_type>::m_log_file.c_str());
        file << "Number of surfaces : " << polygons.size() << std::endl;
        file << "Number of skeleton edges : " << skeleton_edges.size() << std::endl;
        file << "Number of boundary edges : " << boundary_edges.size() << std::endl;
        file << "Number of vertices : " << vertices.size() << std::endl;
        file.close();
    }
    
};

#endif /* fitted_geometry_builder_hpp */



