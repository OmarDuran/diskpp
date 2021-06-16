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
          
        // Duplicated facets are eliminated
        std::sort( facets.begin(), facets.end() );
        facets.erase( std::unique( facets.begin(), facets.end() ), facets.end() );
        
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

template<typename T>
class line_1d_mesh_reader : public
fitted_geometry_builder<disk::generic_mesh<T,1>>
//fitted_geometry_builder<disk::mesh<T, 1, disk::generic_mesh_storage<T, 1>>>

{
    typedef disk::generic_mesh<T,1>                 mesh_type;
//    typedef disk::mesh<T, 1, disk::generic_mesh_storage<T, 1>>                 mesh_type;
    typedef typename mesh_type::point_type          point_type;
    typedef typename mesh_type::node_type           node_type;
    typedef typename mesh_type::edge_type           edge_type;
    
    struct line_cell
    {
        std::vector<size_t>                 m_member_nodes;
        std::set<std::array<size_t, 2>>     m_member_edges;
        bool operator<(const line_cell & other) {
            return m_member_nodes < other.m_member_nodes;
        }
    };
    
    std::vector<point_type>                         points;
    std::vector<node_type>                          vertices;
    std::vector<std::array<size_t, 2>>              facets;
    std::vector<size_t>                             skeleton_edges;
    std::vector<size_t>                             boundary_edges;
    std::vector<line_cell>                          lines;
    std::string line_mesh_file;
    std::set<size_t> bc_points;
    
    void clear_storage() {
        points.clear();
        vertices.clear();
        skeleton_edges.clear();
        boundary_edges.clear();
        lines.clear();
        bc_points.clear();
    }
    
    void reserve_storage(){
        
        std::ifstream input;
        input.open(line_mesh_file.c_str());
        
        size_t n_points, n_lines, n_bc_edges, n_edges;
        if (input.is_open()) {
            std::string line;
            std::getline(input, line);
            std::stringstream(line) >> n_points >> n_lines >> n_bc_edges;
            for(size_t id = 0; id < n_points; id++){
                if(std::getline(input, line)){

                }
                else{
                    break;
                }
            }

            n_edges = 0;
            size_t n_line_vertices;
            for(size_t line_id=0; line_id < n_lines; line_id++)
            {
                if(std::getline(input, line)){
                    std::stringstream(line) >> n_line_vertices;
                    n_edges += n_line_vertices;
                  }
                  else{
                      break;
                  }
            }

            bc_points.clear();
            size_t bc_point_id;
            for(size_t bc_id=0; bc_id < n_bc_edges; bc_id++)
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
            lines.reserve(n_lines);

        }
    }
            
    void validate_edge(std::array<size_t, 2> & edge){
        assert(edge[0] != edge[1]);
        if (edge[0] > edge[1]){
            std::swap(edge[0], edge[1]);
        }
    }
    
public:

    line_1d_mesh_reader() : fitted_geometry_builder<mesh_type>()
    {
        fitted_geometry_builder<mesh_type>::m_dimension = 1;
    }
    
    void set_line_mesh_file(std::string mesh_file){
        line_mesh_file = mesh_file;
    }
    
    // build the mesh
    bool build_mesh(){
        
        clear_storage();
        reserve_storage();
        
        std::ifstream input;
        input.open(line_mesh_file.c_str());
        
        size_t n_points, n_lines, n_bc_points;
        if (input.is_open()) {
            std::string line;
            std::getline(input, line);
            std::stringstream(line) >> n_points >> n_lines >> n_bc_points;

            T xv, yv;
            for(size_t id = 0; id < n_points; id++){
                if(std::getline(input, line)){
                    std::stringstream(line) >> xv >> yv;
//                    point_type point(xv, yv);
                    point_type point(xv);
                    points.push_back(point);
                    vertices.push_back(node_type(point_identifier<1>(id)));
                }
                else{
                    break;
                }
            }

            size_t n_line_vertices, id;
            for(size_t line_id=0; line_id < n_lines; line_id++)
            {
                if(std::getline(input, line)){
                    std::stringstream input_line(line);
                    input_line >> n_line_vertices;

                    line_cell line;
                    std::vector<size_t> member_nodes;
                    for (size_t i = 0; i < n_line_vertices; i++) {
                        input_line >> id;
                        id--;
                        member_nodes.push_back(id);
                    }
                    line.m_member_nodes = member_nodes;

                    assert(member_nodes.size() == n_line_vertices);

                    std::set< std::array<size_t, 2> > member_edges;
                    std::array<size_t, 2> edge;
                    for (size_t i = 0; i < member_nodes.size()-1; i++) {

                        edge = {member_nodes[i],member_nodes[i+1]};

                        validate_edge(edge);
                        facets.push_back( edge );
                        member_edges.insert(edge);

                        bool is_bc_point_l_Q = bc_points.find(member_nodes[i]) != bc_points.end();
                        bool is_bc_point_r_Q = bc_points.find(edge.at(1)) != bc_points.end();
                        if (is_bc_point_l_Q) {
                            boundary_edges.push_back( member_nodes[i] );
                        }
                        if (is_bc_point_r_Q) {
                            boundary_edges.push_back( edge.at(1) );
                        }
                    }

                    line.m_member_edges = member_edges;
                    lines.push_back( line );

                  }
                  else{
                      break;
                  }
            }
        }
          
        // Duplicated facets are eliminated
        std::sort( facets.begin(), facets.end() );
        facets.erase( std::unique( facets.begin(), facets.end() ), facets.end() );
        
        return true;
    }
    
    void move_to_mesh_storage(mesh_type& msh){
        
        auto storage = msh.backend_storage();
        storage->points = std::move(points);
        
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

        storage->boundary_info.resize(vertices.size());
        for (size_t i = 0; i < boundary_edges.size(); i++)
        {
            auto node = node_type(point_identifier<1>(boundary_edges[i]));
            auto position = find_element_id(vertices.begin(), vertices.end(), node);

            if (position.first == false)
            {
                std::cout << "Bad bug at " << __FILE__ << "("
                          << __LINE__ << ")" << std::endl;
                return;
            }

            disk::bnd_info bi{0, true};
            storage->boundary_info.at(position.second) = bi;
        }
        
        storage->nodes = std::move(vertices);
        storage->edges = std::move(edges);
        
    }
            
    // Print in log file relevant mesh information
    void print_log_file(){
        fitted_geometry_builder<mesh_type>::m_n_elements = lines.size();
        std::ofstream file;
        file.open (fitted_geometry_builder<mesh_type>::m_log_file.c_str());
        file << "Number of surfaces : " << lines.size() << std::endl;
        file << "Number of skeleton edges : " << skeleton_edges.size() << std::endl;
        file << "Number of boundary edges : " << boundary_edges.size() << std::endl;
        file << "Number of vertices : " << vertices.size() << std::endl;
        file.close();
    }
    
};

template<typename T>
class gmsh_2d_reader : public fitted_geometry_builder<disk::generic_mesh<T, 2>>
{
    typedef disk::generic_mesh<T,2>                 mesh_type;
    typedef typename mesh_type::point_type          point_type;
    typedef typename mesh_type::node_type           node_type;
    typedef typename mesh_type::edge_type           edge_type;
    typedef typename mesh_type::surface_type        surface_type;
    
    /// gmsh data
    struct gmsh_data {
        
        int m_n_volumes;
        int m_n_surfaces;
        int m_n_curves;
        int m_n_points;
        int m_n_physical_volumes;
        int m_n_physical_surfaces;
        int m_n_physical_curves;
        int m_n_physical_points;
        int m_dimension;
        std::vector<std::map<int,std::vector<int>>> m_dim_entity_tag_and_physical_tag;
        std::vector<std::map<int,std::string>> m_dim_physical_tag_and_name;
        std::vector<std::map<std::string,int>> m_dim_name_and_physical_tag;
        std::vector<std::map<int,int>> m_dim_physical_tag_and_physical_tag;
        std::vector<int> m_entity_index;
        
        gmsh_data() {
            m_n_volumes = 0;
            m_n_surfaces = 0;
            m_n_curves = 0;
            m_n_points = 0;
            m_n_physical_volumes = 0;
            m_n_physical_surfaces = 0;
            m_n_physical_curves = 0;
            m_n_physical_points = 0;
            m_dimension = 0;
            m_dim_entity_tag_and_physical_tag.resize(4);
            m_dim_physical_tag_and_name.resize(4);
            m_dim_name_and_physical_tag.resize(4);
            m_dim_physical_tag_and_physical_tag.resize(4);
        }
        
        gmsh_data(const gmsh_data &other) {
            m_n_volumes = other.m_n_volumes;
            m_n_surfaces = other.m_n_surfaces;
            m_n_curves = other.m_n_curves;
            m_n_points = other.m_n_points;
            m_n_physical_volumes = other.m_n_physical_volumes;
            m_n_physical_surfaces = other.m_n_physical_surfaces;
            m_n_physical_curves = other.m_n_physical_curves;
            m_n_physical_points = other.m_n_physical_points;
            m_dimension = other.m_dimension;
            m_dim_entity_tag_and_physical_tag   = other.m_dim_entity_tag_and_physical_tag;
            m_dim_physical_tag_and_name         = other.m_dim_physical_tag_and_name;
            m_dim_name_and_physical_tag         = other.m_dim_name_and_physical_tag;
            m_dim_physical_tag_and_physical_tag = other.m_dim_physical_tag_and_physical_tag;
        }
        
        gmsh_data &operator=(const gmsh_data &other){
            m_n_volumes = other.m_n_volumes;
            m_n_surfaces = other.m_n_surfaces;
            m_n_curves = other.m_n_curves;
            m_n_points = other.m_n_points;
            m_n_physical_volumes = other.m_n_physical_volumes;
            m_n_physical_surfaces = other.m_n_physical_surfaces;
            m_n_physical_curves = other.m_n_physical_curves;
            m_n_physical_points = other.m_n_physical_points;
            m_dimension = other.m_dimension;
            m_dim_entity_tag_and_physical_tag   = other.m_dim_entity_tag_and_physical_tag;
            m_dim_physical_tag_and_name         = other.m_dim_physical_tag_and_name;
            m_dim_name_and_physical_tag         = other.m_dim_name_and_physical_tag;
            m_dim_physical_tag_and_physical_tag = other.m_dim_physical_tag_and_physical_tag;
            return *this;
        }
        
    };
    
    struct polygon
    {
        std::vector<size_t>                 m_member_nodes;
        std::set<std::array<size_t, 2>>     m_member_edges;
        bool operator<(const polygon & other) {
            return m_member_nodes < other.m_member_nodes;
        }
    };
    
    std::vector<point_type>                         points;
    std::vector<node_type>                          vertices;
    std::vector<std::array<size_t, 2>>              facets;
    std::vector<std::array<size_t, 2>>              skeleton_edges;
    std::vector<std::array<size_t, 2>>              boundary_edges;
    std::vector<polygon>                         polygons;
    std::string poly_mesh_file;
    std::set<size_t> bc_points;
    gmsh_data   mesh_data;
 
    
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

    gmsh_2d_reader() : fitted_geometry_builder<mesh_type>()
    {
        fitted_geometry_builder<mesh_type>::m_dimension = 2;
    }
    
    void set_gmsh_file(std::string mesh_file){
        poly_mesh_file = mesh_file;
    }
    
    int GetNumberofNodes(int & cell_type){
        
        int n_nodes;
        switch (cell_type) {
            case 1:
            {   // Line
                n_nodes = 2;
            }
                break;
            case 2:
            {
                // Triangle
                n_nodes = 3;
            }
                break;
            case 3:
            {
                // Quadrilateral
                n_nodes = 4;
            }
                break;
            case 4:
            {
                // Tetrahedron
                n_nodes = 4;
            }
                break;
            case 5:
            {
                // Hexahedra
                n_nodes = 8;
            }
                break;
            case 6:
            {
                // Prism
                n_nodes = 6;
            }
                break;
            case 7:
            {
                // Pyramid
                n_nodes = 5;
            }
                break;
            case 8:
            {
                // Quadratic Line
                n_nodes = 3;
            }
                break;
            case 9:
            {
                // Quadratic Triangle
                n_nodes = 6;
            }
                break;
            case 10:
            {
                // Quadratic Quadrilateral
                n_nodes = 9;
            }
                break;
            case 11:
            {
                // Quadratic Tetrahedron
                n_nodes = 10;
                
            }
                break;
            case 12:
            {
                // Quadratic Hexahedra
                n_nodes = 20;
            }
                break;
            case 13:
            {
                // Quadratic Prism
                n_nodes = 15;
            }
                break;
            case 15:{
                // Point
                n_nodes = 1;
            }
                break;
            default:
            {
                std::cout << "Cell not impelemented." << std::endl;
                n_nodes = 0;
                assert(false);
            }
                break;
        }
        
        return n_nodes;
    }
    
    // build the mesh
    bool build_mesh(){
        
        int max_dimension = 0;
        {
            
            // reading a general mesh information by filter
            std::ifstream read (poly_mesh_file.c_str());
            bool file_check_Q = !read;
            if(file_check_Q)
            {
                std::cout << "Couldn't open the file." << poly_mesh_file << std::endl;
                assert(!file_check_Q);
            }
            
            if (file_check_Q) {
                std::cout << "File path is wrong." << std::endl;
                assert(!file_check_Q);
            }
            
            std::vector<int64_t> node_map;
            while(read)
            {
                char buf[1024];
                read.getline(buf, 1024);
                std::string str(buf);
                
                if(str == "$MeshFormat" || str == "$MeshFormat\r")
                {
                    read.getline(buf, 1024);
                    std::string str(buf);
                    std::cout << "Reading mesh format = " << str << std::endl;
                    
                }
                
                if(str == "$PhysicalNames" || str == "$PhysicalNames\r" )
                {
                    
                    int64_t n_physical_names;
                    read >> n_physical_names;
                    
                    int dimension, id;
                    std::string name;
                    std::pair<int, std::string> chunk;
                    
                    for (int64_t i_name = 0; i_name < n_physical_names; i_name++) {
                        
                        read.getline(buf, 1024);
                        read >> dimension;
                        read >> id;
                        read >> name;
                        name.erase(0,1);
                        name.erase(name.end()-1,name.end());
                        mesh_data.m_dim_physical_tag_and_name[dimension][id] = name;
                        
                        if(mesh_data.m_dim_name_and_physical_tag[dimension].find(name) == mesh_data.m_dim_name_and_physical_tag[dimension].end())
                        {
                            std::cout << "Automatically associating " << name << " with material id " << id << std::endl;
                            mesh_data.m_dim_name_and_physical_tag[dimension][name] = id;
                        }
                        else
                        {
                            int external_matid = mesh_data.m_dim_name_and_physical_tag[dimension][name];
                            std::cout << "Associating " << name << " with material id " << id <<
                            " with external material id " << external_matid << std::endl;
                        }
                        
                        mesh_data.m_dim_physical_tag_and_physical_tag[dimension][id] = mesh_data.m_dim_name_and_physical_tag[dimension][name];
                        
                        if (max_dimension < dimension) {
                            max_dimension = dimension;
                        }
                    }
                    mesh_data.m_dimension = max_dimension;
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndPhysicalNames" || str_end == "$EndPhysicalNames\r")
                    {
                        std::cout << "Read mesh physical entities = " << n_physical_names << std::endl;
                    }
                    continue;
                }
                
                if(str == "$Entities" || str == "$Entities\r")
                {
                    read >> mesh_data.m_n_points;
                    read >> mesh_data.m_n_curves;
                    read >> mesh_data.m_n_surfaces;
                    read >> mesh_data.m_n_volumes;

                    if(max_dimension < 3 && mesh_data.m_n_volumes > 0) max_dimension = 3;
                    else if(max_dimension < 2 && mesh_data.m_n_surfaces > 0) max_dimension = 2;
                    else if(max_dimension < 1 && mesh_data.m_n_curves > 0) max_dimension = 1;
                    
                    int n_physical_tag;
                    std::pair<int, std::vector<int> > chunk;
                    /// Entity bounding box data
                    T x_min, y_min, z_min;
                    T x_max, y_max, z_max;
                    std::vector<int> n_entities = {mesh_data.m_n_points,mesh_data.m_n_curves,mesh_data.m_n_surfaces,mesh_data.m_n_volumes};
                    std::vector<int> n_entities_with_physical_tag = {0,0,0,0};
                    
                    
                    for (int i_dim = 0; i_dim <4; i_dim++) {
                        for (int64_t i_entity = 0; i_entity < n_entities[i_dim]; i_entity++) {
                            
                            read.getline(buf, 1024);
                            read >> chunk.first;
                            read >> x_min;
                            read >> y_min;
                            read >> z_min;
                            if(i_dim > 0)
                            {
                                read >> x_max;
                                read >> y_max;
                                read >> z_max;
                            }
                            read >> n_physical_tag;
                            chunk.second.resize(n_physical_tag);
                            for (int i_data = 0; i_data < n_physical_tag; i_data++) {
                                read >> chunk.second[i_data];
                            }
                            if(i_dim > 0)
                            {
                                size_t n_bounding_points;
                                read >> n_bounding_points;
                                for (int i_data = 0; i_data < n_bounding_points; i_data++) {
                                    int point_tag;
                                    read >> point_tag;
                                }
                            }
                            n_entities_with_physical_tag[i_dim] += n_physical_tag;
                            mesh_data.m_dim_entity_tag_and_physical_tag[i_dim].insert(chunk);
                        }
                    }

                    mesh_data.m_n_physical_points = n_entities_with_physical_tag[0];
                    mesh_data.m_n_physical_curves = n_entities_with_physical_tag[1];
                    mesh_data.m_n_physical_surfaces = n_entities_with_physical_tag[2];
                    mesh_data.m_n_physical_volumes = n_entities_with_physical_tag[3];
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndEntities" || str_end == "$EndEntities\r")
                    {
                        std::cout << "Read mesh entities = " <<  mesh_data.m_n_points + mesh_data.m_n_curves + mesh_data.m_n_surfaces + mesh_data.m_n_volumes << std::endl;
                        std::cout << "Read mesh entities with physical tags = " <<  mesh_data.m_n_physical_points + mesh_data.m_n_physical_curves + mesh_data.m_n_physical_surfaces + mesh_data.m_n_physical_volumes << std::endl;
                    }
                    continue;
                }
                
                
                if(str == "$Nodes" || str == "$Nodes\r")
                {
                    
                    int64_t n_entity_blocks, n_nodes, min_node_tag, max_node_tag;
                    read >> n_entity_blocks;
                    read >> n_nodes;
                    read >> min_node_tag;
                    read >> max_node_tag;
                    
                    int64_t node_id;
                    T nodecoordX , nodecoordY , nodecoordZ ;
                    points.reserve(max_node_tag);
                    vertices.reserve(max_node_tag);
                    node_map.resize(max_node_tag,-1);
                    int64_t node_c = 0;
                    const int64_t Tnodes = max_node_tag;
            
                    int entity_tag, entity_dim, entity_parametric, entity_nodes;
                    for (int64_t i_block = 0; i_block < n_entity_blocks; i_block++)
                    {
                        read.getline(buf, 1024);
                        read >> entity_dim;
                        read >> entity_tag;
                        read >> entity_parametric;
                        read >> entity_nodes;
                        
                        if (entity_parametric != 0) {
                            std::cout << "Characteristic not implemented." << std::endl;
                            assert(false);
                        }
                        
                        std::vector<int64_t> nodeids(entity_nodes,-1);
                        for (int64_t inode = 0; inode < entity_nodes; inode++) {
                            read >> nodeids[inode];
                            nodeids[inode] --;
                            node_map[nodeids[inode]] = node_c;
                            node_c++;
                        }
                        for (int64_t inode = 0; inode < entity_nodes; inode++) {
                            read >> nodecoordX;
                            read >> nodecoordY;
                            read >> nodecoordZ;
                            
                            int64_t node_id = node_map[nodeids[inode]];
                            point_type point(nodecoordX, nodecoordY);
                            points.push_back(point);
                            vertices.push_back(node_type(point_identifier<2>(node_id)));
                            
                        }
                    }
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndNodes" || str_end == "$EndNodes\r")
                    {
                        std::cout << "Read mesh nodes = " <<  points.size() << std::endl;
                    }
                    continue;
                }
                
                if(str == "$Elements" || str == "$Elements\r")
                {
                    
                    int64_t n_entity_blocks, n_elements, min_element_tag, max_element_tag;
                    read >> n_entity_blocks;
                    read >> n_elements;
                    read >> min_element_tag;
                    read >> max_element_tag;
                    polygons.reserve(n_elements-1);

                    
                    int entity_tag, entity_dim, entity_el_type, entity_elements;
                    for (int64_t i_block = 0; i_block < n_entity_blocks; i_block++)
                    {
                        read.getline(buf, 1024);
                        read >> entity_dim;
                        read >> entity_tag;
                        read >> entity_el_type;
                        read >> entity_elements;
                        
                        if(entity_elements == 0){
                            std::cout << "The entity with tag " << entity_tag << " does not have elements to insert" << std::endl;
                        }
                        
                        for (int64_t iel = 0; iel < entity_elements; iel++) {
                            int physical_identifier;
                            int n_physical_identifier = 0;
                            if(mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim].find(entity_tag) != mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim].end())
                            {
                                n_physical_identifier = mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim][entity_tag].size();
                            }
                            bool physical_identifier_Q = n_physical_identifier != 0;
                            if(physical_identifier_Q)
                            {
                                int gmsh_physical_identifier = mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim][entity_tag][0];
                                physical_identifier = mesh_data.m_dim_physical_tag_and_physical_tag[entity_dim][gmsh_physical_identifier];
                                if(n_physical_identifier !=1){
                                    std::cout << "The entity with tag " << entity_tag << std::endl;
                                    std::cout << "Has associated the following physical tags : " << std::endl;
                                    for (int i_data = 0; i_data < n_physical_identifier; i_data++) {
                                        std::cout << mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim][entity_tag][i_data] << std::endl;
                                    }
                                    
                                    std::cout << "Automatically, the assgined external physical tag = " << physical_identifier << " is used.  The other ones are dropped out." << std::endl;
                                }
                                
                                
                                read.getline(buf, 1024);
                                int el_identifier, n_el_nodes;
                                n_el_nodes = GetNumberofNodes(entity_el_type);
                                read >> el_identifier;
                                std::vector<int> node_identifiers(n_el_nodes);
                                for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                    read >> node_identifiers[i_node];
                                    node_identifiers[i_node]--;
                                }
                                
                                // BC case
                                
                                std::string entity_name = mesh_data.m_dim_physical_tag_and_name[entity_dim][physical_identifier];
                                
                                std::string bc_string = "Gamma";
                                std::string fracture_string = "Fracture";
                                if (entity_name.find(bc_string) != std::string::npos) {
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        bc_points.insert(node_map[node_identifiers[i_node]]);
                                    }
                                    
                                    for (int i_node = 0; i_node < n_el_nodes - 1; i_node++) {
                                        std::array<size_t, 2> edge = {static_cast<unsigned long>(node_map[node_identifiers[i_node]]),static_cast<unsigned long>(node_map[node_identifiers[i_node+1]])};
                                        validate_edge(edge);
                                        boundary_edges.push_back( edge );
                                    }
                                    
                                }else if(entity_name.find(fracture_string) != std::string::npos){ // fracture case
                                    for (int i_node = 0; i_node < n_el_nodes - 1; i_node++) {
                                        std::array<size_t, 2> edge = {static_cast<unsigned long>(node_map[node_identifiers[i_node]]),static_cast<unsigned long>(node_map[node_identifiers[i_node+1]])};
                                        validate_edge(edge);
                                        facets.push_back( edge );
                                    }
                                    
                                }else{ // polygon case
                                    /// Internally the nodes index and element index is converted to zero based indexation
                                    
                                    polygon polygon;
                                    std::vector<size_t> member_nodes;
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        member_nodes.push_back(node_map[node_identifiers[i_node]]);
                                    }
                                    polygon.m_member_nodes = member_nodes;
                                
                                    std::set< std::array<size_t, 2> > member_edges;
                                    std::array<size_t, 2> edge;
                                    for (size_t i = 0; i < member_nodes.size(); i++) {
                                        
                                        if (i == n_el_nodes - 1) {
                                            edge = {member_nodes[i],member_nodes[0]};
                                        }else{
                                            edge = {member_nodes[i],member_nodes[i+1]};
                                        }
                                        
                                        validate_edge(edge);
                                        facets.push_back( edge );
                                        member_edges.insert(edge);

                                    }
                                    
                                    polygon.m_member_edges = member_edges;
                                    polygons.push_back( polygon );
                                    
                                }
                                
                            }else{
                                read.getline(buf, 1024);
                                int el_identifier, n_el_nodes;
                                n_el_nodes = GetNumberofNodes(entity_el_type);
                                read >> el_identifier;
                                std::vector<int> node_identifiers(n_el_nodes);
                                for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                    read >> node_identifiers[i_node];
                                }
                                std::cout << "The entity with tag " << entity_tag << " does not have a physical tag, element " << el_identifier << " skipped " << std::endl;
                            }

                        }
                    }
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndElements" || str_end == "$EndElements\r")
                    {
                        std::cout << "Read mesh cells = " << polygons.size() << std::endl;
                    }
                    continue;
                }
                
            }
            
        }
        
        std::cout << "Gmsh data is read." << std::endl;
        std::cout << "Number of cells " << polygons.size() << std::endl;
        
        // Duplicated facets are eliminated
        std::sort( facets.begin(), facets.end() );
        facets.erase( std::unique( facets.begin(), facets.end() ), facets.end() );
        
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

template<typename T>
class gmsh_3d_reader : public fitted_geometry_builder<disk::generic_mesh<T, 3>>
{
    typedef disk::generic_mesh<T,3>                       mesh_type;
    typedef typename mesh_type::point_type          point_type;
    typedef typename mesh_type::node_type           node_type;
    typedef typename mesh_type::edge_type           edge_type;
    typedef typename mesh_type::surface_type        surface_type;
    typedef typename mesh_type::volume_type         volume_type;

    std::vector<point_type>                         m_points;
    std::vector<node_type>                          m_nodes;
    std::vector<std::pair<size_t, edge_type>>       m_edges;

    std::vector<std::pair<size_t, std::vector<size_t>>>     vol_to_faces;
    std::vector<std::vector<size_t>>                        vol_to_vts;
    std::vector<std::pair<size_t, std::vector<size_t>>>     faces_to_edges;
    std::vector<std::vector<size_t>>                        faces_to_vts;
    
    /// gmsh data
    struct gmsh_data {
        
        int m_n_volumes;
        int m_n_surfaces;
        int m_n_curves;
        int m_n_points;
        int m_n_physical_volumes;
        int m_n_physical_surfaces;
        int m_n_physical_curves;
        int m_n_physical_points;
        int m_dimension;
        std::vector<std::map<int,std::vector<int>>> m_dim_entity_tag_and_physical_tag;
        std::vector<std::map<int,std::string>> m_dim_physical_tag_and_name;
        std::vector<std::map<std::string,int>> m_dim_name_and_physical_tag;
        std::vector<std::map<int,int>> m_dim_physical_tag_and_physical_tag;
        std::vector<int> m_entity_index;
        
        gmsh_data() {
            m_n_volumes = 0;
            m_n_surfaces = 0;
            m_n_curves = 0;
            m_n_points = 0;
            m_n_physical_volumes = 0;
            m_n_physical_surfaces = 0;
            m_n_physical_curves = 0;
            m_n_physical_points = 0;
            m_dimension = 0;
            m_dim_entity_tag_and_physical_tag.resize(4);
            m_dim_physical_tag_and_name.resize(4);
            m_dim_name_and_physical_tag.resize(4);
            m_dim_physical_tag_and_physical_tag.resize(4);
        }
        
        gmsh_data(const gmsh_data &other) {
            m_n_volumes = other.m_n_volumes;
            m_n_surfaces = other.m_n_surfaces;
            m_n_curves = other.m_n_curves;
            m_n_points = other.m_n_points;
            m_n_physical_volumes = other.m_n_physical_volumes;
            m_n_physical_surfaces = other.m_n_physical_surfaces;
            m_n_physical_curves = other.m_n_physical_curves;
            m_n_physical_points = other.m_n_physical_points;
            m_dimension = other.m_dimension;
            m_dim_entity_tag_and_physical_tag   = other.m_dim_entity_tag_and_physical_tag;
            m_dim_physical_tag_and_name         = other.m_dim_physical_tag_and_name;
            m_dim_name_and_physical_tag         = other.m_dim_name_and_physical_tag;
            m_dim_physical_tag_and_physical_tag = other.m_dim_physical_tag_and_physical_tag;
        }
        
        gmsh_data &operator=(const gmsh_data &other){
            m_n_volumes = other.m_n_volumes;
            m_n_surfaces = other.m_n_surfaces;
            m_n_curves = other.m_n_curves;
            m_n_points = other.m_n_points;
            m_n_physical_volumes = other.m_n_physical_volumes;
            m_n_physical_surfaces = other.m_n_physical_surfaces;
            m_n_physical_curves = other.m_n_physical_curves;
            m_n_physical_points = other.m_n_physical_points;
            m_dimension = other.m_dimension;
            m_dim_entity_tag_and_physical_tag   = other.m_dim_entity_tag_and_physical_tag;
            m_dim_physical_tag_and_name         = other.m_dim_physical_tag_and_name;
            m_dim_name_and_physical_tag         = other.m_dim_name_and_physical_tag;
            m_dim_physical_tag_and_physical_tag = other.m_dim_physical_tag_and_physical_tag;
            return *this;
        }
        
    };
    
    struct polygon
    {
        std::vector<size_t>                 m_member_nodes;
        std::vector<std::array<size_t, 2>>  m_member_edges;
        bool operator<(const polygon & other) {
            return m_member_nodes < other.m_member_nodes;
        }
    };

    struct polyhedron
    {
        std::vector<size_t>                 m_member_nodes;
        std::vector<std::array<size_t, 2>>  m_member_edges;
        std::vector<polygon>                m_member_facets;
        bool operator<(const polyhedron & other) {
            return m_member_nodes < other.m_member_nodes;
        }
    };
    
    std::vector<point_type>                         points;
    std::vector<node_type>                          nodes;
    std::vector<edge_type>                          edges;
    std::vector<surface_type>                       surfaces, boundary_surfaces;
    std::vector<volume_type>                        volumes;
    std::vector<polygon>                            polygons;
    std::vector<polygon>                            boundary_polygons;
    std::vector<polyhedron>                         polyhedrons;
    std::vector<std::array<size_t, 2>>              skeleton_edges;
    std::string gmsh_file;
    
    
    
    gmsh_data   mesh_data;
            
    void validate_edge(std::array<size_t, 2> & edge){
        assert(edge[0] != edge[1]);
        if (edge[0] > edge[1]){
            std::swap(edge[0], edge[1]);
        }
    }
    
public:

    gmsh_3d_reader() : fitted_geometry_builder<mesh_type>()
    {
        fitted_geometry_builder<mesh_type>::m_dimension = 3;
    }
    
    void set_gmsh_file(std::string mesh_file){
        gmsh_file = mesh_file;
    }
    
    int GetNumberofNodes(int & cell_type){
        
        int n_nodes;
        switch (cell_type) {
            case 1:
            {   // Line
                n_nodes = 2;
            }
                break;
            case 2:
            {
                // Triangle
                n_nodes = 3;
            }
                break;
            case 3:
            {
                // Quadrilateral
                n_nodes = 4;
            }
                break;
            case 4:
            {
                // Tetrahedron
                n_nodes = 4;
            }
                break;
            case 5:
            {
                // Hexahedra
                n_nodes = 8;
            }
                break;
            case 6:
            {
                // Prism
                n_nodes = 6;
            }
                break;
            case 7:
            {
                // Pyramid
                n_nodes = 5;
            }
                break;
            case 8:
            {
                // Quadratic Line
                n_nodes = 3;
            }
                break;
            case 9:
            {
                // Quadratic Triangle
                n_nodes = 6;
            }
                break;
            case 10:
            {
                // Quadratic Quadrilateral
                n_nodes = 9;
            }
                break;
            case 11:
            {
                // Quadratic Tetrahedron
                n_nodes = 10;
                
            }
                break;
            case 12:
            {
                // Quadratic Hexahedra
                n_nodes = 20;
            }
                break;
            case 13:
            {
                // Quadratic Prism
                n_nodes = 15;
            }
                break;
            case 15:{
                // Point
                n_nodes = 1;
            }
                break;
            default:
            {
                std::cout << "Cell not impelemented." << std::endl;
                n_nodes = 0;
                assert(false);
            }
                break;
        }
        
        return n_nodes;
    }
    
    // build the mesh
    bool build_mesh(){
        
        int max_dimension = 0;
        {
            
            // reading a general mesh information by filter
            std::ifstream read (gmsh_file.c_str());
            bool file_check_Q = !read;
            if(file_check_Q)
            {
                std::cout << "Couldn't open the file." << gmsh_file << std::endl;
                assert(!file_check_Q);
            }
            
            if (file_check_Q) {
                std::cout << "File path is wrong." << std::endl;
                assert(!file_check_Q);
            }
            
            std::vector<int64_t> node_map;
            while(read)
            {
                char buf[1024];
                read.getline(buf, 1024);
                std::string str(buf);
                
                if(str == "$MeshFormat" || str == "$MeshFormat\r")
                {
                    read.getline(buf, 1024);
                    std::string str(buf);
                    std::cout << "Reading mesh format = " << str << std::endl;
                    
                }
                
                if(str == "$PhysicalNames" || str == "$PhysicalNames\r" )
                {
                    
                    int64_t n_physical_names;
                    read >> n_physical_names;
                    
                    int dimension, id;
                    std::string name;
                    std::pair<int, std::string> chunk;
                    
                    for (int64_t i_name = 0; i_name < n_physical_names; i_name++) {
                        
                        read.getline(buf, 1024);
                        read >> dimension;
                        read >> id;
                        read >> name;
                        name.erase(0,1);
                        name.erase(name.end()-1,name.end());
                        mesh_data.m_dim_physical_tag_and_name[dimension][id] = name;
                        
                        if(mesh_data.m_dim_name_and_physical_tag[dimension].find(name) == mesh_data.m_dim_name_and_physical_tag[dimension].end())
                        {
                            std::cout << "Automatically associating " << name << " with material id " << id << std::endl;
                            mesh_data.m_dim_name_and_physical_tag[dimension][name] = id;
                        }
                        else
                        {
                            int external_matid = mesh_data.m_dim_name_and_physical_tag[dimension][name];
                            std::cout << "Associating " << name << " with material id " << id <<
                            " with external material id " << external_matid << std::endl;
                        }
                        
                        mesh_data.m_dim_physical_tag_and_physical_tag[dimension][id] = mesh_data.m_dim_name_and_physical_tag[dimension][name];
                        
                        if (max_dimension < dimension) {
                            max_dimension = dimension;
                        }
                    }
                    mesh_data.m_dimension = max_dimension;
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndPhysicalNames" || str_end == "$EndPhysicalNames\r")
                    {
                        std::cout << "Read mesh physical entities = " << n_physical_names << std::endl;
                    }
                    continue;
                }
                
                if(str == "$Entities" || str == "$Entities\r")
                {
                    read >> mesh_data.m_n_points;
                    read >> mesh_data.m_n_curves;
                    read >> mesh_data.m_n_surfaces;
                    read >> mesh_data.m_n_volumes;

                    if(max_dimension < 3 && mesh_data.m_n_volumes > 0) max_dimension = 3;
                    else if(max_dimension < 2 && mesh_data.m_n_surfaces > 0) max_dimension = 2;
                    else if(max_dimension < 1 && mesh_data.m_n_curves > 0) max_dimension = 1;
                    
                    int n_physical_tag;
                    std::pair<int, std::vector<int> > chunk;
                    /// Entity bounding box data
                    T x_min, y_min, z_min;
                    T x_max, y_max, z_max;
                    std::vector<int> n_entities = {mesh_data.m_n_points,mesh_data.m_n_curves,mesh_data.m_n_surfaces,mesh_data.m_n_volumes};
                    std::vector<int> n_entities_with_physical_tag = {0,0,0,0};
                    
                    
                    for (int i_dim = 0; i_dim <4; i_dim++) {
                        for (int64_t i_entity = 0; i_entity < n_entities[i_dim]; i_entity++) {
                            
                            read.getline(buf, 1024);
                            read >> chunk.first;
                            read >> x_min;
                            read >> y_min;
                            read >> z_min;
                            if(i_dim > 0)
                            {
                                read >> x_max;
                                read >> y_max;
                                read >> z_max;
                            }
                            read >> n_physical_tag;
                            chunk.second.resize(n_physical_tag);
                            for (int i_data = 0; i_data < n_physical_tag; i_data++) {
                                read >> chunk.second[i_data];
                            }
                            if(i_dim > 0)
                            {
                                size_t n_bounding_points;
                                read >> n_bounding_points;
                                for (int i_data = 0; i_data < n_bounding_points; i_data++) {
                                    int point_tag;
                                    read >> point_tag;
                                }
                            }
                            n_entities_with_physical_tag[i_dim] += n_physical_tag;
                            mesh_data.m_dim_entity_tag_and_physical_tag[i_dim].insert(chunk);
                        }
                    }

                    mesh_data.m_n_physical_points = n_entities_with_physical_tag[0];
                    mesh_data.m_n_physical_curves = n_entities_with_physical_tag[1];
                    mesh_data.m_n_physical_surfaces = n_entities_with_physical_tag[2];
                    mesh_data.m_n_physical_volumes = n_entities_with_physical_tag[3];
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndEntities" || str_end == "$EndEntities\r")
                    {
                        std::cout << "Read mesh entities = " <<  mesh_data.m_n_points + mesh_data.m_n_curves + mesh_data.m_n_surfaces + mesh_data.m_n_volumes << std::endl;
                        std::cout << "Read mesh entities with physical tags = " <<  mesh_data.m_n_physical_points + mesh_data.m_n_physical_curves + mesh_data.m_n_physical_surfaces + mesh_data.m_n_physical_volumes << std::endl;
                    }
                    continue;
                }
                
                
                if(str == "$Nodes" || str == "$Nodes\r")
                {
                    
                    int64_t n_entity_blocks, n_nodes, min_node_tag, max_node_tag;
                    read >> n_entity_blocks;
                    read >> n_nodes;
                    read >> min_node_tag;
                    read >> max_node_tag;
                    
                    int64_t node_id;
                    T nodecoordX , nodecoordY , nodecoordZ ;
                    points.reserve(max_node_tag);
                    nodes.reserve(max_node_tag);
                    
                    edges.reserve(max_node_tag);
                    polygons.reserve(max_node_tag);
                    boundary_polygons.reserve(max_node_tag);
                    
                    node_map.resize(max_node_tag,-1);
                    int64_t node_c = 0;
                    const int64_t Tnodes = max_node_tag;
            
                    int entity_tag, entity_dim, entity_parametric, entity_nodes;
                    for (int64_t i_block = 0; i_block < n_entity_blocks; i_block++)
                    {
                        read.getline(buf, 1024);
                        read >> entity_dim;
                        read >> entity_tag;
                        read >> entity_parametric;
                        read >> entity_nodes;
                        
                        if (entity_parametric != 0) {
                            std::cout << "Characteristic not implemented." << std::endl;
                            assert(false);
                        }
                        
                        std::vector<int64_t> nodeids(entity_nodes,-1);
                        for (int64_t inode = 0; inode < entity_nodes; inode++) {
                            read >> nodeids[inode];
                            nodeids[inode] --;
                            node_map[nodeids[inode]] = node_c;
                            node_c++;
                        }
                        for (int64_t inode = 0; inode < entity_nodes; inode++) {
                            read >> nodecoordX;
                            read >> nodecoordY;
                            read >> nodecoordZ;

                            int64_t node_id = node_map[nodeids[inode]];
                            points.push_back( point_type({nodecoordX, nodecoordY, nodecoordZ}) );
                            nodes.push_back( node_type( point_identifier<3>(node_id) ) );
                        }
                    }
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndNodes" || str_end == "$EndNodes\r")
                    {
                        std::cout << "Read mesh nodes = " <<  points.size() << std::endl;
                    }
                    continue;
                }
                
                if(str == "$Elements" || str == "$Elements\r")
                {
                    
                    int64_t n_entity_blocks, n_elements, min_element_tag, max_element_tag;
                    read >> n_entity_blocks;
                    read >> n_elements;
                    read >> min_element_tag;
                    read >> max_element_tag;
                    volumes.reserve(n_elements-1);

                    
                    int entity_tag, entity_dim, entity_el_type, entity_elements;
                    for (int64_t i_block = 0; i_block < n_entity_blocks; i_block++)
                    {
                        read.getline(buf, 1024);
                        read >> entity_dim;
                        read >> entity_tag;
                        read >> entity_el_type;
                        read >> entity_elements;
                        
                        if(entity_elements == 0){
                            std::cout << "The entity with tag " << entity_tag << " does not have elements to insert" << std::endl;
                        }
                        
                        for (int64_t iel = 0; iel < entity_elements; iel++) {
                            int physical_identifier;
                            int n_physical_identifier = 0;
                            if(mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim].find(entity_tag) != mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim].end())
                            {
                                n_physical_identifier = mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim][entity_tag].size();
                            }
                            bool physical_identifier_Q = n_physical_identifier != 0;
                            if(physical_identifier_Q)
                            {
                                int gmsh_physical_identifier = mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim][entity_tag][0];
                                physical_identifier = mesh_data.m_dim_physical_tag_and_physical_tag[entity_dim][gmsh_physical_identifier];
                                if(n_physical_identifier !=1){
                                    std::cout << "The entity with tag " << entity_tag << std::endl;
                                    std::cout << "Has associated the following physical tags : " << std::endl;
                                    for (int i_data = 0; i_data < n_physical_identifier; i_data++) {
                                        std::cout << mesh_data.m_dim_entity_tag_and_physical_tag[entity_dim][entity_tag][i_data] << std::endl;
                                    }
                                    
                                    std::cout << "Automatically, the assgined external physical tag = " << physical_identifier << " is used.  The other ones are dropped out." << std::endl;
                                }
                                
                                
                                read.getline(buf, 1024);
                                int el_identifier, n_el_nodes;
                                n_el_nodes = GetNumberofNodes(entity_el_type);
                                read >> el_identifier;
                                std::vector<int> node_identifiers(n_el_nodes);
                                for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                    read >> node_identifiers[i_node];
                                    node_identifiers[i_node]--;
                                }
                                
                                // BC case
                                
                                std::string entity_name = mesh_data.m_dim_physical_tag_and_name[entity_dim][physical_identifier];
                                
                                std::string bc_string = "Gamma";
                                std::string fracture_string = "Fracture";
                                if (entity_name.find(bc_string) != std::string::npos) {
                                    
                                    std::vector<size_t> member_nodes;
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        member_nodes.push_back(node_map[node_identifiers[i_node]]);
                                    }
                                    
                                    std::vector<std::array<size_t, 2>> member_edges;
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        std::array<size_t, 2> edge;
                                        if (i_node == n_el_nodes - 1) {
                                            edge = {static_cast<unsigned long>(member_nodes[i_node]),static_cast<unsigned long>(member_nodes[0])};
                                        }else{
                                            edge = {static_cast<unsigned long>(member_nodes[i_node]),static_cast<unsigned long>(member_nodes[i_node+1])};
                                        }
                                        validate_edge(edge);
                                        skeleton_edges.push_back( edge );
                                        member_edges.push_back(edge);
                                    }

                                    polygon  facet;
                                    facet.m_member_nodes = member_nodes;
                                    facet.m_member_edges = member_edges;
                                    boundary_polygons.push_back(facet);
                                    
                                    
                                }else if(entity_name.find(fracture_string) != std::string::npos){ // fracture case
                                    
                                    std::vector<size_t> member_nodes;
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        member_nodes.push_back(node_map[node_identifiers[i_node]]);
                                    }
                                    
                                    std::vector<std::array<size_t, 2>> member_edges;
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        std::array<size_t, 2> edge;
                                        if (i_node == n_el_nodes - 1) {
                                            edge = {static_cast<unsigned long>(member_nodes[i_node]),static_cast<unsigned long>(member_nodes[0])};
                                        }else{
                                            edge = {static_cast<unsigned long>(member_nodes[i_node]),static_cast<unsigned long>(member_nodes[i_node+1])};
                                        }
                                        validate_edge(edge);
                                        skeleton_edges.push_back( edge );
                                        member_edges.push_back(edge);
                                    }

                                    polygon  facet;
                                    facet.m_member_nodes = member_nodes;
                                    facet.m_member_edges = member_edges;
                                    polygons.push_back(facet);
                                    
                                }else{ // polygon case
                                                                        
                                    std::vector<size_t> member_nodes;
                                    for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                        member_nodes.push_back(node_map[node_identifiers[i_node]]);
                                    }
                                    
                                    auto simplex_edges = [&](const std::vector<size_t> & nodes)->std::vector<std::array<size_t, 2>> {
                                        
                                        std::vector<std::array<size_t, 2>> edges;
                                        std::array<size_t, 2> edge0 {{nodes[0], nodes[1]}};
                                        validate_edge(edge0);
                                        edges.push_back(edge0);
                                        
                                        std::array<size_t, 2> edge1 {{nodes[1], nodes[2]}};
                                        validate_edge(edge1);
                                        edges.push_back(edge1);
                                        
                                        std::array<size_t, 2> edge2 {{nodes[2], nodes[0]}};
                                        validate_edge(edge2);
                                        edges.push_back(edge2);
                                        
                                        std::array<size_t, 2> edge3 {{nodes[0], nodes[3]}};
                                        validate_edge(edge3);
                                        edges.push_back(edge3);
                                        
                                        std::array<size_t, 2> edge4 {{nodes[2], nodes[3]}};
                                        validate_edge(edge4);
                                        edges.push_back(edge4);
                                        
                                        std::array<size_t, 2> edge5 {{nodes[1], nodes[3]}};
                                        validate_edge(edge5);
                                        edges.push_back(edge5);
                                        
                                        return edges;
                                    };
                                    
                                    auto simplex_facets = [&](const std::vector<size_t> & nodes, const std::vector<std::array<size_t, 2>> & edges)->std::vector<polygon> {
                                        
                                        std::vector<polygon> polygons;
                                        
                                        polygon facet0;
                                        facet0.m_member_nodes = {nodes[0],nodes[1],nodes[2]};
                                        facet0.m_member_edges = {edges[0],edges[1],edges[2]};
                                        polygons.push_back(facet0);
                                        
                                        polygon facet1;
                                        facet1.m_member_nodes = {nodes[0],nodes[2],nodes[3]};
                                        facet1.m_member_edges = {edges[2],edges[4],edges[3]};
                                        polygons.push_back(facet1);
                                        
                                        polygon facet2;
                                        facet2.m_member_nodes = {nodes[1],nodes[2],nodes[3]};
                                        facet2.m_member_edges = {edges[1],edges[4],edges[5]};
                                        polygons.push_back(facet2);
                                        
                                        polygon facet4;
                                        facet4.m_member_nodes = {nodes[0],nodes[1],nodes[3]};
                                        facet4.m_member_edges = {edges[0],edges[5],edges[3]};
                                        polygons.push_back(facet4);
                                        
                                        return polygons;
                                    };
                                    
                                    std::vector<std::array<size_t, 2>> member_edges = simplex_edges(member_nodes);
                                    for (auto & edge : member_edges) {
                                        skeleton_edges.push_back( edge );
                                    }
                                    
                                    std::vector<polygon> member_facets = simplex_facets(member_nodes,member_edges);
                                    for (auto & facet : member_facets) {
                                        polygons.push_back( facet );
                                    }

                                    polyhedron vol;
                                    vol.m_member_nodes  = member_nodes;
                                    vol.m_member_edges  = member_edges;
                                    vol.m_member_facets = member_facets;
                                    polyhedrons.push_back(vol);
                                    
                                }
                                
                            }else{
                                read.getline(buf, 1024);
                                int el_identifier, n_el_nodes;
                                n_el_nodes = GetNumberofNodes(entity_el_type);
                                read >> el_identifier;
                                std::vector<int> node_identifiers(n_el_nodes);
                                for (int i_node = 0; i_node < n_el_nodes; i_node++) {
                                    read >> node_identifiers[i_node];
                                }
                                std::cout << "The entity with tag " << entity_tag << " does not have a physical tag, element " << el_identifier << " skipped " << std::endl;
                            }

                        }
                    }
                    
                    char buf_end[1024];
                    read.getline(buf_end, 1024);
                    read.getline(buf_end, 1024);
                    std::string str_end(buf_end);
                    if(str_end == "$EndElements" || str_end == "$EndElements\r")
                    {
                        std::cout << "Read mesh cells = " << polyhedrons.size() << std::endl;
                    }
                    continue;
                }
                
            }
            
        }
        
        // Duplicated facets are eliminated
        std::sort( skeleton_edges.begin(), skeleton_edges.end() );
        skeleton_edges.erase( std::unique( skeleton_edges.begin(), skeleton_edges.end() ), skeleton_edges.end() );
        
        std::cout << "Gmsh data is read." << std::endl;
        std::cout << "Raw number of nodes  " << nodes.size() << std::endl;
        std::cout << "Raw number of edges  " << skeleton_edges.size() << std::endl;
        std::cout << "Raw number of facets " << polygons.size() << std::endl;
        std::cout << "Raw number of cells  " << polyhedrons.size() << std::endl;
        
        return true;
    }
    
    void move_to_mesh_storage(mesh_type& msh){
        
        auto storage = msh.backend_storage();
        storage->points = std::move(points);
        storage->nodes  = std::move(nodes);
        
        edges.clear();
        edges.reserve(skeleton_edges.size());
        for (size_t i = 0; i < skeleton_edges.size(); i++)
        {
            assert(skeleton_edges[i][0] < skeleton_edges[i][1]);
            auto node1 = typename node_type::id_type(skeleton_edges[i][0]);
            auto node2 = typename node_type::id_type(skeleton_edges[i][1]);

            auto e = edge_type{{node1, node2}};

            e.set_point_ids(skeleton_edges[i].begin(), skeleton_edges[i].end());
            edges.push_back(e);
        }
        std::sort(edges.begin(), edges.end());
        edges.erase( std::unique( edges.begin(), edges.end() ), edges.end() );
        storage->edges = std::move(edges);
        
        auto facet_edges = [&](const polygon & p)->std::vector<typename edge_type::id_type> {
            
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
                    assert(false);
                }

                surface_edges.push_back(edge_id.second);
            }
            
            return surface_edges;
        };
        
        auto should_skip = [&](const surface_type & surface)-> bool {
        
            bool check_Q = false;
            for (auto face_it = surfaces.begin(); face_it != surfaces.end(); face_it++)
            {
                auto face = *face_it;
                
                std::vector<size_t> face_p_ids,p_ids;
                for (auto & id: face.point_ids()) {
                    face_p_ids.push_back(id);
                }
                for (auto & id: surface.point_ids()) {
                    p_ids.push_back(id);
                }
                std::sort(face_p_ids.begin(), face_p_ids.end());
                std::sort(p_ids.begin(), p_ids.end());
                check_Q = std::equal(face_p_ids.begin(), face_p_ids.end(), p_ids.begin());
                if (check_Q) {
                    break;
                }
            }
            return check_Q;
        };
        
        surfaces.reserve( polygons.size() );
        for (auto& p : polygons)
        {
            std::vector<typename edge_type::id_type> surface_edges = facet_edges(p);
            auto surface = surface_type(surface_edges);
            surface.set_point_ids(p.m_member_nodes.begin(), p.m_member_nodes.end());
            bool should_skip_Q = should_skip(surface);
            if (should_skip_Q) {
                continue;
            }
            surfaces.push_back( surface );
        }

        std::sort(surfaces.begin(), surfaces.end());
        storage->surfaces = std::move(surfaces);
        
        
        auto find_surface = [&](const surface_type & surface)-> typename surface_type::id_type {
            
            size_t c = 0;
            bool check_Q = false;
            for (auto face_it = storage->surfaces.begin(); face_it != storage->surfaces.end(); face_it++)
            {
                auto face = *face_it;
                
                std::vector<size_t> face_p_ids,p_ids;
                for (auto & id: face.point_ids()) {
                    face_p_ids.push_back(id);
                }
                for (auto & id: surface.point_ids()) {
                    p_ids.push_back(id);
                }
                std::sort(face_p_ids.begin(), face_p_ids.end());
                std::sort(p_ids.begin(), p_ids.end());
                check_Q = std::equal(face_p_ids.begin(), face_p_ids.end(), p_ids.begin());
                if (check_Q) {
                    break;
                }
                c++;
            }
            if (check_Q) {
                return typename surface_type::id_type(c);
            }else{
                return typename surface_type::id_type(-1);
            }
            
        };
        
        storage->boundary_info.resize(storage->surfaces.size());
        for (auto& p : boundary_polygons)
        {
            std::vector<typename edge_type::id_type> surface_edges = facet_edges(p);
            auto surface = surface_type(surface_edges);
            surface.set_point_ids(p.m_member_nodes.begin(), p.m_member_nodes.end());
 


            auto position = find_surface(surface);
            if (position == -1)
            {
                std::cout << "Bad bug at " << __FILE__ << "("
                          << __LINE__ << ")" << std::endl;
                assert(false);
            }

            disk::bnd_info bi{0, true};
            storage->boundary_info.at(position) = bi;

        }
        
        auto volume_facets = [&](const polyhedron & p)->std::vector<typename surface_type::id_type> {
            
            std::vector<typename surface_type::id_type> volume_surfaces;
            for (auto& p : p.m_member_facets)
            {

                std::vector<typename edge_type::id_type> surface_edges = facet_edges(p);
                auto surface = surface_type(surface_edges);
                surface.set_point_ids(p.m_member_nodes.begin(), p.m_member_nodes.end());
                
                auto surface_id = find_surface(surface);
                if (surface_id == -1)
                {
                    std::cout << "Bad bug at " << __FILE__ << "("
                              << __LINE__ << ")" << std::endl;
                    assert(false);
                }

                volume_surfaces.push_back(surface_id);
            }
            
            return volume_surfaces;
        };
        
        volumes.reserve( polyhedrons.size() );
        for (auto& p : polyhedrons)
        {
            std::vector<typename surface_type::id_type> volume_surfaces = volume_facets(p);
            auto volume = volume_type(volume_surfaces);
            volume.set_point_ids(p.m_member_nodes.begin(), p.m_member_nodes.end());
            volumes.push_back( volume );
        }

        std::sort(volumes.begin(), volumes.end());
        storage->volumes = std::move(volumes);
        
        std::cout << "Nodes: " << storage->nodes.size() << std::endl;
        std::cout << "Edges: " << storage->edges.size() << std::endl;
        std::cout << "Faces: " << storage->surfaces.size() << std::endl;
        std::cout << "Volumes: " << storage->volumes.size() << std::endl;

        polygons.clear();
        boundary_polygons.clear();
        polyhedrons.clear();
        skeleton_edges.clear();

        return true;
        
    }
            
    // Print in log file relevant mesh information
    void print_log_file(){
        fitted_geometry_builder<mesh_type>::m_n_elements = volumes.size();
        std::ofstream file;
        file.open (fitted_geometry_builder<mesh_type>::m_log_file.c_str());
        file << "Number of volumes : " << volumes.size() << std::endl;
        file << "Number of surfaces : " << surfaces.size() << std::endl;
        file << "Number of skeleton edges : " << edges.size() << std::endl;
        file << "Number of boundary edges : " << boundary_surfaces.size() << std::endl;
        file << "Number of nodes : " << nodes.size() << std::endl;
        file.close();
    }
    
};

#endif /* fitted_geometry_builder_hpp */



