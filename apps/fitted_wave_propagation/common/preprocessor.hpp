//
//  preprocessor.hpp
//  acoustics
//
//  Created by Omar Durán on 4/10/20.
//

#pragma once
#ifndef preprocessor_hpp
#define preprocessor_hpp

#include <getopt.h>

class simulation_data
{
    
public:
    
    size_t m_k_degree;
    
    size_t m_n_divs;
    
    bool m_hdg_stabilization_Q;
    
    bool m_render_silo_files_Q;
    
    simulation_data() : m_k_degree(0), m_n_divs(0), m_hdg_stabilization_Q(false), m_render_silo_files_Q(false) {
        
    }
    
    simulation_data(const simulation_data & other){
        
        m_k_degree              = other.m_k_degree;
        m_n_divs                = other.m_n_divs;
        m_hdg_stabilization_Q   = other.m_hdg_stabilization_Q;
        m_render_silo_files_Q   = other.m_render_silo_files_Q;
    }

    const simulation_data & operator=(const simulation_data & other){
        
        // check for self-assignment
        if(&other == this){
            return *this;
        }
        
        m_k_degree              = other.m_k_degree;
        m_n_divs                = other.m_n_divs;
        m_hdg_stabilization_Q   = other.m_hdg_stabilization_Q;
        m_render_silo_files_Q   = other.m_render_silo_files_Q;
        
        return *this;
    }
    
    ~simulation_data(){
        
    }
    
    void print_simulation_data(){
        std::cout << bold << red << "face degree : " << m_k_degree << reset << std::endl;
        std::cout << bold << red << "refinements : " << m_n_divs << reset << std::endl;
        std::cout << bold << red << "stabilization type : " << m_hdg_stabilization_Q << reset << std::endl;
        std::cout << bold << red << "write silo files : " << m_render_silo_files_Q << reset << std::endl;
    }
    
};

class preprocessor {
    
public:
    
    

    static void PrintHelp()
    {
        std::cout <<
                "-k <int>:           Face polynomial degree: default 0\n"
                "-l <int>:           Number of uniform refinements: default 0\n"
                "-s <0-1>:           Stabilization type 0 -> HHO, 1 -> HDG: default 0 \n"
                "-f <string>:        Write silo files to\n"
                "-help:              Show help\n";
        exit(1);
    }

    static simulation_data process_args(int argc, char** argv)
    {
        const char* const short_opts = "k:l:s:f";
        const option long_opts[] = {
                {"degree", required_argument, nullptr, 'k'},
                {"ref", required_argument, nullptr, 'l'},
                {"stab", required_argument, nullptr, 's'},
                {"file", required_argument, nullptr, 'f'},
                {"help", no_argument, nullptr, 'h'},
                {nullptr, no_argument, nullptr, 0}
        };

        size_t k_degree = 0;
        size_t n_divs   = 0;
        bool hdg_Q = false;
        bool silo_files_Q = false;
        
        while (true)
        {
            const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

            if (-1 == opt)
                break;

            switch (opt)
            {
            case 'k':
                k_degree = std::stoi(optarg);
                break;

            case 'l':
                n_divs = std::stoi(optarg);
                break;

            case 's':
                hdg_Q = std::stoi(optarg);
                break;

            case 'f':
                silo_files_Q = std::stoi(optarg);
                break;

            case 'h': // -h or --help
            case '?': // Unrecognized option
            default:
                preprocessor::PrintHelp();
                break;
            }
        }
        
        // populating simulation data
        simulation_data sim_data;
        sim_data.m_k_degree = k_degree;
        sim_data.m_n_divs = n_divs;
        sim_data.m_hdg_stabilization_Q = hdg_Q;
        sim_data.m_render_silo_files_Q = silo_files_Q;
        
        return sim_data;
    }
};

#endif /* preprocessor_hpp */
