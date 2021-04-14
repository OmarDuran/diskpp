
set(md_sources ${md_sources} 
	#${CMAKE_CURRENT_SOURCE_DIR}/fitted_geometry_builders.cpp
PARENT_SCOPE)

set(md_headers ${md_headers} 
	${CMAKE_CURRENT_SOURCE_DIR}/display_settings.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/preprocessor.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/postprocessor.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/assembly_index.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/elastic_material_data.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/fitted_geometry_builders.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/elastic_two_fields_assembler.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/elastic_1d_two_fields_assembler.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/fitted_geometry_builders.hpp	
	${CMAKE_CURRENT_SOURCE_DIR}/linear_solver.hpp
PARENT_SCOPE)

