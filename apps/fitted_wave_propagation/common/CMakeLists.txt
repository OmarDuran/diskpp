
set(fitted_waves_sources ${fitted_waves_sources} 
	#${CMAKE_CURRENT_SOURCE_DIR}/fitted_geometry_builders.cpp
PARENT_SCOPE)

set(fitted_waves_headers ${fitted_waves_headers} 
	${CMAKE_CURRENT_SOURCE_DIR}/display_settings.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/preprocessor.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/postprocessor.hpp	
	${CMAKE_CURRENT_SOURCE_DIR}/scal_analytic_functions.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/fitted_geometry_builders.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/one_field_assembler.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/two_fields_assembler.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/one_field_vectorial_assembler.hpp
PARENT_SCOPE)
