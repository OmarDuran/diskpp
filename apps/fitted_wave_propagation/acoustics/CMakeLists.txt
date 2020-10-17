
set(input_files ${input_files} 
	${CMAKE_CURRENT_SOURCE_DIR}/input_files/elliptic_convergence_test.lua
)

file(COPY ${input_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug/input_files/)
file(COPY ${input_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release/input_files/)

file(COPY ${input_poly_meshes} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug/meshes/)
file(COPY ${input_poly_meshes} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release/meshes/)

add_executable(acoustics acoustics.cpp ${fitted_waves_headers} ${fitted_waves_sources})
target_link_libraries(acoustics ${LINK_LIBS})
