add_executable(md_small_strain md_small_strain.cpp ${md_headers} ${md_sources})
target_link_libraries(md_small_strain ${LINK_LIBS})

file(COPY ${input_poly_meshes} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug/meshes/)
file(COPY ${input_poly_meshes} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release/meshes/)