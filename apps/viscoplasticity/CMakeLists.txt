#add_executable(viscoplasticity_constants viscoplasticity_vector_k0.cpp)
#target_link_libraries(viscoplasticity_constants ${LINK_LIBS})

add_executable(viscoplasticity_vector viscoplasticity_vector.cpp)
target_link_libraries(viscoplasticity_vector ${LINK_LIBS})

add_executable(viscoplasticity_scalar viscoplasticity_scalar.cpp)
target_link_libraries(viscoplasticity_scalar ${LINK_LIBS})

#add_executable(viscoplasticity_vector_old viscoplasticity_vector_old.cpp)
#target_link_libraries(viscoplasticity_vector_old ${LINK_LIBS})
