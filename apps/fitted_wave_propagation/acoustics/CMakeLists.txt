add_executable(acoustics acoustics.cpp ${fitted_waves_headers} ${fitted_waves_sources})
target_link_libraries(acoustics ${LINK_LIBS})
