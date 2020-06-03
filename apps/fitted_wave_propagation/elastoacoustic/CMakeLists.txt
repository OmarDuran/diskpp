add_executable(elastoacoustic elastoacoustic.cpp ${fitted_waves_headers} ${fitted_waves_sources})
target_link_libraries(elastoacoustic ${LINK_LIBS})
