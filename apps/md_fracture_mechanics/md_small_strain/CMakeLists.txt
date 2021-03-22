add_executable(md_small_strain md_small_strain.cpp ${md_headers} ${md_sources})
target_link_libraries(md_small_strain ${LINK_LIBS})
