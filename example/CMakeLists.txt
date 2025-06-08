add_executable(cu_exception cu_exception.cu)
target_link_libraries(cu_exception cu_utility)

add_executable(cu_trace_mem cu_trace_mem.cu)
target_link_libraries(cu_trace_mem cu_utility)

add_executable(cu_ndarray cu_ndarray.cu)
target_link_libraries(cu_ndarray cu_utility)

add_executable(cu_ndarray2 cu_ndarray2.cu)
target_link_libraries(cu_ndarray2 cu_utility)

add_executable(cu_mem_alloc cu_mem_alloc.cu)
target_link_libraries(cu_mem_alloc cu_utility)

add_executable(ndslice ndslice.cu)
target_link_libraries(ndslice cu_utility)

add_executable(cu_fancy_copy cu_fancy_copy.cu)
target_link_libraries(cu_fancy_copy cu_utility)

add_executable(cu_unified_ndarray cu_unified_ndarray.cu)
target_link_libraries(cu_unified_ndarray cu_utility)