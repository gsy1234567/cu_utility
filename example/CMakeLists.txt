add_executable(cu_exception cu_exception.cu)
target_link_libraries(cu_exception cu_utility)

add_executable(cu_trace_mem cu_trace_mem.cu)
target_link_libraries(cu_trace_mem cu_utility)

add_executable(cu_ndarray cu_ndarray.cu)
target_link_libraries(cu_ndarray cu_utility)
