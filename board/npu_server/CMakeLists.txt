# Built as part of examples/qualcomm (see ../steps/build_npu_server.sh): same link set as
# qnn_executor_runner, plus the Module and Tensor extensions.
add_executable(npu_server ${CMAKE_CURRENT_LIST_DIR}/npu_server.cpp)
target_include_directories(npu_server PUBLIC ${_common_include_directories})
target_compile_features(npu_server PRIVATE cxx_std_17)
target_link_libraries(
  npu_server
  qnn_executorch_backend
  executorch_core
  extension_module_static
  extension_tensor
  extension_data_loader
  extension_flat_tensor
  extension_named_data_map
  full_portable_ops_lib
  quantized_ops_lib
)
set_target_properties(npu_server PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
