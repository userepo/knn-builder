# knn-gpu's C++ shim over cuVS's CAGRA build, as a static library its
# build script links. find_package(cuvs) brings the include paths, CCCL
# copy and definitions libcuvs was built with.
cmake_minimum_required(VERSION 3.30.4)
project(knn_cagra LANGUAGES CXX)

find_package(cuvs 26.08.01 EXACT REQUIRED)

add_library(knn_cagra STATIC knn_cagra.cpp)
target_link_libraries(knn_cagra PRIVATE cuvs::cuvs)
set_target_properties(
  knn_cagra
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             POSITION_INDEPENDENT_CODE ON
)
install(TARGETS knn_cagra ARCHIVE DESTINATION lib)
