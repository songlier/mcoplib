#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nvbench::nvbench" for configuration "Release"
set_property(TARGET nvbench::nvbench APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvbench::nvbench PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnvbench.so"
  IMPORTED_SONAME_RELEASE "libnvbench.so"
  )

list(APPEND _cmake_import_check_targets nvbench::nvbench )
list(APPEND _cmake_import_check_files_for_nvbench::nvbench "${_IMPORT_PREFIX}/lib/libnvbench.so" )

# Import target "nvbench::main" for configuration "Release"
set_property(TARGET nvbench::main APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvbench::main PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_OBJECTS_RELEASE "${_IMPORT_PREFIX}/lib/objects-Release/nvbench.main/main.cu.o"
  )

list(APPEND _cmake_import_check_targets nvbench::main )
list(APPEND _cmake_import_check_files_for_nvbench::main "${_IMPORT_PREFIX}/lib/objects-Release/nvbench.main/main.cu.o" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
