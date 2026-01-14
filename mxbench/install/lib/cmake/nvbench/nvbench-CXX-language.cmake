# Enable the requested language, which is only supported
# in the highest directory that 'uses' a language.
# We have to presume all directories use a language
# since linking to a target with language standards
# means `using`

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  if(NOT DEFINED CMAKE_CURRENT_FUNCTION)
    #Can't be called inside a function
    enable_language(CXX)
    return()
  endif()
endif()

# If we aren't in the highest directory we need to hoist up
# all the language information to trick CMake into thinking
# the correct things happened.
# `cmake_langauge(DEFER )` doesn't support calling `enable_language`
# so we have to emulate what it does.
#
# So what we need to do is the following:
#
# For non-root directories:
#   1. Transform each `set` in CMakeCXXCompiler to be a `PARENT_SCOPE`
#      This allows us to propagate up immediate information that is
#      used by commands such target_compile_features.
#
#   2. Include `CMakeCXXInformation` this can't be deferred as the contents
#      are required if any target is constructed
#
# For root directories we only need to include `CMakeCXXInformation`
#

# Expose the language at the current scope
enable_language(CXX)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/PropagateCMakeCXXCompiler.cmake")
  # 1.
  # Take everything that `enable_language` generated and transform all sets to PARENT_SCOPE ()
  # This will allow our parent directory to be able to call CMakeCXXInformation
  file(STRINGS "${CMAKE_BINARY_DIR}/CMakeFiles/${CMAKE_VERSION}/CMakeCXXCompiler.cmake" rapids_code_to_transform)
  set(rapids_code_to_execute "if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)\n")
  foreach( line IN LISTS rapids_code_to_transform)
    if(line MATCHES "[ ]*set")
      string(REPLACE ")" " PARENT_SCOPE)" line "${line}")
    endif()
    string(APPEND rapids_code_to_execute "${line}\n")
  endforeach()
  string(APPEND rapids_code_to_execute "endif()\n")

  # 2.
  # Make sure we call "CMakeCXXInformation" for the current directory
  string(APPEND rapids_code_to_execute "include(CMakeCXXInformation)\n")

  file(WRITE "${CMAKE_BINARY_DIR}/cmake/PropagateCMakeCXXCompiler.cmake" "${rapids_code_to_execute}")
  unset(rapids_code_to_execute)
  unset(rapids_code_to_transform)
endif()

# propagate up one parent_scope
include("${CMAKE_BINARY_DIR}/cmake/PropagateCMakeCXXCompiler.cmake")

# Compute all directories between here and the root of the project
# - Each directory but the root needs to include `PropagateCMakeCXXCompiler.cmake`
# - Since the root directory doesn't have a parent it only needs to include
#   `CMakeCXXInformation`
set(rapids_directories "${CMAKE_CURRENT_SOURCE_DIR}")
get_directory_property(parent_dir DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_DIRECTORY)
while(parent_dir)
  list(APPEND rapids_directories "${parent_dir}")
  get_directory_property(parent_dir DIRECTORY "${parent_dir}" PARENT_DIRECTORY)
endwhile()

foreach(rapids_directory IN LISTS rapids_directories)
  # Make sure we haven't already installed a language hook for this directory
  # Once we found a directory with an existing hook we can safely stop
  # as that means hooks exist from that point up in the graph
  cmake_language(DEFER DIRECTORY "${rapids_directory}" GET_CALL_IDS rapids_existing_calls)
  if(NOT rapids_CXX_hook IN_LIST rapids_existing_calls)
    cmake_language(DEFER DIRECTORY "${rapids_directory}"
                   ID rapids_CXX_hook
                   CALL include "${CMAKE_BINARY_DIR}/cmake/PropagateCMakeCXXCompiler.cmake")
  else()
    break()
  endif()
endforeach()

unset(rapids_existing_calls)
unset(rapids_directories)
unset(rapids_root_directory)
