cmake_minimum_required(VERSION 3.24)

project(Stockfish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(STOCKFISH_ENABLE_WARNINGS "Enable the Stockfish warning set" ON)
option(STOCKFISH_ENABLE_GLIBCXX_DEBUG "Enable libstdc++ debug checks in Debug builds" ON)
option(STOCKFISH_ENABLE_IPO "Enable interprocedural optimization in release-like builds" OFF)
set(STOCKFISH_ARCH "native" CACHE STRING "Stockfish architecture preset")
set_property(
  CACHE STOCKFISH_ARCH
  PROPERTY STRINGS
           native
           x86-64-avx512icl
           x86-64-vnni512
           x86-64-avx512
           x86-64-avxvnni
           x86-64-bmi2
           x86-64-avx2
           x86-64-sse41-popcnt
           x86-64-ssse3
           x86-64-sse3-popcnt
           x86-64
           x86-32-sse41-popcnt
           x86-32-sse2
           x86-32
           armv7
           armv7-neon
           armv8
           armv8-dotprod
           apple-silicon
           general-64
           general-32
           riscv64
           loongarch64
           loongarch64-lsx
           loongarch64-lasx)

set(STOCKFISH_SOURCES
    src/benchmark.cpp
    src/bitboard.cpp
    src/engine.cpp
    src/evaluate.cpp
    src/main.cpp
    src/memory.cpp
    src/misc.cpp
    src/movegen.cpp
    src/movepick.cpp
    src/nnue/features/full_threats.cpp
    src/nnue/features/half_ka_v2_hm.cpp
    src/nnue/network.cpp
    src/nnue/nnue_accumulator.cpp
    src/nnue/nnue_misc.cpp
    src/position.cpp
    src/score.cpp
    src/search.cpp
    src/syzygy/tbprobe.cpp
    src/thread.cpp
    src/timeman.cpp
    src/tt.cpp
    src/tune.cpp
    src/uci.cpp
    src/ucioption.cpp)

add_executable(stockfish ${STOCKFISH_SOURCES})
target_include_directories(stockfish PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_options(stockfish PRIVATE -fno-exceptions)

if(STOCKFISH_ENABLE_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
      stockfish
      PRIVATE
        -Wall
        -Wcast-qual
        -pedantic
        -Wextra
        -Wshadow)
  endif()
endif()

target_compile_definitions(stockfish PRIVATE USE_PTHREADS)

find_package(Threads REQUIRED)
target_link_libraries(stockfish PRIVATE Threads::Threads)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  target_compile_definitions(stockfish PRIVATE IS_64BIT)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(stockfish PRIVATE rt)
endif()

if(STOCKFISH_ENABLE_GLIBCXX_DEBUG)
  target_compile_definitions(
    stockfish
    PRIVATE
      $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
      $<$<CONFIG:Debug>:_GLIBCXX_DEBUG>)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(
    stockfish
    PRIVATE
      $<$<CONFIG:Debug>:-O0;-g3>
      $<$<CONFIG:Release>:-O3;-funroll-loops>
      $<$<CONFIG:RelWithDebInfo>:-O2;-g>)
endif()

include(CheckIPOSupported)
if(STOCKFISH_ENABLE_IPO)
  check_ipo_supported(RESULT stockfish_ipo_supported OUTPUT stockfish_ipo_error)
  if(stockfish_ipo_supported)
    set_property(TARGET stockfish PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "IPO/LTO unavailable: ${stockfish_ipo_error}")
  endif()
endif()

function(stockfish_apply_arch target arch_name)
  set(defs)
  set(opts)

  if(arch_name STREQUAL "general-32")
    set(opts -m32)
  elseif(arch_name STREQUAL "general-64")
    set(defs IS_64BIT)
  elseif(arch_name STREQUAL "x86-32")
    set(opts -m32 -mmmx)
  elseif(arch_name STREQUAL "x86-32-sse2")
    set(defs USE_SSE2)
    set(opts -m32 -msse -msse2)
  elseif(arch_name STREQUAL "x86-32-sse41-popcnt")
    set(defs USE_SSE2 USE_SSSE3 USE_SSE41 USE_POPCNT)
    set(opts -m32 -msse -msse2 -mssse3 -msse4.1 -msse3 -mpopcnt)
  elseif(arch_name STREQUAL "x86-64")
    set(defs USE_SSE2 IS_64BIT)
    set(opts -msse -msse2)
  elseif(arch_name STREQUAL "x86-64-sse3-popcnt")
    set(defs USE_SSE2 USE_POPCNT IS_64BIT)
    set(opts -msse -msse2 -msse3 -mpopcnt)
  elseif(arch_name STREQUAL "x86-64-ssse3")
    set(defs USE_SSE2 USE_SSSE3 IS_64BIT)
    set(opts -msse -msse2 -mssse3)
  elseif(arch_name STREQUAL "x86-64-sse41-popcnt")
    set(defs USE_SSE2 USE_SSSE3 USE_SSE41 USE_POPCNT IS_64BIT)
    set(opts -msse -msse2 -mssse3 -msse4.1 -msse3 -mpopcnt)
  elseif(arch_name STREQUAL "x86-64-avx2")
    set(defs USE_SSE2 USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 IS_64BIT)
    set(opts -msse -msse2 -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi)
  elseif(arch_name STREQUAL "x86-64-bmi2")
    set(defs USE_SSE2 USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_PEXT IS_64BIT)
    set(opts -msse -msse2 -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mbmi2)
  elseif(arch_name STREQUAL "x86-64-avxvnni")
    set(
      defs
      USE_SSE2
      USE_SSSE3
      USE_SSE41
      USE_POPCNT
      USE_AVX2
      USE_PEXT
      USE_VNNI
      USE_AVXVNNI
      IS_64BIT)
    set(opts -msse -msse2 -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mbmi2 -mavxvnni)
  elseif(arch_name STREQUAL "x86-64-avx512")
    set(defs USE_SSE2 USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_PEXT USE_AVX512 IS_64BIT)
    set(
      opts
      -msse
      -msse2
      -mssse3
      -msse4.1
      -msse3
      -mpopcnt
      -mavx2
      -mbmi
      -mbmi2
      -mavx512f
      -mavx512bw
      -mavx512dq
      -mavx512vl)
  elseif(arch_name STREQUAL "x86-64-vnni512")
    set(
      defs
      USE_SSE2
      USE_SSSE3
      USE_SSE41
      USE_POPCNT
      USE_AVX2
      USE_PEXT
      USE_AVX512
      USE_VNNI
      IS_64BIT)
    set(
      opts
      -msse
      -msse2
      -mssse3
      -msse4.1
      -msse3
      -mpopcnt
      -mavx2
      -mbmi
      -mbmi2
      -mavx512f
      -mavx512bw
      -mavx512dq
      -mavx512vl
      -mavx512vnni)
  elseif(arch_name STREQUAL "x86-64-avx512icl")
    set(
      defs
      USE_SSE2
      USE_SSSE3
      USE_SSE41
      USE_POPCNT
      USE_AVX2
      USE_PEXT
      USE_AVX512
      USE_VNNI
      USE_AVX512ICL
      IS_64BIT)
    set(
      opts
      -msse
      -msse2
      -mssse3
      -msse4.1
      -msse3
      -mpopcnt
      -mavx2
      -mbmi
      -mbmi2
      -mavx512f
      -mavx512cd
      -mavx512vl
      -mavx512dq
      -mavx512bw
      -mavx512ifma
      -mavx512vbmi
      -mavx512vbmi2
      -mavx512vpopcntdq
      -mavx512bitalg
      -mavx512vnni
      -mvpclmulqdq
      -mgfni
      -mvaes)
  elseif(arch_name STREQUAL "armv7")
    set(opts -m32)
  elseif(arch_name STREQUAL "armv7-neon")
    set(defs USE_POPCNT USE_NEON=7)
    set(opts -m32)
  elseif(arch_name STREQUAL "armv8")
    set(defs USE_POPCNT USE_NEON=8 IS_64BIT)
  elseif(arch_name STREQUAL "armv8-dotprod")
    set(defs USE_POPCNT USE_NEON=8 USE_NEON_DOTPROD IS_64BIT)
    set(opts -march=armv8.2-a+dotprod)
  elseif(arch_name STREQUAL "apple-silicon")
    set(defs USE_POPCNT USE_NEON=8 USE_NEON_DOTPROD IS_64BIT)
    set(opts -march=armv8.2-a+dotprod)
  elseif(arch_name STREQUAL "riscv64")
    set(defs IS_64BIT)
  elseif(arch_name STREQUAL "loongarch64")
    set(defs IS_64BIT)
  elseif(arch_name STREQUAL "loongarch64-lsx")
    set(defs IS_64BIT)
    set(opts -mlsx)
  elseif(arch_name STREQUAL "loongarch64-lasx")
    set(defs IS_64BIT)
    set(opts -mlsx -mlasx)
  else()
    message(FATAL_ERROR "Unsupported STOCKFISH_ARCH='${arch_name}'")
  endif()

  if(defs)
    target_compile_definitions(${target} PRIVATE ${defs})
  endif()
  if(opts)
    target_compile_options(${target} PRIVATE ${opts})
    target_link_options(${target} PRIVATE ${opts})
  endif()
endfunction()

set(STOCKFISH_EFFECTIVE_ARCH "${STOCKFISH_ARCH}")
if(STOCKFISH_ARCH STREQUAL "native")
  execute_process(
    COMMAND sh "${PROJECT_SOURCE_DIR}/scripts/get_native_properties.sh"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src"
    OUTPUT_VARIABLE STOCKFISH_NATIVE_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(STOCKFISH_NATIVE_ARCH)
    set(STOCKFISH_EFFECTIVE_ARCH "${STOCKFISH_NATIVE_ARCH}")
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(STOCKFISH_EFFECTIVE_ARCH "general-64")
  else()
    set(STOCKFISH_EFFECTIVE_ARCH "general-32")
  endif()
endif()

stockfish_apply_arch(stockfish "${STOCKFISH_EFFECTIVE_ARCH}")

execute_process(
  COMMAND git rev-parse --short=8 HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE STOCKFISH_GIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(STOCKFISH_GIT_SHA)
  target_compile_definitions(stockfish PRIVATE GIT_SHA=${STOCKFISH_GIT_SHA})
endif()

execute_process(
  COMMAND git show -s --date=format:%Y%m%d --format=%cd HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE STOCKFISH_GIT_DATE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(STOCKFISH_GIT_DATE)
  target_compile_definitions(stockfish PRIVATE GIT_DATE=${STOCKFISH_GIT_DATE})
endif()

target_compile_definitions(stockfish PRIVATE ARCH=${STOCKFISH_EFFECTIVE_ARCH})

message(STATUS "Stockfish architecture preset: ${STOCKFISH_ARCH}")
message(STATUS "Stockfish effective architecture: ${STOCKFISH_EFFECTIVE_ARCH}")
