cmake_minimum_required(VERSION 3.19)
project(alphaFM)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/Frame)  # Add the given directories to those the compiler, uses to search for include files
include_directories(src/FTRL)
include_directories(src/Lock)
include_directories(src/Mem)
include_directories(src/Sample)
include_directories(src/Utils)


add_definitions("-Wall -g")

add_executable(alphaFM
        src/Frame/pc_frame.cpp
        src/Utils/utils.cpp
        fm_train.cpp)

target_link_libraries(alphaFM pthread)

# 设置输出的target的属性。可以设置任何kv
set_target_properties(alphaFM
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin   # 设置target的输出文件夹.  如其他类型的target，可设置其他属性
)

# build文件夹外：
# cmake . -B build
# 真正build：
# cmake --build build 


# vscode中： 利用cmake tools插件,调试c++项目
# 1 首先点击cmake插件中Configure, 配置build所需内容，也建了build目录  (通过task.json，运行cmake)
# 2 再点击cmake插件中Build, 构建待调试的可执行文件                   
# 3 点击左边大的Run，里边的gdb Launch进行调试                       （传参等通过launch.json文件）
#   如果直接点cmake插件中生成的可执行文件debug,没法传参。


# 每次修改cmakeLIst,会自动configure。 只需要重新build即可