cmake_minimum_required(VERSION 3.0)
project(OpenCV_CPP_Project)

# Yêu cầu chuẩn C++ 11 trở lên
set(CMAKE_CXX_STANDARD 11)

# Tìm thư viện OpenCV đã cài ở Bước 1
find_package(OpenCV REQUIRED)

# Báo thư viện tìm được (để kiểm tra)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Khai báo file chạy (Executable)
# Thay 'main_cpp' bằng tên file exe bạn muốn
# Thay 'test.cpp' bằng tên file code của bạn (ví dụ erosion.cpp)
add_executable(main_cpp erosion.cpp)

# Liên kết (Link) thư viện OpenCV vào file chạy
target_link_libraries(main_cpp ${OpenCV_LIBS})