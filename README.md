# cu_utility

# Installation
We can use the following commands to install cu_utility.
```
git clone https://github.com/gsy1234567/cu_utility.git
cd cu_utility
mkdir build
cd build
cmake ..
cmake --install ./ --prefix /path/to/install
```
Let cmake to find our library, you can use the following code in cmake.
```
set(cu_utility_DIR /path/to/install/lib/cmake/cu_utility)
find_package(cu_utility REQUIRED)
```
Use the following code to link our library.
```
target_link_libraries(your_target cu_utility::cu_utility)
```
