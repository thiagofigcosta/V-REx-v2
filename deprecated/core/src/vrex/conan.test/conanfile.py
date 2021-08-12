from conans import ConanFile, CMake, tools
import os

class MongocxxdriverConan(ConanFile):
    name = "mongo-cxx-driver"
    version = "3.4.0"
    description = "C++ Driver for MongoDB"
    topics = ("conan", "libmongocxx", "mongodb")
    url = "http://github.com/huangminghuang/conan-mongo-cxx-driver"
    homepage = "https://github.com/mongodb/mongo-cxx-driver"
    author = "Huang-Ming Huang <huangh@objectcomputing.com>"
    license = "MIT"
    exports = ["LICENSE"]
    settings = "os", "compiler", "arch", "build_type"
    
    options = {"shared": [True, False]}
    default_options = "shared=False"
    
    requires = 'mongo-c-driver/1.13.0@huangminghuang/stable'
    generators = "cmake"
    no_copy_source = True
    build_policy = "missing"
    
    def configure(self):
        self.options["mongo-c-driver"].shared = self.options.shared
        
    def requirements(self):
        if self.settings.os == "Windows":
            self.requires("boost/[>= 1.67.0]@conan/stable")
            
    def source(self):
        tools.get("https://github.com/mongodb/mongo-cxx-driver/archive/r{0}.tar.gz".format(self.version))
        extracted_dir = "mongo-cxx-driver-r{0}".format(self.version)
        os.rename(extracted_dir, "sources")
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
        tools.replace_in_file("sources/CMakeLists.txt", "project(MONGO_CXX_DRIVER LANGUAGES CXX)",
                              '''project(MONGO_CXX_DRIVER LANGUAGES CXX)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')
        
        tools.replace_in_file("sources/src/mongocxx/CMakeLists.txt", "add_subdirectory(test)", "")
        tools.replace_in_file("sources/src/bsoncxx/CMakeLists.txt", "add_subdirectory(test)", "")
        
        # fix compiling problem on linux for clang++ with libc++
        tools.replace_in_file("sources/src/mongocxx/options/change_stream.cpp", "auto count", "std::int64_t count")
        
        tools.replace_in_file("sources/src/bsoncxx/cmake/libbsoncxx-config.cmake.in", "# We want to", '''
        
if (@BSONCXX_POLY_USE_BOOST@)
  find_package(Boost REQUIRED)
  set(LIBBSONCXX_INCLUDE_DIRS ${LIBBSONCXX_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
endif()

# We want to''')

        tools.replace_in_file("sources/src/bsoncxx/cmake/libbsoncxx-static-config.cmake.in", "# We want to", '''
        
if (@BSONCXX_POLY_USE_BOOST@)
  find_package(Boost REQUIRED)
  set(LIBBSONCXX_STATIC_INCLUDE_DIRS ${LIBBSONCXX_STATIC_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
endif()

# We want to''')

        if self.settings.os == "Windows":
            tools.replace_in_file("sources/src/bsoncxx/builder/core.cpp", "#include <bsoncxx/builder/core.hpp>", '''
#define _DISABLE_EXTENDED_ALIGNED_STORAGE
#include <bsoncxx/builder/core.hpp>''')
        
    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["BUILD_SHARED_LIBS"] = "ON" if self.options.shared else "OFF"
        cmake.configure(source_dir=os.path.join(self.source_folder,"sources"))
        return cmake

    def build(self):
        self._configure_cmake().build()

    def package(self):
        self.copy(pattern="LICENSE", dst="licenses", src="sources")
        CMake(self).install()

    def package_info(self):
        lib_suffix = "" if self.options.shared else "-static"

        libnames = ['mongocxx', 'bsoncxx']
        self.cpp_info.libs = [ "{}{}".format(name, lib_suffix) for name in libnames ]
        self.cpp_info.includedirs.extend( ['include/{}/v_noabi'.format(name) for name in libnames ] )
            
        if not self.options.shared:
            self.cpp_info.defines.extend(['BSONCXX_STATIC', 'MONGOCXX_STATIC'])