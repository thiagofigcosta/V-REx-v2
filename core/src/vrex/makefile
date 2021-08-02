FORMAT=.cpp
ifeq ($(g++-9 -v &>/dev/null && echo g++-9),g++-9)
CXX=g++-9
else 
CXX=g++
endif
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
SRC=$(wildcard *$(FORMAT)) $(wildcard slide/*$(FORMAT))
OBJS=$(wildcard obj/*.o)
EXEC=bin/vrex
CFLAGS=-c --std=c++11 -Wall -fopenmp -g
INC_FLAGS=-I /usr/local/include/mongocxx/v_noabi/ -I /usr/local/include/bsoncxx/v_noabi/ -I /usr/local/include/libmongoc-1.0 -I /usr/local/include/libbson-1.0 -D BOOST_STACKTRACE_USE_BACKTRACE
LDFLAGS=-L /usr/local/lib -L /usr/lib -lmongocxx -lbsoncxx -ldl -lbacktrace -lboost_system -lboost_filesystem -lboost_program_options -fPIC -fopenmp 
#OPT_FLAGS= -fno-strict-aliasing -O2 #-march=skylake -mtune=intel -mavx2 -mavx512f
OPT_FLAGS=-fno-strict-aliasing -O3 -mtune=intel -mavx2

all: compileLinkAndRun

.SILENT compile:
	$(CXX) $(CFLAGS) $(INC_FLAGS) $(OPT_FLAGS) $(SRC)
	if [ ! -d "obj" ]; then mkdir obj; fi
	@mv *.o obj/

link:
	if [ ! -d "bin" ]; then mkdir bin; fi
	$(CXX) $(OBJS) -o $(EXEC) $(LDFLAGS)

run:
	./$(EXEC)

# to run type > r
# to add break point type > break full_file_path/file.cpp:line_number
# to get stack trace type > where
# to get stack trace type > backtrace
debug: 
	gdb ./$(EXEC)

profile:
	valgrind --leak-check=full --track-origins=yes $(EXEC)

compileAndLink: compile link

compileLinkAndRun: compile link run

linkAndRun: link run

compileLinkAndDebug: compile link debug

compileLinkAndProfile: compile link profile

clean:
	rm -rf *.o obj/*.o $(EXEC)