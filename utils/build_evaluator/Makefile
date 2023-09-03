.PHONY: all clean build
.SUFFIXES: .o .cpp

PROJECT = build_evaluator

CXX = g++
LINK = ld
STRIP = strip

CXXFLAGS = -MMD -MP -Wall -std=c++20 -pedantic -O3
LDFLAGS = -fPIC -lyaml-cpp

OBJS = \
main.o \

all: build

compile: $(OBJS)

build: compile
	$(CXX) $(OBJS) -o $(PROJECT) $(LDFLAGS)
	$(STRIP) $(PROJECT)

clean:
	rm -f $(OBJS)

.cpp.o:
	$(CXX) $< -o $@ -c $(CXXFLAGS)
