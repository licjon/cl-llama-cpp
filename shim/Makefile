LLAMA_CPP_DIR   ?= ../llama.cpp
LLAMA_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build/bin

CXX      ?= c++
CXXFLAGS  = -std=c++17 -O2 -fPIC
INCLUDES  = -I$(LLAMA_CPP_DIR)/common -I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include -I$(LLAMA_CPP_DIR)/vendor
LDFLAGS   = -L$(LLAMA_BUILD_DIR) -lllama-common -lllama -Wl,-rpath,'$$ORIGIN'

SRCS     := $(wildcard *.cpp)
OBJS     := $(SRCS:.cpp=.o)
TARGET    = $(LLAMA_BUILD_DIR)/libllama-common-shim.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
