REDIS_HOME = /glade/u/home/jedwards/sandboxes/cesm2_x_alpha/components/SmartSim/SmartRedis
REDIS_SRC = $(REDIS_HOME)/src/fortran/client.F90 \
	    $(REDIS_HOME)/src/fortran/dataset.F90 \
	    $(REDIS_HOME)/src/fortran/fortran_c_interop.F90

REDIS_OBJ = client.o dataset.o fortran_c_interop.o
MPIFC = mpif90


hello: hello.F90 $(REDIS_OBJ)
	$(MPIFC) $< -o $@ $(REDIS_OBJ) -L$(REDIS_HOME)/install/lib -lsmartredis -Wl,-rpath $(REDIS_HOME)/install/lib

%.o : $(REDIS_HOME)/src/fortran/%.F90
	$(MPIFC) $< -c -o $@ -I $(REDIS_HOME)/install/include


client.o: dataset.o
dataset.o: fortran_c_interop.o
