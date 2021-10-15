SMARTREDIS_FTN = $(SMARTREDIS_FSRC)/client.F90 \
		 $(SMARTREDIS_FSRC)/dataset.F90 \
		 $(SMARTREDIS_FSRC)/fortran_c_interop.F90

SMARTREDIS_OBJ = client.o dataset.o fortran_c_interop.o
MPIFC = mpif90

smartredis_put_get_3D: smartredis_put_get_3D.F90 $(SMARTREDIS_OBJ)
	$(MPIFC) $< -o $@ $(SMARTREDIS_OBJ) -L$(SMARTREDIS_LIB) -lsmartredis -Wl,-rpath $(SMARTREDIS_LIB)

%.o : $(SMARTREDIS_FSRC)/%.F90
	$(MPIFC) $< -c -o $@ -I $(SMARTREDIS_INCLUDE)


client.o: dataset.o
dataset.o: fortran_c_interop.o

clean:
	$(RM) *.o *.mod
