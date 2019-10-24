# for Unicode

OLD := MyMM4x4_14
NEW := MyMM4x4_15

CC := icc
CFLAGS := -g -O2 -msse3 -mavx -mavx2
LDFLAGS := -lm


.c .o:
	@$(CC) -c $(CFLAGS) $<
.c .o:
	@$(CC) -c $(CFLAGS) $<

run: MyMM.o MyMMult.o
	$(CC) $(CFLAGS) $(LDFLAGS) MyMM.o MyMMult.o -o run
	@rm MyMMult.o 
	@rm MyMM.o


	# @echo "" > ./figs/output_$(NEW).m 
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	# @./run >> ./figs/output_$(NEW).m 
	# @python plot.py $(OLD) $(NEW)


clean:
	rm -f *.o
